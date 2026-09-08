"""
Entity profile and inference API endpoints.

Endpoints:
- POST /v1/entities/{entity_id}/derive - Trigger inference derivation for an entity
- GET /v1/entities/{entity_id}/card - Get cached entity profile card
- GET /v1/entities/{entity_id}/insights - Get derived insights for an entity
"""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from scitrera_app_framework import Plugin, Variables, ext_parse_bool

from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from ...config import DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED, MEMORYLAYER_ENTITY_REGISTRY_ENABLED
from ...models import DetailLevel, ReflectInput
from ...models.entity_registry import EntityProvenance, EntityType
from ...services.entity_registry.provenance import summarize_entity_provenance
from ...services.audit import AuditEvent, AuditService
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.cache import CacheService
from ...services.entity_registry import EntityRegistryService, get_entity_registry_service
from ...services.inference import DefaultInferenceService
from ...services.memory import MemoryService
from ...services.reflect import ReflectService
from .. import EXT_MULTI_API_ROUTERS
from .deps import (
    get_audit_service,
    get_auth_service,
    get_authz_service,
    get_cache_service,
    get_inference_service,
    get_memory_service,
    get_reflect_service,
)
from .schemas import (
    EntityBackfillResponse,
    EntityCardResponse,
    EntityDedupeResponse,
    EntityDeriveRequest,
    EntityDeriveResponse,
    EntityEnrichResponse,
    EntityInsightsResponse,
    EntityListResponse,
    EntityMergeRequest,
    EntityResolveResponse,
    EntityResponse,
    ErrorResponse,
    RelatedEntitiesResponse,
    SeedEntitiesRequest,
    SeedEntitiesResponse,
)

router = APIRouter(prefix="/v1/entities", tags=["entities"])

# Cache TTL for entity cards (5 minutes)
ENTITY_CARD_CACHE_TTL = 300


def _card_cache_key(workspace_id: str, entity_id: str) -> str:
    return f"entity_card:{workspace_id}:{entity_id}"


def get_registry_service(v: Variables = Depends(get_variables_dep)) -> EntityRegistryService:
    """Get the entity registry service instance for FastAPI dependency injection."""
    return get_entity_registry_service(v)


def _require_registry_enabled(v: Variables) -> None:
    """Guard the registry CRUD surface behind MEMORYLAYER_ENTITY_REGISTRY_ENABLED.

    The registry ships DARK (default OFF). When disabled, the CRUD endpoints
    return 501 (not implemented/available) — mirroring how the rest of the
    codebase treats the registry as an opt-in capability.
    """
    enabled = v.environ(
        MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
        default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
        type_fn=ext_parse_bool,
    )
    if not enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Entity registry is not enabled (set MEMORYLAYER_ENTITY_REGISTRY_ENABLED=true)",
        )


@router.get(
    "",
    response_model=EntityListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_entities(
    http_request: Request,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to _default)"),
    status_filter: str = Query("active", alias="status", description="Entity status filter: 'active' or 'merged'"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum entities to return"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityListResponse:
    """List canonical entities in a workspace (deterministic order by id)."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        logger.debug("Listing entities in workspace: %s (status=%s, limit=%d)", workspace_id, status_filter, limit)

        entities = await registry_service.list_entities(workspace_id, status=status_filter, limit=limit)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="list",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    metadata={"count": len(entities)},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity list")
        return EntityListResponse(entities=entities, total_count=len(entities))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list entities: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list entities")


@router.get(
    "/resolve",
    response_model=EntityResolveResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "No entity matched the given name"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def resolve_entity(
    http_request: Request,
    name: str = Query(..., description="Surface name to resolve (never created)", min_length=1),
    entity_type: EntityType = Query(EntityType.PERSON, description="Entity type to resolve against"),
    workspace_id: str | None = Query(None, description="Workspace override (defaults to _default)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityResolveResponse:
    """Resolve a surface name/alias to an existing canonical entity (never creates)."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        logger.debug("Resolving entity name=%s type=%s in workspace: %s", name, entity_type, workspace_id)

        try:
            resolution = await registry_service.resolve(
                workspace_id,
                name,
                entity_type,
                allow_create=False,
            )
        except LookupError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=resolution.entity.id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity resolve")
        return EntityResolveResponse(resolution=resolution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve entity %s: %s", name, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to resolve entity")


@router.post(
    "/merge",
    response_model=EntityResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Source or target entity not found"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def merge_entities(
    http_request: Request,
    request: EntityMergeRequest,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to _default)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityResponse:
    """Merge ``source_id`` into ``target_id``; returns the surviving target entity."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        logger.info("Merging entity %s -> %s in workspace: %s", request.source_id, request.target_id, workspace_id)

        try:
            target = await registry_service.merge(
                workspace_id,
                request.source_id,
                request.target_id,
                reason=request.reason,
            )
        except LookupError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=request.target_id,
                    metadata={"merged_from": request.source_id},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity merge")
        return EntityResponse(entity=target)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid entity merge request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to merge entities %s -> %s: %s", request.source_id, request.target_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to merge entities")


@router.post(
    "/backfill",
    response_model=EntityBackfillResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def backfill_entities(
    http_request: Request,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to auth ctx)"),
    batch_size: int = Query(200, ge=1, le=2000, description="Memories enumerated per page"),
    max_memories: int | None = Query(None, ge=1, description="Optional cap on memories processed this call"),
    reset: bool = Query(False, description="Wipe the workspace's existing entities first (pristine rebuild)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityBackfillResponse:
    """Replay entity extraction + accretion over the workspace's EXISTING memories.

    Populates the registry from history for workspaces whose memories predate the
    registry being enabled. Idempotent (safe to re-run); no-op when the registry is
    disabled. Runs inline — for very large workspaces prefer paging via ``max_memories``.
    """
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        logger.info(
            "Entity-registry backfill requested for workspace=%s (batch_size=%d, max=%s, reset=%s)",
            workspace_id, batch_size, max_memories, reset,
        )

        result = await memory_service.backfill_entity_registry(
            workspace_id,
            batch_size=batch_size,
            max_memories=max_memories,
            reset=reset,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    metadata={"backfill": True, **result},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity backfill")
        return EntityBackfillResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to backfill entities for %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to backfill entities")


@router.post(
    "/seed",
    response_model=SeedEntitiesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def seed_entities(
    http_request: Request,
    request: SeedEntitiesRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> SeedEntitiesResponse:
    """Seed a curated catalog of canonical entities into the registry (create-or-update).

    High-precision pre-canonicalization: later extracted mentions attach to a seed
    (and its aliases/description) instead of fragmenting. Each entity_type is
    validated against the ontology entity-type vocabulary; unknown types are skipped
    and reported. Idempotent.
    """
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        # Validate entity types against the ontology vocabulary (the same authority
        # the extractor uses). If the ontology can't be resolved, accept all types.
        ontology = None
        try:
            from ...services.ontology import EXT_ONTOLOGY_SERVICE
            from scitrera_app_framework import get_extension

            ontology = get_extension(EXT_ONTOLOGY_SERVICE, v)
        except Exception:
            logger.debug("OntologyService unavailable for entity seeding; skipping type validation")

        valid: list = []
        skipped_invalid_type: list[str] = []
        for e in request.entities:
            if ontology is not None and not ontology.validate_entity_type(e.entity_type, ctx.tenant_id, workspace_id):
                skipped_invalid_type.append(e.name)
            else:
                valid.append(e)

        result = await registry_service.seed_entities(workspace_id, valid)

        logger.info(
            "Entity seed for workspace=%s: seeded=%d, failed=%d, invalid_type=%d",
            workspace_id, result["seeded"], len(result["failed"]), len(skipped_invalid_type),
        )
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="create",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    metadata={"seed": True, "seeded": result["seeded"]},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity seed")

        return SeedEntitiesResponse(
            workspace_id=workspace_id,
            seeded=result["seeded"],
            failed=result["failed"],
            skipped_invalid_type=skipped_invalid_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to seed entities for %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to seed entities")


@router.post(
    "/enrich",
    response_model=EntityEnrichResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def enrich_entities(
    http_request: Request,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to auth ctx)"),
    types: str = Query("org,person,place", description="CSV of entity types eligible for linking"),
    limit: int = Query(500, ge=1, le=5000, description="Max entities to consider"),
    overwrite: bool = Query(False, description="Re-link entities that already carry the source link"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityEnrichResponse:
    """Link canonical entities to an external KB (Wikidata) and fold ids into provenance.

    Uses the configured entity-linker provider (no-op unless
    MEMORYLAYER_ENTITY_LINKER_PROVIDER=wikidata). Restricted to ``types`` (default
    org/person/place, where external coverage is good). Idempotent + best-effort.
    """
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        from ...services.entity_linker import get_entity_linker_service

        linker = get_entity_linker_service(v)  # resolves to the no-op default unless configured
        eligible = {t.strip() for t in types.split(",") if t.strip()} or None
        source = "wikidata"

        result = await registry_service.enrich_entities(
            workspace_id, linker, limit=limit, eligible_types=eligible, source=source, overwrite=overwrite
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    metadata={"enrich": source, **result},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity enrich")

        return EntityEnrichResponse(
            workspace_id=workspace_id,
            source=source,
            checked=result["checked"],
            enriched=result["enriched"],
            skipped=result["skipped"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to enrich entities for %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to enrich entities")


@router.post(
    "/dedupe",
    response_model=EntityDedupeResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Registry backend does not support dedupe"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def dedupe_entities(
    http_request: Request,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to auth ctx)"),
    threshold: float | None = Query(None, ge=0.0, le=1.0, description="Similarity threshold (default: resolve-time high threshold)"),
    limit: int = Query(1000, ge=1, le=10000, description="Max entities to consider"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityDedupeResponse:
    """Batch-dedupe: cluster transitively-similar entities and merge each cluster into
    its most-established representative. Requires an embedding-capable registry."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        try:
            result = await registry_service.dedupe_workspace(workspace_id, threshold=threshold, limit=limit)
        except NotImplementedError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="The active entity registry backend does not support dedupe (needs name embeddings)",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    metadata={"dedupe": True, **result},
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity dedupe")

        return EntityDedupeResponse(
            workspace_id=workspace_id, clusters=result["clusters"], merged=result["merged"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to dedupe entities for %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to dedupe entities")


@router.get(
    "/{entity_id}",
    response_model=EntityResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Entity not found"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_entity(
    http_request: Request,
    entity_id: str,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to _default)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityResponse:
    """Get a single canonical entity by id."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        logger.debug("Getting entity: %s in workspace: %s", entity_id, workspace_id)

        entity = await registry_service.get(workspace_id, entity_id)
        if entity is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Entity not found: {entity_id}")

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=entity_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity get")
        return EntityResponse(entity=entity)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get entity")


@router.get(
    "/{entity_id}/provenance",
    response_model=EntityProvenance,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Entity not found"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_entity_provenance(
    http_request: Request,
    entity_id: str,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to auth ctx)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EntityProvenance:
    """Return the normalized lineage of an entity: how it originated, which memories
    first surfaced it, and its external-KB links (Wikidata etc.)."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        entity = await registry_service.get(workspace_id, entity_id)
        if entity is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Entity not found: {entity_id}")

        prov = summarize_entity_provenance(entity)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=entity_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity provenance read")
        return prov

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity provenance %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get entity provenance")


@router.get(
    "/{entity_id}/related",
    response_model=RelatedEntitiesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        501: {"model": ErrorResponse, "description": "Entity registry not enabled"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_related_entities(
    http_request: Request,
    entity_id: str,
    workspace_id: str | None = Query(None, description="Workspace override (defaults to auth ctx)"),
    limit: int = Query(10, ge=1, le=100, description="Max related entities to return"),
    min_shared: int = Query(1, ge=1, description="Minimum shared member memories"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    registry_service: EntityRegistryService = Depends(get_registry_service),
    audit_service: AuditService = Depends(get_audit_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> RelatedEntitiesResponse:
    """Return the entity's co-occurrence neighborhood — entities mentioned together
    in shared member memories, ranked by overlap."""
    try:
        _require_registry_enabled(v)
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        related = await registry_service.related_entities(
            workspace_id, entity_id, limit=limit, min_shared=min_shared
        )
        return RelatedEntitiesResponse(entity_id=entity_id, related=related)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get related entities for %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get related entities")


@router.post(
    "/{entity_id}/derive",
    response_model=EntityDeriveResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def derive_entity_insights(
    http_request: Request,
    entity_id: str,
    request: EntityDeriveRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    inference_service: DefaultInferenceService = Depends(get_inference_service),
    cache_service: CacheService = Depends(get_cache_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> EntityDeriveResponse:
    """
    Trigger inference derivation for an entity.

    Analyzes all memories about the entity and derives higher-order insights
    (patterns, preferences, tendencies) that are stored as INFERENCE-subtype memories.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "write", workspace_id=workspace_id)

        logger.info("Deriving insights for entity: %s in workspace: %s", entity_id, workspace_id)

        result = await inference_service.derive_insights(
            workspace_id=workspace_id,
            subject_id=entity_id,
            observer_id=request.observer_id,
            force=request.force,
        )

        # Invalidate entity card cache on new derivation
        card_key = _card_cache_key(workspace_id, entity_id)
        await cache_service.delete(card_key)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=entity_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity derive")
        return EntityDeriveResponse(
            subject_id=entity_id,
            workspace_id=workspace_id,
            insights_created=result.insights_created,
            insights_updated=result.insights_updated,
            source_memory_count=result.source_memory_count,
            insights=result.insights,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to derive insights for entity %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to derive entity insights")


@router.get(
    "/{entity_id}/card",
    response_model=EntityCardResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_entity_card(
    http_request: Request,
    entity_id: str,
    workspace_id: str | None = None,
    force_refresh: bool = False,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    reflect_service: ReflectService = Depends(get_reflect_service),
    inference_service: DefaultInferenceService = Depends(get_inference_service),
    cache_service: CacheService = Depends(get_cache_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> EntityCardResponse:
    """
    Get a cached entity profile card.

    Returns a synthesized view of an entity combining reflection and derived insights.
    Results are cached and lazily recalculated - most calls return quickly from cache.
    Use force_refresh=true to trigger immediate recalculation.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        card_key = _card_cache_key(workspace_id, entity_id)

        # Try cache first unless force refresh
        if not force_refresh:
            cached_card = await cache_service.get(card_key)
            if cached_card is not None:
                logger.debug("Returning cached entity card for: %s", entity_id)
                cached_card["cached"] = True
                return EntityCardResponse(**cached_card)

        logger.info("Generating entity card for: %s in workspace: %s", entity_id, workspace_id)

        # Generate reflection about this entity
        reflect_input = ReflectInput(
            query=f"Comprehensive profile of entity {entity_id}: who they are, their patterns, preferences, and key characteristics",
            subject_id=entity_id,
            detail_level=DetailLevel.OVERVIEW,
            include_sources=True,
            depth=2,
        )

        reflect_result = await reflect_service.reflect(
            workspace_id=workspace_id,
            input=reflect_input,
        )

        # Get existing insights
        insights = await inference_service.get_insights(
            workspace_id=workspace_id,
            subject_id=entity_id,
            limit=20,
        )

        now = datetime.now(UTC).isoformat()
        card_data = {
            "entity_id": entity_id,
            "workspace_id": workspace_id,
            "reflection": reflect_result.reflection,
            "insights": [i.model_dump(mode="json") for i in insights],
            "source_memories": reflect_result.source_memories,
            "confidence": reflect_result.confidence,
            "cached": False,
            "generated_at": now,
        }

        # Cache the card
        await cache_service.set(card_key, card_data, ttl_seconds=ENTITY_CARD_CACHE_TTL)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=entity_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity card read")
        return EntityCardResponse(**card_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity card for %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate entity card")


@router.get(
    "/{entity_id}/insights",
    response_model=EntityInsightsResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_entity_insights(
    http_request: Request,
    entity_id: str,
    workspace_id: str | None = None,
    observer_id: str | None = None,
    limit: int = 20,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    inference_service: DefaultInferenceService = Depends(get_inference_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> EntityInsightsResponse:
    """
    Get derived insights for an entity.

    Returns existing INFERENCE-subtype memories about the entity without
    triggering new derivation. Use POST /derive to generate fresh insights.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "entities", "read", workspace_id=workspace_id)

        logger.debug("Getting insights for entity: %s", entity_id)

        insights = await inference_service.get_insights(
            workspace_id=workspace_id,
            subject_id=entity_id,
            observer_id=observer_id,
            limit=limit,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="entity",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="entity",
                    resource_id=entity_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for entity insights read")
        return EntityInsightsResponse(
            entity_id=entity_id,
            workspace_id=workspace_id,
            insights=insights,
            total_count=len(insights),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get insights for entity %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve entity insights")


class EntitiesAPIPlugin(Plugin):
    """Plugin to register entities API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
