"""
Memory CRUD and recall operations API endpoints.

Endpoints:
- POST /v1/memories - Store a memory
- GET /v1/memories/{memory_id} - Get single memory
- PUT /v1/memories/{memory_id} - Update memory
- DELETE /v1/memories/{memory_id} - Delete memory (soft delete)
- POST /v1/memories/recall - Query memories with mode (rag/llm/hybrid)
- POST /v1/memories/reflect - Synthesize memories
- POST /v1/memories/{memory_id}/decay - Decay importance
- POST /v1/memories/batch - Batch operations (create, update, delete)
"""

import logging
import time as _time

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.generation import GenerationBudgetExceededError, GenerationNotAllowedError
from ...models.memory import (
    DetailLevel,
    Memory,
    MemoryReplaceInput,
    MemoryRevision,
    MemoryType,
    RecallInput,
    ReflectInput,
    RememberInput,
)
from ...models.versioned_resource import VersionedResourcePreconditionFailedError
from ...services.audit import AuditEvent, AuditService
from ...services.authentication import AuthenticationError, AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.ingest import normalize_connector_metadata
from ...services.memory import MemoryService
from ...services.metrics import MetricsService
from ...services.reflect import EXT_REFLECT_SERVICE, ReflectService
from ...services.storage import StorageCapabilityError
from .. import EXT_MULTI_API_ROUTERS
from ._versioned_resource import raise_api_error as _shared_raise_api_error
from ._versioned_resource import required_header as _required_header
from .deps import get_active_session, get_audit_service, get_auth_service, get_authz_service, get_memory_service, get_metrics_service
from .schemas import (
    BatchCreateOp,
    BatchDeleteOp,
    BatchOperationResponse,
    BatchOperationResult,
    BatchUpdateOp,
    ErrorResponse,
    MemoryBatchRequest,
    MemoryCreateRequest,
    MemoryDecayRequest,
    MemoryListResponse,
    MemoryRecallRequest,
    MemoryReflectRequest,
    MemoryResponse,
    MemoryUpdateRequest,
    RecallResult,
    ReflectResult,
)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


class MemoryMutationResponse(BaseModel):
    memory: Memory
    replayed: bool = False


class MemoryRevisionResponse(BaseModel):
    memory: Memory
    action: str
    operation_id: str


class MemoryRevisionListResponse(BaseModel):
    revisions: list[MemoryRevisionResponse]
    next_page_token: str | None = None


def _memory_revision_response(revision: MemoryRevision) -> MemoryRevisionResponse:
    return MemoryRevisionResponse(
        memory=revision.memory,
        action=revision.action,
        operation_id=revision.operation_id,
    )


# Aether machine-principal namespaces. A service/agent connection principal must
# never OWN a memory: a memory user-scoped to a machine id is unrecallable by any
# real user, so machine-committed workspace knowledge belongs to the workspace
# (user_id NULL), not to the connection principal.
_MACHINE_PRINCIPAL_PREFIXES = ("sv::", "ag::")


def _owner_user_id(request_user_id: str | None, ctx_user_id: str | None) -> str | None:
    """Resolve the ``user_id`` to stamp as a new memory's owner.

    An explicit request ``user_id`` wins; otherwise fall back to the context
    user — EXCEPT a machine (service/agent) connection principal, which is
    dropped to None so the memory is workspace-shared rather than orphaned to a
    principal no user recall will ever match. Note: under OBO the context user is
    the real subject (a plain user id), so genuine user commits are unaffected.
    """
    uid = request_user_id if request_user_id is not None else ctx_user_id
    if uid and uid.startswith(_MACHINE_PRINCIPAL_PREFIXES):
        return None
    return uid


def _drop_embeddings(memories: list) -> None:
    """Blank each memory's embedding vector in place, for response serialization.

    The vectors dominate the payload — at 1024 dimensions they are ~8 KB per
    memory, so a default recall of 10 is ~80 KB of data callers almost never
    read. Endpoints omit them unless the caller opts in.

    In place is safe here: these are per-request Pydantic models built from
    storage rows, not shared cache entries.
    """
    for memory in memories:
        memory.embedding = None


# Dependencies for services
async def get_reflect_service(v: Variables = Depends(get_variables_dep)) -> ReflectService:
    """Get reflect service instance for FastAPI dependency injection."""
    return get_extension(EXT_REFLECT_SERVICE, v)


@router.post(
    "",
    response_model=MemoryMutationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_memory(
    http_request: Request,
    response: Response,
    request: MemoryCreateRequest,
    session_id: str = Depends(get_active_session),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryMutationResponse:
    """
    Store a new memory with automatic embedding and classification.

    Authentication:
        - Authorization header: Bearer <api_key> (optional in OSS)
        - X-Session-ID header: Session for workspace context (optional)

    Workspace Resolution:
        1. request.workspace_id (explicit override)
        2. session.workspace_id (from X-Session-ID header)
        3. "_default" (fallback)

    Args:
        http_request: FastAPI request (for headers)
        request: Memory creation request
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Created memory with generated ID and embedding

    Raises:
        HTTPException: If memory creation fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "create", workspace_id=ctx.workspace_id)

        logger.info("Creating memory in workspace: %s, content length: %d", ctx.workspace_id, len(request.content))

        # Connector-shaped metadata is an explicit opt-in to deterministic
        # knowledge-work normalization. Ordinary arbitrary metadata is untouched.
        normalized_metadata = normalize_connector_metadata(request.metadata).metadata

        # Convert request to domain input
        remember_input = RememberInput(
            content=request.content,
            tenant_id=ctx.tenant_id,
            logical_key=request.logical_key,
            type=request.type,
            subtype=request.subtype,
            importance=request.importance,
            tags=request.tags,
            metadata=normalized_metadata,
            refinement_metadata=request.refinement_metadata,
            associations=request.associations,
            relations=request.relations,
            context_id=request.context_id or ctx.context_id,
            observer_id=request.observer_id,
            subject_id=request.subject_id,
            user_id=_owner_user_id(request.user_id, ctx.user_id),
            pinned=request.pinned,
            scope=request.scope,
        )

        operation_id = http_request.headers.get("Idempotency-Key", "").strip()
        expected = http_request.headers.get("If-None-Match", "").strip()
        replayed = False
        _t0 = _time.monotonic()
        if operation_id or expected:
            operation_id = _required_header(http_request, "Idempotency-Key")
            expected = _required_header(http_request, "If-None-Match")
            if expected != "*":
                raise VersionedResourcePreconditionFailedError("conditional memory create requires If-None-Match: *")
            result = await memory_service.remember_versioned(
                workspace_id=ctx.workspace_id,
                input=remember_input,
                tenant_id=ctx.tenant_id,
                user_id=remember_input.user_id,
                operation_id=operation_id,
                expected_etag=expected,
            )
            memory = result.memory
            replayed = result.replayed
        else:
            if request.logical_key is not None:
                raise ValueError("logical_key requires Idempotency-Key and If-None-Match headers")
            memory = await memory_service.remember(
                workspace_id=ctx.workspace_id,
                input=remember_input,
            )

        logger.info("Created memory: %s", memory.id)
        try:
            metrics_service.counter("memorylayer_remember_total", labels={"workspace": ctx.workspace_id})
            metrics_service.histogram(
                "memorylayer_remember_duration_seconds", _time.monotonic() - _t0, labels={"workspace": ctx.workspace_id}
            )
        except Exception:
            logger.debug("Metrics recording failed for memory create")
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="create",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    resource_id=memory.id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory create")
        response.headers["ETag"] = memory.etag
        return MemoryMutationResponse(memory=memory, replayed=replayed)

    except StorageCapabilityError as e:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail={"code": "storage_capability_missing", "capability": e.capability},
        ) from e
    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except GenerationNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": e.code, "activity": e.activity.value, "policy": e.policy.value},
        ) from e
    except GenerationBudgetExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": e.code, "activity": e.activity.value, "message": e.reason},
        ) from e
    except ValueError as e:
        logger.warning("Invalid memory creation request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        _shared_raise_api_error(e, "create memory", __name__)


@router.get(
    "",
    response_model=MemoryListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_memories(
    http_request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum memories to return"),
    offset: int = Query(0, ge=0, description="Number of memories to skip for pagination"),
    type: MemoryType | None = Query(None, description="Filter by cognitive type"),
    subtype: str | None = Query(None, description="Filter by domain subtype"),
    tag: str | None = Query(None, description="Filter by a single tag"),
    context_id: str | None = Query(None, description="Filter by memory context"),
    include_embeddings: bool = Query(
        False,
        description=(
            "Include each memory's raw embedding vector. Off by default: the vectors dominate the payload and are rarely used by callers."
        ),
    ),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryListResponse:
    """
    List/browse memories in a workspace ordered by recency (created_at desc).

    Unlike POST /recall this performs no vector search — it is a plain filtered
    enumeration for browsing. Supports pagination and optional type/subtype/tag/
    context filters.

    Workspace Resolution:
        1. X-Workspace-ID header (explicit override)
        2. session.workspace_id (from X-Session-ID header)
        3. "_default" (fallback)
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        logger.debug("(API) Listing memories in workspace: %s (limit=%d, offset=%d)", ctx.workspace_id, limit, offset)

        memories = await memory_service.list_memories(
            ctx.workspace_id,
            types=[type] if type is not None else None,
            subtypes=[subtype] if subtype else None,
            tags=[tag] if tag else None,
            context_id=context_id,
            limit=limit,
            offset=offset,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="list",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    metadata={"count": len(memories)},
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory list")
        if not include_embeddings:
            _drop_embeddings(memories)
        return MemoryListResponse(memories=memories, total_count=len(memories))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list memories: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list memories")


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_memory(
    http_request: Request,
    response: Response,
    memory_id: str,
    include_deleted: bool = Query(False),
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Retrieve a single memory by ID.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Memory object

    Raises:
        HTTPException: If memory not found
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)

        logger.debug("Getting memory: %s", memory_id)

        # Memory IDs are globally unique; look up without workspace filter
        memory = await memory_service.get_by_id(
            memory_id=memory_id,
            track_access=not include_deleted,
            include_deleted=include_deleted,
        )

        if not memory or (workspace_id is not None and memory.workspace_id != workspace_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory not found: {memory_id}")

        await authz_service.require_authorization(ctx, "memories", "read", resource_id=memory_id, workspace_id=memory.workspace_id)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=memory.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    resource_id=memory_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory read")
        response.headers["ETag"] = memory.etag
        return MemoryResponse(memory=memory)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve memory")


@router.put(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def update_memory(
    http_request: Request,
    response: Response,
    memory_id: str,
    request: MemoryUpdateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Update an existing memory.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        request: Memory update request
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Updated memory object

    Raises:
        HTTPException: If memory not found or update fails
    """
    try:
        # Build request context
        ctx = await auth_service.build_context(http_request, None)

        logger.info("Updating memory: %s", memory_id)

        # Fetch memory first, then authorize against the memory's actual workspace
        existing_memory = await memory_service.get_by_id(memory_id=memory_id)
        if not existing_memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory not found: {memory_id}")

        await authz_service.require_authorization(
            ctx, "memories", "write", resource_id=memory_id, workspace_id=existing_memory.workspace_id
        )

        # Build update kwargs from non-None fields
        update_kwargs = {}
        if request.content is not None:
            update_kwargs["content"] = request.content
        if request.type is not None:
            update_kwargs["type"] = request.type
        if request.subtype is not None:
            update_kwargs["subtype"] = request.subtype
        if request.importance is not None:
            update_kwargs["importance"] = request.importance
        if request.tags is not None:
            update_kwargs["tags"] = request.tags
        if request.metadata is not None:
            update_kwargs["metadata"] = request.metadata
        if request.refinement_metadata is not None:
            update_kwargs["refinement_metadata"] = request.refinement_metadata
        if request.pinned is not None:
            update_kwargs["pinned"] = 1 if request.pinned else 0

        # Update memory via service layer
        updated_memory = await memory_service.update(workspace_id=existing_memory.workspace_id, memory_id=memory_id, **update_kwargs)

        if not updated_memory:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update memory")

        logger.info("Updated memory: %s", memory_id)
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=existing_memory.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    resource_id=memory_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory update")
        response.headers["ETag"] = updated_memory.etag
        return MemoryResponse(memory=updated_memory)

    except HTTPException:
        raise
    except GenerationNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": e.code, "activity": e.activity.value, "policy": e.policy.value},
        ) from e
    except GenerationBudgetExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": e.code, "activity": e.activity.value, "message": e.reason},
        ) from e
    except ValueError as e:
        logger.warning("Invalid memory update request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to update memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update memory")


@router.get(
    "/{memory_id}/revisions",
    response_model=MemoryRevisionListResponse,
    responses={404: {"model": ErrorResponse}},
)
async def list_memory_revisions(
    http_request: Request,
    memory_id: str,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryRevisionListResponse:
    """List immutable semantic memory revisions newest first."""

    try:
        ctx = await auth_service.build_context(http_request, None)
        current = await memory_service.get_by_id(
            memory_id,
            track_access=False,
            include_deleted=True,
        )
        if current is None or current.tenant_id != ctx.tenant_id or (workspace_id is not None and current.workspace_id != workspace_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}",
            )
        await authz_service.require_authorization(
            ctx,
            "memories",
            "read",
            resource_id=memory_id,
            workspace_id=current.workspace_id,
        )
        revisions, next_token = await memory_service.list_revision_page(
            ctx.tenant_id,
            current.workspace_id,
            memory_id,
            limit=limit,
            page_token=page_token,
        )
        return MemoryRevisionListResponse(
            revisions=[_memory_revision_response(item) for item in revisions],
            next_page_token=next_token,
        )
    except Exception as exc:
        _shared_raise_api_error(exc, "list memory revisions", __name__)


@router.put(
    "/{memory_id}/semantic",
    response_model=MemoryMutationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}},
)
async def replace_memory_semantic(
    http_request: Request,
    response: Response,
    memory_id: str,
    request: MemoryReplaceInput,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryMutationResponse:
    """Conditionally replace the complete refinement-owned memory document."""

    try:
        ctx = await auth_service.build_context(http_request, None)
        current = await memory_service.get_by_id(memory_id, track_access=False)
        if current is None or current.tenant_id != ctx.tenant_id or (workspace_id is not None and current.workspace_id != workspace_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}",
            )
        await authz_service.require_authorization(
            ctx,
            "memories",
            "write",
            resource_id=memory_id,
            workspace_id=current.workspace_id,
        )
        result = await memory_service.replace_versioned(
            current.workspace_id,
            memory_id,
            request,
            tenant_id=ctx.tenant_id,
            operation_id=_required_header(http_request, "Idempotency-Key"),
            expected_etag=_required_header(http_request, "If-Match"),
        )
        response.headers["ETag"] = result.memory.etag
        return MemoryMutationResponse(memory=result.memory, replayed=result.replayed)
    except Exception as exc:
        _shared_raise_api_error(exc, "replace semantic memory", __name__)


async def _change_memory_deleted_state(
    *,
    action: str,
    http_request: Request,
    response: Response,
    memory_id: str,
    workspace_id: str | None,
    auth_service: AuthenticationService,
    authz_service: AuthorizationService,
    memory_service: MemoryService,
) -> MemoryMutationResponse:
    ctx = await auth_service.build_context(http_request, None)
    current = await memory_service.get_by_id(
        memory_id,
        track_access=False,
        include_deleted=True,
    )
    if current is None or current.tenant_id != ctx.tenant_id or (workspace_id is not None and current.workspace_id != workspace_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory not found: {memory_id}",
        )
    await authz_service.require_authorization(
        ctx,
        "memories",
        "delete" if action == "delete" else "write",
        resource_id=memory_id,
        workspace_id=current.workspace_id,
    )
    operation_id = _required_header(http_request, "Idempotency-Key")
    expected_etag = _required_header(http_request, "If-Match")
    if action == "delete":
        result = await memory_service.delete_versioned(
            current.workspace_id,
            memory_id,
            tenant_id=ctx.tenant_id,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
    else:
        result = await memory_service.restore_versioned(
            current.workspace_id,
            memory_id,
            tenant_id=ctx.tenant_id,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
    response.headers["ETag"] = result.memory.etag
    return MemoryMutationResponse(memory=result.memory, replayed=result.replayed)


@router.post(
    "/{memory_id}/delete",
    response_model=MemoryMutationResponse,
)
async def delete_memory_versioned(
    http_request: Request,
    response: Response,
    memory_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryMutationResponse:
    try:
        return await _change_memory_deleted_state(
            action="delete",
            http_request=http_request,
            response=response,
            memory_id=memory_id,
            workspace_id=workspace_id,
            auth_service=auth_service,
            authz_service=authz_service,
            memory_service=memory_service,
        )
    except Exception as exc:
        _shared_raise_api_error(exc, "delete semantic memory", __name__)


@router.post(
    "/{memory_id}/restore",
    response_model=MemoryMutationResponse,
)
async def restore_memory_versioned(
    http_request: Request,
    response: Response,
    memory_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryMutationResponse:
    try:
        return await _change_memory_deleted_state(
            action="restore",
            http_request=http_request,
            response=response,
            memory_id=memory_id,
            workspace_id=workspace_id,
            auth_service=auth_service,
            authz_service=authz_service,
            memory_service=memory_service,
        )
    except Exception as exc:
        _shared_raise_api_error(exc, "restore semantic memory", __name__)


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_memory(
    http_request: Request,
    memory_id: str,
    hard: bool = False,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """
    Delete a memory (soft delete by default).

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        hard: If True, permanently delete; if False, soft delete
        auth_service: Authentication service
        memory_service: Memory service instance

    Raises:
        HTTPException: If memory not found or deletion fails
    """
    try:
        # Build request context. Fetch the memory first (IDs are globally
        # unique), then authorize and delete against the memory's ACTUAL
        # workspace — not ctx.workspace_id. Memories can live in non-default
        # workspaces (e.g. USER scope / _global_user); deleting against
        # ctx.workspace_id silently failed (404) for those. Mirrors update_memory.
        ctx = await auth_service.build_context(http_request, None)

        existing_memory = await memory_service.get_by_id(memory_id=memory_id)
        if not existing_memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory not found: {memory_id}")

        await authz_service.require_authorization(
            ctx, "memories", "delete", resource_id=memory_id, workspace_id=existing_memory.workspace_id
        )

        logger.info("Deleting memory: %s (hard=%s)", memory_id, hard)

        success = await memory_service.forget(
            workspace_id=existing_memory.workspace_id,
            memory_id=memory_id,
            hard=hard,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory not found: {memory_id}")

        logger.info("Deleted memory: %s", memory_id)
        try:
            metrics_service.counter("memorylayer_delete_total", labels={"workspace": existing_memory.workspace_id})
        except Exception:
            logger.debug("Metrics recording failed for memory delete")
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="delete",
                    tenant_id=ctx.tenant_id,
                    workspace_id=existing_memory.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    resource_id=memory_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory delete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete memory")


@router.post(
    "/recall",
    response_model=RecallResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def recall_memories(
    http_request: Request,
    request: MemoryRecallRequest,
    session_id: str = Depends(get_active_session),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    logger: logging.Logger = Depends(get_logger),
) -> RecallResult:
    """
    Query memories using vector similarity and optional filters.

    Supports three retrieval modes:
    - RAG: Fast vector similarity search (~30ms)
    - LLM: Query rewriting + enhanced search (~500ms)
    - HYBRID: RAG first, LLM if insufficient (balanced)

    Args:
        http_request: FastAPI request (for headers)
        request: Memory recall request with query and filters
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Recall result with matched memories and metadata

    Raises:
        HTTPException: If recall fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        logger.debug("(API) Recalling memories in workspace: %s, mode: %s, query: %s", ctx.workspace_id, request.mode, request.query[:50])

        # Convert request to domain input
        recall_input = RecallInput(
            query=request.query,
            types=request.types,
            subtypes=request.subtypes,
            tags=request.tags,
            context_id=request.context_id or ctx.context_id,
            observer_id=request.observer_id,
            subject_id=request.subject_id,
            user_id=request.user_id if request.user_id is not None else ctx.user_id,
            include_global=request.include_global,
            include_global_user=request.include_global_user,
            mode=request.mode,
            tolerance=request.tolerance,
            limit=request.limit,
            offset=request.offset,
            min_relevance=request.min_relevance,
            recency_weight=request.recency_weight,
            include_associations=request.include_associations,
            traverse_depth=request.traverse_depth,
            max_expansion=request.max_expansion,
            created_after=request.created_after,
            created_before=request.created_before,
            event_after=request.event_after,
            event_before=request.event_before,
            time_order=request.time_order,
            context=request.context,
            rag_threshold=request.rag_threshold,
            include_archived=request.include_archived,
            exclude_ids=request.exclude_ids,
            budget_tokens=request.budget_tokens,
            include_confidence=request.include_confidence,
            include_relations=request.include_relations,
        )

        # Perform recall
        _t0 = _time.monotonic()
        result = await memory_service.recall(
            workspace_id=ctx.workspace_id,
            input=recall_input,
        )

        logger.debug("Recalled %d memories in %d ms using %s mode", len(result.memories), result.search_latency_ms, result.mode_used)

        if not request.include_embeddings:
            _drop_embeddings(result.memories)

        try:
            metrics_service.counter("memorylayer_recall_total", labels={"workspace": ctx.workspace_id, "mode": request.mode or "default"})
            metrics_service.histogram(
                "memorylayer_recall_duration_seconds", _time.monotonic() - _t0, labels={"workspace": ctx.workspace_id}
            )
            metrics_service.histogram(
                "memorylayer_recall_result_count",
                len(result.memories) if hasattr(result, "memories") else 0,
                labels={"workspace": ctx.workspace_id},
            )
        except Exception:
            logger.debug("Metrics recording failed for memory recall")
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="recall",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    metadata={"query_length": len(request.query), "mode": request.mode},
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory recall")
        return result

    except GenerationNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": e.code, "activity": e.activity.value, "policy": e.policy.value},
        ) from e
    except GenerationBudgetExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": e.code, "activity": e.activity.value, "message": e.reason},
        ) from e
    except ValueError as e:
        logger.warning("Invalid recall request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to recall memories: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to recall memories")


@router.post(
    "/reflect",
    response_model=ReflectResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reflect_memories(
    http_request: Request,
    request: MemoryReflectRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    reflection_service: ReflectService = Depends(get_reflect_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ReflectResult:
    """
    Synthesize memories into a coherent reflection.

    Args:
        http_request: FastAPI request (for headers)
        request: Memory reflection request
        auth_service: Authentication service
        reflection_service: Reflection service instance

    Returns:
        Reflection result with synthesized content

    Raises:
        HTTPException: If reflection fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        logger.info("Reflecting on memories in workspace: %s, query: %s", ctx.workspace_id, request.query[:50])

        # Convert detail_level string to DetailLevel enum
        detail_level = DetailLevel.FULL
        if request.detail_level:
            detail_level_map = {
                "abstract": DetailLevel.ABSTRACT,
                "overview": DetailLevel.OVERVIEW,
                "full": DetailLevel.FULL,
            }
            detail_level = detail_level_map.get(request.detail_level.lower(), DetailLevel.FULL)

        # Convert request to domain input
        reflect_input = ReflectInput(
            query=request.query,
            detail_level=detail_level,
            include_sources=request.include_sources,
            depth=request.depth,
            types=request.types,
            subtypes=request.subtypes,
            tags=request.tags,
            context_id=request.context_id or ctx.context_id,
            observer_id=request.observer_id,
            subject_id=request.subject_id,
        )

        # Perform reflection
        result = await reflection_service.reflect(
            workspace_id=ctx.workspace_id,
            input=reflect_input,
        )

        logger.info("Reflected on %d source memories, generated %d tokens", len(result.source_memories), result.tokens_processed)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="reflect",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory reflect")
        return result

    except GenerationNotAllowedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": e.code, "activity": e.activity.value, "policy": e.policy.value},
        ) from e
    except GenerationBudgetExceededError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"code": e.code, "activity": e.activity.value, "message": e.reason},
        ) from e
    except ValueError as e:
        logger.warning("Invalid reflect request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to reflect memories: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reflect memories")


@router.post(
    "/{memory_id}/decay",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def decay_memory(
    http_request: Request,
    memory_id: str,
    request: MemoryDecayRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Reduce memory importance by decay rate.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        request: Decay request with rate
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Updated memory with decayed importance

    Raises:
        HTTPException: If memory not found or decay fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "write", resource_id=memory_id, workspace_id=ctx.workspace_id)

        logger.info("Decaying memory: %s by rate: %f", memory_id, request.decay_rate)

        updated_memory = await memory_service.decay(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
            decay_rate=request.decay_rate,
        )

        if not updated_memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Memory not found: {memory_id}")

        logger.info("Decayed memory: %s", memory_id)
        return MemoryResponse(memory=updated_memory)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid decay request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to decay memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to decay memory")


@router.post(
    "/batch",
    response_model=BatchOperationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def batch_operations(
    http_request: Request,
    request: MemoryBatchRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> BatchOperationResponse:
    """
    Perform multiple memory operations in a single request.

    Supported operation types:
    - create: Create a new memory
    - update: Update an existing memory
    - delete: Delete a memory

    Args:
        http_request: FastAPI request (for headers)
        request: Batch request with list of operations
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Results for each operation with success/error status

    Raises:
        HTTPException: If batch request is invalid
    """
    try:
        # Build request context and check authorization (batch requires write access)
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        logger.info("Processing batch operations in workspace: %s, count: %d", ctx.workspace_id, len(request.operations))

        results = []
        successful = 0
        failed = 0

        for i, operation in enumerate(request.operations):
            op_type = operation.op

            logger.debug("Processing batch operation %d: %s", i, op_type)

            try:
                # CREATE operation
                if isinstance(operation, BatchCreateOp):
                    normalized_metadata = normalize_connector_metadata(operation.metadata).metadata
                    remember_input = RememberInput(
                        content=operation.content,
                        type=operation.type,
                        subtype=operation.subtype,
                        importance=operation.importance,
                        tags=operation.tags,
                        metadata=normalized_metadata,
                        observer_id=operation.observer_id,
                        subject_id=operation.subject_id,
                    )

                    # Create memory
                    memory = await memory_service.remember(
                        workspace_id=ctx.workspace_id,
                        input=remember_input,
                    )

                    results.append(
                        BatchOperationResult(
                            index=i,
                            type=op_type,
                            status="success",
                            memory_id=memory.id,
                        )
                    )
                    successful += 1

                # UPDATE operation
                elif isinstance(operation, BatchUpdateOp):
                    memory_id = operation.memory_id

                    # Resolve the memory's ACTUAL workspace (IDs are globally
                    # unique). Using ctx.workspace_id silently 404'd memories in
                    # non-default workspaces (e.g. USER scope). Mirrors
                    # single-endpoint update_memory / delete_memory.
                    existing = await memory_service.get_by_id(memory_id=memory_id)
                    if not existing:
                        raise ValueError(f"Memory not found: {memory_id}")

                    # Authorize against the memory's real workspace (not ctx).
                    await authz_service.require_authorization(
                        ctx, "memories", "write", resource_id=memory_id, workspace_id=existing.workspace_id
                    )

                    # Build update kwargs from non-None fields
                    update_kwargs = {}
                    if operation.content is not None:
                        update_kwargs["content"] = operation.content
                    if operation.type is not None:
                        update_kwargs["type"] = operation.type
                    if operation.subtype is not None:
                        update_kwargs["subtype"] = operation.subtype
                    if operation.importance is not None:
                        update_kwargs["importance"] = operation.importance
                    if operation.tags is not None:
                        update_kwargs["tags"] = operation.tags
                    if operation.metadata is not None:
                        update_kwargs["metadata"] = operation.metadata
                    if operation.pinned is not None:
                        update_kwargs["pinned"] = 1 if operation.pinned else 0

                    # Update memory via service layer (memory's own workspace)
                    updated = await memory_service.update(workspace_id=existing.workspace_id, memory_id=memory_id, **update_kwargs)

                    results.append(
                        BatchOperationResult(
                            index=i,
                            type=op_type,
                            status="success",
                            memory_id=memory_id,
                        )
                    )
                    successful += 1

                # DELETE operation
                elif isinstance(operation, BatchDeleteOp):
                    memory_id = operation.memory_id

                    # Resolve the memory's ACTUAL workspace (IDs are globally
                    # unique) before deleting; ctx.workspace_id silently 404'd
                    # memories in non-default workspaces (e.g. USER scope).
                    existing = await memory_service.get_by_id(memory_id=memory_id)
                    if not existing:
                        raise ValueError(f"Memory not found: {memory_id}")

                    # Authorize against the memory's real workspace (not ctx).
                    await authz_service.require_authorization(
                        ctx, "memories", "delete", resource_id=memory_id, workspace_id=existing.workspace_id
                    )

                    # Delete memory (memory's own workspace)
                    success = await memory_service.forget(
                        workspace_id=existing.workspace_id,
                        memory_id=memory_id,
                        hard=operation.hard,
                    )

                    if not success:
                        raise ValueError(f"Memory not found: {memory_id}")

                    results.append(
                        BatchOperationResult(
                            index=i,
                            type=op_type,
                            status="success",
                            memory_id=memory_id,
                        )
                    )
                    successful += 1

            except Exception as e:
                logger.warning("Batch operation %d failed: %s - %s", i, op_type, str(e))
                results.append(
                    BatchOperationResult(
                        index=i,
                        type=op_type,
                        status="error",
                        error=str(e),
                    )
                )
                failed += 1

        logger.info("Completed batch operations: %d successful, %d failed", successful, failed)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="memory",
                    action="batch",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="memory",
                    metadata={"operation_count": len(request.operations)},
                )
            )
        except Exception:
            logger.debug("Audit record failed for memory batch")
        return BatchOperationResponse(
            total_operations=len(request.operations),
            successful=successful,
            failed=failed,
            results=results,
        )

    except ValueError as e:
        logger.warning("Invalid batch request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to process batch operations: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process batch operations")


class MemoriesAPIPlugin(Plugin):
    """Plugin to register memories API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
