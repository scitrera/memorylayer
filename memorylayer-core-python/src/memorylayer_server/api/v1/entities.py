"""
Entity profile and inference API endpoints.

Endpoints:
- POST /v1/entities/{entity_id}/derive - Trigger inference derivation for an entity
- GET /v1/entities/{entity_id}/card - Get cached entity profile card
- GET /v1/entities/{entity_id}/insights - Get derived insights for an entity
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger

from .schemas import (
    EntityDeriveRequest,
    EntityDeriveResponse,
    EntityCardResponse,
    EntityInsightsResponse,
    ErrorResponse,
)
from ...models import ReflectInput, DetailLevel
from ...services.inference import DefaultInferenceService
from ...services.reflect import ReflectService
from ...services.cache import CacheService
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...config import DEFAULT_TENANT_ID
from .deps import (
    get_auth_service, get_authz_service,
    get_inference_service, get_reflect_service, get_cache_service,
    get_audit_service,
)
from ...services.audit import AuditService, AuditEvent

router = APIRouter(prefix="/v1/entities", tags=["entities"])

# Cache TTL for entity cards (5 minutes)
ENTITY_CARD_CACHE_TTL = 300


def _card_cache_key(workspace_id: str, entity_id: str) -> str:
    return f"entity_card:{workspace_id}:{entity_id}"


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
        await authz_service.require_authorization(
            ctx, "entities", "write", workspace_id=workspace_id
        )

        logger.info(
            "Deriving insights for entity: %s in workspace: %s",
            entity_id, workspace_id
        )

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
            await audit_service.record(AuditEvent(
                event_type="entity",
                action="update",
                tenant_id=ctx.tenant_id,
                workspace_id=workspace_id,
                user_id=ctx.user_id,
                resource_type="entity",
                resource_id=entity_id,
            ))
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to derive entity insights"
        )


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
        workspace_id: Optional[str] = None,
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
        await authz_service.require_authorization(
            ctx, "entities", "read", workspace_id=workspace_id
        )

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

        now = datetime.now(timezone.utc).isoformat()
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
            await audit_service.record(AuditEvent(
                event_type="entity",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=workspace_id,
                user_id=ctx.user_id,
                resource_type="entity",
                resource_id=entity_id,
            ))
        except Exception:
            logger.debug("Audit record failed for entity card read")
        return EntityCardResponse(**card_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get entity card for %s: %s", entity_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate entity card"
        )


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
        workspace_id: Optional[str] = None,
        observer_id: Optional[str] = None,
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
        await authz_service.require_authorization(
            ctx, "entities", "read", workspace_id=workspace_id
        )

        logger.debug("Getting insights for entity: %s", entity_id)

        insights = await inference_service.get_insights(
            workspace_id=workspace_id,
            subject_id=entity_id,
            observer_id=observer_id,
            limit=limit,
        )

        try:
            await audit_service.record(AuditEvent(
                event_type="entity",
                action="read",
                tenant_id=ctx.tenant_id,
                workspace_id=workspace_id,
                user_id=ctx.user_id,
                resource_type="entity",
                resource_id=entity_id,
            ))
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entity insights"
        )


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
