"""
Contradiction management API endpoints.

Endpoints:
- GET /v1/workspaces/{workspace_id}/contradictions - List unresolved contradictions
- POST /v1/contradictions/{contradiction_id}/resolve - Resolve a contradiction
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from scitrera_app_framework import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS
from ...services.authentication import AuthenticationError, AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.contradiction import ContradictionService, get_contradiction_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from .schemas import (
    ContradictionListResponse,
    ContradictionResolveRequest,
    ContradictionResponse,
    ErrorResponse,
)
from .deps import get_auth_service, get_authz_service

router = APIRouter(prefix='/v1', tags=["contradictions"])


# Dependencies for services
async def get_contradiction_svc(v: Variables = Depends(get_variables_dep)) -> ContradictionService:
    """Get contradiction service instance from dependency injection."""
    return get_contradiction_service(v)


@router.get(
    "/workspaces/{workspace_id}/contradictions",
    response_model=ContradictionListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_contradictions(
        http_request: Request,
        workspace_id: str,
        limit: int = 10,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        contradiction_service: ContradictionService = Depends(get_contradiction_svc),
        logger: logging.Logger = Depends(get_logger),
) -> ContradictionListResponse:
    """List unresolved contradictions for a workspace."""
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "contradictions", "read", workspace_id=workspace_id
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed for contradictions list: %s", e)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    try:
        records = await contradiction_service.get_unresolved(workspace_id, limit=limit)
        contradictions = [
            ContradictionResponse(
                id=r.id,
                workspace_id=r.workspace_id,
                memory_a_id=r.memory_a_id,
                memory_b_id=r.memory_b_id,
                contradiction_type=r.contradiction_type,
                confidence=r.confidence,
                detection_method=r.detection_method,
                detected_at=r.detected_at,
                resolved_at=r.resolved_at,
                resolution=r.resolution,
            )
            for r in records
        ]
        return ContradictionListResponse(contradictions=contradictions, count=len(contradictions))
    except Exception as e:
        logger.error("Failed to list contradictions: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list contradictions"
        )


@router.post(
    "/contradictions/{contradiction_id}/resolve",
    response_model=ContradictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Contradiction not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def resolve_contradiction(
        http_request: Request,
        contradiction_id: str,
        request: ContradictionResolveRequest,
        workspace_id: str | None = None,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        contradiction_service: ContradictionService = Depends(get_contradiction_svc),
        logger: logging.Logger = Depends(get_logger),
) -> ContradictionResponse:
    """Resolve a contradiction with a chosen strategy."""
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "contradictions", "write", workspace_id=workspace_id or ctx.workspace_id
        )
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed for contradiction resolve: %s", e)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    # Validate resolution strategy
    valid_resolutions = {"keep_a", "keep_b", "keep_both", "merge"}
    if request.resolution not in valid_resolutions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid resolution. Must be one of: {', '.join(sorted(valid_resolutions))}"
        )

    if request.resolution == "merge" and not request.merged_content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="merged_content is required when resolution is 'merge'"
        )

    try:
        effective_workspace_id = workspace_id or ctx.workspace_id
        record = await contradiction_service.resolve(
            workspace_id=effective_workspace_id,
            contradiction_id=contradiction_id,
            resolution=request.resolution,
            merged_content=request.merged_content,
        )

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Contradiction {contradiction_id} not found"
            )

        return ContradictionResponse(
            id=record.id,
            workspace_id=record.workspace_id,
            memory_a_id=record.memory_a_id,
            memory_b_id=record.memory_b_id,
            contradiction_type=record.contradiction_type,
            confidence=record.confidence,
            detection_method=record.detection_method,
            detected_at=record.detected_at,
            resolved_at=record.resolved_at,
            resolution=record.resolution,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve contradiction %s: %s", contradiction_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resolve contradiction"
        )


class ContradictionsAPIPlugin(Plugin):
    """Plugin to register contradiction API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
