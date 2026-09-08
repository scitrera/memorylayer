"""Perspective read-surface API endpoint (P4.3): expose ``get_representation``.

This is a THIN PASS-THROUGH to ``RepresentationService.get_representation`` — the
deterministic (observer, subject) perspective-assembly surface ("what observer O
understands about subject S"). It is a SEPARATE read surface; it does NOT touch
the recall fusion arms and the service it calls MUST NOT call ``recall()`` or
inject members into recall (the leakage-0 contract lives in the service —
``services/representation/base.py``). The endpoint adds NO scoping logic of its
own: leakage-0 is preserved precisely because everything is delegated.

Dark-gated behind ``MEMORYLAYER_REPRESENTATION_ENABLED`` (default OFF): when the
flag is off the endpoint returns 404 (the route exists but behaves as if absent,
mirroring how other dark-gated surfaces hide themselves). Enterprise transparently
serves the maintained-profile / derived-beliefs version via the provider seam
(``MEMORYLAYER_REPRESENTATION_PROVIDER``) — no special-casing here.

Endpoint:
- POST /v1/representation - assemble the (observer, subject) representation
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from scitrera_app_framework import Plugin, Variables, ext_parse_bool, get_extension

from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from ...config import (
    DEFAULT_MEMORYLAYER_REPRESENTATION_ENABLED,
    MEMORYLAYER_REPRESENTATION_ENABLED,
)
from ...models.representation import Representation, UserRepresentation
from ...services.audit import AuditEvent, AuditService
from ...services.authentication import AuthenticationError, AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.representation import EXT_REPRESENTATION_SERVICE, RepresentationService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_audit_service, get_auth_service, get_authz_service
from .schemas import ErrorResponse, RepresentationRequest, UserRepresentationRequest

router = APIRouter(prefix="/v1", tags=["representation"])


def _representation_enabled(v: Variables) -> bool:
    """Read the dark-gate flag (default OFF)."""
    return v.environ(
        MEMORYLAYER_REPRESENTATION_ENABLED,
        default=DEFAULT_MEMORYLAYER_REPRESENTATION_ENABLED,
        type_fn=ext_parse_bool,
    )


async def get_representation_svc(v: Variables = Depends(get_variables_dep)) -> RepresentationService:
    """Get the representation service instance from dependency injection."""
    return get_extension(EXT_REPRESENTATION_SERVICE, v)


@router.post(
    "/representation",
    response_model=Representation,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Representation surface disabled (dark-gated)"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_representation(
    http_request: Request,
    request: RepresentationRequest,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    representation_service: RepresentationService = Depends(get_representation_svc),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> Representation:
    """Assemble what ``observer`` understands about ``subject`` (leakage-0).

    Thin pass-through: builds the request context (for workspace isolation +
    authorization) then delegates to the service. No scoping logic is added here.
    """
    # Dark-gate: behave as if the route does not exist when disabled.
    if not _representation_enabled(v):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Representation surface is disabled (set MEMORYLAYER_REPRESENTATION_ENABLED=true to enable)",
        )

    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "representation", "read", workspace_id=ctx.workspace_id)
    except AuthenticationError as e:
        logger.warning("Authentication failed for representation read: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed for representation read: %s", e)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    try:
        representation = await representation_service.get_representation(
            ctx.workspace_id,
            request.observer,
            request.subject,
            observer_type=request.observer_type,
            subject_type=request.subject_type,
            limit=request.limit,
            include_profile=request.include_profile,
        )
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="representation",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="representation",
                )
            )
        except Exception:
            logger.debug("Audit record failed for representation read")
        return representation
    except Exception as e:
        logger.error("Failed to assemble representation: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to assemble representation")


@router.post(
    "/representation/user",
    response_model=UserRepresentation,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Representation surface disabled (dark-gated)"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_user_representation(
    http_request: Request,
    request: UserRepresentationRequest,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    representation_service: RepresentationService = Depends(get_representation_svc),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> UserRepresentation:
    """Assemble the USER-SCOPE self-representation for a user (cross-user safe).

    Thin pass-through to ``get_user_representation``. Resolves the effective
    user_id from the authenticated context when the request omits it (the caller
    reads their OWN representation); an explicit user_id that differs from the
    caller's identity is a CROSS-USER read and is authorized separately. The
    FORCED user_id filter (the cross-user leakage guard) lives in the service —
    the endpoint adds no scoping logic, only the authz boundary.
    """
    # Dark-gate: behave as if the route does not exist when disabled.
    if not _representation_enabled(v):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Representation surface is disabled (set MEMORYLAYER_REPRESENTATION_ENABLED=true to enable)",
        )

    try:
        ctx = await auth_service.build_context(http_request, request)
        # Resolve the effective user_id: prefer the explicit request value, else
        # fall back to the authenticated identity (read your own representation).
        effective_user_id = request.user_id or ctx.user_id
        # Cross-user read: an explicit user_id that differs from the caller's
        # identity must be separately authorized (resource_id = the target user).
        await authz_service.require_authorization(
            ctx,
            "representation",
            "read",
            resource_id=effective_user_id,
            workspace_id=ctx.workspace_id,
        )
    except AuthenticationError as e:
        logger.warning("Authentication failed for user-representation read: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed for user-representation read: %s", e)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    try:
        representation = await representation_service.get_user_representation(
            effective_user_id or "",
            limit=request.limit,
            include_profile=request.include_profile,
        )
        try:
            await audit_service.record(
                AuditEvent(
                    event_type="representation",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=effective_user_id,
                    resource_type="user_representation",
                )
            )
        except Exception:
            logger.debug("Audit record failed for user-representation read")
        return representation
    except Exception as e:
        logger.error("Failed to assemble user-representation: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to assemble user-representation")


class RepresentationAPIPlugin(Plugin):
    """Plugin to register the representation API route."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
