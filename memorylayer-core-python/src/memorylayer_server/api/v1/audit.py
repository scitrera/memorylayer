"""
Audit log query API endpoints.

Endpoints:
- GET /v1/audit/events - Query audit events
- GET /v1/audit/events/summary - Get audit event counts by type
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request, Query, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS
from ...lifecycle.fastapi import get_logger
from ...services.audit import AuditService
from ...services.authentication import AuthenticationService, AuthenticationError
from ...services.authorization import AuthorizationService

from .deps import get_auth_service, get_authz_service, get_audit_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/audit", tags=["audit"])


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AuditEventResponse(BaseModel):
    """Response schema for a single audit event."""

    id: str
    event_type: str
    action: str
    tenant_id: str
    workspace_id: Optional[str] = None
    user_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    metadata: dict = {}
    timestamp: datetime


class AuditEventsListResponse(BaseModel):
    """Response schema for a list of audit events."""

    events: list[AuditEventResponse]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/events",
    response_model=AuditEventsListResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_audit_events(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    since: Optional[str] = Query(None, description="Return events at or after this ISO datetime"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> AuditEventsListResponse:
    """
    Query audit events for the current tenant.

    Requires admin read authorization.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Optional workspace filter
        event_type: Optional event type filter
        since: Optional ISO datetime lower bound (inclusive)
        limit: Maximum number of events to return (1-1000)
        auth_service: Authentication service
        authz_service: Authorization service
        audit_service: Audit service instance

    Returns:
        List of matching audit events ordered by timestamp descending

    Raises:
        HTTPException: If authentication, authorization, or query fails
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "admin", "read")

        since_dt: Optional[datetime] = None
        if since is not None:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid 'since' datetime format; expected ISO 8601",
                ) from e

        logger.info(
            "Querying audit events: tenant=%s, workspace=%s, event_type=%s, since=%s, limit=%d",
            ctx.tenant_id, workspace_id, event_type, since_dt, limit,
        )

        events = await audit_service.query(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            event_type=event_type,
            since=since_dt,
            limit=limit,
        )

        logger.debug("Audit query returned %d events", len(events))

        return AuditEventsListResponse(
            events=[
                AuditEventResponse(
                    id=e.id,
                    event_type=e.event_type,
                    action=e.action,
                    tenant_id=e.tenant_id,
                    workspace_id=e.workspace_id,
                    user_id=e.user_id,
                    resource_type=e.resource_type,
                    resource_id=e.resource_id,
                    metadata=e.metadata,
                    timestamp=e.timestamp,
                )
                for e in events
            ],
            count=len(events),
        )

    except HTTPException:
        raise
    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Failed to query audit events: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query audit events",
        )


@router.get(
    "/events/summary",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def audit_events_summary(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Filter by workspace ID"),
    since: Optional[str] = Query(None, description="Return events at or after this ISO datetime"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Get audit event counts grouped by event type.

    Requires admin read authorization.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Optional workspace filter
        since: Optional ISO datetime lower bound (inclusive)
        auth_service: Authentication service
        authz_service: Authorization service
        audit_service: Audit service instance

    Returns:
        Dict with ``summary`` (counts by event_type) and ``total`` keys

    Raises:
        HTTPException: If authentication, authorization, or query fails
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "admin", "read")

        since_dt: Optional[datetime] = None
        if since is not None:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid 'since' datetime format; expected ISO 8601",
                ) from e

        logger.info(
            "Querying audit summary: tenant=%s, workspace=%s, since=%s",
            ctx.tenant_id, workspace_id, since_dt,
        )

        # Fetch up to 10 000 events to build the summary; this is an admin-only
        # endpoint so the volume is bounded by the tenant's own data.
        events = await audit_service.query(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            since=since_dt,
            limit=10_000,
        )

        summary: dict[str, int] = {}
        for event in events:
            summary[event.event_type] = summary.get(event.event_type, 0) + 1

        total = sum(summary.values())
        logger.debug("Audit summary: %d event types, %d total events", len(summary), total)

        return {"summary": summary, "total": total}

    except HTTPException:
        raise
    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("Failed to build audit events summary: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build audit events summary",
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

class AuditAPIPlugin(Plugin):
    """Plugin to register audit query routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
