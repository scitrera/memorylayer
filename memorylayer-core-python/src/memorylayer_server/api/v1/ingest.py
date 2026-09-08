"""Source ingest API endpoints (P4.4): the email source adapter.

Email is a first-class ingestion source alongside chat and documents. This
endpoint normalizes an inbound email into a ``RememberInput`` (via
``services.ingest.email.email_to_remember_input``) and hands it to the SAME
``MemoryService.remember()`` pipeline chat + document ingestion use — so the
email gets the identical decompose + enrich + entity-accretion lifecycle through
``enqueue_post_store`` (no parallel enrichment fork).

The sender becomes the ``observer_id`` perspective anchor: emails are prefix-less
like documents (no ``[ts] Speaker:`` dialogue prefix to parse), so the sender
FIELD is the authoritative perspective anchor — exactly the doc-path pattern.

Endpoint:
- POST /v1/ingest/email - ingest an email as a memory
"""

import logging
import time as _time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.lifecycle.fastapi import get_logger

from ...services.audit import AuditEvent, AuditService
from ...services.authentication import AuthenticationError, AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.ingest import email_to_remember_input
from ...services.memory import MemoryService
from ...services.metrics import MetricsService
from .. import EXT_MULTI_API_ROUTERS
from .deps import (
    get_active_session,
    get_audit_service,
    get_auth_service,
    get_authz_service,
    get_memory_service,
    get_metrics_service,
)
from .schemas import EmailIngestRequest, ErrorResponse, MemoryResponse

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


@router.post(
    "/email",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def ingest_email(
    http_request: Request,
    request: EmailIngestRequest,
    session_id: str = Depends(get_active_session),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    memory_service: MemoryService = Depends(get_memory_service),
    audit_service: AuditService = Depends(get_audit_service),
    metrics_service: MetricsService = Depends(get_metrics_service),
    logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """Normalize an email into a memory on the shared remember() pipeline.

    The email is mapped to a ``RememberInput`` (sender -> observer_id, body
    grounded with a ``[date] Subject:`` prefix, source=EMAIL, recipients in
    metadata) and stored via ``memory_service.remember()`` so it rides the same
    decompose + enrich + accretion lifecycle as chat and document memories.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        # Reuse the memory create permission: an ingested email IS a memory write.
        await authz_service.require_authorization(ctx, "memories", "create", workspace_id=ctx.workspace_id)
    except AuthenticationError as e:
        logger.warning("Authentication failed for email ingest: %s", e)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization failed for email ingest: %s", e)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    try:
        remember_input = email_to_remember_input(
            sender=request.sender,
            body=request.body,
            to=request.to,
            subject=request.subject,
            timestamp=request.timestamp,
            thread_id=request.thread_id,
            message_id=request.message_id,
            importance=request.importance,
            context_id=request.context_id or ctx.context_id,
            extra_metadata=request.metadata,
            relations=request.relations,
        )
    except ValueError as e:
        logger.warning("Invalid email ingest request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        _t0 = _time.monotonic()
        memory = await memory_service.remember(workspace_id=ctx.workspace_id, input=remember_input)
        logger.info("Ingested email as memory: %s", memory.id)
        try:
            metrics_service.counter("memorylayer_ingest_email_total", labels={"workspace": ctx.workspace_id})
            metrics_service.histogram(
                "memorylayer_ingest_email_duration_seconds", _time.monotonic() - _t0, labels={"workspace": ctx.workspace_id}
            )
        except Exception:
            logger.debug("Metrics recording failed for email ingest")
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
            logger.debug("Audit record failed for email ingest")
        return MemoryResponse(memory=memory)
    except ValueError as e:
        logger.warning("Invalid email ingest content: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to ingest email: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to ingest email")


class IngestAPIPlugin(Plugin):
    """Plugin to register the source-ingest API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
