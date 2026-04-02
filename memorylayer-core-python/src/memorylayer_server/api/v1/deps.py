"""Shared FastAPI dependencies for v1 API."""

import logging

from fastapi import Depends, Request
from scitrera_app_framework import Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...services.audit import EXT_AUDIT_SERVICE, AuditService
from ...services.authentication import EXT_AUTHENTICATION_SERVICE, AuthenticationService
from ...services.authorization import EXT_AUTHORIZATION_SERVICE, AuthorizationService
from ...services.cache import EXT_CACHE_SERVICE, CacheService
from ...services.chat import EXT_CHAT_SERVICE, ChatService
from ...services.inference import EXT_INFERENCE_SERVICE, DefaultInferenceService
from ...services.memory import EXT_MEMORY_SERVICE, MemoryService
from ...services.metrics import EXT_METRICS_SERVICE, MetricsService
from ...services.reflect import EXT_REFLECT_SERVICE
from ...services.session import EXT_SESSION_SERVICE, SessionService
from ...services.tasks import EXT_TASK_SERVICE, TaskService
from ...services.workspace import EXT_WORKSPACE_SERVICE, WorkspaceService
from ...tasks.session_touch_handler import SESSION_TOUCH_HANDLER_TASK


async def get_task_service(v: Variables = Depends(get_variables_dep)) -> TaskService:
    return get_extension(EXT_TASK_SERVICE, v)


async def get_active_session(
    http_request: Request,
    task_service: TaskService = Depends(get_task_service),
    logger: logging.Logger = Depends(get_logger),
) -> str | None:
    session_id = http_request.headers.get("X-Session-ID")
    if not session_id:
        return None

    try:
        await task_service.schedule_task(SESSION_TOUCH_HANDLER_TASK, {"session_id": session_id})
    except Exception as e:
        logger.debug("Exception scheduling session touch task: %s", e)

    return session_id


async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


def get_session_service(v: Variables = Depends(get_variables_dep)) -> SessionService:
    """FastAPI dependency wrapper for session service."""
    return get_extension(EXT_SESSION_SERVICE, v)


def get_workspace_service(v: Variables = Depends(get_variables_dep)) -> WorkspaceService:
    """FastAPI dependency wrapper for workspace service."""
    return get_extension(EXT_WORKSPACE_SERVICE, v)


def get_memory_service(v: Variables = Depends(get_variables_dep)) -> MemoryService:
    """FastAPI dependency wrapper for memory service."""
    return get_extension(EXT_MEMORY_SERVICE, v)


def get_inference_service(v: Variables = Depends(get_variables_dep)) -> DefaultInferenceService:
    """FastAPI dependency wrapper for inference service."""
    return get_extension(EXT_INFERENCE_SERVICE, v)


def get_reflect_service(v: Variables = Depends(get_variables_dep)):
    """FastAPI dependency wrapper for reflect service."""
    return get_extension(EXT_REFLECT_SERVICE, v)


def get_cache_service(v: Variables = Depends(get_variables_dep)) -> CacheService:
    """FastAPI dependency wrapper for cache service."""
    return get_extension(EXT_CACHE_SERVICE, v)


def get_chat_service(v: Variables = Depends(get_variables_dep)) -> ChatService:
    """FastAPI dependency wrapper for chat service."""
    return get_extension(EXT_CHAT_SERVICE, v)


def get_audit_service(v: Variables = Depends(get_variables_dep)) -> AuditService:
    """FastAPI dependency wrapper for audit service."""
    return get_extension(EXT_AUDIT_SERVICE, v)


def get_metrics_service(v: Variables = Depends(get_variables_dep)) -> MetricsService:
    """FastAPI dependency wrapper for metrics service."""
    return get_extension(EXT_METRICS_SERVICE, v)
