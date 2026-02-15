"""Shared FastAPI dependencies for v1 API."""

import logging

from fastapi import Depends, Request
from scitrera_app_framework import Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...services.tasks import TaskService, EXT_TASK_SERVICE
from ...services.authentication import AuthenticationService, EXT_AUTHENTICATION_SERVICE
from ...services.authorization import AuthorizationService, EXT_AUTHORIZATION_SERVICE
from ...services.session import SessionService, EXT_SESSION_SERVICE
from ...services.workspace import WorkspaceService, EXT_WORKSPACE_SERVICE
from ...services.memory import MemoryService, EXT_MEMORY_SERVICE
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
