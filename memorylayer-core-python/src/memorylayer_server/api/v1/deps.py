"""Shared FastAPI dependencies for v1 API."""

import logging

from fastapi import Depends, Request
from scitrera_app_framework import Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...tasks.session_touch_handler import SESSION_TOUCH_HANDLER_TASK
from ...services.tasks import TaskService, EXT_TASK_SERVICE


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
