"""Session touch task handler."""
from typing import Optional

from scitrera_app_framework import Variables

from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services.session import EXT_SESSION_SERVICE, SessionService

SESSION_TOUCH_HANDLER_TASK = "session_touch"


async def handle_session_touch(
        session_service: SessionService,
        session_id: str,
) -> None:
    session = await session_service.get(session_id)
    logger = session_service.logger
    if not session:
        logger.debug("Session %s not found for touch", session_id)
        return

    try:
        await session_service.touch_session(session.workspace_id, session_id)
        logger.debug("Extended session %s TTL in workspace %s", session_id, session.workspace_id)
    except Exception as e:
        logger.warning("Failed to extend session %s TTL: %s", session_id, e)


class SessionTouchHandler(TaskHandlerPlugin):

    def get_task_type(self) -> str:
        return SESSION_TOUCH_HANDLER_TASK

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        session_service: SessionService = self.get_extension(EXT_SESSION_SERVICE, v)

        session_id = payload.get("session_id")
        if not session_id:
            session_service.logger.debug("Session touch: no session_id in payload")
            return

        await handle_session_touch(session_service, session_id)
