"""Contradiction check task handler for on-demand contradiction detection."""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..services.contradiction import ContradictionService, EXT_CONTRADICTION_SERVICE
from ..services.tasks import TaskHandlerPlugin, TaskSchedule


class ContradictionCheckTaskHandler(TaskHandlerPlugin):
    """
    On-demand contradiction check task handler.

    Triggered after a new memory is stored to check for contradictions
    with existing memories. No recurring schedule - runs only when
    explicitly submitted via the task service.
    """

    def get_task_type(self) -> str:
        return 'check_contradictions'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        # No recurring schedule - triggered on-demand after remember
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        contradiction_service: ContradictionService = self.get_extension(
            EXT_CONTRADICTION_SERVICE, v
        )
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get('workspace_id')
        memory_id = payload.get('memory_id')

        if not workspace_id or not memory_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, memory_id=%s",
                workspace_id, memory_id,
            )
            return

        logger.info("Checking contradictions for memory %s in workspace %s", memory_id, workspace_id)
        contradictions = await contradiction_service.check_new_memory(workspace_id, memory_id)

        if contradictions:
            logger.info(
                "Found %d contradiction(s) for memory %s in workspace %s",
                len(contradictions), memory_id, workspace_id,
            )
        else:
            logger.debug(
                "No contradictions found for memory %s in workspace %s",
                memory_id, workspace_id,
            )
