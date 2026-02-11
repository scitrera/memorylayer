"""Decay task handler for periodic background decay."""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..tasks import TaskHandlerPlugin, TaskSchedule
from .base import DecayService, EXT_DECAY_SERVICE


class DecayTaskHandler(TaskHandlerPlugin):
    """
    Periodic decay task handler.

    Runs every 6 hours to decay importance and archive stale memories.
    """

    def get_task_type(self) -> str:
        return 'decay_memories'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return TaskSchedule(
            interval_seconds=6 * 3600,  # Every 6 hours
            default_payload={},
        )

    async def handle(self, payload: dict) -> None:
        decay_service: DecayService = self.get_extension(EXT_DECAY_SERVICE, self._v)
        logger: Logger = get_logger(self._v, name=self.get_task_type())

        workspace_id = payload.get('workspace_id')
        if workspace_id:
            logger.info("Running decay for workspace %s", workspace_id)
            result = await decay_service.decay_workspace(workspace_id)
            archived = await decay_service.archive_stale_memories(workspace_id)
            logger.info(
                "Decay complete for workspace %s: %d decayed, %d archived",
                workspace_id, result.decayed, archived
            )
        else:
            logger.info("Running decay for all workspaces")
            result = await decay_service.decay_all_workspaces()
            logger.info(
                "Decay complete: %d processed, %d decayed, %d archived",
                result.processed, result.decayed, result.archived
            )

    def initialize(self, v, logger) -> 'DecayTaskHandler':
        self._v = v
        return self
