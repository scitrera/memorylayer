"""
Tier Generation Task Handler.

Background task handler for generating memory tiers (abstract, overview)
via the TaskService infrastructure.
"""
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger, Variables

from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services.semantic_tiering import SemanticTieringService, EXT_SEMANTIC_TIERING_SERVICE


class TierGenerationTaskHandler(TaskHandlerPlugin):
    """
    Task handler for background tier generation.

    Processes 'generate_tiers' tasks by delegating to the TierGenerationService.
    This is an on-demand handler (no recurring schedule).
    """

    def get_task_type(self) -> str:
        return 'generate_tiers'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None  # On-demand only, not recurring

    async def handle(self, v: Variables, payload: dict) -> None:
        memory_id = payload['memory_id']
        workspace_id = payload['workspace_id']

        tier_service: SemanticTieringService = self.get_extension(EXT_SEMANTIC_TIERING_SERVICE, v)
        logger: Logger = get_logger(v, name=self.get_task_type())

        logger.debug("Generating tiers for memory %s in workspace %s", memory_id, workspace_id)
        await tier_service.generate_tiers(memory_id, workspace_id)
        logger.debug("Completed tier generation for memory %s", memory_id)
