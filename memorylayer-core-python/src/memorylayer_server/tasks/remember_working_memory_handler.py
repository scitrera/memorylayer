"""Write-behind task handler for persisting working memory entries as long-term memories.

Scheduled by set_working_memory() to asynchronously store each working memory
entry via the standard remember pipeline. Working memories are stored with
type=WORKING, which naturally skips fact decomposition.
"""
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services._constants import EXT_MEMORY_SERVICE
from ..models import RememberInput, MemoryType


class RememberWorkingMemoryHandler(TaskHandlerPlugin):
    """
    On-demand task handler for persisting working memory entries.

    Triggered after a working memory entry is written to asynchronously
    store it as a long-term memory via the standard remember pipeline.

    No recurring schedule -- runs only when explicitly submitted via the
    task service.
    """

    def get_task_type(self) -> str:
        return 'remember_working_memory'

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        # No recurring schedule - triggered on-demand after set_working_memory
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get('workspace_id')
        session_id = payload.get('session_id')
        key = payload.get('key')
        content = payload.get('content')
        context_id = payload.get('context_id')
        importance = payload.get('importance', 0.5)

        if not workspace_id or not content:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, content=%s",
                workspace_id, content,
            )
            return

        # Resolve memory service from the framework
        from ..services.memory import MemoryService
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)

        # Build RememberInput
        remember_input = RememberInput(
            content=content,
            type=MemoryType.WORKING,
            importance=importance,
            metadata={"session_id": session_id, "working_memory_key": key},
            context_id=context_id,
        )

        # Store memory via remember pipeline
        try:
            memory = await memory_service.remember(
                workspace_id=workspace_id,
                input=remember_input
            )
            logger.info(
                "Persisted working memory entry as memory %s (session: %s, key: %s)",
                memory.id, session_id, key,
            )
        except Exception as e:
            logger.warning(
                "Failed to persist working memory entry (session: %s, key: %s): %s",
                session_id, key, e,
            )
