"""Write-behind full-text reindex task handler.

Triggered after a memory's content or aliases change, to reconcile any
materialized full-text index (e.g. SQLite FTS5) that the update path does not
touch inline. Delegates to ``StorageBackend.reindex_memory`` (a no-op on
backends that build their full-text representation at query time).
"""

from logging import Logger

from scitrera_app_framework import Variables, get_logger

from ..services.storage import EXT_STORAGE_BACKEND, StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule


class ReindexMemoryTaskHandler(TaskHandlerPlugin):
    """On-demand handler that rebuilds a memory's full-text index row."""

    def get_task_type(self) -> str:
        return "reindex_memory"

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        # No recurring schedule -- runs only when submitted after an update.
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get("workspace_id")
        memory_id = payload.get("memory_id")
        if not workspace_id or not memory_id:
            logger.warning("Missing required payload fields: workspace_id=%s, memory_id=%s", workspace_id, memory_id)
            return

        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        await storage.reindex_memory(workspace_id, memory_id)
        logger.debug("Reindexed full-text index for memory %s", memory_id)
