"""Chat thread cleanup task handler.

Periodic background task that cleans up expired chat threads.
"""

from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ..services.storage import EXT_STORAGE_BACKEND, StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule

MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL = "MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL"
DEFAULT_CLEANUP_INTERVAL: int = 3600


async def periodic_chat_thread_cleanup_task(
    storage: StorageBackend,
    logger: Logger,
) -> None:
    """
    Task to clean up expired chat threads.

    Fetches all expired threads from storage and deletes them along with their messages.

    Args:
        storage: Storage backend for thread operations
        logger: Logger instance
    """
    logger.debug("Chat Thread Cleanup Task started")

    try:
        cleaned_count = 0

        # Get expired threads in batches
        while True:
            expired_threads = await storage.list_expired_threads(limit=100)
            if not expired_threads:
                break

            for thread in expired_threads:
                try:
                    logger.debug("Deleting expired chat thread %s from workspace %s", thread.id, thread.workspace_id)
                    await storage.delete_thread(thread.workspace_id, thread.id)
                    cleaned_count += 1
                except Exception as e:
                    logger.warning("Failed to delete expired thread %s: %s", thread.id, e)

        if cleaned_count > 0:
            logger.info("Background cleanup removed %d expired chat threads", cleaned_count)

    except Exception as e:
        logger.warning("Chat thread cleanup task failed: %s", e)


class ChatThreadCleanupTaskHandlerPlugin(TaskHandlerPlugin):
    """Task handler for periodic chat thread cleanup."""

    def get_task_type(self) -> str:
        return "cleanup_expired_threads"

    def get_schedule(self, v: Variables) -> Optional["TaskSchedule"]:
        interval: int = v.environ(MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL, default=DEFAULT_CLEANUP_INTERVAL, type_fn=int)
        return TaskSchedule(interval_seconds=interval, default_payload={})

    async def handle(self, v: Variables, payload: dict):
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        logger = get_logger(name=self.get_task_type(), v=v)
        return await periodic_chat_thread_cleanup_task(storage=storage, logger=logger)
