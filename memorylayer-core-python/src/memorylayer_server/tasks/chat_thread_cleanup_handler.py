"""Chat thread cleanup task handler.

Periodic background task that enforces thread lifecycle:

1. Absolute expiry: threads with ``expires_at`` in the past are hard-deleted
   (the explicit per-thread ephemerality knob; unchanged).
2. Idle policy: threads carry a per-thread ``idle_action`` (``'hide'`` or
   ``'delete'``). Once a thread has been idle longer than the server idle
   threshold (``MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS``, measured on
   ``updated_at``), the action is applied — 'delete' removes it, 'hide'
   archives it (sets ``hidden_at``).
3. Post-archive grace purge: hidden threads older than
   ``MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS`` (measured on ``hidden_at``)
   are removed for space. ``0``/negative disables this. Covers both manually
   archived and auto-hidden threads. Revival (a new append) clears
   ``hidden_at`` and bumps ``updated_at``, so the clock never goes stale.

Decompose-before-delete: any time a thread with undecomposed messages is about
to be removed (or archived), a ``chat_decomposition`` task is scheduled and the
DELETE is skipped for this cycle. The thread is removed on a later cycle once
its watermark has advanced — so we never drop unextracted content. (A thread
whose decomposition perpetually fails is never deleted; accepted tradeoff.)
"""

from datetime import UTC, datetime, timedelta
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ..config import (
    DEFAULT_MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS,
    DEFAULT_MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS,
    MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS,
    MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS,
)
from ..services._constants import EXT_TASK_SERVICE
from ..services.storage import EXT_STORAGE_BACKEND, StorageBackend
from ..services.tasks import TaskHandlerPlugin, TaskSchedule, TaskService
from .chat_decomposition_handler import CHAT_DECOMPOSITION_TASK

MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL = "MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL"
DEFAULT_CLEANUP_INTERVAL: int = 3600

# Max threads processed per idle/grace sweep per run. Deferred (decompose-first)
# or failed threads simply retry on the next interval — so we use a single
# bounded batch rather than draining, which avoids a tight re-fetch loop when a
# delete is deferred.
_SWEEP_BATCH_LIMIT = 500


async def periodic_chat_thread_cleanup_task(
    storage: StorageBackend,
    task_service: Optional[TaskService],
    v: Variables,
    logger: Logger,
) -> None:
    """Run the chat-thread lifecycle sweeps. See module docstring."""
    logger.debug("Chat Thread Cleanup Task started")

    now = datetime.now(UTC)
    idle_days = v.environ(
        MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS,
        default=DEFAULT_MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS,
        type_fn=int,
    )
    hidden_delete_days = v.environ(
        MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS,
        default=DEFAULT_MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS,
        type_fn=int,
    )
    idle_before = now - timedelta(days=idle_days)

    async def _schedule_decompose(thread) -> None:
        if task_service is None:
            return
        try:
            await task_service.schedule_task(
                CHAT_DECOMPOSITION_TASK,
                {"workspace_id": thread.workspace_id, "thread_id": thread.id, "user_id": thread.user_id},
            )
        except Exception as e:
            logger.warning("Failed to schedule decomposition for thread %s: %s", thread.id, e)

    async def _decompose_or_delete(thread) -> str:
        """Delete the thread, or defer (schedule decomposition) if it has an
        undecomposed tail. Returns 'deleted' | 'deferred' | 'error'."""
        if thread.unprocessed_count > 0:
            await _schedule_decompose(thread)
            return "deferred"
        try:
            await storage.delete_thread(thread.workspace_id, thread.id, user_id=thread.user_id)
            return "deleted"
        except Exception as e:
            logger.warning("Failed to delete thread %s: %s", thread.id, e)
            return "error"

    # ------------------------------------------------------------------
    # 1. Absolute expires_at -> hard delete (drains; these always delete).
    # ------------------------------------------------------------------
    expired_deleted = 0
    try:
        while True:
            batch = await storage.list_expired_threads(limit=100)
            if not batch:
                break
            removed_any = False
            for thread in batch:
                try:
                    await storage.delete_thread(thread.workspace_id, thread.id, user_id=thread.user_id)
                    expired_deleted += 1
                    removed_any = True
                except Exception as e:
                    logger.warning("Failed to delete expired thread %s: %s", thread.id, e)
            if not removed_any:
                break  # avoid a tight loop if a batch can't be deleted
    except Exception as e:
        logger.warning("Expired-thread sweep failed: %s", e)

    # ------------------------------------------------------------------
    # 2. idle_action='delete' past idle threshold -> decompose-or-delete.
    # ------------------------------------------------------------------
    idle_deleted = idle_deferred = 0
    try:
        for thread in await storage.list_idle_threads(
            idle_before, idle_action="delete", only_hidden=None, limit=_SWEEP_BATCH_LIMIT
        ):
            outcome = await _decompose_or_delete(thread)
            idle_deleted += outcome == "deleted"
            idle_deferred += outcome == "deferred"
    except Exception as e:
        logger.warning("Idle delete-action sweep failed: %s", e)

    # ------------------------------------------------------------------
    # 3. Post-archive grace purge (hidden_at older than the grace window).
    #    Disabled when hidden_delete_days <= 0.
    # ------------------------------------------------------------------
    grace_deleted = grace_deferred = 0
    if hidden_delete_days > 0:
        hidden_before = now - timedelta(days=hidden_delete_days)
        try:
            for thread in await storage.list_hidden_threads(hidden_before, limit=_SWEEP_BATCH_LIMIT):
                outcome = await _decompose_or_delete(thread)
                grace_deleted += outcome == "deleted"
                grace_deferred += outcome == "deferred"
        except Exception as e:
            logger.warning("Hidden-thread grace purge failed: %s", e)

    # ------------------------------------------------------------------
    # 4. idle_action='hide' past idle threshold, not yet hidden -> archive.
    #    Schedule a decomposition for any undecomposed tail first (hide is
    #    non-destructive, so we always proceed to hide).
    # ------------------------------------------------------------------
    hidden = 0
    try:
        for thread in await storage.list_idle_threads(
            idle_before, idle_action="hide", only_hidden=False, limit=_SWEEP_BATCH_LIMIT
        ):
            if thread.unprocessed_count > 0:
                await _schedule_decompose(thread)
            try:
                await storage.hide_thread(thread.workspace_id, thread.id, user_id=thread.user_id)
                hidden += 1
            except Exception as e:
                logger.warning("Failed to hide thread %s: %s", thread.id, e)
    except Exception as e:
        logger.warning("Idle hide-action sweep failed: %s", e)

    if expired_deleted or idle_deleted or idle_deferred or grace_deleted or grace_deferred or hidden:
        logger.info(
            "Chat thread cleanup: expired_deleted=%d idle_deleted=%d idle_deferred=%d "
            "grace_deleted=%d grace_deferred=%d hidden=%d",
            expired_deleted, idle_deleted, idle_deferred, grace_deleted, grace_deferred, hidden,
        )


class ChatThreadCleanupTaskHandlerPlugin(TaskHandlerPlugin):
    """Task handler for periodic chat thread lifecycle cleanup."""

    def get_task_type(self) -> str:
        return "cleanup_expired_threads"

    def get_schedule(self, v: Variables) -> Optional["TaskSchedule"]:
        interval: int = v.environ(MEMORYLAYER_BACKGROUND_CHAT_THREAD_CLEANUP_INTERVAL, default=DEFAULT_CLEANUP_INTERVAL, type_fn=int)
        return TaskSchedule(interval_seconds=interval, default_payload={})

    async def handle(self, v: Variables, payload: dict):
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        # Task service is needed to schedule decomposition before delete; resolve
        # best-effort so the sweep still runs (without deferral) if it's absent.
        try:
            task_service: Optional[TaskService] = self.get_extension(EXT_TASK_SERVICE, v)
        except Exception:
            task_service = None
        logger = get_logger(name=self.get_task_type(), v=v)
        return await periodic_chat_thread_cleanup_task(storage=storage, task_service=task_service, v=v, logger=logger)
