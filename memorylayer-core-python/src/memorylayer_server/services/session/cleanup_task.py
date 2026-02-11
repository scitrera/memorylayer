"""Session cleanup task handler.

Periodic background task that cleans up expired sessions,
optionally auto-committing working memory before deletion.
"""
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, ext_parse_bool, get_logger

from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from ..tasks import TaskHandlerPlugin, TaskSchedule
from .base import EXT_SESSION_SERVICE, SessionService

MEMORYLAYER_BACKGROUND_SESSION_CLEANUP_ENABLED = 'MEMORYLAYER_BACKGROUND_SESSION_CLEANUP_ENABLED'
DEFAULT_CLEANUP_ENABLED = True

MEMORYLAYER_BACKGROUND_SESSION_CLEANUP_INTERVAL = 'MEMORYLAYER_BACKGROUND_SESSION_CLEANUP_INTERVAL'
DEFAULT_CLEANUP_INTERVAL: float = 300

MEMORYLAYER_BACKGROUND_SESSION_AUTO_COMMIT = 'MEMORYLAYER_BACKGROUND_SESSION_AUTO_COMMIT'
DEFAULT_AUTO_COMMIT_ENABLED = True


async def periodic_session_cleanup_task(
        storage: StorageBackend,
        session_service: Optional['SessionService'],
        auto_commit_enabled: bool,
        logger: Logger,
) -> None:
    """
    Task to clean up expired sessions.

    If auto_commit_enabled is True and a session_service is provided,
    sessions with auto_commit=True will have their working memory
    committed to long-term storage before deletion.

    Args:
        storage: Storage backend for session operations
        session_service: Session service for commit operations (optional)
        auto_commit_enabled: Whether to auto-commit before cleanup
        logger: Logger instance
    """
    logger.debug("Session Cleanup Task (auto_commit=%s)", auto_commit_enabled)

    try:
        if auto_commit_enabled and session_service:
            # Get expired sessions before deleting them
            expired_sessions = await storage.list_expired_sessions(limit=100)

            committed_count = 0
            for session in expired_sessions:
                if session.auto_commit and session.committed_at is None:
                    try:
                        logger.debug(
                            "Auto-committing expired session %s before cleanup",
                            session.id
                        )
                        await session_service.commit_session(
                            session.workspace_id,
                            session.id
                        )
                        committed_count += 1
                    except Exception as e:
                        logger.warning(
                            "Auto-commit failed for expired session %s: %s",
                            session.id,
                            e
                        )

            if committed_count > 0:
                logger.info(
                    "Auto-committed %d expired sessions before cleanup",
                    committed_count
                )

        # Now delete all expired sessions
        count = await storage.cleanup_all_expired_sessions()
        if count > 0:
            logger.info("Background cleanup removed %d expired sessions", count)

    except Exception as e:
        logger.warning("Session cleanup failed: %s", e)


class SessionCleanupTaskHandlerPlugin(TaskHandlerPlugin):
    """Task handler for periodic session cleanup."""

    def get_task_type(self) -> str:
        return 'session_cleanup'

    def get_schedule(self, v: Variables) -> Optional['TaskSchedule']:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        session_service: SessionService = self.get_extension(EXT_SESSION_SERVICE, v)

        interval: int = v.environ(
            MEMORYLAYER_BACKGROUND_SESSION_CLEANUP_INTERVAL,
            default=DEFAULT_CLEANUP_INTERVAL,
            type_fn=int
        )
        auto_commit_enabled: bool = v.environ(
            MEMORYLAYER_BACKGROUND_SESSION_AUTO_COMMIT,
            default=DEFAULT_AUTO_COMMIT_ENABLED,
            type_fn=ext_parse_bool
        )
        return TaskSchedule(interval_seconds=interval, default_payload={
            'storage': storage,
            'session_service': session_service,
            'auto_commit_enabled': auto_commit_enabled,
            'logger': get_logger(name=self.get_task_type(), v=v)
        })

    async def handle(self, payload: dict):
        return await periodic_session_cleanup_task(**payload)
