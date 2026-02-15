"""Persistent session service using storage backend."""
import json
from datetime import datetime, timezone, timedelta
from logging import Logger
from typing import Optional, Any, List, TYPE_CHECKING

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import SessionServicePluginBase, SessionService, CommitResult, CommitOptions
from ..storage import StorageBackend, EXT_STORAGE_BACKEND
from ..extraction import ExtractionService, EXT_EXTRACTION_SERVICE, ExtractionOptions
from ..deduplication import DeduplicationService, EXT_DEDUPLICATION_SERVICE
from ..memory import MemoryService, EXT_MEMORY_SERVICE
from ..contradiction import ContradictionService, EXT_CONTRADICTION_SERVICE
from ...models import Session, WorkingMemory
from ...models.session import SessionBriefing


class PersistentSessionService(SessionService):
    """Storage-backed session service.

    Sessions persist across server restarts using the configured
    StorageBackend (SQLite, PostgreSQL, etc.).

    This is the recommended session service for production deployments
    where session data should survive server restarts.
    """

    def __init__(
            self,
            storage: StorageBackend,
            v: Variables = None,
            extraction_service: Optional[ExtractionService] = None,
            deduplication_service: Optional[DeduplicationService] = None,
            memory_service: Optional[MemoryService] = None,
            contradiction_service: Optional[ContradictionService] = None,
            task_service: Optional['TaskService'] = None,
            default_touch_ttl: int = 3600,
    ):
        self.storage = storage
        self.extraction_service = extraction_service
        self.deduplication_service = deduplication_service
        self._memory_service = memory_service
        self.contradiction_service = contradiction_service
        self.task_service = task_service
        self.default_touch_ttl = default_touch_ttl
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized PersistentSessionService with storage backend")

    async def create_session(
            self,
            workspace_id: str,
            session: Session,
            context_id: Optional[str] = None
    ) -> Session:
        """Store a new session in storage backend."""
        return await self.storage.create_session(workspace_id, session)

    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """Retrieve session from storage if not expired."""
        return await self.storage.get_session(workspace_id, session_id)

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID without workspace filter."""
        return await self.storage.get_session_by_id(session_id)

    async def delete_session(self, workspace_id: str, session_id: str, skip_auto_commit: bool = False) -> bool:
        """Delete session and all context from storage.

        If the session has auto_commit=True, working memory is committed
        to long-term storage before deletion.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to delete
            skip_auto_commit: If True, skip auto-commit even if session has auto_commit=True

        Returns:
            True if session was deleted, False if not found
        """
        # Get session to check auto_commit flag
        session = await self.storage.get_session(workspace_id, session_id)

        if session is None:
            return False

        # Auto-commit if enabled and not already committed
        if not skip_auto_commit and session.auto_commit and session.committed_at is None:
            try:
                self.logger.info(
                    "Auto-committing session %s before deletion (auto_commit=True)",
                    session_id
                )
                await self.commit_session(workspace_id, session_id)
            except Exception as e:
                self.logger.warning(
                    "Auto-commit failed for session %s, proceeding with deletion: %s",
                    session_id,
                    e
                )

        return await self.storage.delete_session(workspace_id, session_id)

    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """Set working memory in storage backend."""
        # Verify session exists
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        result = await self.storage.set_working_memory(
            workspace_id, session_id, key, value, ttl_seconds
        )

        # Write-behind: persist to long-term memory via background task
        content_str = value if isinstance(value, str) else json.dumps(value, default=str)
        await self.task_service.schedule_task(
            'remember_working_memory',
            {
                'workspace_id': workspace_id,
                'session_id': session_id,
                'key': key,
                'content': content_str,
                'context_id': session.context_id if hasattr(session, 'context_id') else None,
            },
        )

        return result

    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional[WorkingMemory]:
        """Get working memory from storage backend."""
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return None

        return await self.storage.get_working_memory(workspace_id, session_id, key)

    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> List[WorkingMemory]:
        """Get all working memory from storage backend."""
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return []

        return await self.storage.get_all_working_memory(workspace_id, session_id)

    async def cleanup_expired(self, workspace_id: str) -> int:
        """Cleanup expired sessions. Should be called periodically."""
        count = await self.storage.cleanup_expired_sessions(workspace_id)
        if count > 0:
            self.logger.info("Cleaned up %d expired sessions in workspace %s", count, workspace_id)
        return count

    async def get_briefing(
            self,
            workspace_id: str,
            lookback_minutes: int = 60,
            detail_level: str = "abstract",
            limit: int = 10,
            include_memories: bool = True,
            include_contradictions: bool = True,
    ) -> SessionBriefing:
        """
        Generate a session briefing with workspace summary and recent activity.

        Uses storage backend to provide comprehensive statistics about memories,
        associations, and recent activity. Advanced features like contradiction
        detection and LLM-enhanced summaries require custom implementations.

        Args:
            workspace_id: Workspace identifier
            lookback_minutes: Time window for recent memories (default 60)
            detail_level: Memory detail level - abstract, overview, or full
            limit: Maximum memories to include
            include_memories: Whether to include memory content
            include_contradictions: Whether to detect contradictions

        Returns:
            SessionBriefing with workspace summary and activity
        """
        # Get workspace statistics from storage
        stats = await self.storage.get_workspace_stats(workspace_id)

        # Build workspace summary with real data from storage
        workspace_summary = {
            "total_memories": stats.get("total_memories", 0),
            "recent_memories": 0,  # Will calculate below
            "active_topics": [],
            "total_categories": stats.get("total_categories", 0),
            "total_associations": stats.get("total_associations", 0),
            "memory_types": stats.get("memory_types", {}),
        }

        # Get recent memories if requested
        memories = []
        if include_memories:
            now = datetime.now(timezone.utc)
            created_after = now - timedelta(minutes=lookback_minutes)
            memories = await self.storage.get_recent_memories(
                workspace_id, created_after=created_after,
                limit=limit, detail_level=detail_level
            )

            # Update workspace summary with recent count
            workspace_summary["recent_memories"] = len(memories)

        # Build recent activity list
        recent_activity = []
        # Note: Storage backend doesn't track detailed session activity
        # Custom implementations can enhance this with actual activity tracking
        recent_activity.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": f"Workspace stats: {workspace_summary['total_memories']} total memories",
            "memories_created": 0,
            "key_decisions": [],
        })

        self.logger.debug(
            "Generated briefing for workspace %s: %d memories, %d associations, %d recent memories",
            workspace_id,
            workspace_summary["total_memories"],
            workspace_summary["total_associations"],
            workspace_summary["recent_memories"]
        )

        # Get unresolved contradictions
        contradictions_detected = []
        if include_contradictions and self.contradiction_service:
            try:
                records = await self.contradiction_service.get_unresolved(workspace_id, limit=3)
                for record in records:
                    contradictions_detected.append({
                        "id": record.id,
                        "memory_a_id": record.memory_a_id,
                        "memory_b_id": record.memory_b_id,
                        "type": record.contradiction_type,
                        "confidence": record.confidence,
                    })
            except Exception as e:
                self.logger.warning("Failed to get contradictions for briefing: %s", e)

        return SessionBriefing(
            workspace_summary=workspace_summary,
            recent_activity=recent_activity,
            open_threads=[],  # Advanced feature - empty for OSS
            contradictions_detected=contradictions_detected,
            memories=memories,
        )

    async def commit_session(
            self,
            workspace_id: str,
            session_id: str,
            options: Optional[CommitOptions] = None
    ) -> CommitResult:
        """
        Finalize a session and mark it as committed.

        Working memories are persisted to long-term storage via write-behind
        as they are written (in set_working_memory). This method simply marks
        the session as committed and returns statistics.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier
            options: Commit options (retained for API compatibility)

        Returns:
            CommitResult with session statistics

        Raises:
            ValueError: If session doesn't exist or has expired
        """
        # Verify session exists and is not expired
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired in workspace {workspace_id}")

        # Count working memory entries
        working_memory_list = await self.get_all_working_memory(workspace_id, session_id)
        memory_count = len(working_memory_list)

        # Mark session as committed
        committed_at = datetime.now(timezone.utc)
        session.committed_at = committed_at

        # Persist the committed_at timestamp
        try:
            await self.storage.update_session(
                workspace_id,
                session_id,
                committed_at=committed_at
            )
        except Exception as e:
            self.logger.warning(
                "Failed to persist committed_at for session %s: %s",
                session_id, e
            )

        self.logger.info(
            "Committed session %s: %d working memory entries (persisted via write-behind)",
            session_id, memory_count,
        )

        return CommitResult(
            session_id=session_id,
            committed_at=committed_at,
            memories_committed=memory_count,
            memories_extracted=memory_count,
            memories_deduplicated=0,
            success=True,
        )

    async def touch_session(
            self,
            workspace_id: str,
            session_id: str,
            extend_seconds: int | None = None,
    ) -> 'Session':
        """Extend session TTL using sliding window.

        Resets expires_at to now + TTL. If extend_seconds is provided,
        it overrides the server default for this call.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to extend
            extend_seconds: TTL override for this call (uses server default if None)

        Returns:
            Updated session with new expires_at

        Raises:
            ValueError: If session not found in storage
        """
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found in workspace {workspace_id}")

        ttl = extend_seconds if extend_seconds is not None else self.default_touch_ttl
        new_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        updated = await self.storage.update_session(
            workspace_id, session_id, expires_at=new_expires_at
        )
        if updated is None:
            raise ValueError(f"Failed to update session {session_id} in storage")

        self.logger.info(
            "Refreshed session %s TTL to %d seconds, new expiration: %s",
            session_id, ttl, updated.expires_at.isoformat()
        )
        return updated

    async def list_sessions(
            self,
            workspace_id: str,
            context_id: str | None = None,
            include_expired: bool = False
    ) -> list['Session']:
        """List sessions for a workspace.

        Args:
            workspace_id: Workspace boundary
            context_id: Optional context filter
            include_expired: Whether to include expired sessions

        Returns:
            List of sessions
        """
        return await self.storage.list_sessions(
            workspace_id, context_id=context_id, include_expired=include_expired
        )


class PersistentSessionServicePlugin(SessionServicePluginBase):
    """Plugin for persistent session service."""
    PROVIDER_NAME = 'persistent'

    def get_dependencies(self, v: Variables):
        from .._constants import EXT_TASK_SERVICE
        return (EXT_STORAGE_BACKEND, EXT_EXTRACTION_SERVICE, EXT_DEDUPLICATION_SERVICE, EXT_MEMORY_SERVICE, EXT_CONTRADICTION_SERVICE, EXT_TASK_SERVICE)

    def initialize(self, v: Variables, logger: Logger) -> SessionService:
        from ..tasks import TaskService
        from .._constants import EXT_TASK_SERVICE
        from ...config import MEMORYLAYER_SESSION_TOUCH_TTL, DEFAULT_MEMORYLAYER_SESSION_TOUCH_TTL

        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        extraction_service: ExtractionService = self.get_extension(EXT_EXTRACTION_SERVICE, v)
        deduplication_service: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)
        contradiction_service: ContradictionService = self.get_extension(EXT_CONTRADICTION_SERVICE, v)
        task_service: TaskService = self.get_extension(EXT_TASK_SERVICE, v)

        default_touch_ttl: int = v.environ(
            MEMORYLAYER_SESSION_TOUCH_TTL,
            default=DEFAULT_MEMORYLAYER_SESSION_TOUCH_TTL,
            type_fn=int,
        )

        return PersistentSessionService(
            storage=storage,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            memory_service=memory_service,
            contradiction_service=contradiction_service,
            task_service=task_service,
            default_touch_ttl=default_touch_ttl,
            v=v
        )
