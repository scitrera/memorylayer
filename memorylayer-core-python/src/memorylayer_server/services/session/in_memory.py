"""Default session service implementation."""
import json
from datetime import datetime, timezone, timedelta
from logging import Logger
from typing import Optional, Any, TYPE_CHECKING

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import SessionServicePluginBase, SessionService, CommitOptions, CommitResult
from ...models import Session, WorkingMemory
from ...models.session import SessionBriefing

if TYPE_CHECKING:
    from ..storage import StorageBackend
    from ..extraction import ExtractionService
    from ..deduplication import DeduplicationService
    from ..memory import MemoryService
    from ..contradiction import ContradictionService
    from ..tasks import TaskService


class InMemorySessionService(SessionService):
    """
    In-memory session management service.

    Sessions are temporary by design and use TTL-based expiration.
    All data is stored in memory and lost on service restart.
    """

    def __init__(
        self,
        v: Variables = None,
        storage: Optional['StorageBackend'] = None,
        extraction_service: Optional['ExtractionService'] = None,
        deduplication_service: Optional['DeduplicationService'] = None,
        memory_service: Optional['MemoryService'] = None,
        contradiction_service: Optional['ContradictionService'] = None,
        task_service: Optional['TaskService'] = None,
        default_touch_ttl: int = 3600,
    ):
        """Initialize in-memory session storage.

        Args:
            v: Variables for dependency injection
            storage: Optional storage backend for briefing/commit operations
            extraction_service: Optional extraction service for commit operations
            deduplication_service: Optional deduplication service
            memory_service: Optional memory service for commit operations
            contradiction_service: Optional contradiction service for briefing
            default_touch_ttl: Default TTL in seconds for touch_session calls
        """
        # Store sessions: {workspace_id:session_id -> Session}
        self._sessions: dict[str, Session] = {}
        # Store working memory: {workspace_id:session_id -> {key -> WorkingMemory}}
        self._working_memory: dict[str, dict[str, WorkingMemory]] = {}

        # Optional service dependencies
        self.storage = storage
        self.extraction_service = extraction_service
        self.deduplication_service = deduplication_service
        self._memory_service = memory_service
        self.contradiction_service = contradiction_service
        self.task_service = task_service
        self.default_touch_ttl = default_touch_ttl

        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized SessionService with in-memory storage")

    def _make_key(self, workspace_id: str, session_id: str) -> str:
        """Create composite key for session storage."""
        return f"{workspace_id}:{session_id}"

    async def create_session(
        self,
        workspace_id: str,
        session: Session,
        context_id: Optional[str] = None
    ) -> Session:
        """
        Store a new session.

        Args:
            workspace_id: Workspace identifier
            session: Session object to store
            context_id: Optional context ID (uses session.context_id if not provided)

        Returns:
            The stored session

        Note:
            If a session with the same ID already exists, it will be replaced.
        """
        key = self._make_key(workspace_id, session.id)
        self._sessions[key] = session
        # Initialize empty working memory dict for this session
        self._working_memory[key] = {}
        self.logger.info(
            "Created session: %s in workspace: %s, context: %s",
            session.id,
            workspace_id,
            session.context_id
        )
        return session

    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """
        Retrieve session if it exists and has not expired.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier

        Returns:
            Session object if found and not expired, None otherwise

        Note:
            Expired sessions are automatically removed when accessed.
        """
        key = self._make_key(workspace_id, session_id)
        session = self._sessions.get(key)

        if session is None:
            self.logger.debug("Session not found: %s in workspace: %s", session_id, workspace_id)
            return None

        # Check expiration
        if session.is_expired:
            self.logger.info("Session expired: %s in workspace: %s, removing", session_id, workspace_id)
            # Clean up expired session
            await self.delete_session(workspace_id, session_id)
            return None

        self.logger.debug("Retrieved session: %s in workspace: %s", session_id, workspace_id)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID without workspace filter.

        Searches all sessions. Within a tenant's session service,
        session IDs are globally unique.
        """
        for key, session in self._sessions.items():
            if key.endswith(f":{session_id}"):
                if session.is_expired:
                    self.logger.info("Session expired: %s, removing", session_id)
                    await self.delete_session(session.workspace_id, session_id)
                    return None
                return session
        return None

    async def delete_session(self, workspace_id: str, session_id: str, skip_auto_commit: bool = False) -> bool:
        """
        Delete session and all its context entries.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier
            skip_auto_commit: If True, skip auto-commit (in-memory service doesn't persist anyway)

        Returns:
            True if session was deleted, False if it didn't exist
        """
        key = self._make_key(workspace_id, session_id)

        # Get session to check auto_commit flag
        session = self._sessions.get(key)

        # Auto-commit if enabled and services are available
        if session and not skip_auto_commit and session.auto_commit and session.committed_at is None:
            if self.extraction_service and self._memory_service:
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
            else:
                self.logger.debug(
                    "Session %s has auto_commit=True but no extraction/memory services configured",
                    session_id
                )

        # Remove session
        session_existed = key in self._sessions
        if session_existed:
            del self._sessions[key]

        # Remove all working memory entries
        if key in self._working_memory:
            del self._working_memory[key]

        if session_existed:
            self.logger.info("Deleted session: %s in workspace: %s", session_id, workspace_id)
        else:
            self.logger.debug("Session not found for deletion: %s in workspace: %s", session_id, workspace_id)

        return session_existed

    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """
        Set a working memory key-value pair within a session.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier
            key: Working memory key
            value: Working memory value (must be JSON-serializable)
            ttl_seconds: Optional TTL override (inherits from session if None)

        Returns:
            The created/updated WorkingMemory

        Raises:
            ValueError: If session doesn't exist or has expired
        """
        # Verify session exists and is not expired
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired in workspace {workspace_id}")

        session_key = self._make_key(workspace_id, session_id)

        # Get or initialize working memory dict for this session
        if session_key not in self._working_memory:
            self._working_memory[session_key] = {}

        # Check if updating existing entry
        existing = self._working_memory[session_key].get(key)
        now = datetime.now(timezone.utc)

        if existing:
            # Update existing entry
            entry = WorkingMemory(
                session_id=session_id,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                created_at=existing.created_at,
                updated_at=now
            )
            self.logger.debug("Updated working memory key: %s in session: %s", key, session_id)
        else:
            # Create new entry
            entry = WorkingMemory(
                session_id=session_id,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
                created_at=now,
                updated_at=now
            )
            self.logger.debug("Created working memory key: %s in session: %s", key, session_id)

        self._working_memory[session_key][key] = entry

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

        return entry

    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional[WorkingMemory]:
        """
        Get a specific working memory entry.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier
            key: Working memory key

        Returns:
            WorkingMemory if found, None if session expired or key doesn't exist
        """
        # Verify session exists and is not expired
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return None

        session_key = self._make_key(workspace_id, session_id)
        entries = self._working_memory.get(session_key, {})
        entry = entries.get(key)

        if entry:
            self.logger.debug("Retrieved working memory key: %s from session: %s", key, session_id)
        else:
            self.logger.debug("Working memory key not found: %s in session: %s", key, session_id)

        return entry

    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> list[WorkingMemory]:
        """
        Get all working memory entries for a session.

        Args:
            workspace_id: Workspace identifier
            session_id: Session identifier

        Returns:
            List of WorkingMemory objects (empty if session expired or has no entries)
        """
        # Verify session exists and is not expired
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return []

        session_key = self._make_key(workspace_id, session_id)
        entries = self._working_memory.get(session_key, {})

        self.logger.debug(
            "Retrieved %d working memory entries from session: %s",
            len(entries),
            session_id
        )

        return list(entries.values())

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

        Uses storage backend when available for comprehensive statistics.
        Falls back to basic in-memory session counting when storage is not available.

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
        # If storage backend is available, use it for comprehensive stats
        if self.storage is not None:
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

        # Fall back to basic in-memory stats when storage is not available
        # Count active (non-expired) sessions in this workspace
        active_sessions = []
        for key, session in self._sessions.items():
            if key.startswith(f"{workspace_id}:") and not session.is_expired:
                active_sessions.append(session)

        # Calculate recent activity (sessions created in last 24 hours)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        recent_sessions = [s for s in active_sessions if s.created_at >= cutoff]

        # Build basic workspace summary
        workspace_summary = {
            "total_sessions": len(active_sessions),
            "recent_sessions": len(recent_sessions),
            "total_memories": 0,  # Not available without storage backend
            "recent_memories": 0,
            "active_topics": [],
            "total_categories": 0,
            "total_associations": 0,
        }

        # Build recent activity list (simple version for OSS)
        recent_activity = []
        for session in sorted(recent_sessions, key=lambda s: s.created_at, reverse=True)[:5]:
            recent_activity.append({
                "timestamp": session.created_at.isoformat(),
                "session_id": session.id,
                "summary": f"Session {session.id} created",
                "metadata": session.metadata,
            })

        self.logger.debug(
            "Generated briefing for workspace %s: %d active sessions, %d recent",
            workspace_id,
            len(active_sessions),
            len(recent_sessions)
        )

        return SessionBriefing(
            workspace_summary=workspace_summary,
            recent_activity=recent_activity,
            open_threads=[],  # Advanced feature - empty for OSS
            contradictions_detected=[],  # Advanced feature - empty for OSS
            memories=[],  # In-memory service has no storage backend
        )

    async def commit_session(
        self,
        workspace_id: str,
        session_id: str,
        options: Optional['CommitOptions'] = None
    ) -> 'CommitResult':
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
        from .base import CommitResult

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
        extend_seconds: Optional[int] = None
    ) -> Session:
        """Extend session TTL using sliding window.

        Resets expires_at to now + TTL. If extend_seconds is provided,
        it overrides the server default for this call.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to extend
            extend_seconds: TTL override for this call (uses server default if None)

        Returns:
            Updated session with new expiration time

        Raises:
            ValueError: If session not found or already expired
        """
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        ttl = extend_seconds if extend_seconds is not None else self.default_touch_ttl
        session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        # Update in storage
        key = self._make_key(workspace_id, session_id)
        self._sessions[key] = session

        self.logger.info(
            "Refreshed session %s TTL to %d seconds, new expiration: %s",
            session_id,
            ttl,
            session.expires_at.isoformat()
        )

        return session

    async def list_sessions(
        self,
        workspace_id: str,
        context_id: Optional[str] = None,
        include_expired: bool = False
    ) -> list[Session]:
        """
        List sessions in a workspace.

        Args:
            workspace_id: Workspace to list sessions for
            context_id: Optional filter by context
            include_expired: Whether to include expired sessions

        Returns:
            List of sessions matching criteria
        """
        sessions = []
        for key, session in self._sessions.items():
            if not key.startswith(f"{workspace_id}:"):
                continue

            # Check expiration filter
            if not include_expired and session.is_expired:
                continue

            # Check context filter
            if context_id and session.context_id != context_id:
                continue

            sessions.append(session)

        return sessions


class InMemorySessionServicePlugin(SessionServicePluginBase):
    """In-memory session service plugin (no persistence)."""
    PROVIDER_NAME = 'in-memory'

    def get_dependencies(self, v: Variables):
        from ..storage import EXT_STORAGE_BACKEND
        from ..extraction import EXT_EXTRACTION_SERVICE
        from ..deduplication import EXT_DEDUPLICATION_SERVICE
        from ..memory import EXT_MEMORY_SERVICE
        from ..contradiction import EXT_CONTRADICTION_SERVICE
        from .._constants import EXT_TASK_SERVICE
        return (EXT_STORAGE_BACKEND, EXT_EXTRACTION_SERVICE, EXT_DEDUPLICATION_SERVICE, EXT_MEMORY_SERVICE, EXT_CONTRADICTION_SERVICE, EXT_TASK_SERVICE)

    def initialize(self, v: Variables, logger: Logger) -> SessionService:
        from ..storage import StorageBackend, EXT_STORAGE_BACKEND
        from ..extraction import ExtractionService, EXT_EXTRACTION_SERVICE
        from ..deduplication import DeduplicationService, EXT_DEDUPLICATION_SERVICE
        from ..memory import MemoryService, EXT_MEMORY_SERVICE
        from ..contradiction import ContradictionService, EXT_CONTRADICTION_SERVICE
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

        return InMemorySessionService(
            v=v,
            storage=storage,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            memory_service=memory_service,
            contradiction_service=contradiction_service,
            task_service=task_service,
            default_touch_ttl=default_touch_ttl,
        )
