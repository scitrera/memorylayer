"""Default session service implementation."""
from datetime import datetime, timezone, timedelta
from logging import Logger
from typing import Optional, Any

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import SessionServicePluginBase, SessionService
from ...models import Session, WorkingMemory
from ...models.session import SessionBriefing


class InMemorySessionService(SessionService):
    """
    In-memory session management service.

    Sessions are temporary by design and use TTL-based expiration.
    All data is stored in memory and lost on service restart.
    """

    def __init__(self, v: Variables = None):
        """Initialize in-memory session storage."""
        # Store sessions: {workspace_id:session_id -> Session}
        self._sessions: dict[str, Session] = {}
        # Store working memory: {workspace_id:session_id -> {key -> WorkingMemory}}
        self._working_memory: dict[str, dict[str, WorkingMemory]] = {}
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

        # In-memory: auto_commit is logged but doesn't persist to long-term storage
        if session and not skip_auto_commit and session.auto_commit and session.committed_at is None:
            self.logger.debug(
                "Session %s has auto_commit=True but in-memory service doesn't persist commits",
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

        For in-memory sessions, this provides basic stats without storage backend access.
        Advanced features like contradictions and LLM-enhanced summaries require custom implementations.

        Args:
            workspace_id: Workspace identifier
            lookback_minutes: Time window for recent memories (default 60)
            detail_level: Memory detail level - abstract, overview, or full
            limit: Maximum memories to include
            include_memories: Whether to include memory content
            include_contradictions: Whether to detect contradictions

        Returns:
            SessionBriefing with basic workspace summary
        """
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
        Commit session working memory to long-term storage.

        For in-memory sessions, this is a placeholder that returns a basic result.
        Full implementation requires storage backend integration.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to commit
            options: Optional commit configuration

        Returns:
            CommitResult with basic statistics

        Raises:
            ValueError: If session not found or already expired
        """
        from .base import CommitResult

        # Verify session exists
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        # Get working memory
        working_memory_entries = await self.get_all_working_memory(workspace_id, session_id)

        self.logger.info(
            "Commit session %s: %d working memory entries (in-memory placeholder)",
            session_id,
            len(working_memory_entries)
        )

        # Placeholder: return basic result without actual memory creation
        return CommitResult(
            session_id=session_id,
            committed_at=datetime.now(timezone.utc),
            memories_committed=len(working_memory_entries),
            associations_committed=0,
            success=True
        )

    async def touch_session(
        self,
        workspace_id: str,
        session_id: str,
        extend_seconds: Optional[int] = None
    ) -> Session:
        """
        Extend session TTL.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to extend
            extend_seconds: Additional seconds to add (uses 3600 if None)

        Returns:
            Updated session with new expiration time

        Raises:
            ValueError: If session not found or already expired
        """
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        # Extend TTL
        extend_by = extend_seconds or 3600
        session.expires_at = session.expires_at + timedelta(seconds=extend_by)

        # Update in storage
        key = self._make_key(workspace_id, session_id)
        self._sessions[key] = session

        self.logger.info(
            "Extended session %s TTL by %d seconds, new expiration: %s",
            session_id,
            extend_by,
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
    """Default session service plugin."""
    PROVIDER_NAME = 'in-memory'

    def initialize(self, v: Variables, logger: Logger) -> SessionService:
        return InMemorySessionService(v=v)
