"""
Session Service - Working memory management with TTL-based expiration.

This service manages temporary sessions and their context data in memory.
Sessions automatically expire based on their TTL and are not persisted to disk.

Operations:
- create_session: Register a new working memory session
- get_session: Retrieve session if not expired
- delete_session: Remove session and all its context
- set_context: Store key-value data within a session
- get_context: Retrieve specific context entry
- get_all_context: Get all context entries for a session
- commit_session: Commit session working memory to long-term storage (v2)
- touch_session: Extend session TTL (v2)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any, List, Dict, TYPE_CHECKING

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_SESSION_SERVICE, DEFAULT_MEMORYLAYER_SESSION_SERVICE
from ...models import Session, WorkingMemory

if TYPE_CHECKING:
    from ...models.session import SessionBriefing

from .._constants import EXT_SESSION_SERVICE


class CommitResult:
    """Result of a session commit operation."""

    def __init__(
            self,
            session_id: str,
            committed_at: datetime,
            memories_committed: int = 0,
            associations_committed: int = 0,
            success: bool = True,
            error: Optional[str] = None,
            memories_extracted: int = 0,
            memories_deduplicated: int = 0,
            extraction_summary: Optional[Dict[str, Any]] = None
    ):
        self.session_id = session_id
        self.committed_at = committed_at
        self.memories_committed = memories_committed
        self.associations_committed = associations_committed
        self.success = success
        self.error = error
        self.memories_extracted = memories_extracted
        self.memories_deduplicated = memories_deduplicated
        self.extraction_summary = extraction_summary or {}


class CommitOptions:
    """Options for session commit operation."""

    def __init__(
            self,
            include_working_memory: bool = True,
            importance_threshold: Optional[float] = None,
            delete_after_commit: bool = False,
            tags: Optional[List[str]] = None
    ):
        self.include_working_memory = include_working_memory
        self.importance_threshold = importance_threshold
        self.delete_after_commit = delete_after_commit
        self.tags = tags or []


class SessionService(ABC):
    """Interface for session service."""

    logger: logging.Logger = None

    @abstractmethod
    async def create_session(
            self,
            workspace_id: str,
            session: Session,
            context_id: Optional[str] = None
    ) -> Session:
        """Store a new session.

        Args:
            workspace_id: Workspace boundary for the session
            session: Session object to create
            context_id: Optional context ID (uses session.context_id or _default if not provided)

        Returns:
            Created session with any generated fields
        """
        pass

    @abstractmethod
    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """Retrieve session if not expired."""
        pass

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID without workspace filter.

        This method allows looking up a session when the workspace is not known,
        such as when resolving from the X-Session-ID header during authentication.

        Within a tenant's storage backend, session IDs are globally unique.
        """
        pass

    @abstractmethod
    async def delete_session(self, workspace_id: str, session_id: str, skip_auto_commit: bool = False) -> bool:
        """Delete session and all its context.

        If the session has auto_commit=True and skip_auto_commit=False,
        working memory is committed to long-term storage before deletion.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to delete
            skip_auto_commit: If True, skip auto-commit even if session has auto_commit=True

        Returns:
            True if session was deleted, False if not found
        """
        pass

    @abstractmethod
    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """Store key-value data within a session's working memory."""
        pass

    @abstractmethod
    async def get_working_memory(self, workspace_id: str, session_id: str, key: str) -> Optional[WorkingMemory]:
        """Retrieve specific working memory entry."""
        pass

    @abstractmethod
    async def get_all_working_memory(self, workspace_id: str, session_id: str) -> List[WorkingMemory]:
        """Get all working memory entries for a session."""
        pass

    @abstractmethod
    async def get_briefing(
            self,
            workspace_id: str,
            lookback_minutes: int = 60,
            detail_level: str = "abstract",
            limit: int = 10,
            include_memories: bool = True,
            include_contradictions: bool = True,
    ) -> 'SessionBriefing':
        """Generate a session briefing with workspace summary and recent activity.

        Args:
            workspace_id: Workspace identifier
            lookback_minutes: Time window for recent memories (default 60)
            detail_level: Memory detail level - abstract, overview, or full
            limit: Maximum memories to include
            include_memories: Whether to include memory content
            include_contradictions: Whether to detect contradictions
        """
        pass

    # ========================================
    # v2 Session Lifecycle Methods
    # ========================================

    @abstractmethod
    async def commit_session(
            self,
            workspace_id: str,
            session_id: str,
            options: Optional[CommitOptions] = None
    ) -> CommitResult:
        """Commit session working memory to long-term storage.

        This moves working memory entries to persistent memory storage,
        making them available for future recall operations.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to commit
            options: Optional commit configuration

        Returns:
            CommitResult with commit statistics and status
        """
        pass

    @abstractmethod
    async def touch_session(
            self,
            workspace_id: str,
            session_id: str,
            extend_seconds: Optional[int] = None
    ) -> Session:
        """Extend session TTL.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to extend
            extend_seconds: Additional seconds to add (uses default TTL if None)

        Returns:
            Updated session with new expiration time

        Raises:
            ValueError: If session not found or already expired
        """
        pass

    @abstractmethod
    async def list_sessions(
            self,
            workspace_id: str,
            context_id: Optional[str] = None,
            include_expired: bool = False
    ) -> List[Session]:
        """List sessions in a workspace.

        Args:
            workspace_id: Workspace to list sessions for
            context_id: Optional filter by context
            include_expired: Whether to include expired sessions

        Returns:
            List of sessions matching criteria
        """
        pass


# noinspection PyAbstractClass
class SessionServicePluginBase(Plugin):
    """Base plugin for session service - extensible for custom implementations."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_SESSION_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_SESSION_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_SESSION_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_SESSION_SERVICE, DEFAULT_MEMORYLAYER_SESSION_SERVICE)
