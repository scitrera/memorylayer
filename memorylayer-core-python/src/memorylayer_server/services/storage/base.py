"""Abstract storage backend interface."""
from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Optional, Any

from scitrera_app_framework import get_logger, get_extension
from scitrera_app_framework.api import Variables, Plugin, enabled_option_pattern

from ...config import MEMORYLAYER_STORAGE_BACKEND, DEFAULT_MEMORYLAYER_STORAGE_BACKEND

from ...models.memory import Memory, RememberInput, RecallInput, RecallResult
from ...models.association import Association, AssociateInput, GraphQueryInput, GraphQueryResult
from ...models.workspace import Workspace, Context
if TYPE_CHECKING:
    from ...models import Session, WorkingMemory

from .._constants import EXT_STORAGE_BACKEND


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    """

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)

    # Lifecycle
    @abstractmethod
    async def connect(self) -> None:
        """Initialize storage connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close storage connection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        pass

    # Memory operations
    @abstractmethod
    async def create_memory(self, workspace_id: str, input: RememberInput) -> Memory:
        """Store a new memory."""
        pass

    @abstractmethod
    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Optional[Memory]:
        """Get memory by ID within a workspace. Set track_access=False for internal reads that should not affect decay tracking."""
        pass

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Optional[Memory]:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        raise NotImplementedError("Subclass should implement get_memory_by_id")

    @abstractmethod
    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Optional[Memory]:
        """Update memory fields."""
        pass

    @abstractmethod
    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        pass

    @abstractmethod
    async def search_memories(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            offset: int = 0,
            min_relevance: float = 0.5,
            types: Optional[list[str]] = None,
            subtypes: Optional[list[str]] = None,
            tags: Optional[list[str]] = None,
            include_archived: bool = False,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search, returns (memory, relevance_score) tuples."""
        pass

    @abstractmethod
    async def full_text_search(
            self,
            workspace_id: str,
            query: str,
            limit: int = 10,
            offset: int = 0,
    ) -> list[Memory]:
        """Full-text search on memory content."""
        pass

    @abstractmethod
    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Optional[Memory]:
        """Get memory by content hash for deduplication."""
        pass

    @abstractmethod
    async def get_recent_memories(
        self,
        workspace_id: str,
        created_after: datetime,
        limit: int = 10,
        detail_level: str = "abstract",
        offset: int = 0,
    ) -> list:
        """Get recent memories ordered by creation time (newest first).

        Args:
            workspace_id: Workspace boundary
            created_after: Only return memories created after this time
            limit: Maximum number of memories to return
            detail_level: Level of detail - "abstract", "overview", or "full"
            offset: Number of memories to skip (for pagination)

        Returns:
            List of dicts with memory data, newest first
        """
        pass

    # Association operations
    @abstractmethod
    async def create_association(self, workspace_id: str, input: AssociateInput) -> Association:
        """Create graph edge between memories."""
        pass

    @abstractmethod
    async def get_associations(
            self,
            workspace_id: str,
            memory_id: str,
            direction: str = "both",  # outgoing, incoming, both
            relationships: Optional[list[str]] = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        pass

    @abstractmethod
    async def traverse_graph(
            self,
            workspace_id: str,
            start_id: str,
            max_depth: int = 3,
            relationships: Optional[list[str]] = None,
            direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop graph traversal."""
        pass

    # Workspace operations
    @abstractmethod
    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace."""
        pass

    @abstractmethod
    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        pass

    # Context operations
    @abstractmethod
    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        pass

    @abstractmethod
    async def get_context(self, workspace_id: str, context_id: str) -> Optional[Context]:
        """Get context by ID."""
        pass

    @abstractmethod
    async def list_contexts(self, workspace_id: str) -> list[Context]:
        """List all contexts in a workspace."""
        pass

    @abstractmethod
    async def list_workspaces(self) -> list[Workspace]:
        """List all workspaces."""
        pass

    # Statistics
    @abstractmethod
    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        pass

    # Session operations (for persistent sessions)
    @abstractmethod
    async def create_session(self, workspace_id: str, session: 'Session') -> 'Session':
        """Store a new session."""
        pass

    @abstractmethod
    async def get_session(self, workspace_id: str, session_id: str) -> Optional['Session']:
        """Get session by ID (returns None if not found or expired)."""
        pass

    @abstractmethod
    async def get_session_by_id(self, session_id: str) -> Optional['Session']:
        """Get session by ID without workspace filter.

        Useful when looking up a session from the X-Session-ID header
        when the workspace is not yet known.
        """
        pass

    @abstractmethod
    async def delete_session(self, workspace_id: str, session_id: str) -> bool:
        """Delete session and all its context."""
        pass

    @abstractmethod
    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> 'WorkingMemory':
        """Set working memory key-value within session."""
        pass

    @abstractmethod
    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional['WorkingMemory']:
        """Get specific working memory entry."""
        pass

    @abstractmethod
    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> list['WorkingMemory']:
        """Get all working memory entries for session."""
        pass

    @abstractmethod
    async def cleanup_expired_sessions(self, workspace_id: str) -> int:
        """Delete all expired sessions. Returns number cleaned up."""
        pass

    async def cleanup_all_expired_sessions(self) -> int:
        """Delete all expired sessions across all workspaces. Returns number cleaned up."""
        # Default implementation: no-op (subclasses should override for efficiency)
        return 0

    async def list_expired_sessions(self, limit: int = 100) -> list['Session']:
        """List expired sessions that need cleanup.

        Used by the cleanup task to retrieve sessions before deletion,
        enabling auto-commit of working memory before cleanup.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of expired sessions
        """
        # Default implementation: empty list (subclasses should override)
        return []

    async def update_session(
            self,
            workspace_id: str,
            session_id: str,
            **updates
    ) -> Optional['Session']:
        """Update session fields.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to update
            **updates: Fields to update (e.g., committed_at, expires_at)

        Returns:
            Updated session or None if not found
        """
        # Default implementation: no-op (subclasses should override)
        return None

    async def list_sessions(
            self,
            workspace_id: str,
            context_id: str | None = None,
            include_expired: bool = False,
    ) -> list['Session']:
        """List sessions for a workspace.

        Args:
            workspace_id: Workspace boundary
            context_id: Optional context filter
            include_expired: Whether to include expired sessions

        Returns:
            List of sessions
        """
        return []

    # Decay service support methods (non-abstract with default no-op implementations)

    async def get_memories_for_decay(
        self,
        workspace_id: str,
        min_age_days: int = 7,
        exclude_pinned: bool = True,
    ) -> list[Memory]:
        """Get memories eligible for importance decay. Override in subclasses."""
        return []

    async def get_archival_candidates(
        self,
        workspace_id: str,
        max_importance: float = 0.2,
        max_access_count: int = 3,
        min_age_days: int = 90,
    ) -> list[Memory]:
        """Get memories eligible for archival. Override in subclasses."""
        return []

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs. Override in subclasses."""
        return []

    # Contradiction service support methods (non-abstract with default no-op implementations)

    async def create_contradiction(self, contradiction: 'ContradictionRecord') -> 'ContradictionRecord':
        """Store a contradiction record. Override in subclasses."""
        return contradiction

    async def get_contradiction(self, workspace_id: str, contradiction_id: str) -> Optional['ContradictionRecord']:
        """Get a specific contradiction. Override in subclasses."""
        return None

    async def get_unresolved_contradictions(self, workspace_id: str, limit: int = 10) -> list['ContradictionRecord']:
        """Get unresolved contradictions. Override in subclasses."""
        return []

    async def resolve_contradiction(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: Optional[str] = None,
    ) -> Optional['ContradictionRecord']:
        """Resolve a contradiction. Override in subclasses."""
        return None


# noinspection PyAbstractClass
class StoragePluginBase(Plugin):
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_STORAGE_BACKEND}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_STORAGE_BACKEND

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_STORAGE_BACKEND, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_STORAGE_BACKEND, DEFAULT_MEMORYLAYER_STORAGE_BACKEND)

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, StorageBackend):
            try:
                await value.connect()
                logger.info("Storage backend '%s' connected successfully.", self.PROVIDER_NAME)
            except Exception as e:
                logger.error("Error connecting storage backend '%s': %s", self.PROVIDER_NAME, e)
                raise
        return

    async def async_stopping(self, v: Variables, logger: Logger, value: object | None) -> None:
        if isinstance(value, StorageBackend):
            try:
                await value.disconnect()
                logger.info("Storage backend '%s' disconnected successfully.", self.PROVIDER_NAME)
            except Exception as e:
                logger.error("Error disconnecting storage backend '%s': %s", self.PROVIDER_NAME, e)
        return
