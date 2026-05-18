"""Abstract storage backend interface."""

from abc import ABC, abstractmethod
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any, Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import DEFAULT_MEMORYLAYER_STORAGE_BACKEND, MEMORYLAYER_STORAGE_BACKEND
from ...models.association import AssociateInput, Association, GraphQueryResult
from ...models.memory import Memory, RememberInput
from ...models.workspace import Context, Workspace

if TYPE_CHECKING:
    from ...models import Session, WorkingMemory
    from ...models.chat import ChatMessage, ChatThread, MessageInput
    from ...models.data_provider import DataProvider
    from ...models.document import Document, DocumentPage, IngestionJob
    from ...models.mcp_server import McpServer
    from ...models.skill import Skill, SkillFile

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
    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID within a workspace. Set track_access=False for internal reads that should not affect decay tracking."""
        pass

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        raise NotImplementedError("Subclass should implement get_memory_by_id")

    @abstractmethod
    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
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
        types: list[str] | None = None,
        subtypes: list[str] | None = None,
        tags: list[str] | None = None,
        include_archived: bool = False,
        observer_id: str | None = None,
        subject_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        user_id: str | None = None,
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
        context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search on memory content.

        Args:
            workspace_id: Workspace boundary
            query: Search text
            limit: Maximum results
            offset: Pagination offset
            context_id: If provided, restrict results to this context partition
        """
        pass

    @abstractmethod
    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Memory | None:
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
        relationships: list[str] | None = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        pass

    @abstractmethod
    async def traverse_graph(
        self,
        workspace_id: str,
        start_id: str,
        max_depth: int = 3,
        relationships: list[str] | None = None,
        direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop graph traversal."""
        pass

    async def get_associations_batch(
        self,
        workspace_id: str,
        memory_ids: list[str],
        direction: str = "outgoing",
        relationships: list[str] | None = None,
    ) -> list[Association]:
        """Get associations for multiple memories in one call.

        Default implementation loops over get_associations(). Storage backends
        should override with a single batched query for efficiency.

        Args:
            workspace_id: Workspace boundary
            memory_ids: Memory IDs to fetch associations for
            direction: outgoing, incoming, or both
            relationships: Filter to these relationship types

        Returns:
            Deduplicated list of associations across all requested memories.
        """
        seen: set[str] = set()
        result: list[Association] = []
        for mem_id in memory_ids:
            assocs = await self.get_associations(
                workspace_id=workspace_id,
                memory_id=mem_id,
                direction=direction,
                relationships=relationships,
            )
            for a in assocs:
                if a.id not in seen:
                    seen.add(a.id)
                    result.append(a)
        return result

    async def delete_association(self, workspace_id: str, association_id: str) -> bool:
        """Delete an association by ID.

        Args:
            workspace_id: Workspace boundary
            association_id: Association to delete

        Returns:
            True if deleted, False if not found
        """
        return False

    async def update_association(
        self,
        workspace_id: str,
        association_id: str,
        metadata: dict | None = None,
        strength: float | None = None,
    ) -> bool:
        """Update an existing association's metadata and/or strength.

        Args:
            workspace_id: Workspace boundary
            association_id: Association to update
            metadata: New metadata dict (replaces existing if provided)
            strength: New strength value (replaces existing if provided)

        Returns:
            True if updated, False if not found
        """
        return False

    # Filtered memory search (non-vector)
    async def search_memories_by_filter(
        self,
        workspace_id: str,
        *,
        subtypes: list[str] | None = None,
        tags: list[str] | None = None,
        metadata_filter: dict[str, str] | None = None,
        status: str = "active",
        context_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by subtype, tags, and/or metadata without requiring embeddings.

        This enables efficient filtered queries (e.g., find all RPG nodes by subtype)
        without relying on full-text search hacks or fetching all memories.

        Args:
            workspace_id: Workspace boundary
            subtypes: Filter to memories with these subtypes
            tags: Filter to memories containing all of these tags
            metadata_filter: Exact-match filter on metadata keys (e.g., {"rpg_node_id": "src/main.py"})
            status: Memory status filter (default "active")
            context_id: Filter to memories in this context (None = all contexts)
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching Memory objects
        """
        return []

    # Workspace operations
    @abstractmethod
    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace."""
        pass

    @abstractmethod
    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get workspace by ID."""
        pass

    # Context operations
    @abstractmethod
    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        pass

    @abstractmethod
    async def get_context(self, workspace_id: str, context_id: str) -> Context | None:
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

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all associated data. Override in subclasses."""
        return False

    async def update_workspace(self, workspace_id: str, **updates) -> Optional[Workspace]:
        """Update workspace fields (name, settings, etc.).

        Args:
            workspace_id: Workspace to update
            **updates: Fields to update (e.g., name="New Name", settings={...})

        Returns:
            Updated workspace or None if not found
        """
        return None

    # Statistics
    @abstractmethod
    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        pass

    # Session operations (for persistent sessions)
    @abstractmethod
    async def create_session(self, workspace_id: str, session: "Session") -> "Session":
        """Store a new session."""
        pass

    @abstractmethod
    async def get_session(self, workspace_id: str, session_id: str) -> Optional["Session"]:
        """Get session by ID (returns None if not found or expired)."""
        pass

    @abstractmethod
    async def get_session_by_id(self, session_id: str) -> Optional["Session"]:
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
        self, workspace_id: str, session_id: str, key: str, value: Any, ttl_seconds: int | None = None
    ) -> "WorkingMemory":
        """Set working memory key-value within session."""
        pass

    @abstractmethod
    async def get_working_memory(self, workspace_id: str, session_id: str, key: str) -> Optional["WorkingMemory"]:
        """Get specific working memory entry."""
        pass

    @abstractmethod
    async def get_all_working_memory(self, workspace_id: str, session_id: str) -> list["WorkingMemory"]:
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

    async def list_expired_sessions(self, limit: int = 100) -> list["Session"]:
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

    async def update_session(self, workspace_id: str, session_id: str, **updates) -> Optional["Session"]:
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
    ) -> list["Session"]:
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
        max_importance: float = 0.3,
        max_access_count: int = 5,
        older_than_days: int = 90,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories eligible for archival. Override in subclasses."""
        return []

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs. Override in subclasses."""
        return []

    # Contradiction service support methods (non-abstract with default no-op implementations)

    async def create_contradiction(self, contradiction: "ContradictionRecord") -> "ContradictionRecord":
        """Store a contradiction record. Override in subclasses."""
        return contradiction

    async def get_contradiction(self, workspace_id: str, contradiction_id: str) -> Optional["ContradictionRecord"]:
        """Get a specific contradiction. Override in subclasses."""
        return None

    async def get_unresolved_contradictions(self, workspace_id: str, limit: int = 10) -> list["ContradictionRecord"]:
        """Get unresolved contradictions. Override in subclasses."""
        return []

    async def resolve_contradiction(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: str | None = None,
    ) -> Optional["ContradictionRecord"]:
        """Resolve a contradiction. Override in subclasses."""
        return None

    # Chat history operations (non-abstract with default no-op implementations)

    async def create_thread(self, thread: "ChatThread") -> "ChatThread":
        """Store a new chat thread. Override in subclasses."""
        return thread

    async def get_thread(self, workspace_id: str, thread_id: str) -> Optional["ChatThread"]:
        """Get chat thread by ID. Override in subclasses."""
        return None

    async def list_threads(
        self,
        workspace_id: str,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        scope_filter: str | None = None,
    ) -> list["ChatThread"]:
        """List chat threads in a workspace. Override in subclasses."""
        return []

    async def update_thread(self, workspace_id: str, thread_id: str, **updates) -> Optional["ChatThread"]:
        """Update thread fields. Override in subclasses."""
        return None

    async def delete_thread(self, workspace_id: str, thread_id: str) -> bool:
        """Delete a thread and all its messages. Override in subclasses."""
        return False

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        messages: list["MessageInput"],
    ) -> list["ChatMessage"]:
        """Append messages to a thread. Override in subclasses."""
        return []

    async def get_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        after_index: int | None = None,
        order: str = "asc",
    ) -> list["ChatMessage"]:
        """Get messages from a thread. Override in subclasses."""
        return []

    async def get_message_count(self, workspace_id: str, thread_id: str) -> int:
        """Get total message count for a thread. Override in subclasses."""
        return 0

    async def delete_message(self, workspace_id: str, thread_id: str, message_id: str) -> bool:
        """Delete a single message by ID within a thread and workspace.

        Returns True if the message was found and deleted, False if not found.
        Idempotent: a missing message returns False without raising.
        Override in subclasses.
        """
        return False

    async def list_expired_threads(self, limit: int = 100) -> list["ChatThread"]:
        """List expired chat threads across all workspaces.

        Enables efficient cleanup of expired threads via background tasks.

        Args:
            limit: Maximum number of threads to return

        Returns:
            List of expired ChatThread objects
        """
        # Default implementation: empty list (subclasses should override)
        return []

    # Document operations (non-abstract with default NotImplementedError)

    async def create_document(self, workspace_id: str, doc: "Document") -> "Document":
        """Store a new document record. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def get_document(self, workspace_id: str, doc_id: str) -> "Document | None":
        """Get document by ID within a workspace. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def list_documents(
        self,
        workspace_id: str,
        status: str | None = None,
        document_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list["Document"], int]:
        """List documents in a workspace. Returns (documents, total_count). Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def update_document(self, workspace_id: str, doc_id: str, **updates) -> "Document | None":
        """Update document fields. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def delete_document(self, workspace_id: str, doc_id: str, delete_memories: bool = False) -> bool:
        """Delete a document and optionally cascade to memories. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def find_document_by_hash(self, workspace_id: str, content_hash: str) -> "Document | None":
        """Find document by content hash for deduplication. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    async def get_document_memories(self, workspace_id: str, doc_id: str) -> list[Memory]:
        """Get all memories created from a document. Override in subclasses."""
        raise NotImplementedError("Document storage not implemented by this backend")

    # Document page operations

    async def create_page(self, workspace_id: str, document_id: str, page: "DocumentPage") -> "DocumentPage":
        """Store a document page. Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def get_pages(self, document_id: str, workspace_id: str | None = None) -> list["DocumentPage"]:
        """Get all pages for a document, ordered by page_no. Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def get_page(self, page_id: str) -> "DocumentPage | None":
        """Get a single page by ID. Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def update_page(self, page_id: str, **updates) -> "DocumentPage | None":
        """Update page fields (transcript, embedding, etc.). Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def search_pages_by_maxsim(
        self,
        workspace_id: str,
        query_multivector: list[list[float]],
        limit: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple["DocumentPage", float]]:
        """Search document pages by ColPali MaxSim (ColBERT-style late interaction).

        Returns ``(page, score)`` tuples sorted by descending score. Backends
        without multi-vector page support should leave this as a
        ``NotImplementedError``. The OSS SQLite backend scores in Python after
        loading candidate page multivectors; the Enterprise PostgreSQL backend
        pushes the scoring into the database for higher throughput at scale.

        Args:
            workspace_id: Workspace scope.
            query_multivector: Query multi-vector embedding (list of token vectors).
            limit: Maximum results to return.
            doc_ids: Optional list of document IDs to restrict search.
        """
        raise NotImplementedError("MaxSim page search not implemented by this backend")

    # Ingestion job operations

    async def create_job(self, job: "IngestionJob") -> "IngestionJob":
        """Store an ingestion job. Override in subclasses."""
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    async def get_job(self, job_id: str, workspace_id: str | None = None) -> "IngestionJob | None":
        """Get ingestion job by ID. Override in subclasses."""
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    async def list_jobs(
        self,
        workspace_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list["IngestionJob"]:
        """List ingestion jobs for a workspace. Override in subclasses."""
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    async def update_job(self, job_id: str, **updates) -> "IngestionJob | None":
        """Update ingestion job fields (status, progress, etc.). Override in subclasses."""
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    # Data provider operations

    async def create_data_provider(self, workspace_id: str, provider: "DataProvider") -> "DataProvider":
        """Store a data provider. Override in subclasses."""
        raise NotImplementedError("Data provider storage not implemented by this backend")

    async def get_data_provider(self, workspace_id: str, provider_id: str) -> "DataProvider | None":
        """Get data provider by ID. Override in subclasses."""
        raise NotImplementedError("Data provider storage not implemented by this backend")

    async def list_data_providers(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list["DataProvider"], int]:
        """List data providers for a workspace. Returns (providers, total_count). Override in subclasses."""
        raise NotImplementedError("Data provider storage not implemented by this backend")

    async def update_data_provider(self, workspace_id: str, provider_id: str, **updates) -> "DataProvider | None":
        """Update data provider fields. Override in subclasses."""
        raise NotImplementedError("Data provider storage not implemented by this backend")

    async def delete_data_provider(self, workspace_id: str, provider_id: str) -> bool:
        """Delete a data provider. Override in subclasses."""
        raise NotImplementedError("Data provider storage not implemented by this backend")

    # Knowledgebase article operations

    async def store_kb_article(
        self,
        workspace_id: str,
        article_id: str,
        article_type: str,
        title: str,
        content_md: str,
        metadata: dict | None = None,
    ) -> dict:
        """Store a knowledgebase article. Override in subclasses."""
        raise NotImplementedError("Knowledgebase storage not implemented by this backend")

    async def get_kb_article(self, workspace_id: str, article_id: str) -> dict | None:
        """Get a knowledgebase article by ID. Override in subclasses."""
        raise NotImplementedError("Knowledgebase storage not implemented by this backend")

    async def list_kb_articles(
        self,
        workspace_id: str,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List knowledgebase articles for a workspace. Override in subclasses."""
        raise NotImplementedError("Knowledgebase storage not implemented by this backend")

    async def delete_kb_articles(self, workspace_id: str) -> int:
        """Delete all knowledgebase articles for a workspace (for regeneration). Override in subclasses."""
        raise NotImplementedError("Knowledgebase storage not implemented by this backend")

    async def store_graph_analysis(self, workspace_id: str, analysis_json: dict) -> dict:
        """Cache a graph analysis result. Override in subclasses."""
        raise NotImplementedError("Graph analysis storage not implemented by this backend")

    async def get_graph_analysis(self, workspace_id: str) -> dict | None:
        """Get cached graph analysis for a workspace. Override in subclasses."""
        raise NotImplementedError("Graph analysis storage not implemented by this backend")

    # Skill operations

    async def create_skill(self, skill: "Skill") -> "Skill":
        """Store a new skill. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def get_skill(self, workspace_id: str, skill_id: str) -> "Skill | None":
        """Get skill by ID within a workspace. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def get_skill_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: Optional[str] = None,
    ) -> "Skill | None":
        """Get skill by name within a workspace, optionally filtering by user scope. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def list_skills(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list[str]] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["Skill"]:
        """List skills in a workspace with optional filters. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def find_skills_by_name(
        self,
        name: str,
        scope_filters: list[dict],
    ) -> list["Skill"]:
        """Find skills by name across multiple scopes for precedence resolution.

        Each entry in scope_filters is a dict with ``workspace_id`` and
        optional ``user_id`` keys. Returns all matching skills across the
        given scopes so the caller can apply precedence ordering.
        Override in subclasses.
        """
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def update_skill(
        self,
        workspace_id: str,
        skill_id: str,
        updates: dict,
    ) -> "Skill | None":
        """Update skill fields. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Delete a skill and cascade to its files. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def upsert_skill_file(self, skill_file: "SkillFile") -> "SkillFile":
        """Insert or update a file within a skill bundle. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def get_skill_file(self, skill_id: str, path: str) -> "SkillFile | None":
        """Get a single skill file by skill ID and relative path. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def list_skill_files(self, skill_id: str) -> list["SkillFile"]:
        """List all files belonging to a skill bundle. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def delete_skill_file(self, skill_id: str, path: str) -> bool:
        """Delete a single skill file by path. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    # MCP server operations

    async def create_mcp_server(self, server: "McpServer") -> "McpServer":
        """Store a new MCP server record. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def get_mcp_server(self, workspace_id: str, server_id: str) -> "McpServer | None":
        """Get MCP server by ID within a workspace. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def get_mcp_server_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: Optional[str] = None,
    ) -> "McpServer | None":
        """Get MCP server by name within a workspace, optionally filtering by user scope. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        transport: Optional[str] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["McpServer"]:
        """List MCP servers in a workspace with optional filters. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def find_mcp_servers_by_name(
        self,
        name: str,
        scope_filters: list[dict],
    ) -> list["McpServer"]:
        """Find MCP servers by name across multiple scopes for precedence resolution.

        Each entry in scope_filters is a dict with ``workspace_id`` and
        optional ``user_id`` keys. Returns all matching servers across the
        given scopes so the caller can apply precedence ordering.
        Override in subclasses.
        """
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def update_mcp_server(
        self,
        workspace_id: str,
        server_id: str,
        updates: dict,
    ) -> "McpServer | None":
        """Update MCP server fields. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def delete_mcp_server(self, workspace_id: str, server_id: str) -> bool:
        """Delete an MCP server record. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")


# noinspection PyAbstractClass
class StoragePluginBase(Plugin):
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_STORAGE_BACKEND}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_STORAGE_BACKEND

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_STORAGE_BACKEND, self_attr="PROVIDER_NAME")

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
