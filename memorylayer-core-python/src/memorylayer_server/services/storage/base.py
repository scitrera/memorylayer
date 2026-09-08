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
    from ...models.context_pack import SessionCheckpoint, SessionCheckpointInput, SessionContextEvent
    from ...models.data_provider import DataProvider
    from ...models.document import Document, DocumentPage, IngestionJob
    from ...models.entity_relation import EntityRelation, EntityRelationEvidence, EntityRelationPath
    from ...models.mcp_server import McpServer
    from ...models.memory import MemoryMutation, MemoryMutationResult, MemoryRevision
    from ...models.skill import Skill, SkillFile, SkillMutation, SkillMutationResult, SkillRevision
    from ...models.versioned_resource import (
        VersionedResource,
        VersionedResourceMutation,
        VersionedResourceMutationResult,
        VersionedResourceRevision,
    )

from .._constants import EXT_STORAGE_BACKEND

# A deterministic, comparable "last changed" token for a workspace's graph inputs.
# Tuple of (max_memory_updated_at_iso, max_assoc_created_at_iso, memory_count,
# association_count). Treated as OPAQUE by callers: only equality is meaningful
# (see StorageBackend.get_workspace_change_watermark for the staleness rationale).
WorkspaceChangeWatermark = tuple[str, str, int, int]


class StorageCapabilityError(NotImplementedError):
    """Raised when a backend cannot honor an additive durable contract."""

    def __init__(self, capability: str):
        self.capability = capability
        super().__init__(f"Storage backend does not support required capability: {capability}")


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
    async def get_memory(
        self,
        workspace_id: str,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID within a workspace, optionally including its tombstone."""
        pass

    async def get_memory_by_id(
        self,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        raise NotImplementedError("Subclass should implement get_memory_by_id")

    @abstractmethod
    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
        """Update memory fields."""
        pass

    async def get_memories_by_ids(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        include_deleted: bool = False,
    ) -> list[Memory]:
        """Batch-hydrate memories. Backends used for relation recall must override."""
        results: list[Memory] = []
        for memory_id in memory_ids:
            memory = await self.get_memory(
                workspace_id,
                memory_id,
                track_access=False,
                include_deleted=include_deleted,
            )
            if memory is not None:
                results.append(memory)
        return results

    async def list_source_memories(
        self,
        workspace_id: str,
        source_memory_id: str,
    ) -> list[Memory]:
        """Return active memories derived from one raw source record."""
        return []

    @abstractmethod
    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        pass

    async def mutate_memory(self, mutation: "MemoryMutation") -> "MemoryMutationResult":
        """Atomically mutate a native memory head and append its semantic revision."""
        raise NotImplementedError("Versioned memory storage not implemented by this backend")

    async def get_memory_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> "MemoryMutationResult | None":
        """Return an exact accepted semantic memory mutation for an idempotent retry."""
        raise NotImplementedError("Versioned memory storage not implemented by this backend")

    async def list_memory_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> "list[MemoryRevision]":
        """List immutable semantic memory revisions newest first."""
        raise NotImplementedError("Versioned memory storage not implemented by this backend")

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

    async def search_memories_by_entities(
        self,
        workspace_id: str,
        query_embedding: list[float],
        entities: list[str],
        limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Entity-anchored vector search.

        Restrict the candidate set to memories whose ``metadata['speaker']`` is
        one of ``entities`` (the speaker is the discriminating, validated signal),
        then rank that set by vector similarity to ``query_embedding``. Returns
        (memory, relevance) tuples like ``search_memories``. Powers the
        entity-anchored RRF fusion channel (see
        MemoryService._fuse_entity_results).

        The base raises so backends without entity metadata support fail loudly;
        the recall path treats NotImplementedError as "degrade to vector-only".
        """
        raise NotImplementedError("Storage backend does not implement search_memories_by_entities")

    # ------------------------------------------------------------------
    # Cue-anchor extension seam (enterprise cue retrieval channel).
    #
    # A cue anchor is a short "[entity] + [aspect]" semantic key generated per
    # memory (see ExtractionService.generate_cue_anchors) and embedded. The
    # enterprise PostgreSQL backend persists (cue_text, embedding) rows in a
    # cue_anchors table and searches them by vector similarity, dereferencing
    # back to the primary memory for the RRF cue fusion arm
    # (MemoryService._fuse_cue_results). Both methods are concrete base no-ops so
    # OSS/relational backends inherit unchanged and the cue channel is a clean
    # no-op unless an enterprise backend overrides them. Ships DARK (default OFF).
    # ------------------------------------------------------------------

    async def search_cue_anchors(
        self,
        workspace_id: str,
        query_embedding: list[float],
        limit: int,
    ) -> list[tuple[Memory, float]]:
        """Return primary memories reached via cue-anchor similarity, best-first.

        Searches the workspace's cue anchors by vector similarity to
        ``query_embedding``, dereferences each hit back to its parent memory, and
        returns ``(memory, score)`` tuples ranked best-first (deduped by memory
        id, keeping the best score). Base is a no-op returning ``[]`` so backends
        without cue-anchor storage inherit unchanged; the enterprise pgvector
        backend overrides it.
        """
        return []

    async def store_cue_anchors(
        self,
        workspace_id: str,
        memory_id: str,
        cues: list[dict],
    ) -> None:
        """Persist structured cue-anchor rows for a memory.

        Called at ingest (behind the cue-channel flag) with the cue anchors
        generated for ``memory_id``. Each cue dict carries::

            {"cue": str, "embedding": list[float], "entity_id": str | None}

        where ``cue`` is the ``"[entity] [aspect]"`` text, ``embedding`` its
        vector, and ``entity_id`` the canonical Entity the cue names (resolved
        against the entity registry when available, else ``None``). Persisting
        ``entity_id`` links the cue arm and the entity-anchored arm to a shared
        entity vocabulary. Base is a no-op returning ``None`` so backends without
        cue-anchor storage inherit unchanged; the enterprise pgvector backend
        overrides it.
        """
        return None

    async def expand_via_cues(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        limit: int = 20,
        threshold: float = 0.85,
    ) -> list[tuple[Memory, float]]:
        """Reach OTHER memories that share a thematic cue with ``memory_ids``.

        Given a set of memory ids (the agentic-recall frontier), fetch those
        memories' cue-anchor embeddings, find similar cue anchors (cosine
        similarity ``>= threshold``) belonging to DIFFERENT memories, and
        dereference those hits back to their parent memories. Returns
        ``(memory, score)`` tuples deduped by memory id (keeping the best score),
        ranked best-first, capped at ``limit``. This is the "relaxed frontier"
        second source for the agentic EXPAND step: it reaches memories linked by
        shared cue theme, not just by graph edges.

        Base is a no-op returning ``[]`` so backends without cue-anchor storage
        inherit unchanged; the enterprise pgvector backend overrides it. Ships
        DARK (only consulted when the cue channel is enabled).
        """
        return []

    async def reindex_memory(self, workspace_id: str, memory_id: str) -> None:
        """Reconcile any materialized full-text index for a memory after its
        content/aliases changed.

        Default is a no-op: backends that build their full-text representation at
        query time (e.g. PostgreSQL tsvector, in-memory substring) have no
        materialized index to reconcile. Backends with a materialized FTS table
        (e.g. SQLite FTS5) override this to rebuild the memory's index row.
        """
        return None

    async def get_timeline(
        self,
        workspace_id: str,
        event_after: str | None = None,
        event_before: str | None = None,
        ascending: bool = True,
        limit: int = 50,
        offset: int = 0,
        types: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Return memories ordered by effective event time (event_time or created_at).

        Powers explicit timeline browsing and temporal-neighbour lookups. Backends
        override this with an event-time-ordered query; the base raises so an
        unsupported backend fails loudly rather than silently returning nothing.
        """
        raise NotImplementedError("Storage backend does not implement get_timeline")

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

        Watermark-visibility contract (implementations MUST honour this):
        ``memory_associations`` has no ``updated_at`` column, so an in-place
        edge update is invisible to ``get_workspace_change_watermark`` on its
        own.  Implementations MUST therefore bump ``updated_at`` on both the
        source and target memories after updating the edge row.  This advances
        ``max_memory_updated_at`` in the watermark tuple, making the change
        visible to the KB skip-generate gate (Gate A) and the AGE
        watermark-gated materialize gate (Gate B).

        Phase 1 fix 1.6 decoupled the recall recency boost from
        ``updated_at``, so these memory-row bumps do NOT produce a feedback
        loop in recall ranking.

        The base returns False (not-found); backends without an associations
        table inherit this safe no-op.
        """
        return False

    async def count_associations_by_relationship(self, workspace_id: str) -> dict[str, int]:
        """Return per-relationship edge counts for a workspace.

        Powers ``GraphQueryService.relationship_rollup`` without loading every
        edge into Python — backends implement this as a single
        ``GROUP BY relationship`` aggregate. The base loops over a one-shot
        ``get_associations_batch`` of an empty set, which is insufficient on its
        own, so backends with an associations table MUST override. The base
        raises so an unsupported backend fails loudly.
        """
        raise NotImplementedError("Storage backend does not implement count_associations_by_relationship")

    async def get_workspace_change_watermark(self, workspace_id: str) -> "WorkspaceChangeWatermark | None":
        """Return a deterministic "last changed" token for a workspace's graph inputs.

        The token is a comparable tuple
        ``(max_memory_updated_at_iso, max_assoc_created_at_iso, memory_count,
        association_count)`` over the workspace's ACTIVE memories and ALL its
        associations. It is the dirty-watermark used by the P2 strategy-C skip
        gates (KB skip-generate and AGE watermark-gated materialize): two
        watermarks compare EQUAL iff no graph-relevant change has occurred since.

        Why the two counts are part of the token (staleness guard): associations
        have only a ``created_at`` column — an association DELETE advances no
        timestamp, so a timestamp-only watermark would NOT detect edge removal and
        could serve stale results. ``association_count`` (and ``memory_count``)
        make add/delete detectable. Callers MUST treat the token as opaque and
        only test EQUALITY (not ordering): equality is the sole safe
        "unchanged" predicate because a count drop is not ordered against a
        timestamp advance.

        Returns ``None`` when the watermark cannot be computed; callers MUST then
        fall back to doing the full work (fail-safe — never skip on ambiguity).
        Backends without an associations/memories table do not override and
        inherit this ``NotImplementedError`` so callers fall back safely.
        """
        raise NotImplementedError("Storage backend does not implement get_workspace_change_watermark")

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
        user_id: str | None = None,
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
            user_id: Filter to memories owned by this user_id (the user-scope
                partition inside the ``_global_user`` workspace). When set, ONLY
                rows whose top-level ``user_id`` column equals this value are
                returned — the same forced read boundary the recall fan-out uses
                to keep one user's user-global memories private from another.
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

    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Delete a context within a workspace.

        Contexts are hard-deleted (no soft-delete column). Memories that
        reference the context keep their (reserved/unused) context_id; deleting
        a context does not remove its memories. Returns True if a row was
        deleted, False if the context did not exist in this workspace. Override
        in subclasses.
        """
        return False

    @abstractmethod
    async def list_workspaces(
        self,
        *,
        tags: list[str] | None = None,
        match: str = "all",
    ) -> list[Workspace]:
        """List workspaces, optionally filtered by tag.

        Args:
            tags: Optional tag filter. When provided, only workspaces carrying the tag(s)
                are returned. Empty/None means no filtering.
            match: 'all' requires every tag; 'any' requires at least one.
        """
        pass

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all associated data. Override in subclasses."""
        return False

    async def update_workspace(self, workspace_id: str, **updates) -> Workspace | None:
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

    # These additive contracts are concrete so existing backend subclasses stay
    # import-compatible and fail explicitly when a new capability is requested.
    def supports_capability(self, capability: str) -> bool:
        return False

    async def create_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        input: "SessionCheckpointInput",
    ) -> tuple["SessionCheckpoint", bool]:
        raise StorageCapabilityError("session_checkpoints")

    async def get_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
    ) -> "SessionCheckpoint | None":
        raise StorageCapabilityError("session_checkpoints")

    async def list_session_checkpoints(
        self,
        workspace_id: str,
        session_id: str,
        *,
        limit: int = 20,
    ) -> "list[SessionCheckpoint]":
        raise StorageCapabilityError("session_checkpoints")

    async def update_session_checkpoint_status(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
        *,
        index_status: str | None = None,
        enrichment_status: str | None = None,
    ) -> "SessionCheckpoint | None":
        raise StorageCapabilityError("session_checkpoints")

    async def append_context_event(
        self,
        workspace_id: str,
        session_id: str | None,
        event_kind: str,
        subject_kind: str,
        subject_id: str,
        metadata: dict | None = None,
    ) -> "SessionContextEvent":
        raise StorageCapabilityError("session_context_events")

    async def list_context_events(
        self,
        workspace_id: str,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
    ) -> "list[SessionContextEvent]":
        raise StorageCapabilityError("session_context_events")

    async def get_context_event_bounds(
        self,
        workspace_id: str,
        session_id: str,
    ) -> tuple[int, int]:
        raise StorageCapabilityError("session_context_events")

    async def upsert_entity_relation(
        self,
        relation: "EntityRelation",
        evidence: "EntityRelationEvidence",
    ) -> tuple["EntityRelation", bool]:
        raise StorageCapabilityError("entity_relations")

    async def traverse_entity_relations(
        self,
        workspace_id: str,
        seed_entity_ids: list[str],
        *,
        relationships: list[str] | None,
        direction: str,
        max_hops: int,
        max_edges: int,
    ) -> "list[EntityRelationPath]":
        raise StorageCapabilityError("entity_relations")

    async def deactivate_relation_evidence_for_memory(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> int:
        raise StorageCapabilityError("entity_relations")

    async def reconcile_entity_relations_after_merge(
        self,
        workspace_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> int:
        raise StorageCapabilityError("entity_relations")

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

    SUPERSEDED_SCAN_LIMIT = 10000

    async def get_superseded_memory_ids(self, workspace_id: str, memory_ids: list[str]) -> set[str]:
        """Of ``memory_ids``, which are the STALE side of an unresolved contradiction.

        "Stale" means the record names a *different* memory as ``newer_memory_id``. A
        contradiction with no direction recorded supersedes nothing — it says two memories
        conflict, not which one is current — so it is skipped rather than guessed at.

        Resolved contradictions are excluded: resolution is the operator saying the
        conflict is dealt with, and continuing to penalise afterwards would make the
        resolution invisible.

        This default derives the answer from ``get_unresolved_contradictions`` so every
        backend gets correct behaviour for free; backends should override with an indexed
        query. Bounded by ``SUPERSEDED_SCAN_LIMIT`` so a pathological workspace degrades to
        under-penalising rather than to a slow recall.
        """
        if not memory_ids:
            return set()
        wanted = set(memory_ids)
        superseded: set[str] = set()
        for record in await self.get_unresolved_contradictions(workspace_id, limit=self.SUPERSEDED_SCAN_LIMIT):
            newer = record.newer_memory_id
            if not newer:
                continue
            for candidate in (record.memory_a_id, record.memory_b_id):
                if candidate in wanted and candidate != newer:
                    superseded.add(candidate)
        return superseded

    # Chat history operations (non-abstract with default no-op implementations)

    async def create_thread(self, thread: "ChatThread") -> "ChatThread":
        """Store a new chat thread. Override in subclasses."""
        return thread

    async def get_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> Optional["ChatThread"]:
        """Get chat thread by ID, scoped by owner. Override in subclasses.

        ``user_id`` scopes the lookup to a specific owner: threads are identified
        by ``(workspace_id, COALESCE(user_id,''), id)`` so a shared client id like
        ``"_default"`` in the ``_user_chat`` sentinel resolves to the right owner's
        thread. ``None`` matches rows with a NULL ``user_id`` (workspace-owned).
        """
        return None

    async def list_threads(
        self,
        workspace_id: str,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        scope_filter: str | None = None,
        ownership_filter: str | None = None,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list["ChatThread"]:
        """List chat threads in a workspace. Override in subclasses.

        ``include_hidden=False`` (default) excludes archived/hidden threads
        (``hidden_at`` set); pass True for an "Archived" view.
        ``parent_thread=None`` (default) returns only top-level threads
        (``parent_thread IS NULL``); pass a parent id to list its sub-threads.
        """
        return []

    async def list_user_threads(
        self,
        tenant_id: str,
        user_id: str,
        ownership: str = "user",
        scope_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list["ChatThread"]:
        """List chat threads owned by a user across all workspaces.

        Keyed on (tenant_id, user_id, ownership) — does NOT filter by workspace_id.
        Used for the user-session-scoped right rail. Override in subclasses.
        ``include_hidden=False`` (default) excludes archived/hidden threads.
        ``parent_thread=None`` (default) returns only top-level threads.
        """
        return []

    async def update_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None, **updates) -> Optional["ChatThread"]:
        """Update thread fields, scoped by owner. Override in subclasses."""
        return None

    async def delete_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> bool:
        """Delete a thread and all its messages, scoped by owner. Override in subclasses."""
        return False

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        messages: list["MessageInput"],
        user_id: str | None = None,
    ) -> list["ChatMessage"]:
        """Append messages to a thread, scoped by owner. Override in subclasses."""
        return []

    async def get_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        after_index: int | None = None,
        order: str = "asc",
        user_id: str | None = None,
    ) -> list["ChatMessage"]:
        """Get messages from a thread, scoped by owner. Override in subclasses."""
        return []

    async def get_message_count(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> int:
        """Get total message count for a thread, scoped by owner. Override in subclasses."""
        return 0

    async def delete_message(self, workspace_id: str, thread_id: str, message_id: str, user_id: str | None = None) -> bool:
        """Delete a single message by ID within a thread and workspace, scoped by owner.

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

    async def list_idle_threads(
        self,
        updated_before: "datetime",
        *,
        idle_action: str,
        only_hidden: bool | None = None,
        limit: int = 100,
    ) -> list["ChatThread"]:
        """List threads whose idle policy is due, across all workspaces.

        Returns threads with ``idle_action == idle_action`` and
        ``updated_at < updated_before``. ``only_hidden``: ``None`` = any,
        ``True`` = only currently-hidden, ``False`` = only currently-visible.
        Override in subclasses.
        """
        return []

    async def list_hidden_threads(
        self,
        hidden_before: "datetime",
        limit: int = 100,
    ) -> list["ChatThread"]:
        """List hidden/archived threads hidden before ``hidden_before``.

        Drives the post-archive grace purge (``hidden_at < hidden_before``).
        Override in subclasses.
        """
        return []

    async def hide_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> Optional["ChatThread"]:
        """Archive (hide) a thread by setting ``hidden_at`` without bumping
        ``updated_at`` (hiding is not activity), scoped by owner. Override in subclasses.
        """
        return None

    async def unhide_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> Optional["ChatThread"]:
        """Restore (un-hide) a thread by clearing ``hidden_at``, scoped by owner. Override in subclasses."""
        return None

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

    async def try_claim_document(
        self,
        document_id: str,
        workspace_id: str,
        ttl_seconds: int,
    ) -> bool:
        """Atomically claim a document for (re)processing via a conditional UPDATE.

        Flips the row to ``processing`` (stamping ``processing_started_at=now``)
        only when it is not already owned by a fresh in-flight worker — i.e. when
        its status is not ``processing``, or it has no ``processing_started_at``,
        or that timestamp is older than ``ttl_seconds`` (the in-flight freshness
        window). Returns ``True`` iff exactly one row was updated (the caller won
        the claim); ``False`` means another worker owns it (NO-OP). Closes the
        read-then-write race the freshness check alone leaves open. Override in
        subclasses.
        """
        raise NotImplementedError("Document storage not implemented by this backend")

    # Document page operations

    async def create_page(self, workspace_id: str, document_id: str, page: "DocumentPage") -> "DocumentPage":
        """Store a document page. Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def get_pages(self, document_id: str, workspace_id: str | None = None) -> list["DocumentPage"]:
        """Get all pages for a document, ordered by page_no. Override in subclasses."""
        raise NotImplementedError("Document page storage not implemented by this backend")

    async def delete_pages(self, document_id: str, workspace_id: str | None = None) -> int:
        """Delete all pages for a document, returning the row count. Override in subclasses."""
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

    async def get_embeddings_batch(
        self,
        workspace_id: str,
        memory_ids: list[str],
    ) -> dict[str, list[float]]:
        """Return ``{memory_id: single-vector embedding}`` for the memories that
        have one. Base default loops ``get_memory`` (N calls); backends may
        override with a single query for throughput. Best-effort — memories
        without an embedding are omitted. Used by KB graph densification.
        """
        out: dict[str, list[float]] = {}
        for mid in memory_ids:
            try:
                mem = await self.get_memory(workspace_id, mid, track_access=False)
            except Exception:
                mem = None
            emb = getattr(mem, "embedding", None) if mem is not None else None
            if emb:
                out[mid] = list(emb)
        return out

    async def get_multivectors_batch(
        self,
        workspace_id: str,
        memory_ids: list[str],
    ) -> dict[str, list[list[float]]]:
        """Return ``{memory_id: ColPali multivector}`` for the memories that carry
        one. Base default loops ``get_memory`` (N calls); backends may override
        with a single query. Best-effort. Used by KB graph densification (MaxSim).
        """
        out: dict[str, list[list[float]]] = {}
        for mid in memory_ids:
            try:
                mem = await self.get_memory(workspace_id, mid, track_access=False)
            except Exception:
                mem = None
            mv = getattr(mem, "multivector", None) if mem is not None else None
            if mv:
                out[mid] = mv
        return out

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

    async def list_active_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] = ("queued", "running"),
    ) -> list["IngestionJob"]:
        """Return non-terminal ingestion jobs whose ``document_ids`` overlap any
        id in ``document_ids``.

        Used to enforce at-most-one in-flight job per document: the create paths
        supersede overlapping jobs before minting a new one, and the
        document->completed transition closes the losers. ``statuses`` bounds
        "non-terminal" (default the two in-flight states). Override in subclasses.
        """
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    async def list_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list["IngestionJob"]:
        """Return ingestion jobs whose ``document_ids`` overlap any id in
        ``document_ids``, newest first.

        The unfiltered form of :meth:`list_active_jobs_for_documents`:
        ``statuses=None`` means ANY status. Diagnostic rather than operational —
        an operator inspecting one document wants its whole job history, where
        the failed and superseded attempts are usually what explains its state.
        Override in subclasses.
        """
        raise NotImplementedError("Ingestion job storage not implemented by this backend")

    async def cancel_orphaned_ingestion_jobs(self) -> int:
        """Cancel every ``queued``/``running`` ingestion job whose referenced
        documents are ALL already ``completed`` (an orphan safety-net sweep).

        Returns the number of jobs transitioned to ``cancelled``. A job with a
        missing referenced document is left untouched (its liveness is
        undeterminable). Override in subclasses.
        """
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

    async def delete_kb_article(self, workspace_id: str, article_id: str) -> bool:
        """Delete a single knowledgebase article by id (for stale-article GC).

        Returns True if an article was deleted, False if it did not exist.
        Override in subclasses.
        """
        raise NotImplementedError("Knowledgebase storage not implemented by this backend")

    # Entity registry operations (canonical entity nodes + aliases + members)
    #
    # All return/accept plain dicts (mirroring the KB-article precedent above);
    # the EntityRegistryService maps these to/from the models.entity_registry
    # DTOs. Entity dicts carry: id, workspace_id, entity_type, canonical_name,
    # normalized_name, aliases (list[str]), confidence, provenance (dict),
    # representative_memory_id, status, merged_into, created_at, updated_at.
    # Member dicts carry: entity_id, memory_id, role, confidence.

    async def store_entity(self, entity: dict) -> dict:
        """Insert a canonical entity row (and its initial aliases). Override in subclasses.

        The entity dict MAY carry an optional ``name_embedding`` (list[float])
        for backends that support embedding-fuzzy resolution (enterprise PG).
        Relational backends without a vector column ignore it.
        """
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def find_entities_by_name_embedding(
        self,
        workspace_id: str,
        entity_type: str,
        embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """Approximate-nearest-neighbor search over active entities' name embeddings.

        Returns active entities of ``entity_type`` in ``workspace_id`` whose
        ``name_embedding`` is within cosine ``min_score`` of ``embedding``,
        ordered best-first, each dict carrying an extra ``score`` key (cosine
        similarity in ``[0, 1]``). Only the enterprise pgvector backend
        implements this; relational/SQLite backends raise (the OSS ``resolve``
        never calls it, so OSS stays exact+alias only).
        """
        raise NotImplementedError("Embedding-fuzzy entity search is not implemented by this backend (enterprise pgvector only)")

    async def get_entity(self, workspace_id: str, entity_id: str) -> dict | None:
        """Get a canonical entity by id (with aliases folded in). Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def find_entity_by_normalized_name(
        self,
        workspace_id: str,
        entity_type: str,
        normalized_name: str,
    ) -> dict | None:
        """Find the active entity matching (workspace, type, normalized_name). Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def find_entities_by_normalized_name_any_type(
        self,
        workspace_id: str,
        normalized_name: str,
    ) -> list[dict]:
        """Find ALL active entities matching (workspace, normalized_name), any type.

        Type-agnostic counterpart to ``find_entity_by_normalized_name``: powers
        the accretion name-first resolution path that unifies a name seen both as
        a speaker (PERSON) and as a third-party mention (CONCEPT) onto a single
        canonical node. Returns active entities deterministically ordered
        PERSON-first, then by entity id (so the name-first resolver can prefer an
        existing PERSON node and detect a CONCEPT node to promote/merge). Override
        in subclasses.
        """
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def find_entities_by_normalized_alias(
        self,
        workspace_id: str,
        normalized_alias: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Find active entities whose aliases include normalized_alias. Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def add_entity_alias(
        self,
        workspace_id: str,
        entity_id: str,
        alias: str,
        normalized_alias: str,
        source: str = "manual",
    ) -> None:
        """Add an alias to an entity (idempotent on normalized_alias). Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def add_entity_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        role: str = "mention",
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> dict:
        """Attach a memory to an entity as a member (idempotent on (entity_id, memory_id, role)). Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def list_entity_members(
        self,
        workspace_id: str,
        entity_id: str,
        role: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List member rows for an entity, optionally filtered by role. Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def reassign_entity_members(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> int:
        """Reassign all members from source_id to target_id (for merge). Returns rows moved. Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def list_workspace_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 10000,
    ) -> list[dict]:
        """List ALL canonical entities for a workspace (registry-wide enumeration).

        Powers the workspace-scoped consumers of the entity registry that need
        the FULL entity set rather than a single-entity lookup: the enterprise
        AGE Entity-vertex materialization and the OSS relational
        ``entity_neighborhood`` fallback. Returns entity dicts (same shape
        as ``get_entity``, aliases folded in), ordered deterministically by
        entity id. Defaults to ``status="active"`` so merged tombstones are
        excluded. Override in subclasses; the base raises so an unsupported
        backend fails loudly.
        """
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def list_workspace_entity_members(
        self,
        workspace_id: str,
        *,
        role: str | None = None,
        limit: int = 100000,
    ) -> list[dict]:
        """List ALL member edges for a workspace (registry-wide enumeration).

        Workspace-scoped counterpart to ``list_entity_members`` (which is
        per-entity): returns every ``(entity_id, memory_id, role, confidence)``
        member row in the workspace, optionally filtered by ``role``, ordered
        deterministically by ``(entity_id, memory_id, role)``. Used to bulk-load
        the MENTIONS edge set for AGE materialization (C1) and the OSS relational
        ``entity_neighborhood`` (C2) without an N+1 per-entity loop. Override in
        subclasses; the base raises so an unsupported backend fails loudly.
        """
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def update_entity(
        self,
        workspace_id: str,
        entity_id: str,
        **updates,
    ) -> dict | None:
        """Update mutable entity fields (status/merged_into/confidence/representative_memory_id/...). Override in subclasses."""
        raise NotImplementedError("Entity registry storage not implemented by this backend")

    async def delete_workspace_entities(self, workspace_id: str) -> int:
        """Hard-delete ALL canonical entities (+ their aliases/members) in a workspace.

        Returns the number of entities deleted. Override in subclasses; the base
        raises so an unsupported backend fails loudly rather than silently no-op'ing
        a requested registry wipe.
        """
        raise NotImplementedError("Entity registry storage not implemented by this backend")

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

    async def get_skill(
        self,
        workspace_id: str,
        skill_id: str,
        *,
        include_deleted: bool = False,
    ) -> "Skill | None":
        """Get skill by ID within a workspace. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def get_skill_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: str | None = None,
    ) -> "Skill | None":
        """Get skill by name within a workspace, optionally filtering by user scope. Override in subclasses."""
        raise NotImplementedError("Skill storage not implemented by this backend")

    async def list_skills(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        include_global: bool = False,
    ) -> list["Skill"]:
        """List skills in a workspace with optional filters. Override in subclasses.

        When ``include_global`` is True, tenant-shared skills living in the
        ``_global`` workspace (``user_id IS NULL``) are unioned into the result
        alongside ``workspace_id``'s own skills. The global union is skipped when
        ``workspace_id`` is already ``_global`` or when a specific ``user_id``
        filter is supplied (global skills are never user-scoped).
        """
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

    async def mutate_skill(self, mutation: "SkillMutation") -> "SkillMutationResult":
        """Atomically mutate a native skill head and append its revision/result."""
        raise NotImplementedError("Versioned skill storage not implemented by this backend")

    async def get_skill_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> "SkillMutationResult | None":
        """Return an exact accepted skill mutation for an idempotent retry."""
        raise NotImplementedError("Versioned skill storage not implemented by this backend")

    async def list_skill_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        skill_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> "list[SkillRevision]":
        """List immutable skill-manifest revisions newest first."""
        raise NotImplementedError("Versioned skill storage not implemented by this backend")

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
        user_id: str | None = None,
    ) -> "McpServer | None":
        """Get MCP server by name within a workspace, optionally filtering by user scope. Override in subclasses."""
        raise NotImplementedError("MCP server storage not implemented by this backend")

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
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

    # Internal versioned-resource operations. Public APIs must wrap these in a
    # typed domain service rather than exposing an untyped JSON bucket.

    async def mutate_versioned_resource(
        self,
        mutation: "VersionedResourceMutation",
    ) -> "VersionedResourceMutationResult":
        """Atomically mutate a head, immutable revision, and idempotency record."""
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")

    async def get_versioned_resource_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> "VersionedResourceMutationResult | None":
        """Return an accepted operation result, rejecting conflicting key reuse."""
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")

    async def get_versioned_resource(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> "VersionedResource | None":
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")

    async def get_versioned_resource_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> "VersionedResource | None":
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")

    async def list_versioned_resources(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        before_sequence: int | None = None,
        include_deleted: bool = False,
    ) -> "list[VersionedResource]":
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")

    async def list_versioned_resource_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> "list[VersionedResourceRevision]":
        raise NotImplementedError("Versioned-resource storage not implemented by this backend")


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
