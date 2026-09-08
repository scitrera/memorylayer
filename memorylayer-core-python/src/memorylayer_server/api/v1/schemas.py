"""
API request/response schemas for MemoryLayer.ai endpoints.

These schemas define the HTTP API interface separate from core domain models.
"""

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from memorylayer_server.models.association import (
    Association,
    GraphPath,  # noqa: F401 — re-exported for associations.py
    GraphQueryResult,  # noqa: F401 — re-exported for associations.py
    RelationshipCategory,
)
from memorylayer_server.models.context_pack import (
    ContextDelta,  # noqa: F401 — re-exported for sessions.py
    ContextDeltaInput,  # noqa: F401 — re-exported for sessions.py
    ContextPack,  # noqa: F401 — re-exported for sessions.py
    ContextPackInput,  # noqa: F401 — re-exported for sessions.py
    SessionCheckpoint,  # noqa: F401 — re-exported for sessions.py
    SessionCheckpointInput,  # noqa: F401 — re-exported for sessions.py
)
from memorylayer_server.models.entity_registry import (
    Entity,  # noqa: F401 — re-exported for entities.py
    EntityResolution,  # noqa: F401 — re-exported for entities.py
    EntityType,
    RelatedEntity,
    SeedEntity,
)
from memorylayer_server.models.entity_relation import EntityRelationInput
from memorylayer_server.models.memory import (
    Memory,
    MemoryScope,
    MemoryType,
    RecallMode,
    RecallResult,  # noqa: F401 — re-exported for memories.py
    ReflectResult,  # noqa: F401 — re-exported for memories.py
    SearchTolerance,
)
from memorylayer_server.models.representation import (
    Representation,  # noqa: F401 — re-exported for representation.py
)
from memorylayer_server.models.session import Session, SessionBriefing
from memorylayer_server.models.workspace import Context, Workspace


# Memory API Schemas
class MemoryCreateRequest(BaseModel):
    """Request schema for creating a memory."""

    content: str = Field(..., description="Memory content to store", min_length=1)
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    type: MemoryType | None = Field(None, description="Cognitive type (auto-classified if omitted)")
    subtype: str | None = Field(None, description="Domain-specific classification (OSS-known or plugin-contributed)")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance (0.0-1.0)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    logical_key: str | None = Field(
        None,
        description="Stable workspace/owner-scoped key for conditional refinement",
    )
    refinement_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Semantic metadata protected by the memory resource ETag",
    )
    pinned: bool = Field(False, description="Pin the memory against retention decay")
    associations: list[str] = Field(default_factory=list, description="Memory IDs to associate with")
    relations: list[EntityRelationInput] = Field(
        default_factory=list,
        description="Typed structural entity relations evidenced by this memory",
    )
    context_id: str | None = Field(None, description="Target memory context")
    observer_id: str | None = Field(None, description="Entity doing the observing/remembering (agent ID, user ID, etc.)")
    subject_id: str | None = Field(None, description="Entity this memory is about")
    user_id: str | None = Field(None, description="User scope for this memory")
    scope: MemoryScope = Field(
        MemoryScope.WORKSPACE,
        description="Storage scope: 'workspace' (default, local) or 'user' (cross-workspace user-global; requires a user_id)",
    )


class MemoryUpdateRequest(BaseModel):
    """Request schema for updating a memory."""

    content: str | None = Field(None, description="Updated content", min_length=1)
    type: MemoryType | None = Field(None, description="Updated cognitive type")
    subtype: str | None = Field(None, description="Updated domain classification")
    importance: float | None = Field(None, ge=0.0, le=1.0, description="Updated importance")
    tags: list[str] | None = Field(None, description="Updated tags")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")
    refinement_metadata: dict[str, Any] | None = Field(
        None,
        description="Updated semantic refinement metadata",
    )
    pinned: bool | None = Field(None, description="Pin/unpin memory (pinned memories are exempt from decay)")


class MemoryRecallRequest(BaseModel):
    """Request schema for querying memories."""

    query: str = Field(..., description="Natural language query", min_length=1)
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    types: list[MemoryType] = Field(default_factory=list, description="Filter by cognitive types")
    subtypes: list[str] = Field(default_factory=list, description="Filter by domain subtypes")
    tags: list[str] = Field(default_factory=list, description="Filter by tags (AND logic)")
    context_id: str | None = Field(None, description="Filter by memory context")
    observer_id: str | None = Field(None, description="Filter by observer entity")
    subject_id: str | None = Field(None, description="Filter by subject entity")
    user_id: str | None = Field(None, description="Filter by user")
    include_global: bool = Field(True, description="Include the _global workspace in search")
    include_global_user: bool = Field(
        True,
        description=(
            "Include the user-scoped global workspace (_global_user) in search, filtered by the same "
            "user_id (per-user preferences follow the user across workspaces). Only effective when user_id is set."
        ),
    )
    mode: RecallMode | None = Field(None, description="Retrieval strategy (None = server default)")
    tolerance: SearchTolerance | None = Field(None, description="Search precision (None = server default)")
    limit: int = Field(10, ge=1, le=100, description="Maximum memories to return")
    offset: int = Field(0, ge=0, description="Number of results to skip for pagination")
    include_embeddings: bool = Field(
        False,
        description=(
            "Include each memory's raw embedding vector in the response. Off by default: the vectors "
            "dominate the payload (at 1024 dimensions they are roughly 8 KB per memory) and callers "
            "almost never use them. Enable for debugging or offline vector analysis."
        ),
    )
    min_relevance: float | None = Field(None, ge=0.0, le=1.0, description="Minimum relevance score (None = server default)")
    recency_weight: float | None = Field(
        None, ge=0.0, le=1.0, description="Weight for recency boosting (0.0=disabled, 1.0=full). None = server default."
    )
    include_associations: bool | None = Field(None, description="Include linked memories (None = server default)")
    traverse_depth: int | None = Field(None, ge=0, le=5, description="Multi-hop graph traversal depth (None = server default)")
    max_expansion: int | None = Field(None, ge=1, le=500, description="Max memories discovered via graph expansion (None = server default)")
    created_after: datetime | None = Field(None, description="Filter memories created after this time")
    created_before: datetime | None = Field(None, description="Filter memories created before this time")
    event_after: datetime | None = Field(None, description="Keep memories whose effective event time is >= this")
    event_before: datetime | None = Field(None, description="Keep memories whose effective event time is <= this")
    time_order: str | None = Field(None, description="Order results by effective event time: 'asc' or 'desc' (None = by relevance)")
    context: list[dict[str, str]] = Field(default_factory=list, description="Recent conversation context")
    rag_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Use LLM if RAG confidence < threshold")
    detail_level: str | None = Field(None, description="Detail level: abstract, overview, or full (None = server default)")
    include_archived: bool = Field(False, description="Include archived memories in recall results")
    exclude_ids: list[str] = Field(default_factory=list, description="Memory IDs to exclude from results (already shown to user)")
    budget_tokens: int | None = Field(None, ge=1, description="Hard estimated-token budget for returned content")
    include_confidence: bool = Field(True, description="Include deterministic retrieval confidence")
    include_relations: bool = Field(True, description="Allow bounded typed-relation retrieval when enabled")


class MemoryReflectRequest(BaseModel):
    """Request schema for synthesizing memories."""

    query: str = Field(..., description="What to reflect on", min_length=1)
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    detail_level: str | None = Field(None, description="Level of detail: abstract, overview, full (None = server default)")
    include_sources: bool = Field(True, description="Include source memory references")
    depth: int = Field(2, ge=1, le=5, description="Association traversal depth")
    types: list[MemoryType] = Field(default_factory=list, description="Filter by types")
    subtypes: list[str] = Field(default_factory=list, description="Filter by subtypes")
    tags: list[str] = Field(default_factory=list, description="Filter by tags")
    context_id: str | None = Field(None, description="Filter by memory context")
    observer_id: str | None = Field(None, description="Filter by observer entity")
    subject_id: str | None = Field(None, description="Filter by subject entity")


class MemoryDecayRequest(BaseModel):
    """Request schema for decaying a memory."""

    decay_rate: float = Field(0.1, ge=0.0, le=1.0, description="Decay rate to apply")


class BatchCreateOp(BaseModel):
    """Batch create operation."""

    op: Literal["create"] = Field(..., description="Operation type")
    content: str = Field(..., description="Memory content", min_length=1)
    type: MemoryType | None = Field(None, description="Cognitive type")
    subtype: str | None = Field(None, description="Domain classification")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance")
    tags: list[str] = Field(default_factory=list, description="Tags")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    observer_id: str | None = Field(None, description="Observer entity")
    subject_id: str | None = Field(None, description="Subject entity")


class BatchUpdateOp(BaseModel):
    """Batch update operation."""

    op: Literal["update"] = Field(..., description="Operation type")
    memory_id: str = Field(..., description="Memory ID to update")
    content: str | None = Field(None, description="Updated content", min_length=1)
    type: MemoryType | None = Field(None, description="Updated type")
    subtype: str | None = Field(None, description="Updated subtype")
    importance: float | None = Field(None, ge=0.0, le=1.0, description="Updated importance")
    tags: list[str] | None = Field(None, description="Updated tags")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")
    pinned: bool | None = Field(None, description="Pin/unpin memory")


class BatchDeleteOp(BaseModel):
    """Batch delete operation."""

    op: Literal["delete"] = Field(..., description="Operation type")
    memory_id: str = Field(..., description="Memory ID to delete")
    hard: bool = Field(False, description="Hard delete (permanent)")


BatchOperation = Annotated[
    BatchCreateOp | BatchUpdateOp | BatchDeleteOp,
    Field(discriminator="op"),
]


class MemoryBatchRequest(BaseModel):
    """Request schema for batch memory operations."""

    operations: list[BatchOperation] = Field(..., description="List of typed batch operations (create, update, delete)")


class MemoryResponse(BaseModel):
    """Response schema for single memory."""

    memory: Memory


class MemoryListResponse(BaseModel):
    """Response schema for memory list."""

    memories: list[Memory]
    total_count: int


class BatchOperationResult(BaseModel):
    """Result of a single batch operation."""

    index: int = Field(..., description="Operation index in batch")
    type: str = Field(..., description="Operation type")
    status: str = Field(..., description="success or error")
    memory_id: str | None = Field(None, description="Memory ID for create/update operations")
    error: str | None = Field(None, description="Error message if failed")


class BatchOperationResponse(BaseModel):
    """Response schema for batch operations."""

    total_operations: int = Field(..., description="Total operations in batch")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    results: list[BatchOperationResult] = Field(..., description="Results for each operation")


# Association API Schemas
class AssociationCreateRequest(BaseModel):
    """Request schema for creating an association (source from URL path)."""

    target_id: str = Field(..., description="Target memory ID")
    relationship: str = Field(..., description="Relationship type (e.g., SIMILAR_TO, CAUSES, SOLVES)")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Relationship strength")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")


class AssociationCreateFullRequest(BaseModel):
    """Request schema for creating an association with both source and target in body."""

    source_id: str = Field(..., description="Source memory ID")
    target_id: str = Field(..., description="Target memory ID")
    relationship: str = Field(..., description="Relationship type (e.g., SIMILAR_TO, CAUSES, SOLVES)")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Relationship strength")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class AssociationUpdateRequest(BaseModel):
    """Request schema for updating an association's strength and/or metadata.

    Only ``strength`` and ``metadata`` are mutable — the storage seam
    (``update_association``) does not support changing the relationship type
    (re-typing an edge is a delete + create)."""

    strength: float | None = Field(None, ge=0.0, le=1.0, description="New relationship strength (None = unchanged)")
    metadata: dict[str, Any] | None = Field(None, description="New metadata dict, replaces existing (None = unchanged)")


class AssociationListRequest(BaseModel):
    """Request schema for listing associations."""

    relationships: list[str] | None = Field(None, description="Filter by relationship types (e.g., SIMILAR_TO, CAUSES)")
    direction: str = Field("both", pattern="^(outgoing|incoming|both)$", description="Association direction")


class MemoryTraverseRequest(BaseModel):
    """Request schema for traversing from a specific memory."""

    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum traversal depth")
    relationship_types: list[str] = Field(default_factory=list, description="Filter by specific relationship types (empty = all)")
    direction: str = Field("both", pattern="^(outgoing|incoming|both)$", description="Traversal direction: outgoing, incoming, both")
    min_strength: float = Field(0.0, ge=0.0, le=1.0, description="Minimum edge strength")


class GraphTraverseRequest(BaseModel):
    """Request schema for graph traversal."""

    start_memory_id: str = Field(..., description="Starting memory for traversal")
    relationship_types: list[str] = Field(default_factory=list, description="Filter by specific relationship types")
    relationship_categories: list[RelationshipCategory] = Field(default_factory=list, description="Filter by relationship categories")
    max_depth: int = Field(3, ge=1, le=5, description="Maximum traversal depth")
    direction: str = Field("both", pattern="^(outgoing|incoming|both)$", description="Traversal direction")
    min_strength: float = Field(0.0, ge=0.0, le=1.0, description="Minimum edge strength")
    max_paths: int = Field(100, ge=1, le=1000, description="Maximum paths to return")
    max_nodes: int = Field(50, ge=1, le=500, description="Maximum nodes in result")


class AssociationResponse(BaseModel):
    """Response schema for single association."""

    association: Association


class AssociationListResponse(BaseModel):
    """Response schema for association list."""

    associations: list[Association]
    total_count: int


# Session API Schemas
class SessionCreateRequest(BaseModel):
    """Request schema for creating a session."""

    session_id: str | None = Field(None, description="Client-provided session ID (generated if omitted)")
    workspace_id: str | None = Field(None, description="Workspace ID (auto-created if doesn't exist, defaults to _default)")
    ttl_seconds: int = Field(3600, ge=60, le=86400, description="Session TTL in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    context_id: str | None = Field(None, description="Context to bind session to (defaults to _default, auto-created if needed)")
    working_memory: dict[str, Any] | None = Field(None, description="Initial working memory key-value pairs")
    briefing: bool = Field(False, description="Include briefing with relevant memories")
    briefing_options: dict | None = Field(None, description="Briefing options: lookback_hours, detail_level, limit")


class CommitOptions(BaseModel):
    """Options for session commit."""

    min_importance: float = Field(0.5, ge=0.0, le=1.0, description="Minimum importance threshold")
    deduplicate: bool = Field(True, description="Enable deduplication of extracted memories")
    categories: list[str] | None = Field(None, description="Category names or None for all")
    max_memories: int = Field(50, ge=1, le=500, description="Maximum memories to extract")


class CommitResponse(BaseModel):
    """Response from session commit."""

    session_id: str = Field(..., description="Session ID that was committed")
    memories_extracted: int = Field(..., description="Total memories extracted from session")
    memories_deduplicated: int = Field(..., description="Number of duplicates removed")
    memories_created: int = Field(..., description="Number of new memories created")
    breakdown: dict[str, int] = Field(..., description="Memory count by category")
    extraction_time_ms: int = Field(..., description="Time taken for extraction in milliseconds")


class TouchRequest(BaseModel):
    """Request to extend session TTL."""

    extend_seconds: int = Field(3600, ge=60, le=86400, description="Seconds to extend TTL")


class WorkingMemorySetRequest(BaseModel):
    """Request schema for setting working memory."""

    key: str = Field(..., description="Working memory key", min_length=1)
    value: Any = Field(..., description="Working memory value (JSON-serializable)")
    ttl_seconds: int | None = Field(None, description="Optional TTL override")


class SessionListResponse(BaseModel):
    """Response schema for session list."""

    sessions: list[Session]
    total_count: int


class SessionResponse(BaseModel):
    """Response schema for single session."""

    session: Session


class SessionStartResponse(BaseModel):
    """Response schema for session creation with optional briefing."""

    session: Session
    briefing: SessionBriefing | None = None


class WorkingMemoryResponse(BaseModel):
    """Response schema for working memory entry."""

    key: str
    value: Any
    ttl_seconds: int | None = None
    created_at: datetime
    updated_at: datetime


class SessionBriefingResponse(BaseModel):
    """Response schema for session briefing."""

    briefing: SessionBriefing


# Workspace API Schemas
class WorkspaceCreateRequest(BaseModel):
    """Request schema for creating a workspace."""

    id: str | None = Field(None, description="Optional workspace ID. If omitted, a unique ID is generated.")
    name: str = Field(..., description="Workspace name", min_length=1)
    settings: dict[str, Any] = Field(default_factory=dict, description="Workspace-level settings")
    tags: list[str] = Field(default_factory=list, description="Workspace tags for discovery/lookup")


class WorkspaceUpdateRequest(BaseModel):
    """Request schema for updating a workspace."""

    name: str | None = Field(None, description="Updated workspace name", min_length=1)
    settings: dict[str, Any] | None = Field(None, description="Updated settings")
    tags: list[str] | None = Field(None, description="Updated tags")


class WorkspaceResponse(BaseModel):
    """Response schema for single workspace."""

    workspace: Workspace


class WorkspaceListResponse(BaseModel):
    """Response schema for listing workspaces."""

    workspaces: list[Workspace]


# Context API Schemas
class ContextCreateRequest(BaseModel):
    """Request schema for creating a context within a workspace."""

    id: str | None = Field(None, description="Optional context ID. If omitted, a unique ID is generated.")
    name: str = Field(..., description="Context name (unique within workspace)", min_length=1)
    description: str | None = Field(None, description="Context description")
    settings: dict[str, Any] = Field(default_factory=dict, description="Context-level settings")


class ContextResponse(BaseModel):
    """Response schema for a single context."""

    context: Context


class ContextListResponse(BaseModel):
    """Response schema for listing contexts in a workspace."""

    contexts: list[Context]


class TokenSummary(BaseModel):
    """Token usage summary for detail level recall."""

    returned: int
    full_would_be: int
    savings_percent: float


# Error Responses
class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error details")


# Contradiction API Schemas
class ContradictionResponse(BaseModel):
    """Response model for a contradiction."""

    id: str = Field(..., description="Contradiction ID")
    workspace_id: str = Field(..., description="Workspace ID")
    memory_a_id: str = Field(..., description="First memory ID in the contradiction")
    memory_b_id: str = Field(..., description="Second memory ID in the contradiction")
    contradiction_type: str | None = Field(None, description="Type of contradiction (e.g., 'negation', 'value_conflict')")
    confidence: float | None = Field(None, description="Detection confidence score (0.0-1.0)")
    detection_method: str | None = Field(None, description="Method used to detect the contradiction")
    detected_at: datetime | None = Field(None, description="When the contradiction was detected")
    resolved_at: datetime | None = Field(None, description="When the contradiction was resolved")
    resolution: str | None = Field(None, description="Resolution strategy applied")


class ContradictionListResponse(BaseModel):
    """Response model for listing contradictions."""

    contradictions: list[ContradictionResponse] = Field(..., description="List of contradictions")
    count: int = Field(..., description="Number of contradictions returned")


class ContradictionResolveRequest(BaseModel):
    """Request model for resolving a contradiction."""

    resolution: str = Field(..., description="Resolution strategy: 'keep_a', 'keep_b', 'keep_both', or 'merge'")
    merged_content: str | None = Field(None, description="Merged content (required when resolution is 'merge')")


class ContradictionScanRequest(BaseModel):
    """Request model for triggering a workspace contradiction scan."""

    batch_size: int | None = Field(None, ge=1, le=500, description="Number of memories to process per batch (server default if omitted)")


class ContradictionScanResponse(BaseModel):
    """Response from a workspace contradiction scan."""

    workspace_id: str = Field(..., description="Workspace that was scanned")
    contradictions_found: int = Field(..., description="Number of new contradictions detected")
    contradictions: list[ContradictionResponse] = Field(..., description="Newly detected contradictions")


# Representation API Schemas (dark-gated read surface; see api/v1/representation.py)
class RepresentationRequest(BaseModel):
    """Request schema for the perspective read surface.

    A thin pass-through to ``RepresentationService.get_representation`` — every
    field maps 1:1 onto the service signature. No scoping logic lives here (the
    leakage-0 contract is the service's; the endpoint must not add scoping)."""

    observer: str = Field(..., description="The observing entity (surface name; resolved via the registry, never created)", min_length=1)
    subject: str = Field(..., description="The entity the representation is about (surface name; resolved, never created)", min_length=1)
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    observer_type: EntityType | None = Field(None, description="Optional entity type hint for resolving the observer")
    subject_type: EntityType | None = Field(None, description="Optional entity type hint for resolving the subject")
    limit: int = Field(20, ge=1, le=200, description="Max observations to return")
    include_profile: bool = Field(True, description="Attach a deterministic profile assembled from the scoped observations")


class UserRepresentationRequest(BaseModel):
    """Request schema for the USER-SCOPE self-representation surface.

    A thin pass-through to ``RepresentationService.get_user_representation`` — a
    user's cross-workspace preferences/traits assembled from their ``_global_user``
    memories ("personality follows the user"). No scoping logic lives here: the
    FORCED ``user_id`` filter (the cross-user leakage guard) is the service's.

    ``user_id`` is OPTIONAL in the request: when omitted the endpoint resolves it
    from the authenticated context (the caller reads their OWN representation). An
    explicit ``user_id`` that differs from the caller's identity is a cross-user
    read and must be separately authorized."""

    user_id: str | None = Field(
        None, description="User to assemble the representation for; defaults to the authenticated user_id", min_length=1
    )
    workspace_id: str | None = Field(
        None, description="Workspace override for the request context (the assembly itself reads only _global_user)"
    )
    limit: int = Field(20, ge=1, le=200, description="Max observations to return")
    include_profile: bool = Field(True, description="Attach a deterministic profile assembled from the scoped user-global observations")


# Email ingest API Schema (cross-source ingestion; see api/v1/ingest.py)
class EmailIngestRequest(BaseModel):
    """Request schema for ingesting an email as a memory.

    Maps onto ``services.ingest.email.email_to_remember_input``; the resulting
    memory rides the SAME ``remember()`` pipeline chat/doc use. ``sender`` becomes
    the ``observer_id`` perspective anchor (emails are prefix-less, like docs)."""

    sender: str = Field(..., description="Email from/sender — becomes the observer_id perspective anchor", min_length=1)
    body: str = Field(..., description="Email body text", min_length=1)
    to: list[str] = Field(default_factory=list, description="Recipients stored in metadata and mapped to exact-name sent_to relations")
    subject: str | None = Field(None, description="Email subject line (folded into grounded content + metadata)")
    timestamp: datetime | None = Field(None, description="When the email was sent — becomes the memory's event_time")
    thread_id: str | None = Field(None, description="Conversation/thread id — preferred source_thread_id linkage")
    message_id: str | None = Field(None, description="Message id — fallback source_thread_id; stamped in metadata")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance (0.0-1.0)")
    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    context_id: str | None = Field(None, description="Target memory context")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source metadata; knowledge-work fields are normalized deterministically",
    )
    relations: list[EntityRelationInput] = Field(
        default_factory=list,
        description="Additional typed relations evidenced by this email",
    )


# ============================================
# Context Environment API Schemas
# ============================================


class ContextExecuteRequest(BaseModel):
    """Request to execute code in a session's sandbox environment."""

    code: str = Field(..., description="Python code to execute", min_length=1)
    result_var: str | None = Field(None, description="Store expression result in this variable")
    return_result: bool = Field(True, description="Include result value in response")
    max_return_chars: int = Field(10_000, ge=100, le=100_000, description="Maximum chars for result serialization")


class ContextExecuteResponse(BaseModel):
    """Response from code execution."""

    output: str = Field("", description="Captured stdout output")
    result: str | None = Field(None, description="Expression result (string preview)")
    error: str | None = Field(None, description="Error message if execution failed")
    variables_changed: list[str] = Field(default_factory=list, description="Variables created or modified")


class ContextInspectResponse(BaseModel):
    """Response from inspecting sandbox state."""

    variable: str | None = Field(None, description="Specific variable name if inspecting one")
    type: str | None = Field(None, description="Variable type name")
    preview: str | None = Field(None, description="Value preview string")
    size_bytes: int | None = Field(None, description="Estimated size in bytes")
    variable_count: int | None = Field(None, description="Total variable count (all-vars mode)")
    variables: dict[str, Any] | None = Field(None, description="Variable info map (all-vars mode)")
    total_size_bytes: int | None = Field(None, description="Total size across all variables")
    error: str | None = Field(None, description="Error message if inspection failed")


class ContextLoadRequest(BaseModel):
    """Request to load memories into the sandbox."""

    var: str = Field(..., description="Variable name to store results in", min_length=1)
    query: str = Field(..., description="Memory recall query", min_length=1)
    limit: int = Field(50, ge=1, le=500, description="Maximum memories to recall")
    types: list[str] | None = Field(None, description="Filter by memory types")
    tags: list[str] | None = Field(None, description="Filter by tags")
    min_relevance: float | None = Field(None, ge=0.0, le=1.0, description="Minimum relevance score")
    include_embeddings: bool = Field(False, description="Include embedding vectors")


class ContextLoadResponse(BaseModel):
    """Response from loading memories."""

    count: int = Field(0, description="Number of memories loaded")
    variable: str | None = Field(None, description="Variable name where memories were stored")
    query: str | None = Field(None, description="The recall query used")
    total_available: int | None = Field(None, description="Total matching memories available")
    error: str | None = Field(None, description="Error message if load failed")


class ContextInjectRequest(BaseModel):
    """Request to inject a value into the sandbox."""

    key: str = Field(..., description="Variable name", min_length=1)
    value: Any = Field(..., description="Value to inject (JSON-serializable)")
    parse_json: bool = Field(False, description="Parse value string as JSON")


class ContextInjectResponse(BaseModel):
    """Response from injecting a value."""

    variable: str | None = Field(None, description="Variable name")
    type: str | None = Field(None, description="Value type name")
    preview: str | None = Field(None, description="Value preview string")
    error: str | None = Field(None, description="Error message if injection failed")


class ContextQueryRequest(BaseModel):
    """Request to query the LLM with sandbox context."""

    prompt: str = Field(..., description="Prompt for the LLM", min_length=1)
    variables: list[str] = Field(default_factory=list, description="Variable names to include as context")
    max_context_chars: int | None = Field(None, ge=100, le=500_000, description="Maximum chars for variable context")
    result_var: str | None = Field(None, description="Store LLM response in this variable")


class ContextQueryResponse(BaseModel):
    """Response from LLM query."""

    response: str | None = Field(None, description="LLM response text")
    variables_used: list[str] = Field(default_factory=list, description="Variables included in context")
    result_var: str | None = Field(None, description="Variable where response was stored")
    error: str | None = Field(None, description="Error message if query failed")


class ContextRLMRequest(BaseModel):
    """Request to run a Recursive Language Model (RLM) loop."""

    goal: str = Field(..., description="Natural language description of the goal", min_length=1)
    memory_query: str | None = Field(None, description="Optional memory query to load initial data")
    memory_limit: int = Field(100, ge=1, le=500, description="Maximum memories to load")
    max_iterations: int = Field(10, ge=1, le=50, description="Maximum reasoning iterations")
    variables: list[str] | None = Field(None, description="Variable names to include in context")
    result_var: str | None = Field(None, description="Store final result in this variable")
    detail_level: str = Field("standard", description="Detail level: minimal, standard, verbose")


class ContextRLMResponse(BaseModel):
    """Response from RLM execution."""

    result: str | None = Field(None, description="Final result")
    iterations: int = Field(0, description="Number of iterations performed")
    trace: list[dict[str, Any]] = Field(default_factory=list, description="Execution trace per iteration")
    error: str | None = Field(None, description="Error message if RLM failed")
    goal_achieved: bool = Field(False, description="Whether the goal was achieved")


class ContextStatusResponse(BaseModel):
    """Response from status check."""

    exists: bool = Field(False, description="Whether the environment exists")
    variable_count: int = Field(0, description="Number of variables in sandbox")
    variables: list[str] | None = Field(None, description="Variable names")
    total_size_bytes: int = Field(0, description="Total size of all variables")
    memory_limit_bytes: int | None = Field(None, description="Memory limit in bytes")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Environment metadata")


# ============================================
# Workspace Export/Import API Schemas
# ============================================


class MemoryExportItem(BaseModel):
    """Serialized memory for export."""

    id: str
    content: str
    content_hash: str
    type: str
    subtype: str | None = None
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    abstract: str | None = None
    overview: str | None = None
    session_id: str | None = None
    observer_id: str | None = None
    subject_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class AssociationExportItem(BaseModel):
    """Serialized association for export."""

    source_id: str
    target_id: str
    relationship_type: str
    strength: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkspaceExportData(BaseModel):
    """Export envelope for workspace data."""

    version: str = "1.0"
    exported_at: str
    workspace_id: str
    memories: list[MemoryExportItem] = Field(default_factory=list)
    associations: list[AssociationExportItem] = Field(default_factory=list)
    total_memories: int = 0
    total_associations: int = 0
    offset: int = 0
    limit: int = 0


class WorkspaceImportRequest(BaseModel):
    """Import request body."""

    data: WorkspaceExportData


class WorkspaceImportResult(BaseModel):
    """Import operation results."""

    imported: int = 0
    skipped_duplicates: int = 0
    errors: int = 0
    details: list[str] = Field(default_factory=list)


# ============================================
# Entity API Schemas
# ============================================


class EntityDeriveRequest(BaseModel):
    """Request to trigger inference derivation for an entity."""

    workspace_id: str | None = Field(None, description="Workspace override (defaults to _default)")
    observer_id: str | None = Field(None, description="Optional observer perspective filter")
    force: bool = Field(False, description="Force re-derivation even if recent insights exist")


class EntityDeriveResponse(BaseModel):
    """Response from inference derivation."""

    subject_id: str = Field(..., description="Entity that was analyzed")
    workspace_id: str = Field(..., description="Workspace")
    insights_created: int = Field(0, description="Number of new insights derived")
    insights_updated: int = Field(0, description="Number of existing insights updated")
    source_memory_count: int = Field(0, description="Number of source memories analyzed")
    insights: list[Memory] = Field(default_factory=list, description="Derived insight memories")


class EntityCardResponse(BaseModel):
    """Cached entity profile card - synthesized view of an entity."""

    entity_id: str = Field(..., description="Entity identifier")
    workspace_id: str = Field(..., description="Workspace")
    reflection: str = Field(..., description="Synthesized entity profile")
    insights: list[Memory] = Field(default_factory=list, description="Derived insights about the entity")
    source_memories: list[str] = Field(default_factory=list, description="Source memory IDs")
    confidence: float = Field(0.0, description="Confidence in the profile")
    cached: bool = Field(False, description="Whether this was served from cache")
    generated_at: str | None = Field(None, description="When this card was generated/last refreshed")


class EntityInsightsResponse(BaseModel):
    """Response for listing entity insights."""

    entity_id: str = Field(..., description="Entity identifier")
    workspace_id: str = Field(..., description="Workspace")
    insights: list[Memory] = Field(default_factory=list, description="Derived insights")
    total_count: int = Field(0, description="Total insights found")


# --- Entity Registry CRUD (gated by MEMORYLAYER_ENTITY_REGISTRY_ENABLED) ---


class EntityResponse(BaseModel):
    """Response schema for a single canonical entity."""

    entity: Entity


class EntityListResponse(BaseModel):
    """Response schema for listing canonical entities."""

    entities: list[Entity] = Field(default_factory=list, description="Canonical entities")
    total_count: int = Field(0, description="Number of entities returned")


class EntityResolveResponse(BaseModel):
    """Response schema for resolving a surface name to a canonical entity."""

    resolution: EntityResolution


class EntityMergeRequest(BaseModel):
    """Request schema for merging one canonical entity into another."""

    source_id: str = Field(..., description="Entity to merge FROM (tombstoned on success)")
    target_id: str = Field(..., description="Entity to merge INTO (survives)")
    reason: str = Field(..., description="Audit reason recorded in the merged entity's provenance", min_length=1)


class EntityBackfillResponse(BaseModel):
    """Response schema for the entity-registry backfill of existing memories."""

    workspace_id: str = Field(..., description="Workspace that was backfilled")
    cleared: int = Field(0, description="Existing entities wiped first (when reset=true)")
    scanned: int = Field(0, description="Existing memories enumerated")
    accreted: int = Field(0, description="Memories fed through entity extraction + accretion")


class SeedEntitiesRequest(BaseModel):
    """Request schema for seeding a curated catalog of canonical entities."""

    entities: list[SeedEntity] = Field(..., description="Canonical entities to seed (create-or-update)")
    workspace_id: str | None = Field(None, description="Workspace override (defaults to auth ctx)")


class SeedEntitiesResponse(BaseModel):
    """Response schema for seeding canonical entities."""

    workspace_id: str = Field(..., description="Workspace seeded")
    seeded: int = Field(0, description="Entities created or updated")
    failed: list[str] = Field(default_factory=list, description="Names whose upsert raised")
    skipped_invalid_type: list[str] = Field(
        default_factory=list, description="Names skipped because their entity_type is not in the ontology vocabulary"
    )


class EntityEnrichResponse(BaseModel):
    """Response schema for external-KB entity enrichment (e.g. Wikidata linking)."""

    workspace_id: str = Field(..., description="Workspace enriched")
    source: str = Field(..., description="External KB source, e.g. 'wikidata'")
    checked: int = Field(0, description="Eligible entities considered")
    enriched: int = Field(0, description="Entities linked + provenance updated")
    skipped: int = Field(0, description="No confident link / already linked / store failure")


class EntityDedupeResponse(BaseModel):
    """Response schema for a batch entity dedup pass."""

    workspace_id: str = Field(..., description="Workspace deduped")
    clusters: int = Field(0, description="Duplicate clusters found (size > 1)")
    merged: int = Field(0, description="Entities merged into a cluster representative")


class RelatedEntitiesResponse(BaseModel):
    """Response schema for an entity's co-occurrence neighborhood."""

    entity_id: str = Field(..., description="The source entity")
    related: list[RelatedEntity] = Field(default_factory=list, description="Related entities, most-shared first")


# ============================================
# Chat History API Schemas
# ============================================

from memorylayer_server.models.chat import (
    ChatMessage,
    ChatThread,
)


class ThreadCreateRequest(BaseModel):
    """Request schema for creating a chat thread.

    ``thread_id`` intentionally NOT exposed: ids are server-owned. Letting
    callers supply ids forces global PK collision handling on a key that
    has no business being globally unique (workspace already isolates
    every other resource), and a malicious or careless caller could trip
    creates in unrelated workspaces. Server always generates via
    ``generate_id("thread")``.
    """

    workspace_id: str | None = Field(None, description="Workspace override (defaults to session workspace or _default)")
    user_id: str | None = Field(None, description="User scope for this thread")
    context_id: str | None = Field(None, description="Context within workspace (defaults to _default)")
    observer_id: str | None = Field(None, description="Observer entity ID (typically the AI agent)")
    subject_id: str | None = Field(None, description="Subject entity ID (typically the human user)")
    title: str | None = Field(None, description="Optional display title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    expires_at: datetime | None = Field(None, description="Optional expiration (None = permanent)")
    idle_action: str | None = Field(None, description="Idle policy: null (never idles out) | 'hide' | 'delete'")
    scope: str | None = Field(None, description="Surface scope: 'web' | 'office' | None (≡ 'web')")
    ownership: str = Field("user", description="Thread ownership: 'user' | 'workspace'")
    parent_thread: str | None = Field(None, description="Parent thread id to create a sub-thread (inherits parent workspace + ownership)")


class ThreadUpdateRequest(BaseModel):
    """Request schema for updating a chat thread."""

    title: str | None = Field(None, description="Updated display title")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")
    idle_action: str | None = Field(None, description="Idle policy: null (never idles out) | 'hide' | 'delete'")


class ThreadResponse(BaseModel):
    """Response schema for a single chat thread."""

    thread: ChatThread


class ThreadListResponse(BaseModel):
    """Response schema for listing chat threads."""

    threads: list[ChatThread]
    total_count: int


class MessageCreateRequest(BaseModel):
    """A single message in an append request."""

    id: str | None = Field(default=None, description="Optional client-supplied id; server generates one when omitted.")
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: Any = Field(..., description="Message content — string or structured content blocks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class MessagesAppendRequest(BaseModel):
    """Request schema for appending messages to a thread."""

    messages: list[MessageCreateRequest] = Field(..., min_length=1, description="Messages to append")


class MessagesAppendResponse(BaseModel):
    """Response schema for appended messages."""

    messages: list[ChatMessage]
    thread_id: str
    new_message_count: int = Field(..., description="Total message count after append")


class MessageListResponse(BaseModel):
    """Response schema for listing messages."""

    messages: list[ChatMessage]
    thread_id: str
    total_count: int


class ThreadWithMessagesResponse(BaseModel):
    """Response schema for full thread retrieval with messages."""

    thread: ChatThread
    messages: list[ChatMessage]
    total_messages: int


class ThreadDecomposeResponse(BaseModel):
    """Response schema for decomposition trigger."""

    thread_id: str
    workspace_id: str
    messages_processed: int
    memories_created: int
    from_index: int
    to_index: int
