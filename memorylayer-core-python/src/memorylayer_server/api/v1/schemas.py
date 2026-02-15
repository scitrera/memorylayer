"""
API request/response schemas for MemoryLayer.ai endpoints.

These schemas define the HTTP API interface separate from core domain models.
"""
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from memorylayer_server.models.memory import (
    MemoryStatus, MemoryType, MemorySubtype, RecallMode, SearchTolerance,
    Memory, RecallResult, ReflectResult
)
from memorylayer_server.models.association import (
    RelationshipCategory,
    Association, GraphQueryResult, GraphPath
)
from memorylayer_server.models.session import Session, SessionBriefing
from memorylayer_server.models.workspace import Workspace

# Memory API Schemas
class MemoryCreateRequest(BaseModel):
    """Request schema for creating a memory."""

    content: str = Field(..., description="Memory content to store", min_length=1)
    workspace_id: Optional[str] = Field(None, description="Workspace override (defaults to session workspace or _default)")
    type: Optional[MemoryType] = Field(None, description="Cognitive type (auto-classified if omitted)")
    subtype: Optional[MemorySubtype] = Field(None, description="Domain-specific classification")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance (0.0-1.0)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    associations: list[str] = Field(default_factory=list, description="Memory IDs to associate with")
    context_id: Optional[str] = Field(None, description="Target memory context")


class MemoryUpdateRequest(BaseModel):
    """Request schema for updating a memory."""

    content: Optional[str] = Field(None, description="Updated content", min_length=1)
    type: Optional[MemoryType] = Field(None, description="Updated cognitive type")
    subtype: Optional[MemorySubtype] = Field(None, description="Updated domain classification")
    importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Updated importance")
    tags: Optional[list[str]] = Field(None, description="Updated tags")
    metadata: Optional[dict[str, Any]] = Field(None, description="Updated metadata")
    pinned: Optional[bool] = Field(None, description="Pin/unpin memory (pinned memories are exempt from decay)")


class MemoryRecallRequest(BaseModel):
    """Request schema for querying memories."""

    query: str = Field(..., description="Natural language query", min_length=1)
    workspace_id: Optional[str] = Field(None, description="Workspace override (defaults to session workspace or _default)")
    types: list[MemoryType] = Field(default_factory=list, description="Filter by cognitive types")
    subtypes: list[MemorySubtype] = Field(default_factory=list, description="Filter by domain subtypes")
    tags: list[str] = Field(default_factory=list, description="Filter by tags (AND logic)")
    context_id: Optional[str] = Field(None, description="Filter by memory context")
    mode: Optional[RecallMode] = Field(None, description="Retrieval strategy (None = server default)")
    tolerance: Optional[SearchTolerance] = Field(None, description="Search precision (None = server default)")
    limit: int = Field(10, ge=1, le=100, description="Maximum memories to return")
    min_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score (None = server default)")
    recency_weight: Optional[float] = Field(None, ge=0.0, le=1.0,
                                            description="Weight for recency boosting (0.0=disabled, 1.0=full). None = server default.")
    include_associations: Optional[bool] = Field(None, description="Include linked memories (None = server default)")
    traverse_depth: Optional[int] = Field(None, ge=0, le=5, description="Multi-hop graph traversal depth (None = server default)")
    max_expansion: Optional[int] = Field(None, ge=1, le=500, description="Max memories discovered via graph expansion (None = server default)")
    created_after: Optional[datetime] = Field(None, description="Filter memories created after this time")
    created_before: Optional[datetime] = Field(None, description="Filter memories created before this time")
    context: list[dict[str, str]] = Field(default_factory=list, description="Recent conversation context")
    rag_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Use LLM if RAG confidence < threshold")
    detail_level: Optional[str] = Field(None, description="Detail level: abstract, overview, or full (None = server default)")
    include_archived: bool = Field(False, description="Include archived memories in recall results")


class MemoryReflectRequest(BaseModel):
    """Request schema for synthesizing memories."""

    query: str = Field(..., description="What to reflect on", min_length=1)
    workspace_id: Optional[str] = Field(None, description="Workspace override (defaults to session workspace or _default)")
    detail_level: Optional[str] = Field(None, description="Level of detail: abstract, overview, full (None = server default)")
    include_sources: bool = Field(True, description="Include source memory references")
    depth: int = Field(2, ge=1, le=5, description="Association traversal depth")
    types: list[MemoryType] = Field(default_factory=list, description="Filter by types")
    subtypes: list[MemorySubtype] = Field(default_factory=list, description="Filter by subtypes")
    tags: list[str] = Field(default_factory=list, description="Filter by tags")
    context_id: Optional[str] = Field(None, description="Filter by memory context")


class MemoryDecayRequest(BaseModel):
    """Request schema for decaying a memory."""

    decay_rate: float = Field(0.1, ge=0.0, le=1.0, description="Decay rate to apply")


class MemoryBatchRequest(BaseModel):
    """Request schema for batch memory operations."""

    operations: list[dict[str, Any]] = Field(
        ...,
        description="List of operations with 'type' and 'data' fields"
    )


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
    memory_id: Optional[str] = Field(None, description="Memory ID for create/update operations")
    error: Optional[str] = Field(None, description="Error message if failed")


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
    workspace_id: Optional[str] = Field(None, description="Workspace override (defaults to session workspace or _default)")


class AssociationCreateFullRequest(BaseModel):
    """Request schema for creating an association with both source and target in body."""

    source_id: str = Field(..., description="Source memory ID")
    target_id: str = Field(..., description="Target memory ID")
    relationship: str = Field(..., description="Relationship type (e.g., SIMILAR_TO, CAUSES, SOLVES)")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Relationship strength")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")


class AssociationListRequest(BaseModel):
    """Request schema for listing associations."""

    relationships: Optional[list[str]] = Field(
        None,
        description="Filter by relationship types (e.g., SIMILAR_TO, CAUSES)"
    )
    direction: str = Field(
        "both",
        pattern="^(outgoing|incoming|both)$",
        description="Association direction"
    )


class MemoryTraverseRequest(BaseModel):
    """Request schema for traversing from a specific memory."""

    workspace_id: Optional[str] = Field(None, description="Workspace override (defaults to session workspace or _default)")
    max_depth: int = Field(2, ge=1, le=5, description="Maximum traversal depth")
    relationship_types: list[str] = Field(
        default_factory=list,
        description="Filter by specific relationship types (empty = all)"
    )
    direction: str = Field(
        "both",
        pattern="^(outgoing|incoming|both)$",
        description="Traversal direction: outgoing, incoming, both"
    )
    min_strength: float = Field(0.0, ge=0.0, le=1.0, description="Minimum edge strength")


class GraphTraverseRequest(BaseModel):
    """Request schema for graph traversal."""

    start_memory_id: str = Field(..., description="Starting memory for traversal")
    relationship_types: list[str] = Field(
        default_factory=list,
        description="Filter by specific relationship types"
    )
    relationship_categories: list[RelationshipCategory] = Field(
        default_factory=list,
        description="Filter by relationship categories"
    )
    max_depth: int = Field(3, ge=1, le=5, description="Maximum traversal depth")
    direction: str = Field(
        "both",
        pattern="^(outgoing|incoming|both)$",
        description="Traversal direction"
    )
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

    session_id: Optional[str] = Field(None, description="Client-provided session ID (generated if omitted)")
    workspace_id: Optional[str] = Field(None, description="Workspace ID (auto-created if doesn't exist, defaults to _default)")
    ttl_seconds: int = Field(3600, ge=60, le=86400, description="Session TTL in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    context_id: Optional[str] = Field(None, description="Context to bind session to (defaults to _default, auto-created if needed)")
    working_memory: Optional[dict[str, Any]] = Field(None, description="Initial working memory key-value pairs")
    briefing: bool = Field(False, description="Include briefing with relevant memories")
    briefing_options: Optional[dict] = Field(None, description="Briefing options: lookback_hours, detail_level, limit")


class CommitOptions(BaseModel):
    """Options for session commit."""

    min_importance: float = Field(0.5, ge=0.0, le=1.0, description="Minimum importance threshold")
    deduplicate: bool = Field(True, description="Enable deduplication of extracted memories")
    categories: Optional[list[str]] = Field(None, description="Category names or None for all")
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
    ttl_seconds: Optional[int] = Field(None, description="Optional TTL override")


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
    briefing: Optional[SessionBriefing] = None


class WorkingMemoryResponse(BaseModel):
    """Response schema for working memory entry."""

    key: str
    value: Any
    ttl_seconds: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class SessionBriefingResponse(BaseModel):
    """Response schema for session briefing."""

    briefing: SessionBriefing


# Workspace API Schemas
class WorkspaceCreateRequest(BaseModel):
    """Request schema for creating a workspace."""

    name: str = Field(..., description="Workspace name", min_length=1)
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace-level settings"
    )


class WorkspaceUpdateRequest(BaseModel):
    """Request schema for updating a workspace."""

    name: Optional[str] = Field(None, description="Updated workspace name", min_length=1)
    settings: Optional[dict[str, Any]] = Field(None, description="Updated settings")


class WorkspaceResponse(BaseModel):
    """Response schema for single workspace."""

    workspace: Workspace


class WorkspaceListResponse(BaseModel):
    """Response schema for listing workspaces."""

    workspaces: list[Workspace]


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
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")


# Contradiction API Schemas
class ContradictionResponse(BaseModel):
    """Response model for a contradiction."""

    id: str = Field(..., description="Contradiction ID")
    workspace_id: str = Field(..., description="Workspace ID")
    memory_a_id: str = Field(..., description="First memory ID in the contradiction")
    memory_b_id: str = Field(..., description="Second memory ID in the contradiction")
    contradiction_type: Optional[str] = Field(None, description="Type of contradiction (e.g., 'negation', 'value_conflict')")
    confidence: Optional[float] = Field(None, description="Detection confidence score (0.0-1.0)")
    detection_method: Optional[str] = Field(None, description="Method used to detect the contradiction")
    detected_at: Optional[datetime] = Field(None, description="When the contradiction was detected")
    resolved_at: Optional[datetime] = Field(None, description="When the contradiction was resolved")
    resolution: Optional[str] = Field(None, description="Resolution strategy applied")


class ContradictionListResponse(BaseModel):
    """Response model for listing contradictions."""

    contradictions: list[ContradictionResponse] = Field(..., description="List of contradictions")
    count: int = Field(..., description="Number of contradictions returned")


class ContradictionResolveRequest(BaseModel):
    """Request model for resolving a contradiction."""

    resolution: str = Field(..., description="Resolution strategy: 'keep_a', 'keep_b', 'keep_both', or 'merge'")
    merged_content: Optional[str] = Field(None, description="Merged content (required when resolution is 'merge')")


# ============================================
# Context Environment API Schemas
# ============================================

class ContextExecuteRequest(BaseModel):
    """Request to execute code in a session's sandbox environment."""

    code: str = Field(..., description="Python code to execute", min_length=1)
    result_var: Optional[str] = Field(None, description="Store expression result in this variable")
    return_result: bool = Field(True, description="Include result value in response")
    max_return_chars: int = Field(10_000, ge=100, le=100_000, description="Maximum chars for result serialization")


class ContextExecuteResponse(BaseModel):
    """Response from code execution."""

    output: str = Field('', description="Captured stdout output")
    result: Optional[str] = Field(None, description="Expression result (string preview)")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    variables_changed: list[str] = Field(default_factory=list, description="Variables created or modified")


class ContextInspectResponse(BaseModel):
    """Response from inspecting sandbox state."""

    variable: Optional[str] = Field(None, description="Specific variable name if inspecting one")
    type: Optional[str] = Field(None, description="Variable type name")
    preview: Optional[str] = Field(None, description="Value preview string")
    size_bytes: Optional[int] = Field(None, description="Estimated size in bytes")
    variable_count: Optional[int] = Field(None, description="Total variable count (all-vars mode)")
    variables: Optional[dict[str, Any]] = Field(None, description="Variable info map (all-vars mode)")
    total_size_bytes: Optional[int] = Field(None, description="Total size across all variables")
    error: Optional[str] = Field(None, description="Error message if inspection failed")


class ContextLoadRequest(BaseModel):
    """Request to load memories into the sandbox."""

    var: str = Field(..., description="Variable name to store results in", min_length=1)
    query: str = Field(..., description="Memory recall query", min_length=1)
    limit: int = Field(50, ge=1, le=500, description="Maximum memories to recall")
    types: Optional[list[str]] = Field(None, description="Filter by memory types")
    tags: Optional[list[str]] = Field(None, description="Filter by tags")
    min_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum relevance score")
    include_embeddings: bool = Field(False, description="Include embedding vectors")


class ContextLoadResponse(BaseModel):
    """Response from loading memories."""

    count: int = Field(0, description="Number of memories loaded")
    variable: Optional[str] = Field(None, description="Variable name where memories were stored")
    query: Optional[str] = Field(None, description="The recall query used")
    total_available: Optional[int] = Field(None, description="Total matching memories available")
    error: Optional[str] = Field(None, description="Error message if load failed")


class ContextInjectRequest(BaseModel):
    """Request to inject a value into the sandbox."""

    key: str = Field(..., description="Variable name", min_length=1)
    value: Any = Field(..., description="Value to inject (JSON-serializable)")
    parse_json: bool = Field(False, description="Parse value string as JSON")


class ContextInjectResponse(BaseModel):
    """Response from injecting a value."""

    variable: Optional[str] = Field(None, description="Variable name")
    type: Optional[str] = Field(None, description="Value type name")
    preview: Optional[str] = Field(None, description="Value preview string")
    error: Optional[str] = Field(None, description="Error message if injection failed")


class ContextQueryRequest(BaseModel):
    """Request to query the LLM with sandbox context."""

    prompt: str = Field(..., description="Prompt for the LLM", min_length=1)
    variables: list[str] = Field(default_factory=list, description="Variable names to include as context")
    max_context_chars: Optional[int] = Field(None, ge=100, le=500_000, description="Maximum chars for variable context")
    result_var: Optional[str] = Field(None, description="Store LLM response in this variable")


class ContextQueryResponse(BaseModel):
    """Response from LLM query."""

    response: Optional[str] = Field(None, description="LLM response text")
    variables_used: list[str] = Field(default_factory=list, description="Variables included in context")
    result_var: Optional[str] = Field(None, description="Variable where response was stored")
    error: Optional[str] = Field(None, description="Error message if query failed")


class ContextRLMRequest(BaseModel):
    """Request to run a Recursive Language Model (RLM) loop."""

    goal: str = Field(..., description="Natural language description of the goal", min_length=1)
    memory_query: Optional[str] = Field(None, description="Optional memory query to load initial data")
    memory_limit: int = Field(100, ge=1, le=500, description="Maximum memories to load")
    max_iterations: int = Field(10, ge=1, le=50, description="Maximum reasoning iterations")
    variables: Optional[list[str]] = Field(None, description="Variable names to include in context")
    result_var: Optional[str] = Field(None, description="Store final result in this variable")
    detail_level: str = Field("standard", description="Detail level: minimal, standard, verbose")


class ContextRLMResponse(BaseModel):
    """Response from RLM execution."""

    result: Optional[str] = Field(None, description="Final result")
    iterations: int = Field(0, description="Number of iterations performed")
    trace: list[dict[str, Any]] = Field(default_factory=list, description="Execution trace per iteration")
    error: Optional[str] = Field(None, description="Error message if RLM failed")
    goal_achieved: bool = Field(False, description="Whether the goal was achieved")


class ContextStatusResponse(BaseModel):
    """Response from status check."""

    exists: bool = Field(False, description="Whether the environment exists")
    variable_count: int = Field(0, description="Number of variables in sandbox")
    variables: Optional[list[str]] = Field(None, description="Variable names")
    total_size_bytes: int = Field(0, description="Total size of all variables")
    memory_limit_bytes: Optional[int] = Field(None, description="Memory limit in bytes")
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
    subtype: Optional[str] = None
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    abstract: Optional[str] = None
    overview: Optional[str] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


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
