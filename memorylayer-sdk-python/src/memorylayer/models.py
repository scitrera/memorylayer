"""Pydantic models for MemoryLayer.ai SDK."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .types import MemoryType


@dataclass
class PrincipalRef:
    """A reference to a principal (user, service, agent, task)."""

    type: str
    id: str


@dataclass
class AuthorityContext:
    """OBO authority context emitted as X-Aether-* headers."""

    grant_id: str
    subject: PrincipalRef | None = None


class Memory(BaseModel):
    """A memory entry."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    workspace_id: str
    space_id: str | None = None
    user_id: str | None = None
    content: str
    type: MemoryType
    subtype: str | None = None
    importance: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    access_count: int = 0
    last_accessed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    # Additional server fields
    content_hash: str | None = None
    context_id: str | None = None
    tenant_id: str = "_default"
    decay_factor: float = 1.0
    pinned: bool = False
    status: str = "active"
    abstract: str | None = None
    overview: str | None = None
    session_id: str | None = None
    source_memory_id: str | None = None
    observer_id: str | None = None
    subject_id: str | None = None
    relevance_score: float | None = None
    boosted_score: float | None = None
    source_scope: str | None = None
    embedding: list[float] | None = None
    archived_at: datetime | None = None
    match_signals: list[str] | None = None
    relation_write_result: "EntityRelationWriteResult | None" = None


class EntityRelationInput(BaseModel):
    source_entity_id: str | None = None
    source_entity_name: str | None = None
    target_entity_id: str | None = None
    target_entity_name: str | None = None
    relationship: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_span_start: int | None = None
    source_span_end: int | None = None


class EntityRelationWriteResult(BaseModel):
    resolved: int = 0
    unresolved: int = 0
    rejected: int = 0
    duplicate: int = 0
    relation_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class BudgetSummary(BaseModel):
    requested: int | None = None
    used: int = 0
    estimator: str = ""
    truncated_items: int = 0
    omitted_items: int = 0


class GenerationSummary(BaseModel):
    policy: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class EntityRelationPath(BaseModel):
    seed_entity_id: str
    entity_ids: list[str] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence_memory_ids: list[str] = Field(default_factory=list)


class Association(BaseModel):
    """A relationship between two memories."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    workspace_id: str
    source_id: str
    target_id: str
    relationship: str
    strength: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RecallResult(BaseModel):
    """Result from a recall (search) query."""

    memories: list[Memory]
    total_count: int
    query_tokens: int | None = None
    search_latency_ms: int | None = None
    mode_used: str | None = None
    retrieval_confidence: Literal["strong", "moderate", "weak"] = "weak"
    confidence_reasons: list[str] = Field(default_factory=list)
    budget_summary: BudgetSummary | None = None
    generation_summary: GenerationSummary | None = None
    relation_paths: list[EntityRelationPath] = Field(default_factory=list)


class ReflectResult(BaseModel):
    """Result from a reflect (synthesis) query."""

    reflection: str
    source_memories: list[str] = Field(default_factory=list)  # Memory IDs
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_processed: int | None = None


class Session(BaseModel):
    """A working memory session."""

    id: str
    workspace_id: str
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime
    created_at: datetime


class SessionActivity(BaseModel):
    """Recent session activity summary."""

    timestamp: datetime
    summary: str
    memories_created: int
    key_decisions: list[str] = Field(default_factory=list)


class OpenThread(BaseModel):
    """An open topic/thread of work."""

    topic: str
    status: str
    last_activity: datetime
    key_memories: list[str] = Field(default_factory=list)


class ContradictionDetection(BaseModel):
    """Detected contradiction between memories."""

    memory_a: str
    memory_b: str
    relationship: str
    needs_resolution: bool


class SessionBriefing(BaseModel):
    """Session briefing with recent activity and context."""

    workspace_summary: dict[str, Any]
    recent_activity: list[SessionActivity] = Field(default_factory=list)
    open_threads: list[OpenThread] = Field(default_factory=list)
    contradictions_detected: list[ContradictionDetection] = Field(default_factory=list)


class SessionCheckpoint(BaseModel):
    id: str
    workspace_id: str
    session_id: str
    raw_memory_id: str
    source_kind: str
    source_sequence: int | None = None
    source_boundary: int | None = None
    content_hash: str
    byte_count: int
    capture_status: str
    index_status: str
    enrichment_status: str
    idempotency_key: str
    created_at: datetime
    updated_at: datetime


class ContextPackItem(BaseModel):
    id: str
    kind: str
    content: str
    importance: float = 0.5
    event_time: datetime | None = None
    match_signals: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    tombstone: bool = False


class ContextPack(BaseModel):
    rendered: str
    items: list[ContextPackItem] = Field(default_factory=list)
    open_threads: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_contradictions: list[dict[str, Any]] = Field(default_factory=list)
    budget_summary: BudgetSummary
    generation_summary: GenerationSummary
    cursor: str
    degradation_notices: list[str] = Field(default_factory=list)


class ContextDelta(BaseModel):
    rendered: str
    items: list[ContextPackItem] = Field(default_factory=list)
    budget_summary: BudgetSummary
    generation_summary: GenerationSummary
    cursor: str
    has_more: bool = False
    degradation_notices: list[str] = Field(default_factory=list)


class Workspace(BaseModel):
    """A workspace (tenant boundary)."""

    id: str
    tenant_id: str
    name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class MemorySpace(BaseModel):
    """A memory space within a workspace."""

    id: str
    workspace_id: str
    name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class MemoryListResponse(BaseModel):
    """Result from ``list_memories`` (GET /v1/memories) — a plain filtered
    enumeration (no vector search), ordered by recency."""

    memories: list[Memory]
    total_count: int


# ------------------------------------------------------------------ #
# API token models
# ------------------------------------------------------------------ #


class TokenInfo(BaseModel):
    """An API token record (without the plaintext secret)."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    principal_type: str = "User"
    workspace_patterns: list[str] = Field(default_factory=lambda: ["*"])
    scopes: list[str] = Field(default_factory=lambda: ["*"])
    created_at: str
    expires_at: str | None = None
    revoked: bool = False


class TokenCreateResult(TokenInfo):
    """A newly created token, including the one-time plaintext secret.

    The plaintext ``token`` is only returned at creation time — store it
    securely; it cannot be retrieved again.
    """

    token: str


# ------------------------------------------------------------------ #
# Entity registry models (gated by MEMORYLAYER_ENTITY_REGISTRY_ENABLED) #
# ------------------------------------------------------------------ #


class Entity(BaseModel):
    """A canonical, workspace-scoped entity node."""

    model_config = ConfigDict(extra="allow")

    id: str
    workspace_id: str
    entity_type: str
    canonical_name: str
    normalized_name: str
    aliases: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    provenance: dict[str, Any] = Field(default_factory=dict)
    representative_memory_id: str | None = None
    status: str = "active"
    merged_into: str | None = None
    created_at: datetime
    updated_at: datetime


class EntityResolution(BaseModel):
    """Result of resolving a surface name to a canonical entity."""

    model_config = ConfigDict(extra="allow")

    entity: Entity
    matched_via: str
    score: float = 1.0


# ------------------------------------------------------------------ #
# Chat history models
# ------------------------------------------------------------------ #


class ChatMessageContent(BaseModel):
    """Structured content block within a chat message."""

    type: str
    text: str | None = None
    data: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    """A single message in a chat thread."""

    id: str
    thread_id: str
    message_index: int
    role: str
    content: str | list[ChatMessageContent]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ChatThread(BaseModel):
    """A conversation thread."""

    id: str
    workspace_id: str
    tenant_id: str = "_default"
    user_id: str | None = None
    context_id: str = "_default"
    observer_id: str | None = None
    subject_id: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    last_decomposed_at: datetime | None = None
    last_decomposed_index: int = 0
    expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    scope: str | None = None
    ownership: str = "user"


class ChatThreadWithMessages(BaseModel):
    """Thread with messages inlined."""

    thread: ChatThread
    messages: list[ChatMessage] = Field(default_factory=list)
    total_messages: int = 0


class DecompositionResult(BaseModel):
    """Result from triggering decomposition."""

    thread_id: str
    workspace_id: str
    messages_processed: int = 0
    memories_created: int = 0
    from_index: int = 0
    to_index: int = 0


# ------------------------------------------------------------------ #
# Document models (Enterprise)
# ------------------------------------------------------------------ #


class DocumentPage(BaseModel):
    """A page from an ingested document (Enterprise)."""

    id: str
    document_id: str
    workspace_id: str
    page_no: int
    image_storage_path: str | None = None
    transcript: str | None = None
    transcript_model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    relevance_score: float | None = None


class DocumentInfo(BaseModel):
    """Document metadata returned from the API (Enterprise)."""

    id: str
    workspace_id: str
    filename: str
    document_type: str
    content_hash: str
    size_bytes: int
    mime_type: str | None = None
    status: str
    target_context_id: str = "_default"
    page_count: int = 0
    chunk_count: int = 0
    memory_ids: list[str] = Field(default_factory=list)
    storage_path: str | None = None
    retain_original: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


class JobInfo(BaseModel):
    """Ingestion job status (Enterprise)."""

    id: str
    workspace_id: str
    document_ids: list[str] = Field(default_factory=list)
    status: str
    progress_percent: int = 0
    documents_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class PageSearchResult(BaseModel):
    """Result from a document page search (Enterprise)."""

    pages: list[DocumentPage]
    total_count: int
    query: str


# ------------------------------------------------------------------ #
# Dataset models (Enterprise)
# ------------------------------------------------------------------ #


class DatasetColumn(BaseModel):
    """Column-level schema and statistics from dataset profiling (Enterprise)."""

    name: str
    dtype: str
    column_type: str = "unknown"
    nullable: bool = False
    null_count: int = 0
    null_percent: float = 0.0
    unique_count: int = 0

    # Numeric stats
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    median_value: float | None = None
    std_value: float | None = None
    p25_value: float | None = None
    p75_value: float | None = None

    # String stats
    min_length: int | None = None
    max_length: int | None = None
    avg_length: float | None = None

    # Categorical stats
    top_values: list[dict[str, Any]] | None = None

    # Time series detection
    is_temporal: bool = False
    temporal_resolution: str | None = None
    temporal_range_start: str | None = None
    temporal_range_end: str | None = None

    # Distribution
    histogram: dict[str, Any] | None = None


class DatasetInfo(BaseModel):
    """Dataset metadata returned from the API (Enterprise)."""

    id: str
    workspace_id: str
    name: str
    filename: str
    format: str
    content_hash: str
    size_bytes: int
    status: str
    target_context_id: str = "_default"
    row_count: int = 0
    column_count: int = 0
    columns: list[DatasetColumn] = Field(default_factory=list)
    memory_ids: list[str] = Field(default_factory=list)
    profile_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    profiling_started_at: datetime | None = None
    profiling_completed_at: datetime | None = None


class DatasetJobInfo(BaseModel):
    """Dataset processing job status (Enterprise)."""

    id: str
    workspace_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    status: str
    progress_percent: int = 0
    datasets_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DatasetSliceResult(BaseModel):
    """Result of a dataset slice query (Enterprise)."""

    dataset_id: str
    columns: list[str]
    dtypes: list[str] = Field(default_factory=list)
    rows: list[list[Any]]
    total_matching: int
    returned_count: int
    sql_executed: str | None = None
