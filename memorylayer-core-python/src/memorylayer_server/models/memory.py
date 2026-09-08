"""
Memory domain models for MemoryLayer.ai.

Defines cognitive types, domain subtypes, and core memory data structures.
"""

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

from .context_pack import BudgetSummary
from .entity_relation import EntityRelationInput, EntityRelationPath, EntityRelationWriteResult
from .generation import GenerationSummary


class MemoryType(str, Enum):
    """Cognitive classification of memory (how memory is structured)."""

    EPISODIC = "episodic"  # Specific events/interactions
    SEMANTIC = "semantic"  # Facts, concepts, relationships
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"  # Current task context (session-scoped)


class MemorySubtype(str, Enum):
    """Domain classification of memory (what the memory is about)."""

    # Developer-focused taxonomy inspired by Memory-Graph
    SOLUTION = "solution"  # Working fixes to problems
    PROBLEM = "problem"  # Issues encountered
    CODE_PATTERN = "code_pattern"  # Reusable patterns
    FIX = "fix"  # Bug fixes with context
    ERROR = "error"  # Error patterns and resolutions
    WORKFLOW = "workflow"  # Process knowledge
    PREFERENCE = "preference"  # User/project preferences
    DECISION = "decision"  # Architectural decisions

    # v2 additions for extended memory types
    PROFILE = "profile"  # Person/entity profiles
    ENTITY = "entity"  # Named entities (people, places, things)
    EVENT = "event"  # Significant events or milestones
    DIRECTIVE = "directive"  # User instructions/constraints ("always do X", "never do Y")

    # v3 additions for entity attribution and inference
    INFERENCE = "inference"  # Derived insight/conclusion from patterns across memories

    # Fact channel: LLM-extracted (subject, relation, object) triples stored as
    # their own retrievable memories, RRF-fused with the raw-turn pool at recall.
    FACT = "fact"  # Atomic extracted fact (own retrieval channel; see MemoryService._fuse_fact_results)


# OSS-known memory subtypes, grouped by memory type. The "*" key means
# "applies to any memory type" -- the OSS subtypes are flat (any subtype
# may appear with any memory type) and the ontology service treats this
# as the default scope for OSS-known subtypes.
#
# Plugin contributors (e.g. RPG) may extend this set at startup via
# OntologyContributorPlugin.get_subtypes(). The 9 RPG-specific subtypes
# (rpg_directory, rpg_file, rpg_class, rpg_function, rpg_method,
# rpg_component, rpg_module, rpg_package, rpg_interface) live in the
# RPG plugin contributor and are no longer part of the OSS enum.
OSS_KNOWN_SUBTYPES: dict[str, set[str]] = {
    "*": {
        "solution",
        "problem",
        "code_pattern",
        "fix",
        "error",
        "workflow",
        "preference",
        "decision",
        "profile",
        "entity",
        "event",
        "directive",
        "inference",
        "fact",
        "skill",
        "skill_reference",
        "skill_addendum",
        "mcp_server",
    },
}


class MemoryScope(str, Enum):
    """Storage scope for a remembered memory.

    WORKSPACE (default): the memory lives in the caller's workspace — today's
    behavior, unchanged.

    USER: the memory is routed to the user-scoped global workspace
    (config.GLOBAL_USER_WORKSPACE_ID, "_global_user"), partitioned by the
    writer's user_id, so it follows the user across their workspaces without
    leaking across users. A user_id is REQUIRED for USER scope because the
    cross-user read boundary is the forced user_id filter on the recall
    fan-out; a USER-scope write with no user_id is rejected rather than stored
    as an unfilterable global row.
    """

    WORKSPACE = "workspace"
    USER = "user"


class RecallMode(str, Enum):
    """Retrieval strategy for memory queries."""

    RAG = "rag"  # Fast vector similarity search
    LLM = "llm"  # Deep semantic retrieval with query rewriting
    HYBRID = "hybrid"  # Combine both strategies
    AGENTIC = "agentic"  # Iterative LLM-controlled EXPAND/RE_QUERY/STOP loop (enterprise)


class SearchTolerance(str, Enum):
    """Search precision setting affecting fuzzy matching."""

    LOOSE = "loose"  # Fuzzy matching, broader results, lower relevance threshold
    MODERATE = "moderate"  # Balanced precision/recall (default)
    STRICT = "strict"  # Exact matching, high relevance threshold


class DetailLevel(str, Enum):
    """Level of detail to return for hierarchical memories."""

    ABSTRACT = "abstract"  # Brief summary only
    OVERVIEW = "overview"  # High-level overview (tier 2)
    FULL = "full"  # Complete memory content


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory."""

    ACTIVE = "active"
    ARCHIVED = "archived"  # Excluded from default recall, can be restored
    DELETED = "deleted"  # Soft-deleted


class SourceType(str, Enum):
    """Types of sources that can produce memories."""

    MEMORY = "memory"  # Fact decomposition
    SESSION = "session"  # Working memory commit
    DOCUMENT = "document"  # Document ingestion
    PAGE = "page"  # Document page
    THREAD = "thread"  # Chat history decomposition
    DATASET = "dataset"  # Dataset profiling/summarization
    EMAIL = "email"  # Email message ingestion (cross-source: prefix-less,
    # observer_id-anchored on the sender like docs; recipients available for
    # subject/mention extraction)


# OSS-known source types. The email source adapter (and any future producer)
# stamps ``metadata['source']`` with one of these values so downstream consumers
# (coverage/attribution scoring, cross-source assembly) can read the producing
# source straight off the memory. Kept as a set of the enum's string values so
# membership checks are cheap and stable across backends.
KNOWN_SOURCE_TYPES: frozenset[str] = frozenset(st.value for st in SourceType)

_MEMORY_LOGICAL_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")


def validate_memory_logical_key(value: str | None) -> str | None:
    """Validate the stable key used by conditional refinement operations."""

    if value is None:
        return None
    value = value.strip()
    if not _MEMORY_LOGICAL_KEY_RE.fullmatch(value):
        raise ValueError(
            "Memory logical_key must be 1-200 characters using letters, digits, '.', '_', '/', or '-'"
        )
    return value


class Memory(BaseModel):
    """Core memory entity with content, metadata, and lifecycle tracking."""

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Unique memory identifier")
    logical_key: str | None = Field(
        None,
        description="Stable workspace/owner-scoped key for conditional refinement",
    )
    workspace_id: str = Field(..., description="Workspace this memory belongs to")
    tenant_id: str = Field(..., description="Tenant this memory belongs to")
    # context_id is RESERVED / unused as a retrieval filter today and the DB
    # column is nullable (FK→contexts ON DELETE SET NULL). Write paths persist
    # NULL for the "_default" sentinel, so this must be Optional — a non-optional
    # str rejects the NULL on read-back (pydantic string_type error).
    context_id: str | None = Field(None, description="Context for logical grouping; reserved/unused as a filter, usually NULL")
    user_id: str | None = Field(None, description="Optional user scope")

    # Entity attribution (v3) - "who remembers what about whom"
    observer_id: str | None = Field(None, description="Entity doing the observing/remembering (agent ID, user ID, etc.)")
    subject_id: str | None = Field(None, description="Entity the memory is about")

    # Content
    content: str = Field(..., description="The memory content")
    content_hash: str = Field(..., description="SHA-256 hash for deduplication")

    # Classification
    type: MemoryType = Field(..., description="Cognitive type of memory")
    subtype: str | None = Field(None, description="Domain-specific classification (OSS-known or plugin-contributed)")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance (0.0-1.0, affects retention/ranking)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    refinement_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-authored semantic metadata protected by the resource ETag",
    )

    # v2 additions for hierarchical memory
    abstract: str | None = Field(None, description="Brief summary/abstract of memory content")
    overview: str | None = Field(None, description="High-level overview (tier 3)")
    session_id: str | None = Field(None, description="Associated session ID")
    source_memory_id: str | None = Field(None, description="Parent memory this fact was decomposed from")

    # Document provenance - traces memory back to source document/page
    source_document_id: str | None = Field(None, description="Document this memory was derived from")
    source_page_id: str | None = Field(None, description="Document page this memory was extracted from")
    source_dataset_id: str | None = Field(None, description="Dataset this memory was derived from")
    source_thread_id: str | None = Field(None, description="Chat thread this memory was decomposed from")

    category: str | None = Field(None, description="User-defined category")

    # Vector embedding (optional - computed async or stored separately)
    embedding: list[float] | None = Field(None, description="Vector embedding for similarity search")

    # Lifecycle & access tracking
    access_count: int = Field(0, ge=0, description="Number of times memory was accessed")
    last_accessed_at: datetime | None = Field(None, description="Last access timestamp")
    decay_factor: float = Field(1.0, ge=0.0, le=1.0, description="Memory decay over time")
    status: MemoryStatus = Field(MemoryStatus.ACTIVE, description="Memory lifecycle status")
    pinned: bool = Field(False, description="Pinned memories are exempt from decay and archival")

    # Semantic resource revisioning. Embeddings, recall/access projections,
    # importance decay, and ordinary operational metadata deliberately do not
    # participate in this compare-and-swap token.
    revision: int = Field(0, ge=0, description="Authoritative semantic revision")
    etag: str = Field("", description="Opaque semantic compare-and-swap token")

    # Locality-aware ranking metadata (populated during recall)
    source_scope: str | None = Field(None, description="Scope of memory source (same_context, same_workspace, global_workspace, other)")
    relevance_score: float | None = Field(None, description="Base relevance score from vector similarity")
    boosted_score: float | None = Field(None, description="Relevance score after locality boost applied")

    # Evidence contract (populated during recall): which retrieval signals
    # surfaced this memory — any of "vector", "keyword", "alias", "backlink".
    # Lets agents see *why* a memory was recalled, not just that it was.
    match_signals: list[str] | None = Field(None, description="Retrieval signals that matched this memory during recall")

    # Trust scoring (populated during recall)
    trust_score: float | None = Field(None, ge=0.0, le=1.0, description="Composite trust score (0.0-1.0)")
    trust_signals: dict | None = Field(None, description="Component trust scores used to compute trust_score")

    # Freshness metadata (populated during recall)
    freshness_score: float | None = Field(None, description="Exponential freshness score (1.0=new, 0.0=very old)")
    staleness_warning: str | None = Field(None, description="Staleness tier: none, mild, moderate, severe")
    age_days: float | None = Field(None, description="Age of memory in days since creation")

    # Temporal: when the memory's content is *about* (event time), distinct from
    # created_at (= when it was recorded). Powers the timeline index and temporal
    # recall; falls back to created_at when unset (see MemoryService timeline ops).
    event_time: datetime | None = Field(None, description="When the memory's content is about (event time), distinct from created_at")

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Last update timestamp")
    deleted_at: datetime | None = Field(None, description="Durable soft-deletion timestamp")
    relation_write_result: EntityRelationWriteResult | None = Field(
        None,
        description="Result of typed structural relation ingestion for this write",
    )

    @field_validator("logical_key")
    @classmethod
    def validate_logical_key(cls, value: str | None) -> str | None:
        return validate_memory_logical_key(value)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty."""
        if not v or not v.strip():
            raise ValueError("Memory content cannot be empty")
        return v.strip()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Normalize tags (lowercase, no duplicates)."""
        return sorted(set(tag.lower().strip() for tag in v if tag.strip()))


class RememberInput(BaseModel):
    """Request model for creating a new memory."""

    content: str = Field(..., description="The memory content to store")
    type: MemoryType | None = Field(None, description="Cognitive type (auto-classified if omitted)")
    subtype: str | None = Field(None, description="Domain-specific classification (OSS-known or plugin-contributed)")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Memory importance (0.0-1.0)")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    refinement_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Client-authored semantic metadata protected by the resource ETag",
    )
    associations: list[str] = Field(default_factory=list, description="Memory IDs to associate with")
    relations: list[EntityRelationInput] = Field(
        default_factory=list,
        description="Typed structural entity relations evidenced by this memory",
    )

    # Optional overrides (usually auto-computed)
    context_id: str | None = Field(None, description="Target context (default: _default)")
    tenant_id: str | None = Field(None, description="Resolved tenant authority")
    user_id: str | None = Field(None, description="User scope override")
    logical_key: str | None = Field(
        None,
        description="Stable workspace/owner-scoped key for conditional refinement",
    )
    pinned: bool = Field(False, description="Pin the memory against retention decay")
    session_id: str | None = Field(None, description="Session that produced this memory")
    source_memory_id: str | None = Field(None, description="Raw parent memory for a deterministic segment")

    # Storage scope (TRI-STATE — Slice 2):
    #   * None  (UNSET, default): the caller expressed no preference. The memory
    #     is ELIGIBLE for the subtype map (PREFERENCE/DIRECTIVE) and, when the
    #     autoclassify knob is on, the preference-vs-episodic classifier. If
    #     neither promotes it, it falls back to WORKSPACE (today's behavior).
    #   * WORKSPACE: an EXPLICIT request to stay workspace-local. Bypasses the
    #     subtype map and the classifier (explicit caller intent always wins).
    #   * USER: an EXPLICIT request to route into the user-scoped global
    #     workspace (_global_user) so it follows the writer's user_id across
    #     workspaces. Requires a user_id. Bypasses the classifier.
    # See MemoryScope and MemoryService._route_user_scope().
    scope: MemoryScope | None = Field(
        None,
        description="Storage scope: None=unset (eligible for auto-routing), WORKSPACE=workspace-local, USER=user-global",
    )

    # Entity attribution (v3)
    observer_id: str | None = Field(None, description="Entity doing the observing/remembering")
    subject_id: str | None = Field(None, description="Entity this memory is about")

    # Document provenance
    source_document_id: str | None = Field(None, description="Source document ID for provenance tracking")
    source_page_id: str | None = Field(None, description="Source page ID for provenance tracking")
    source_dataset_id: str | None = Field(None, description="Source dataset ID for provenance tracking")
    source_thread_id: str | None = Field(None, description="Source thread ID for provenance tracking")

    # Temporal: when the content is about, if different from now (event time)
    event_time: datetime | None = Field(None, description="When the memory's content is about (event time); distinct from creation time")

    @field_validator("logical_key")
    @classmethod
    def validate_logical_key(cls, value: str | None) -> str | None:
        return validate_memory_logical_key(value)


class MemoryReplaceInput(BaseModel):
    """Complete semantic document for a conditional memory replacement."""

    content: str = Field(..., min_length=1)
    type: MemoryType
    subtype: str | None = None
    tags: list[str] = Field(default_factory=list)
    refinement_metadata: dict[str, Any] = Field(default_factory=dict)
    pinned: bool = False

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Memory content cannot be empty")
        return value.strip()

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        return sorted(set(tag.lower().strip() for tag in value if tag.strip()))


class MemoryMutation(BaseModel):
    """One atomic native-memory semantic mutation."""

    action: Literal["create", "replace", "delete", "restore"]
    memory: Memory
    operation_id: str
    request_hash: str
    expected_etag: str


class MemoryMutationResult(BaseModel):
    memory: Memory
    replayed: bool = False


class MemoryRevision(BaseModel):
    """Immutable accepted semantic memory revision."""

    memory: Memory
    sequence: int
    action: Literal["create", "replace", "delete", "restore"]
    operation_id: str
    request_hash: str


class RecallInput(BaseModel):
    """Request model for querying memories."""

    query: str = Field(..., description="Natural language query or search text")

    # Filters
    types: list[MemoryType] = Field(default_factory=list, description="Filter by cognitive types")
    subtypes: list[str] = Field(default_factory=list, description="Filter by domain subtypes")
    tags: list[str] = Field(default_factory=list, description="Filter by tags (AND logic)")
    context_id: str | None = Field(None, description="Filter by context")
    user_id: str | None = Field(None, description="Filter by user")
    observer_id: str | None = Field(None, description="Filter by observer entity")
    subject_id: str | None = Field(None, description="Filter by subject entity")
    include_global: bool = Field(True, description="Include _global workspace in search")
    include_global_user: bool = Field(
        True,
        description=(
            "Include user-scoped global workspace (_global_user) in search, filtered by "
            "the same user_id. Lets per-user preferences follow the user across workspaces "
            "without leaking across users. Only effective when user_id is set."
        ),
    )

    # Retrieval settings
    mode: RecallMode | None = Field(None, description="Retrieval strategy (None = server default)")
    tolerance: SearchTolerance | None = Field(None, description="Search precision (None = server default)")
    limit: int = Field(10, ge=1, le=100, description="Maximum memories to return")
    offset: int = Field(0, ge=0, description="Number of results to skip for pagination")
    min_relevance: float | None = Field(None, ge=0.0, le=1.0, description="Minimum relevance score (None = server default)")
    recency_weight: float | None = Field(
        None, ge=0.0, le=1.0, description="Weight for recency boosting (0.0=disabled, 1.0=full). None = server default."
    )
    detail_level: DetailLevel | None = Field(None, description="Level of detail to return (None = server default)")

    # Graph traversal (None = use server default from env config)
    include_associations: bool | None = Field(None, description="Include linked memories (None = server default)")
    traverse_depth: int | None = Field(None, ge=0, le=5, description="Multi-hop graph traversal depth (None = server default)")
    max_expansion: int | None = Field(None, ge=1, le=500, description="Max memories discovered via graph expansion (None = server default)")

    # Time range filters (by creation time)
    created_after: datetime | None = Field(None, description="Filter memories created after this time")
    created_before: datetime | None = Field(None, description="Filter memories created before this time")

    # Temporal filters/ordering (by effective event time = event_time or created_at)
    event_after: datetime | None = Field(None, description="Keep memories whose effective event time is >= this")
    event_before: datetime | None = Field(None, description="Keep memories whose effective event time is <= this")
    time_order: str | None = Field(None, description="Order results by effective event time: 'asc' or 'desc' (None = by relevance)")

    # LLM mode options
    context: list[dict[str, str]] = Field(default_factory=list, description="Recent conversation context for query rewriting (LLM mode)")

    # Hybrid mode options
    rag_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Use LLM if RAG confidence < threshold (hybrid mode)")

    # Status filtering
    include_archived: bool = Field(False, description="Include archived memories in recall results")

    # Already-surfaced filtering
    exclude_ids: list[str] = Field(default_factory=list, description="Memory IDs to exclude from results (already shown to user)")

    # Trajectory tracing
    trace: bool = Field(False, description="Enable trajectory logging for this recall")
    budget_tokens: int | None = Field(None, ge=1, description="Hard estimated-token budget for returned memory content")
    include_confidence: bool = Field(True, description="Include deterministic retrieval-confidence metadata")
    include_relations: bool = Field(True, description="Allow bounded typed-relation retrieval when enabled by the server")


class RecallResult(BaseModel):
    """Response model for memory recall queries."""

    memories: list[Memory] = Field(..., description="Retrieved memories")
    total_count: int = Field(..., description="Total matching memories (may exceed returned count)")
    query_tokens: int = Field(0, description="Tokens used in query processing")
    search_latency_ms: int = Field(0, description="Search latency in milliseconds")
    mode_used: RecallMode = Field(..., description="Actual retrieval mode used")

    # LLM mode metadata
    query_rewritten: str | None = Field(None, description="Rewritten query (LLM mode)")
    sufficiency_reached: bool | None = Field(None, description="Whether search stopped early (LLM mode)")

    # Locality-aware ranking metadata
    source_scope: str | None = Field(None, description="Scope of memory source (same_context, same_workspace, global_workspace, other)")
    boosted_score: float | None = Field(None, description="Relevance score after locality boost applied")

    # Token efficiency metadata (for detail_level support)
    token_summary: dict[str, Any] | None = Field(None, description="Token usage summary when using detail_level")

    # Trajectory tracing
    trajectory: dict | None = Field(None, description="Trajectory data if trace=True")

    # Trust scoring
    drift_caveat: str | None = Field(None, description="Warning when one or more recalled memories have low trust scores")

    # Freshness metadata
    freshness_metadata: dict | None = Field(None, description="Aggregate freshness statistics for returned memories")

    # Query intent routing (rule-based, no LLM): labels that classified this query
    query_intent: list[str] | None = Field(None, description="Classified query intent labels used for retrieval routing")
    retrieval_confidence: Literal["strong", "moderate", "weak"] = "weak"
    confidence_reasons: list[str] = Field(default_factory=list)
    budget_summary: BudgetSummary | None = None
    generation_summary: GenerationSummary | None = None
    relation_paths: list[EntityRelationPath] = Field(default_factory=list)
    relation_write_result: EntityRelationWriteResult | None = None


class ReflectInput(BaseModel):
    """Request model for synthesizing memories."""

    query: str = Field(..., description="What to reflect on")
    detail_level: DetailLevel | None = Field(None, description="Level of detail for reflection output (None = server default)")
    include_sources: bool = Field(True, description="Include source memory references")
    depth: int = Field(2, ge=1, le=5, description="Association traversal depth")

    # Optional filters (same as RecallInput)
    types: list[MemoryType] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context_id: str | None = None
    user_id: str | None = None
    observer_id: str | None = None
    subject_id: str | None = None


class ReflectResult(BaseModel):
    """Response model for memory reflection/synthesis."""

    reflection: str = Field(..., description="Synthesized reflection content")
    source_memories: list[str] = Field(default_factory=list, description="Source memory IDs")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in synthesis")
    tokens_processed: int = Field(0, description="Total tokens used")


# Default per-section token budget (~2K tokens each)
DEFAULT_SESSION_SECTION_TOKEN_BUDGET = 2048

# Canonical section names for structured session memory
SESSION_MEMORY_SECTION_NAMES = (
    "context",
    "decisions",
    "learnings",
    "errors",
    "progress",
    "open_items",
)


class SessionMemorySections(BaseModel):
    """Structured session memory organized into semantic categories.

    Divides working memory content across named sections so that related
    information is grouped together and individual section token budgets
    can be enforced independently.

    Each section is a list of string entries (facts, notes, items).
    The total_tokens property provides a rough token-count estimate across
    all sections using the standard len(content) / 4 heuristic.
    """

    sections: dict[str, list[str]] = Field(
        default_factory=lambda: {name: [] for name in SESSION_MEMORY_SECTION_NAMES},
        description="Named memory sections mapping section name to list of entries",
    )
    section_token_budget: int = Field(
        DEFAULT_SESSION_SECTION_TOKEN_BUDGET,
        ge=128,
        description="Per-section token budget (approximate, character-based estimate)",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Estimated total tokens across all sections (len(content) / 4)."""
        total_chars = sum(len(entry) for entries in self.sections.values() for entry in entries)
        return total_chars // 4

    def add_entry(self, section: str, entry: str) -> bool:
        """Add an entry to a section if the section budget allows.

        Args:
            section: Section name (must be a known section)
            entry: Text entry to append

        Returns:
            True if the entry was added, False if the section budget was exceeded
            or the section name is unknown.
        """
        if section not in self.sections:
            return False

        section_tokens = sum(len(e) for e in self.sections[section]) // 4
        entry_tokens = len(entry) // 4

        if section_tokens + entry_tokens > self.section_token_budget:
            return False

        self.sections[section].append(entry)
        return True
