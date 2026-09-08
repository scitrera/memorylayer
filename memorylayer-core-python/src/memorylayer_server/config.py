"""Configuration management for MemoryLayer.ai using Pydantic Settings."""

from enum import Enum

# ============================================
# Data Home Directory
# ============================================
MEMORYLAYER_DATA_DIR = "MEMORYLAYER_DATA_DIR"

# ============================================
# Server Configuration
# ============================================
MEMORYLAYER_SERVER_HOST = "MEMORYLAYER_SERVER_HOST"
DEFAULT_MEMORYLAYER_SERVER_HOST = "127.0.0.1"
MEMORYLAYER_SERVER_PORT = "MEMORYLAYER_SERVER_PORT"
DEFAULT_MEMORYLAYER_SERVER_PORT = 61001


# ============================================
# Embedding Providers
# ============================================
class EmbeddingProviderType(str, Enum):
    """Available embedding provider types.

    The legacy in-process providers ``local`` (sentence-transformers),
    ``colpali`` (colpali-engine), and ``qwen3-vl`` (qwen-vl-utils) were
    removed. All self-hosted/multi-vector workloads now route through
    the ``embed_server`` provider, which delegates to ``memorylayer-embed-server``.
    See ``_LEGACY_EMBEDDING_PROVIDERS`` below for the migration guard.
    """

    OPENAI = "openai"  # OpenAI API (cloud, text-only; also works with any OpenAI-compatible endpoint)
    GOOGLE = "google"  # Google GenAI API (cloud, text-only)
    EMBED_SERVER = "embed_server"  # Delegates to memorylayer-embed-server (self-hosted, all modalities)
    MOCK = "mock"  # Mock provider for testing only (deterministic hash-based)
    HASH = "hash"  # Deterministic lexical feature-hashing provider (eval/testing only, carries token-level signal)


# Legacy provider names that were removed when heavy ML moved to memorylayer-embed-server.
# Any of these in MEMORYLAYER_EMBEDDING_PROVIDER must trigger a hard error
# at config-load time (see assert_supported_embedding_provider).
_LEGACY_EMBEDDING_PROVIDERS: frozenset[str] = frozenset({"local", "colpali", "qwen3-vl"})


def assert_supported_embedding_provider(value: str) -> None:
    """Raise ValueError with migration guidance if ``value`` is a removed provider name."""
    if value in _LEGACY_EMBEDDING_PROVIDERS:
        raise ValueError(
            f"MEMORYLAYER_EMBEDDING_PROVIDER={value!r} was removed. "
            f"Heavy ML (sentence-transformers, colpali-engine, qwen-vl-utils) "
            f"now lives in the memorylayer-embed-server package. "
            f"Set MEMORYLAYER_EMBEDDING_PROVIDER=embed_server and run "
            f"a memorylayer-embed-server peer (configure MEMORYLAYER_EMBED_SERVER_URL "
            f"or MEMORYLAYER_EMBED_TRANSPORT=aether). For cloud, use 'openai' or 'google'."
        )


MEMORYLAYER_EMBEDDING_PROVIDER = "MEMORYLAYER_EMBEDDING_PROVIDER"
# Default 'hash': a bare `pip install memorylayer-server && memorylayer serve` must
# WORK, with no peer container and no API key. Every other provider needs external
# infrastructure — 'embed_server' a GPU sidecar, 'openai'/'google' a paid key — so
# using one as the default meant a first-run `POST /v1/memories` failed with a
# generic 500 and a connection error buried in the log.
#
# 'hash' is lexical (feature-hashed bag-of-words), NOT semantic. It is a working
# default, not a good embedder, and the provider logs a loud startup warning saying
# so. Deployments that want real retrieval set the provider explicitly; the published
# Docker image pins MEMORYLAYER_EMBEDDING_PROVIDER=embed_server so it keeps its
# GPU-backed behavior and can never silently fall back to lexical matching.
DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER = EmbeddingProviderType.HASH
MEMORYLAYER_EMBEDDING_MODEL = "MEMORYLAYER_EMBEDDING_MODEL"
MEMORYLAYER_EMBEDDING_DIMENSIONS = "MEMORYLAYER_EMBEDDING_DIMENSIONS"
# Default vector dimension for embed_server's text embeddings, matching the embed
# server's own default single-vector model (sentence-transformers/all-MiniLM-L6-v2,
# 384-d). Operators MUST override this to match their server-side model when they
# change it — e.g. 2048 for the vLLM Qwen3-VL-Embedding-2B backend.
#
# ⚠ This is not merely an operational setting: the dimension is baked into every
# stored vector. Mixing dimensions inside one workspace makes recall fail outright
# on sqlite-vec (vec_distance_cosine raises, returning NO results for the whole
# query) and silently drop the older memories on the pure-Python fallback. Changing
# it on a populated deployment means re-embedding. See the README.
DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER = 384
MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED = "MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED"
DEFAULT_MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED = True

# ============================================
# Embedding Service
# ============================================
MEMORYLAYER_EMBEDDING_SERVICE = "MEMORYLAYER_EMBEDDING_SERVICE"
DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE = "default"

# ============================================
# Storage Backend
# ============================================
MEMORYLAYER_STORAGE_BACKEND = "MEMORYLAYER_STORAGE_BACKEND"
DEFAULT_MEMORYLAYER_STORAGE_BACKEND = "sqlite"

MEMORYLAYER_SQLITE_STORAGE_PATH = "MEMORYLAYER_SQLITE_STORAGE_PATH"
DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH = "memorylayer.db"

# Turso/libSQL storage backend (alternative to SQLite with native vector support)
MEMORYLAYER_TURSO_MODE = "MEMORYLAYER_TURSO_MODE"
DEFAULT_MEMORYLAYER_TURSO_MODE = "local"  # local, remote, replica

MEMORYLAYER_TURSO_DB_PATH = "MEMORYLAYER_TURSO_DB_PATH"
DEFAULT_MEMORYLAYER_TURSO_DB_PATH = "memorylayer.db"

MEMORYLAYER_TURSO_URL = "MEMORYLAYER_TURSO_URL"  # remote/replica mode
MEMORYLAYER_TURSO_AUTH_TOKEN = "MEMORYLAYER_TURSO_AUTH_TOKEN"  # remote/replica mode

MEMORYLAYER_TURSO_SYNC_INTERVAL = "MEMORYLAYER_TURSO_SYNC_INTERVAL"
DEFAULT_MEMORYLAYER_TURSO_SYNC_INTERVAL = "60"  # seconds, replica mode only

MEMORYLAYER_TURSO_VECTOR_INDEX = "MEMORYLAYER_TURSO_VECTOR_INDEX"
DEFAULT_MEMORYLAYER_TURSO_VECTOR_INDEX = "false"  # opt-in DiskANN indexing

# ============================================
# Memory Service
# ============================================
MEMORYLAYER_MEMORY_SERVICE = "MEMORYLAYER_MEMORY_SERVICE"
DEFAULT_MEMORYLAYER_MEMORY_SERVICE = "default"

# ============================================
# Reflection Service
# ============================================
MEMORYLAYER_REFLECT_SERVICE = "MEMORYLAYER_REFLECT_SERVICE"
DEFAULT_MEMORYLAYER_REFLECT_SERVICE = "default"

# ============================================
# Session Service
# ============================================
MEMORYLAYER_SESSION_SERVICE = "MEMORYLAYER_SESSION_SERVICE"
# Default 'persistent': sessions (and the token-budget extraction state that drives
# automatic memory extraction) survive a restart, using the storage backend that is
# already configured. 'in-memory' silently dropped all of it on every restart, which
# is a surprising default for a memory product. Set 'in-memory' explicitly for
# ephemeral/stateless deployments.
DEFAULT_MEMORYLAYER_SESSION_SERVICE = "persistent"

MEMORYLAYER_SESSION_IMPLICIT_CREATE = "MEMORYLAYER_SESSION_IMPLICIT_CREATE"
DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE = True

# Whether authentication may materialise a workspace it was merely asked to
# resolve. This backs the OSS "just works" pattern (MCP derives a workspace from
# the git repo name and expects the first remember() to land), so it defaults on
# — but note that even when enabled, creation is limited to unsafe HTTP methods
# and to well-formed ids (see services/authentication/base.py). Deployments that
# create workspaces explicitly can turn it off entirely.
MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE = "MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE"
DEFAULT_MEMORYLAYER_WORKSPACE_IMPLICIT_CREATE = True

MEMORYLAYER_SESSION_TOUCH_TTL = "MEMORYLAYER_SESSION_TOUCH_TTL"
DEFAULT_MEMORYLAYER_SESSION_TOUCH_TTL = 3600

# Token-budget-aware extraction thresholds
MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL = "MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL = 12000

MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT = "MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT = 10000

MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH = "MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH = 5000

# Generative enrichment is centrally enforced by LLMService. The compatibility
# default retains existing behavior; deployments can select ``adaptive`` or the
# hard zero-generation ``deterministic`` policy without removing an LLM provider.
MEMORYLAYER_ENRICHMENT_POLICY = "MEMORYLAYER_ENRICHMENT_POLICY"
DEFAULT_MEMORYLAYER_ENRICHMENT_POLICY = "generative"

# Comma-separated GenerationActivity values authorized under ``adaptive``.
# An empty allowlist makes adaptive ordinary paths deterministic while explicit
# deployments opt individual activities in.
MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES = "MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES"
DEFAULT_MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES = "reflection,synthesis"

MEMORYLAYER_GENERATION_MAX_CALLS = "MEMORYLAYER_GENERATION_MAX_CALLS"
DEFAULT_MEMORYLAYER_GENERATION_MAX_CALLS = 8

MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS = "MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS"
DEFAULT_MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS = 100_000

MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS = "MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS"
DEFAULT_MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS = 16_384

# Additive deterministic-memory features. Read paths can be disabled without
# deleting their durable checkpoint, event, or relation records.
MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED = "MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED"
DEFAULT_MEMORYLAYER_EXTRACTIVE_TIERS_ENABLED = True
MEMORYLAYER_SESSION_CHECKPOINT_CAPTURE_ENABLED = "MEMORYLAYER_SESSION_CHECKPOINT_CAPTURE_ENABLED"
DEFAULT_MEMORYLAYER_SESSION_CHECKPOINT_CAPTURE_ENABLED = True
MEMORYLAYER_SESSION_CHECKPOINT_MAX_BYTES = "MEMORYLAYER_SESSION_CHECKPOINT_MAX_BYTES"
DEFAULT_MEMORYLAYER_SESSION_CHECKPOINT_MAX_BYTES = 1_048_576
MEMORYLAYER_CONTEXT_PACK_ENABLED = "MEMORYLAYER_CONTEXT_PACK_ENABLED"
DEFAULT_MEMORYLAYER_CONTEXT_PACK_ENABLED = True
MEMORYLAYER_CONTEXT_CURSOR_SECRET = "MEMORYLAYER_CONTEXT_CURSOR_SECRET"
DEFAULT_MEMORYLAYER_CONTEXT_CURSOR_SECRET = "memorylayer-context-cursor-v1"
MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS = "MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS"
DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS = 30
MEMORYLAYER_RELATIONAL_RECALL_ENABLED = "MEMORYLAYER_RELATIONAL_RECALL_ENABLED"
# The deterministic-memory-relations-qwen1960 gate (2026-09-02) improved
# relation-query Recall@5 by 19.8 points with zero generic-query regression;
# unsupported storage providers take the capability-checked no-op path.
DEFAULT_MEMORYLAYER_RELATIONAL_RECALL_ENABLED = True
MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED = "MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED"
DEFAULT_MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED = True
MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT = "MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT"
DEFAULT_MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT = 0

# ============================================
# Workspace Service
# ============================================
MEMORYLAYER_WORKSPACE_SERVICE = "MEMORYLAYER_WORKSPACE_SERVICE"
DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE = "default"

# ============================================
# Association Service
# ============================================
MEMORYLAYER_ASSOCIATION_SERVICE = "MEMORYLAYER_ASSOCIATION_SERVICE"
DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE = "default"

MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = "MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD"
DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 0.85

# Similarity at/above which a candidate is treated as a near-duplicate and
# labeled ``duplicate_of`` deterministically, WITHOUT an LLM classification
# call. Purely embedding-driven short-circuit for the auto-association tier-0.
MEMORYLAYER_ASSOCIATION_DUPLICATE_THRESHOLD = "MEMORYLAYER_ASSOCIATION_DUPLICATE_THRESHOLD"
DEFAULT_MEMORYLAYER_ASSOCIATION_DUPLICATE_THRESHOLD = 0.97

# Whether the LLM relationship classifier runs at all. When False (or no LLM /
# ontology service is wired), every non-duplicate auto-association edge falls
# back to ``similar_to`` at zero LLM cost. When True, ambiguous candidates are
# classified in a SINGLE batched LLM call per new memory (not one call each).
MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED = "MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED"
DEFAULT_MEMORYLAYER_ASSOCIATION_LLM_CLASSIFY_ENABLED = True

# ============================================
# Authentication Service
# ============================================
MEMORYLAYER_AUTHENTICATION_SERVICE = "MEMORYLAYER_AUTHENTICATION_SERVICE"
DEFAULT_MEMORYLAYER_AUTHENTICATION_SERVICE = "default"  # Open authentication (allow all)

# When the aether authenticator runs and the gateway X-Auth-Tenant-ID header is
# absent, the default (fail-closed) behavior is to reject the request with a 401
# rather than silently falling back to DEFAULT_TENANT_ID. Setting this flag truthy
# re-enables the legacy default-tenant fallback for local development WITHOUT the
# gateway in front (logged loudly). Leave OFF in any environment fronted by the
# auth-proxy so unauthenticated requests fail closed.
MEMORYLAYER_AUTH_ALLOW_DEFAULT_TENANT = "MEMORYLAYER_AUTH_ALLOW_DEFAULT_TENANT"
DEFAULT_MEMORYLAYER_AUTH_ALLOW_DEFAULT_TENANT = False

# ============================================
# Authorization Service
# ============================================
MEMORYLAYER_AUTHORIZATION_SERVICE = "MEMORYLAYER_AUTHORIZATION_SERVICE"
DEFAULT_MEMORYLAYER_AUTHORIZATION_SERVICE = "default"  # Open permissions (allow all)


# ============================================
# Reranker Service
# ============================================
class RerankerProviderType(str, Enum):
    """Available reranker provider types.

    The legacy in-process providers ``local`` (sentence-transformers
    CrossEncoder) and ``qwen3-vl`` were removed; all self-hosted reranking
    now routes through ``embed_server`` (MaxSim via memorylayer-embed-server).
    See ``_LEGACY_RERANKER_PROVIDERS`` below.
    """

    LLM = "llm"  # Use LLM service for reranking
    HYDE = "hyde"  # Hypothetical Document Embeddings (LLM + embedding)
    RRF = "rrf"  # Reciprocal Rank Fusion (embedding-only multi-query); re-embeds candidates per query — slow + net-negative on LongMemEval
    EMBED_SERVER = "embed_server"  # Delegates MaxSim reranking to memorylayer-embed-server
    NONE = "none"  # Disabled (no reranking) (default)


_LEGACY_RERANKER_PROVIDERS: frozenset[str] = frozenset({"local", "qwen3-vl"})


def assert_supported_reranker_provider(value: str) -> None:
    """Raise ValueError with migration guidance if ``value`` is a removed reranker name."""
    if value in _LEGACY_RERANKER_PROVIDERS:
        raise ValueError(
            f"MEMORYLAYER_RERANKER_PROVIDER={value!r} was removed. "
            f"Self-hosted reranking now routes through memorylayer-embed-server. "
            f"Set MEMORYLAYER_RERANKER_PROVIDER=embed_server (or one of "
            f"'rrf', 'llm', 'hyde', 'none')."
        )


MEMORYLAYER_RERANKER_PROVIDER = "MEMORYLAYER_RERANKER_PROVIDER"
# Default 'none': hybrid fusion alone beat hybrid+rrf on LongMemEval, and rrf
# re-embeds every candidate per query (slow). Revisit if a cross-encoder lands.
DEFAULT_MEMORYLAYER_RERANKER_PROVIDER = "none"

MEMORYLAYER_RERANKER_SERVICE = "MEMORYLAYER_RERANKER_SERVICE"
DEFAULT_MEMORYLAYER_RERANKER_SERVICE = "default"

MEMORYLAYER_RERANKER_PRELOAD_ENABLED = "MEMORYLAYER_RERANKER_PRELOAD_ENABLED"
DEFAULT_MEMORYLAYER_RERANKER_PRELOAD_ENABLED = True

# ============================================
# Cache Service
# ============================================
MEMORYLAYER_CACHE_SERVICE = "MEMORYLAYER_CACHE_SERVICE"
DEFAULT_MEMORYLAYER_CACHE_SERVICE = "lru"

# Per-pod tenant identity. MemoryLayer runs single-tenant per pod; this names
# the tenant that this pod serves. Currently used only as caller-asserted LLM
# attribution (stamped as the ``X-Scitrera-Tenant`` header on outgoing calls to
# the MLflow AI Gateway). Empty by default so the header is simply omitted when
# unset — no behavioral change for deployments that do not set it.
MEMORYLAYER_TENANT_ID = "MEMORYLAYER_TENANT_ID"
DEFAULT_MEMORYLAYER_TENANT_ID = ""

# Hosts allowed to receive the caller-asserted identity headers (X-Scitrera-Source,
# X-Scitrera-Tenant, X-Scitrera-Task-Id). Comma-separated; exact hosts or a leading
# wildcard label ("gateway.internal", "*.example.internal"); a scheme, port or path
# may be included and is ignored.
#
# EMPTY BY DEFAULT, AND EMPTY MEANS STAMP NOTHING. These headers name the operator's
# internal tenant to whoever receives them, so they must only reach a first-party
# gateway. Defaulting to "stamp nowhere" is what keeps a profile pointed at a public
# provider (api.openai.com, Fireworks, any OpenAI-compatible vendor) from being sent
# an internal tenant slug. Enterprise sets a default allowlist for its own gateway.
MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS = "MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS"
DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS = ""

# Default tenant and workspace constants
# Use underscore prefix for all reserved/system entities
DEFAULT_TENANT_ID = "_default"
DEFAULT_WORKSPACE_ID = "_default"
GLOBAL_WORKSPACE_ID = "_global"
# User-scoped global workspace. Unlike GLOBAL_WORKSPACE_ID (tenant-wide shared
# memories), GLOBAL_USER_WORKSPACE_ID partitions memories by user_id inside a
# single workspace — it is the natural home for per-user preferences and
# profile facts that should follow a user across their workspaces without
# leaking across users. Callers opt in to cross-workspace recall by setting
# RecallInput.include_global_user=True (default).
GLOBAL_USER_WORKSPACE_ID = "_global_user"

# When enabled, the user-global subtypes PREFERENCE and DIRECTIVE are
# auto-routed to user scope (the _global_user workspace, partitioned by
# user_id) on the remember() path even when the caller does not set an
# explicit scope. An explicit RememberInput.scope=USER always wins; this
# knob only governs the implicit subtype-based mapping. Disable-able so
# Slice 2 can tune/turn off automatic promotion without code changes.
MEMORYLAYER_USER_SCOPE_SUBTYPES = "MEMORYLAYER_USER_SCOPE_SUBTYPES"
DEFAULT_MEMORYLAYER_USER_SCOPE_SUBTYPES = True

# --- Slice 2: automatic preference-vs-episodic classification -----------------
# These knobs gate an OPTIONAL classifier that runs at remember()-time and
# decides whether a memory with NO explicit scope (scope unset) is a durable
# user preference/personality trait (-> route to USER scope, follows the user
# across workspaces) vs an episodic/workspace fact (-> stays workspace-scoped).
#
# KNOB 1 (master enable). MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED
#   Effect: when True, AND the caller gave no explicit scope, AND the subtype
#   map did not already decide, the classifier runs and may route to USER scope.
#   Default FALSE (opt-in). A misclassified episodic memory wrongly follows the
#   user across every workspace, so automatic classification stays OFF until an
#   operator validates it. When False the classifier NEVER runs: zero added
#   latency, behavior is byte-identical to Slice 1. Hard-disable == set False.
MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED = "MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED"
DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED = False

# KNOB 2 (confidence gate). MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD
#   Effect: only route to USER scope when the classifier returns
#   is_user_preference=True AND confidence >= this floor. Default 0.85 is
#   deliberately conservative: a false positive pollutes the user's
#   cross-workspace profile (expensive), a false negative just leaves the
#   memory workspace-scoped (cheap). Operators may raise it; lowering it below
#   the default in code is discouraged (keep the safety bar high).
MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD = "MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD"
DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD = 0.85

# ============================================
# Context ID Default
# ============================================
DEFAULT_CONTEXT_ID = "_default"

# ============================================
# Semantic Tiering Service
# ============================================
MEMORYLAYER_SEMANTIC_TIERING_SERVICE = "MEMORYLAYER_SEMANTIC_TIERING_SERVICE"
DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE = "default"

MEMORYLAYER_SEMANTIC_TIERING_ENABLED = "MEMORYLAYER_SEMANTIC_TIERING_ENABLED"
DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_ENABLED = True

# Skip the tier LLM call when the source text is already at or below the tier's own
# target length — summarizing text that is shorter than its summary spends an LLM call
# to produce something no more compact than the input.
#
# Defaults are the sizes the tier prompts actually ask for ("a single short sentence" /
# "a 2-3 sentence overview"), and match the truncation lengths the no-LLM fallbacks
# already use. Measured against a conversational corpus (median memory 145 chars), these
# skip 76% of abstract calls and 100% of overview calls; long-document ingestion is
# unaffected (0.9% skippable). Set to 0 to disable skipping and always call the LLM.
MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS = "MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS"
DEFAULT_MEMORYLAYER_TIER_ABSTRACT_SKIP_CHARS = 200

MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS = "MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS"
DEFAULT_MEMORYLAYER_TIER_OVERVIEW_SKIP_CHARS = 500

# ============================================
# Deduplication Service
# ============================================
MEMORYLAYER_DEDUPLICATION_SERVICE = "MEMORYLAYER_DEDUPLICATION_SERVICE"
DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE = "default"

# ============================================
# Ontology Service
# ============================================
MEMORYLAYER_ONTOLOGY_SERVICE = "MEMORYLAYER_ONTOLOGY_SERVICE"
DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE = "default"

# Entity linker provider (canonical entity -> external KB). "default" = no-op
# (linking disabled); the enterprise "wikidata" provider is opt-in per tenant.
MEMORYLAYER_ENTITY_LINKER_PROVIDER = "MEMORYLAYER_ENTITY_LINKER_PROVIDER"
DEFAULT_MEMORYLAYER_ENTITY_LINKER_PROVIDER = "default"

# Deployment-level entity-type vocabulary extension (the ontology owns entity
# types; the extractor derives its NER labels from it). CSV of "type" or
# "type:ner_label" — a bare "type" defaults the NER label to the type name;
# "type:" (empty label) adds a non-NER type. Per-tenant because MemoryLayer is
# deployed one instance per tenant. Example (engineering/P&ID domain):
#   MEMORYLAYER_ENTITY_TYPES="equipment,chemical,standard,method,dataset,model"
MEMORYLAYER_ENTITY_TYPES = "MEMORYLAYER_ENTITY_TYPES"
DEFAULT_MEMORYLAYER_ENTITY_TYPES = ""

# ============================================
# Extraction Service
# ============================================
MEMORYLAYER_EXTRACTION_SERVICE = "MEMORYLAYER_EXTRACTION_SERVICE"
DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE = "default"

# ============================================
# Inference Service (entity insight derivation)
# ============================================
MEMORYLAYER_INFERENCE_SERVICE = "MEMORYLAYER_INFERENCE_SERVICE"
DEFAULT_MEMORYLAYER_INFERENCE_SERVICE = "default"

# ============================================
# Task Service
# ============================================
MEMORYLAYER_TASK_PROVIDER = "MEMORYLAYER_TASK_PROVIDER"
DEFAULT_MEMORYLAYER_TASK_PROVIDER = "asyncio"

# ============================================
# Recall Scoring: Recency Boost
# ============================================
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RECENCY_HALF_LIFE_HOURS = 168

# ============================================
# Recall Scoring: Freshness Annotation
# ============================================
MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS = "MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS"
DEFAULT_MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS = 7.0

# ============================================
# Recall Scoring: Tolerance Floors
# ============================================
# Per-tolerance relevance floor applied by recall (see
# MemoryService._get_relevance_threshold). Operators can tune search
# sensitivity per deployment without changing caller behavior; defaults match
# the previously hardcoded values.
MEMORYLAYER_TOLERANCE_FLOOR_STRICT = "MEMORYLAYER_TOLERANCE_FLOOR_STRICT"
DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_STRICT = 0.6

MEMORYLAYER_TOLERANCE_FLOOR_MODERATE = "MEMORYLAYER_TOLERANCE_FLOOR_MODERATE"
DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_MODERATE = 0.3

MEMORYLAYER_TOLERANCE_FLOOR_LOOSE = "MEMORYLAYER_TOLERANCE_FLOOR_LOOSE"
DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_LOOSE = 0.15

# ============================================
# Recall Scoring: Scope Boosts
# ============================================
MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT = "MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT"
DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT = 1.5

MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE = "MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE"
DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE = 1.2

# ============================================
# Decay Service
# ============================================
MEMORYLAYER_DECAY_PROVIDER = "MEMORYLAYER_DECAY_PROVIDER"
DEFAULT_MEMORYLAYER_DECAY_PROVIDER = "default"

# ============================================
# Contradiction Service
# ============================================
# ``default`` = deterministic regex negation pairs (zero inference).
# ``llm``     = fused single call returning duplicates AND contradictions per stored
#               memory. Measured on LongMemEval knowledge-update: 0.0% -> ~49% of
#               supersessions detected. See docs/DESIGN_graphiti_adoption.md.
MEMORYLAYER_CONTRADICTION_PROVIDER = "MEMORYLAYER_CONTRADICTION_PROVIDER"
DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER = "default"

# Candidate neighbourhood searched for conflicting memories.
#
# The floor is the BINDING CONSTRAINT on detection, not the detector: a pair outside
# this window is never shown to any detector. Simulated over LongMemEval
# knowledge-update, the superseded fact is reachable for 51.3% of items at 0.7 but
# 93.6% at 0.5 — the median old<->new similarity is 0.707, sitting directly on the
# 0.7 threshold.
#
# The floor and detector quality are COUPLED, so they are not lowered independently:
# a lower floor shows the detector more pairs to be wrong about, and the regex
# detector's precision cannot absorb that. Hence two defaults — the deterministic
# provider keeps 0.7, and the ``llm`` provider (which measurably can absorb it) uses
# 0.5. Setting the env var explicitly overrides whichever provider is active.
MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE = "MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE"
DEFAULT_MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE = 0.7
DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MIN_RELEVANCE = 0.5

MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT = "MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT"
DEFAULT_MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT = 20

# Token budget for the fused call. Deliberately generous: the models used here are
# REASONING models that emit a long chain before the JSON, and a truncated reply parses
# as "no contradictions found" — a failure indistinguishable from a clean negative. The
# service treats finish_reason == "length" as an error rather than an empty result.
MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS = "MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS"
DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS = 5000

# What recall does with a memory a later memory has superseded.
#
# Until now the answer was "nothing": contradictions were detected on every store,
# written with a supersession direction, and never consulted at read time — so a fact we
# had positively identified as stale was still retrieved at full score.
#
#   off      current behaviour; supersession is recorded but not applied
#   demote   multiply the stale memory's score by _PENALTY (default)
#   exclude  drop stale memories from results entirely
#
# MEASURED end-to-end on LongMemEval knowledge-update (78 items, LLM-judged QA):
#
#   off      74.4%
#   demote   69.2%   (vs off: fixed 4, broke 8; McNemar p=0.39)
#   exclude  76.9%   (vs off: fixed 6, broke 4; McNemar p=0.75)
#
# NO arm is a statistically significant improvement, so this stays `off` by default.
#
# `demote` measured WORST, which inverts the intuition that it is the cautious choice.
# Demoting leaves the stale memory in the retrieved context, just lower — the answering
# model still sees both the old and the new value, so you pay the ranking distortion
# without removing the thing that confuses it. `exclude` at least takes the stale fact out
# of the window. If enabling this at all, prefer `exclude`.
MEMORYLAYER_RECALL_SUPERSESSION_MODE = "MEMORYLAYER_RECALL_SUPERSESSION_MODE"
DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_MODE = "off"

SUPERSESSION_MODE_OFF = "off"
SUPERSESSION_MODE_DEMOTE = "demote"
SUPERSESSION_MODE_EXCLUDE = "exclude"

MEMORYLAYER_RECALL_SUPERSESSION_PENALTY = "MEMORYLAYER_RECALL_SUPERSESSION_PENALTY"
DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_PENALTY = 0.5

# ============================================
# Fact Decomposition
# ============================================
MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = "MEMORYLAYER_FACT_DECOMPOSITION_ENABLED"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = True

# Re-run the post-store pipeline when an incoming fact MERGES into an existing
# memory, rather than creating a new one. Default OFF.
#
# The argument for running it: the survivor's content changed, so its tiers and
# associations are arguably stale. The argument against, which is why it is off:
# the memory was already enriched, so the work is overwhelmingly re-derivation
# of what exists -- a full batched relationship classification whose edges then
# all collide with uq_association. Merges are not rare either; decomposing many
# similar pages produces many similar facts, so this fires continuously.
#
# Turn on if association quality visibly suffers from merged content going
# un-reassociated; the association pass now skips already-linked candidates, so
# the cost of doing so is far lower than it was.
MEMORYLAYER_ENRICH_ON_MERGE = "MEMORYLAYER_ENRICH_ON_MERGE"
DEFAULT_MEMORYLAYER_ENRICH_ON_MERGE = False

MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = "MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = 80

# Max completion tokens for the fact-decomposition LLM call. The model is a
# reasoning model, so a small budget truncates the JSON fact array on large
# inputs — default is generous; raise further via env for exceptionally big jobs.
MEMORYLAYER_FACT_DECOMPOSITION_MAX_TOKENS = "MEMORYLAYER_FACT_DECOMPOSITION_MAX_TOKENS"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MAX_TOKENS = 16384

# Reasoning effort for the fact-decomposition call, default OFF.
#
# Decomposition is a transformation — split text into standalone statements —
# not multi-step inference, so reasoning buys little. It costs a great deal:
# thinking tokens are drawn from the SAME completion budget as the JSON array,
# which is what forced the generous cap above and why the parser carries a
# truncated-array recovery path. A page yielding ~40 facts pays that overhead
# once per page, on every page of every document.
#
# The one inferential step is resolving relative dates against a supplied
# reference time, which is stated explicitly in the prompt and needs no extended
# deliberation. Set a real effort level ("low"/"medium"/"high") if extraction
# quality measurably needs it; "none" disables thinking on models that support
# the control and is ignored by models that do not.
MEMORYLAYER_FACT_DECOMPOSITION_REASONING_EFFORT = "MEMORYLAYER_FACT_DECOMPOSITION_REASONING_EFFORT"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_REASONING_EFFORT = "none"

# Max completion tokens for the 6-category memory-extraction LLM call
# (extract_from_session). Generative/variable — a JSON array of N memories — so
# a small budget truncates the array on rich sessions. Sibling of the
# fact-decomposition cap; env-tunable with a generous default.
MEMORYLAYER_MEMORY_EXTRACTION_MAX_TOKENS = "MEMORYLAYER_MEMORY_EXTRACTION_MAX_TOKENS"
DEFAULT_MEMORYLAYER_MEMORY_EXTRACTION_MAX_TOKENS = 8192

# Per-call completion caps for the smaller internal LLM steps. Each was a
# hardcoded literal sized for the VISIBLE output, but a reasoning model spends
# hidden thinking tokens against the same budget first — so the tiny originals
# (20/50/100/200/250) could truncate to empty. Now env-tunable with generous
# defaults (raising a cap is free for non-reasoning models — max_tokens is a
# ceiling). Originals noted per key.
MEMORYLAYER_MEMORY_CLASSIFY_MAX_TOKENS = "MEMORYLAYER_MEMORY_CLASSIFY_MAX_TOKENS"
DEFAULT_MEMORYLAYER_MEMORY_CLASSIFY_MAX_TOKENS = 1024  # was 20 (classify_content: one label)

MEMORYLAYER_CUE_ANCHOR_MAX_TOKENS = "MEMORYLAYER_CUE_ANCHOR_MAX_TOKENS"
DEFAULT_MEMORYLAYER_CUE_ANCHOR_MAX_TOKENS = 1024  # was 200 (generate_cue_anchors: <=3 anchors)

MEMORYLAYER_RERANK_MAX_TOKENS = "MEMORYLAYER_RERANK_MAX_TOKENS"
DEFAULT_MEMORYLAYER_RERANK_MAX_TOKENS = 1024  # was 50 (LLM rerank: comma-separated indices)

MEMORYLAYER_ONTOLOGY_MAX_TOKENS = "MEMORYLAYER_ONTOLOGY_MAX_TOKENS"
DEFAULT_MEMORYLAYER_ONTOLOGY_MAX_TOKENS = 1024  # was 250 (relationship classification JSON)

MEMORYLAYER_RLM_EVAL_MAX_TOKENS = "MEMORYLAYER_RLM_EVAL_MAX_TOKENS"
DEFAULT_MEMORYLAYER_RLM_EVAL_MAX_TOKENS = 1024  # was 100 (goal-achieved eval)

MEMORYLAYER_INFERENCE_MAX_TOKENS = "MEMORYLAYER_INFERENCE_MAX_TOKENS"
DEFAULT_MEMORYLAYER_INFERENCE_MAX_TOKENS = 4096  # was 2048 (entity-insight synthesis; generative)

# ============================================
# Context Environment Service
# ============================================
MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE = "MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE"
DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE = "default"

MEMORYLAYER_CONTEXT_EXECUTOR = "MEMORYLAYER_CONTEXT_EXECUTOR"
DEFAULT_MEMORYLAYER_CONTEXT_EXECUTOR = "smolagents"

MEMORYLAYER_CONTEXT_MAX_OPERATIONS = "MEMORYLAYER_CONTEXT_MAX_OPERATIONS"
DEFAULT_MEMORYLAYER_CONTEXT_MAX_OPERATIONS = 1_000_000

MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS = "MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS"
DEFAULT_MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS = 30

MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS = "MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS"
DEFAULT_MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS = 50_000

MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS = "MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS"
DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS = 8192

MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES = "MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES"
DEFAULT_MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MB

MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS = "MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS"
DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS = 10

MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS = "MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS"
DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS = 120

MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP = "MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP"
DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP = 0

MEMORYLAYER_CONTEXT_EXEC_HARD_CAP = "MEMORYLAYER_CONTEXT_EXEC_HARD_CAP"
DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP = 0

# ============================================
# Chat History Service
# ============================================
MEMORYLAYER_CHAT_SERVICE = "MEMORYLAYER_CHAT_SERVICE"
DEFAULT_MEMORYLAYER_CHAT_SERVICE = "default"

MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD = "MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD"
DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD = 10

MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL = "MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL"
DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL = 300  # seconds

MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE = "MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE"
DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE = 20

MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP = "MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP"
DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP = 5

# Idle thread policy: how long (days) a thread may sit idle (no new messages,
# measured on updated_at) before its per-thread ``idle_action`` ('hide'/'delete')
# is applied by the background cleanup task.
MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS = "MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS"
DEFAULT_MEMORYLAYER_CHAT_THREAD_IDLE_RETENTION_DAYS = 7

# Grace window (days) after a thread is hidden/archived before it is purged for
# space. Measured against ``hidden_at`` (revival clears hidden_at, so the clock
# never goes stale). 0 or negative DISABLES auto-purge of hidden threads.
MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS = "MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS"
DEFAULT_MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS = 0

# ============================================
# Audit Service
# ============================================
MEMORYLAYER_AUDIT_SERVICE = "MEMORYLAYER_AUDIT_SERVICE"
DEFAULT_MEMORYLAYER_AUDIT_SERVICE = "noop"

# ============================================
# Rate Limiting Service
# ============================================
MEMORYLAYER_RATE_LIMIT_SERVICE = "MEMORYLAYER_RATE_LIMIT_SERVICE"
DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE = "noop"

# Rate limit defaults (requests per window)
MEMORYLAYER_RATE_LIMIT_REQUESTS = "MEMORYLAYER_RATE_LIMIT_REQUESTS"
DEFAULT_MEMORYLAYER_RATE_LIMIT_REQUESTS = 100

MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS = "MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS"
DEFAULT_MEMORYLAYER_RATE_LIMIT_WINDOW_SECONDS = 60

# ============================================
# Metrics / Observability Service
# ============================================
MEMORYLAYER_METRICS_SERVICE = "MEMORYLAYER_METRICS_SERVICE"
DEFAULT_MEMORYLAYER_METRICS_SERVICE = "noop"

# ============================================
# LLM Query Rewriting
# ============================================
MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED = "MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED"
DEFAULT_MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED = False  # Query rewrite sounds good in theory, but doesn't do well in ambiguous contexts

# ============================================
# Memory Consolidation
# ============================================
MEMORYLAYER_CONSOLIDATION_ENABLED = "MEMORYLAYER_CONSOLIDATION_ENABLED"
DEFAULT_MEMORYLAYER_CONSOLIDATION_ENABLED = False

MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE = "MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE"
DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE = 3

MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE = "MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE"
DEFAULT_MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE = 0.3

MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY = "MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY"
DEFAULT_MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY = 0.85

# ============================================
# Document Ingestion Service
# ============================================
MEMORYLAYER_DOCUMENT_PROVIDER = "MEMORYLAYER_DOCUMENT_PROVIDER"
DEFAULT_MEMORYLAYER_DOCUMENT_PROVIDER = "default"

MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE = "MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE"
DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ============================================
# Embed Server Client (relocated to OSS in Phase 3 of the Aether convergence;
# the embed server itself lives at oss/memorylayer-embed-server/).
# ============================================
MEMORYLAYER_EMBED_SERVER_URL = "MEMORYLAYER_EMBED_SERVER_URL"
DEFAULT_MEMORYLAYER_EMBED_SERVER_URL = "http://localhost:61051"
MEMORYLAYER_EMBED_SERVER_TIMEOUT = "MEMORYLAYER_EMBED_SERVER_TIMEOUT"
DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT = 300

# Transport switch. ``http`` (default) calls the embed server directly via
# ``MEMORYLAYER_EMBED_SERVER_URL``; ``aether`` issues proxy_http_async calls
# through the AetherServiceConnection against the configured target topic
# (default sv::memorylayer-embed::default). ``aether`` is what enables
# cross-DC GPU placement under mTLS.
MEMORYLAYER_EMBED_TRANSPORT = "MEMORYLAYER_EMBED_TRANSPORT"
DEFAULT_MEMORYLAYER_EMBED_TRANSPORT = "http"
MEMORYLAYER_EMBED_AETHER_TARGET = "MEMORYLAYER_EMBED_AETHER_TARGET"
DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET = "sv::memorylayer-embed::default"

# Idle-timeout for streaming RPCs over Aether (milliseconds). Maps to
# ``proxy_http_async(stream_idle_timeout_ms=...)``; ``0`` lets the
# Aether client choose its own default (typically 30 seconds).
MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS = "MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS"
DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS = 30000

# Plugin-selection key for the embed-server-client extension.
MEMORYLAYER_EMBED_SERVER_SERVICE = "MEMORYLAYER_EMBED_SERVER_SERVICE"
DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE = "default"

# ============================================
# Data Provider Service
# ============================================
MEMORYLAYER_DATA_PROVIDER_PROVIDER = "MEMORYLAYER_DATA_PROVIDER_PROVIDER"
DEFAULT_MEMORYLAYER_DATA_PROVIDER_PROVIDER = "local"

# ============================================
# Skills Service
# ============================================
MEMORYLAYER_SKILLS_PROVIDER = "MEMORYLAYER_SKILLS_PROVIDER"
DEFAULT_MEMORYLAYER_SKILLS_PROVIDER = "default"

# ============================================
# Graph Analysis Service
# ============================================
MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER = "MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER"
DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER = "default"

# ============================================
# Graph Query Service (recall/RAG-facing read seam; P2 Track A)
# ============================================
MEMORYLAYER_GRAPH_QUERY_PROVIDER = "MEMORYLAYER_GRAPH_QUERY_PROVIDER"
DEFAULT_MEMORYLAYER_GRAPH_QUERY_PROVIDER = "default"

# Entity materialization: project the entity registry into the enterprise AGE graph
# as Entity vertices + MENTIONS edges (Entity->Memory). Ships DARK: default OFF.
# Even when ON it is a clean no-op unless the entity registry is also enabled
# (MEMORYLAYER_ENTITY_REGISTRY_ENABLED) and non-empty. The OSS relational
# entity_neighborhood fallback does NOT depend on this flag — it reads the
# registry tables directly. Enterprise-AGE only; the OSS/default graph-query and
# the NetworkX analysis providers are untouched by this flag.
MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED = "MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED"
DEFAULT_MEMORYLAYER_GRAPH_ENTITY_MATERIALIZE_ENABLED = False

# Fragment materialization: project decomposed-fact memories into the enterprise AGE
# graph as Fragment vertices + DERIVED_FROM edges (Fragment->source Memory).
# Ships DARK: default OFF. Even when ON it is a clean no-op unless the fact
# channel produced subtype="fact" memories (metadata {"kind":"fact",
# "source_id":<parent>}). The OSS relational fragment-traversal fallback does NOT
# depend on this flag — it reads the fact memories + their source_id directly.
# Enterprise-AGE only; the OSS/default graph-query and the NetworkX analysis
# providers are untouched. NOT wired into recall — this is a dark, MEASURABLE
# capability whose eval (does fragment-traversal beat the fact channel E) is a
# later measurement, not a default-on win.
MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED = "MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED"
DEFAULT_MEMORYLAYER_GRAPH_FRAGMENT_MATERIALIZE_ENABLED = False

# ============================================
# Knowledgebase Service
# ============================================
MEMORYLAYER_KNOWLEDGEBASE_PROVIDER = "MEMORYLAYER_KNOWLEDGEBASE_PROVIDER"
DEFAULT_MEMORYLAYER_KNOWLEDGEBASE_PROVIDER = "default"

# Incremental KB rendering (Lever 3): community-stability matching + per-article content-hash skip.
# Jaccard member-set overlap threshold for treating a current community as "the same" as a prior one.
MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD = "MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD"
DEFAULT_MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD = 0.5
# Per-member content version source for the article content-hash: "content" (sha256 of member
# content, default — correct at any sub-second resolution) or "updated_at" (cheaper but only
# ticks once per second, risking stale cache on same-second edits; opt-in for large workspaces
# where the content-hash cost matters).
MEMORYLAYER_KB_CONTENT_VERSION_MODE = "MEMORYLAYER_KB_CONTENT_VERSION_MODE"
DEFAULT_MEMORYLAYER_KB_CONTENT_VERSION_MODE = "content"
# Minimum community size (member count) to generate a community article for.
# Louvain on a sparse association graph produces many size-1 "communities" that
# are just individual memories (e.g. one document page) — noisy and heavily
# overlapping. Skip communities smaller than this so only meaningful clusters
# become articles. 2 drops true singletons; raise for more substance.
MEMORYLAYER_KB_MIN_COMMUNITY_SIZE = "MEMORYLAYER_KB_MIN_COMMUNITY_SIZE"
DEFAULT_MEMORYLAYER_KB_MIN_COMMUNITY_SIZE = 2
# Max completion tokens for a community's title+summary LLM call. Must be large
# enough for a REASONING model to finish: the gateway strips/​separates the
# model's chain-of-thought and returns clean content, but a too-small budget
# truncates mid-reasoning so the raw "Thinking Process:" text leaks into the
# article. 4096 comfortably covers ~1.7k reasoning + a short summary.
MEMORYLAYER_KB_SUMMARY_MAX_TOKENS = "MEMORYLAYER_KB_SUMMARY_MAX_TOKENS"
DEFAULT_MEMORYLAYER_KB_SUMMARY_MAX_TOKENS = 8192
# How many community members to fetch as context for a community article. The
# old fixed cap of 20 under-represented large communities; raise it and let the
# char budget below bound the actual prompt size.
MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS = "MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS"
DEFAULT_MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS = 40
# Char budget for the member block fed to the community summary prompt. Members
# are added (each capped) in order until this budget is reached — deterministic
# truncation so a large community doesn't blow the context window.
MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS = "MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS"
DEFAULT_MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS = 10000

# --- KB graph densification (experimental, in-process) -----------------------
# The base association graph (explicit PART_OF/contradiction/auto-assoc edges) is
# sparse for document-derived memories, so Louvain yields near-singleton
# communities. When enabled, the graph is augmented (in-process) with weighted
# edges from up to three signals so related memories actually cluster:
#   * cosine  — single-vector cosine kNN across ALL memories (universal baseline)
#   * maxsim  — ColPali MaxSim kNN across page-backed memories (reliable for docs)
#   * entity  — co-occurrence: memories sharing a registry entity (cross-type glue)
# Each signal's weights are min-max normalized to ~[0,1] then scaled by its
# weight so one signal doesn't swamp the others. Default OFF (opt-in per
# experiment); loads vectors in-process — NOT tuned for very large workspaces.
MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED = "MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED"
DEFAULT_MEMORYLAYER_KB_GRAPH_DENSIFY_ENABLED = False

# Cosine-kNN signal (single-vector).
MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED = "MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED"
DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_ENABLED = True
MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD = "MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD"
DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_THRESHOLD = 0.82
MEMORYLAYER_KB_DENSIFY_COSINE_K = "MEMORYLAYER_KB_DENSIFY_COSINE_K"
DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_K = 8
MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT = "MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT"
DEFAULT_MEMORYLAYER_KB_DENSIFY_COSINE_WEIGHT = 1.0

# MaxSim-kNN signal (page multivector). MaxSim scores live on a different scale
# than cosine; threshold 0.0 keeps the top-k regardless (calibrate after
# eyeballing real scores on a workspace).
MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED = "MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED"
DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_ENABLED = True
MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD = "MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD"
DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_THRESHOLD = 0.0
MEMORYLAYER_KB_DENSIFY_MAXSIM_K = "MEMORYLAYER_KB_DENSIFY_MAXSIM_K"
DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_K = 8
MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT = "MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT"
DEFAULT_MEMORYLAYER_KB_DENSIFY_MAXSIM_WEIGHT = 1.5

# Entity co-occurrence signal. Entities with more than MAX_GROUP members are
# skipped (they would over-connect the graph into one blob).
MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED = "MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED"
DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_ENABLED = True
MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT = "MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT"
DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_WEIGHT = 1.0
MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP = "MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP"
DEFAULT_MEMORYLAYER_KB_DENSIFY_ENTITY_MAX_GROUP = 50

# ============================================
# Entity Registry Service (canonical entity nodes + aliases + member accretion)
# ============================================
MEMORYLAYER_ENTITY_REGISTRY_PROVIDER = "MEMORYLAYER_ENTITY_REGISTRY_PROVIDER"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_PROVIDER = "default"
# Ships DARK: default OFF. When enabled, ingest-time enrichment accretes
# extracted entities into the registry (exact+alias resolution + member rows).
# Validated/flipped on later — flag-gated so it can land without affecting
# recall/ingest behavior.
MEMORYLAYER_ENTITY_REGISTRY_ENABLED = "MEMORYLAYER_ENTITY_REGISTRY_ENABLED"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED = False

# Registry-backed entity-expansion recall channel. Distinct from the accretion
# flag above: this is the registry's RETRIEVAL consumer. When enabled (and a
# registry service is wired), recall resolves the query's entities to canonical
# entities (resolve allow_create=False — never create from a query string),
# pulls each canonical entity's MEMBER memories (which include memories linked
# via aliases and fuzzy/LLM-merged surface forms — the value over the
# metadata-string entity-anchor channel), ranks them by vector similarity, and
# RRF-fuses them as an additive arm. Ships DARK: default OFF, flag-off recall is
# byte-identical to pre-channel behaviour. OSS-first (works with the exact+alias
# registry); enterprise inherits richer canonical entities via the same path.
MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED = "MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED = False

# Reciprocal Rank Fusion constant for the registry-expansion channel.
MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K = "MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K = 60

# Depth of the registry-expansion candidate pool (per query). Bounds the total
# member memories pulled across all resolved query entities. Mirrors the
# entity-anchor pool size (the member set is small, so a generous pool is cheap
# and lets the channel promote a member that sits deep in the vector ranking).
MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL = "MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL = 200

# Query-relevance floor for registry-expansion members. A resolved canonical
# entity can own hundreds of member memories (e.g. every turn that mentions a
# frequent speaker); injecting the long tail of low-relevance members floods the
# RRF pool and DESTROYS recall (measured: LoCoMo recall@5 0.517 -> 0.083 with a
# 200-member unfiltered pool). Only members whose cosine similarity to the query
# is >= this floor are fused. Default 0.0 preserves the legacy (unfiltered)
# behavior; set > 0 (e.g. 0.5) to make the channel additive instead of harmful.
MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE = "MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE"
DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE = 0.0

# ============================================
# Representation Service (P3 perspective slice 1: "what observer O understands
# about subject S")
# ============================================
# A SCOPED ASSEMBLY surface, NOT a recall channel: it intersects the observer's
# self-authored memories with the subject's mentions to assemble a tight,
# leakage-safe perspective. It NEVER calls recall() and NEVER injects members
# into recall (which would flood/crowd out gold results).
MEMORYLAYER_REPRESENTATION_PROVIDER = "MEMORYLAYER_REPRESENTATION_PROVIDER"
DEFAULT_MEMORYLAYER_REPRESENTATION_PROVIDER = "default"
# Ships DARK: default OFF. The flag gates FUTURE call-site wiring/consumers of
# the representation surface (mirrors the entity-registry pattern). The service
# method itself always works when invoked directly.
MEMORYLAYER_REPRESENTATION_ENABLED = "MEMORYLAYER_REPRESENTATION_ENABLED"
DEFAULT_MEMORYLAYER_REPRESENTATION_ENABLED = False
