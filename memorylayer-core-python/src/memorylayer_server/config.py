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
DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER = EmbeddingProviderType.EMBED_SERVER
MEMORYLAYER_EMBEDDING_MODEL = "MEMORYLAYER_EMBEDDING_MODEL"
MEMORYLAYER_EMBEDDING_DIMENSIONS = "MEMORYLAYER_EMBEDDING_DIMENSIONS"
# Default vector dimension for embed_server's text embeddings. Operators can
# override via MEMORYLAYER_EMBEDDING_DIMENSIONS to match their server-side model.
DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER = 1024
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
DEFAULT_MEMORYLAYER_SESSION_SERVICE = "in-memory"

MEMORYLAYER_SESSION_IMPLICIT_CREATE = "MEMORYLAYER_SESSION_IMPLICIT_CREATE"
DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE = True

MEMORYLAYER_SESSION_TOUCH_TTL = "MEMORYLAYER_SESSION_TOUCH_TTL"
DEFAULT_MEMORYLAYER_SESSION_TOUCH_TTL = 3600

# Token-budget-aware extraction thresholds
MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL = "MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL = 12000

MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT = "MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT = 10000

MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH = "MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH"
DEFAULT_MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH = 5000

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

# ============================================
# Authentication Service
# ============================================
MEMORYLAYER_AUTHENTICATION_SERVICE = "MEMORYLAYER_AUTHENTICATION_SERVICE"
DEFAULT_MEMORYLAYER_AUTHENTICATION_SERVICE = "default"  # Open authentication (allow all)

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
    RRF = "rrf"  # Reciprocal Rank Fusion (embedding-only multi-query) (default)
    EMBED_SERVER = "embed_server"  # Delegates MaxSim reranking to memorylayer-embed-server
    NONE = "none"  # Disabled (no reranking)


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
DEFAULT_MEMORYLAYER_RERANKER_PROVIDER = "rrf"

MEMORYLAYER_RERANKER_SERVICE = "MEMORYLAYER_RERANKER_SERVICE"
DEFAULT_MEMORYLAYER_RERANKER_SERVICE = "default"

MEMORYLAYER_RERANKER_PRELOAD_ENABLED = "MEMORYLAYER_RERANKER_PRELOAD_ENABLED"
DEFAULT_MEMORYLAYER_RERANKER_PRELOAD_ENABLED = True

# ============================================
# Cache Service
# ============================================
MEMORYLAYER_CACHE_SERVICE = "MEMORYLAYER_CACHE_SERVICE"
DEFAULT_MEMORYLAYER_CACHE_SERVICE = "lru"

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
MEMORYLAYER_CONTRADICTION_PROVIDER = "MEMORYLAYER_CONTRADICTION_PROVIDER"
DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER = "default"

# ============================================
# Fact Decomposition
# ============================================
MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = "MEMORYLAYER_FACT_DECOMPOSITION_ENABLED"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = True

MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = "MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH"
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = 80

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
DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS = 4096

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
# Knowledgebase Service
# ============================================
MEMORYLAYER_KNOWLEDGEBASE_PROVIDER = "MEMORYLAYER_KNOWLEDGEBASE_PROVIDER"
DEFAULT_MEMORYLAYER_KNOWLEDGEBASE_PROVIDER = "default"
