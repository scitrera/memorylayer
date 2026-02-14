"""Configuration management for MemoryLayer.ai using Pydantic Settings."""

from enum import Enum
from pathlib import Path

# ============================================
# Data Home Directory
# ============================================
MEMORYLAYER_DATA_DIR = 'MEMORYLAYER_DATA_DIR'

# ============================================
# Server Configuration
# ============================================
MEMORYLAYER_SERVER_HOST = 'MEMORYLAYER_SERVER_HOST'
DEFAULT_MEMORYLAYER_SERVER_HOST = '127.0.0.1'
MEMORYLAYER_SERVER_PORT = 'MEMORYLAYER_SERVER_PORT'
DEFAULT_MEMORYLAYER_SERVER_PORT = 61001


# ============================================
# Embedding Providers
# ============================================
class EmbeddingProviderType(str, Enum):
    """Available embedding provider types."""

    OPENAI = "openai"  # OpenAI API (cloud, text-only; also works with any OpenAI-compatible endpoint)
    GOOGLE = "google"  # Google GenAI API (cloud, text-only)
    LOCAL = "local"  # sentence-transformers (self-hosted, text-only)
    MOCK = "mock"  # Mock provider for testing only (deterministic hash-based)


MEMORYLAYER_EMBEDDING_PROVIDER = 'MEMORYLAYER_EMBEDDING_PROVIDER'
DEFAULT_MEMORYLAYER_EMBEDDING_PROVIDER = EmbeddingProviderType.LOCAL
MEMORYLAYER_EMBEDDING_MODEL = 'MEMORYLAYER_EMBEDDING_MODEL'
MEMORYLAYER_EMBEDDING_DIMENSIONS = 'MEMORYLAYER_EMBEDDING_DIMENSIONS'
MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED = 'MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED'
DEFAULT_MEMORYLAYER_EMBEDDING_PRELOAD_ENABLED = True

# ============================================
# Embedding Service
# ============================================
MEMORYLAYER_EMBEDDING_SERVICE = 'MEMORYLAYER_EMBEDDING_SERVICE'
DEFAULT_MEMORYLAYER_EMBEDDING_SERVICE = 'default'

# ============================================
# Storage Backend
# ============================================
MEMORYLAYER_STORAGE_BACKEND = 'MEMORYLAYER_STORAGE_BACKEND'
DEFAULT_MEMORYLAYER_STORAGE_BACKEND = 'sqlite'

MEMORYLAYER_SQLITE_STORAGE_PATH = 'MEMORYLAYER_SQLITE_STORAGE_PATH'
DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH = "memorylayer.db"

# ============================================
# Memory Service
# ============================================
MEMORYLAYER_MEMORY_SERVICE = 'MEMORYLAYER_MEMORY_SERVICE'
DEFAULT_MEMORYLAYER_MEMORY_SERVICE = 'default'

# ============================================
# Reflection Service
# ============================================
MEMORYLAYER_REFLECT_SERVICE = 'MEMORYLAYER_REFLECT_SERVICE'
DEFAULT_MEMORYLAYER_REFLECT_SERVICE = 'default'

# ============================================
# Session Service
# ============================================
MEMORYLAYER_SESSION_SERVICE = 'MEMORYLAYER_SESSION_SERVICE'
DEFAULT_MEMORYLAYER_SESSION_SERVICE = 'in-memory'

MEMORYLAYER_SESSION_IMPLICIT_CREATE = 'MEMORYLAYER_SESSION_IMPLICIT_CREATE'
DEFAULT_MEMORYLAYER_SESSION_IMPLICIT_CREATE = True

# ============================================
# Workspace Service
# ============================================
MEMORYLAYER_WORKSPACE_SERVICE = 'MEMORYLAYER_WORKSPACE_SERVICE'
DEFAULT_MEMORYLAYER_WORKSPACE_SERVICE = 'default'

# ============================================
# Association Service
# ============================================
MEMORYLAYER_ASSOCIATION_SERVICE = 'MEMORYLAYER_ASSOCIATION_SERVICE'
DEFAULT_MEMORYLAYER_ASSOCIATION_SERVICE = 'default'

MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 'MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD'
DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD = 0.85

# ============================================
# Authorization Service
# ============================================
MEMORYLAYER_AUTHORIZATION_SERVICE = 'MEMORYLAYER_AUTHORIZATION_SERVICE'
DEFAULT_MEMORYLAYER_AUTHORIZATION_SERVICE = 'default'  # Open permissions (allow all)


# ============================================
# Reranker Service
# ============================================
class RerankerProviderType(str, Enum):
    """Available reranker provider types."""
    LLM = "llm"  # Use LLM service for reranking
    HYDE = "hyde"  # Hypothetical Document Embeddings (LLM + embedding) (default)
    LOCAL = "local"  # sentence-transformers CrossEncoder (self-hosted)
    NONE = "none"  # Disabled (no reranking)


MEMORYLAYER_RERANKER_PROVIDER = 'MEMORYLAYER_RERANKER_PROVIDER'
DEFAULT_MEMORYLAYER_RERANKER_PROVIDER = 'hyde'

MEMORYLAYER_RERANKER_SERVICE = 'MEMORYLAYER_RERANKER_SERVICE'
DEFAULT_MEMORYLAYER_RERANKER_SERVICE = 'default'

MEMORYLAYER_RERANKER_PRELOAD_ENABLED = 'MEMORYLAYER_RERANKER_PRELOAD_ENABLED'
DEFAULT_MEMORYLAYER_RERANKER_PRELOAD_ENABLED = True

# ============================================
# Cache Service
# ============================================
MEMORYLAYER_CACHE_SERVICE = 'MEMORYLAYER_CACHE_SERVICE'
DEFAULT_MEMORYLAYER_CACHE_SERVICE = 'default'

# Default tenant and workspace constants
# Use underscore prefix for all reserved/system entities
DEFAULT_TENANT_ID = "_default"
DEFAULT_WORKSPACE_ID = "_default"
GLOBAL_WORKSPACE_ID = "_global"

# ============================================
# Context ID Default
# ============================================
DEFAULT_CONTEXT_ID = "_default"

# ============================================
# Semantic Tiering Service
# ============================================
MEMORYLAYER_SEMANTIC_TIERING_SERVICE = 'MEMORYLAYER_SEMANTIC_TIERING_SERVICE'
DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_SERVICE = 'default'

MEMORYLAYER_SEMANTIC_TIERING_ENABLED = 'MEMORYLAYER_SEMANTIC_TIERING_ENABLED'
DEFAULT_MEMORYLAYER_SEMANTIC_TIERING_ENABLED = True

# ============================================
# Deduplication Service
# ============================================
MEMORYLAYER_DEDUPLICATION_SERVICE = 'MEMORYLAYER_DEDUPLICATION_SERVICE'
DEFAULT_MEMORYLAYER_DEDUPLICATION_SERVICE = 'default'

# ============================================
# Ontology Service
# ============================================
MEMORYLAYER_ONTOLOGY_SERVICE = 'MEMORYLAYER_ONTOLOGY_SERVICE'
DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE = 'default'

# ============================================
# Extraction Service
# ============================================
MEMORYLAYER_EXTRACTION_SERVICE = 'MEMORYLAYER_EXTRACTION_SERVICE'
DEFAULT_MEMORYLAYER_EXTRACTION_SERVICE = 'default'

# ============================================
# Task Service
# ============================================
MEMORYLAYER_TASK_PROVIDER = 'MEMORYLAYER_TASK_PROVIDER'
DEFAULT_MEMORYLAYER_TASK_PROVIDER = 'asyncio'

# ============================================
# Recall Scoring: Recency Boost
# ============================================
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RECENCY_HALF_LIFE_HOURS = 168

# ============================================
# Decay Service
# ============================================
MEMORYLAYER_DECAY_PROVIDER = 'MEMORYLAYER_DECAY_PROVIDER'
DEFAULT_MEMORYLAYER_DECAY_PROVIDER = 'default'

# ============================================
# Contradiction Service
# ============================================
MEMORYLAYER_CONTRADICTION_PROVIDER = 'MEMORYLAYER_CONTRADICTION_PROVIDER'
DEFAULT_MEMORYLAYER_CONTRADICTION_PROVIDER = 'default'

# ============================================
# Fact Decomposition
# ============================================
MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = 'MEMORYLAYER_FACT_DECOMPOSITION_ENABLED'
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED = True

MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = 'MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH'
DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH = 80

# ============================================
# Context Environment Service
# ============================================
MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE = 'MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE'
DEFAULT_MEMORYLAYER_CONTEXT_ENVIRONMENT_SERVICE = 'default'

MEMORYLAYER_CONTEXT_EXECUTOR = 'MEMORYLAYER_CONTEXT_EXECUTOR'
DEFAULT_MEMORYLAYER_CONTEXT_EXECUTOR = 'smolagents'

MEMORYLAYER_CONTEXT_MAX_OPERATIONS = 'MEMORYLAYER_CONTEXT_MAX_OPERATIONS'
DEFAULT_MEMORYLAYER_CONTEXT_MAX_OPERATIONS = 1_000_000

MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS = 'MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS'
DEFAULT_MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS = 30

MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS = 'MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS'
DEFAULT_MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS = 50_000

MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS = 'MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS'
DEFAULT_MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS = 4096

MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES = 'MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES'
DEFAULT_MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256 MB

MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS = 'MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS'
DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS = 10

MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS = 'MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS'
DEFAULT_MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS = 120

MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP = 'MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP'
DEFAULT_MEMORYLAYER_CONTEXT_EXEC_SOFT_CAP = 0

MEMORYLAYER_CONTEXT_EXEC_HARD_CAP = 'MEMORYLAYER_CONTEXT_EXEC_HARD_CAP'
DEFAULT_MEMORYLAYER_CONTEXT_EXEC_HARD_CAP = 0
