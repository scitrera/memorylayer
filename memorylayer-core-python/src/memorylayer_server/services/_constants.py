"""
Centralized extension point constants for all MemoryLayer services.

All EXT_* constants are defined here to avoid circular import issues.
Individual service base modules re-export the relevant constants for
backward compatibility.
"""

# ============================================
# Storage
# ============================================
EXT_STORAGE_BACKEND = 'memorylayer-primary-storage'

# ============================================
# Authentication & Authorization
# ============================================
EXT_AUTHENTICATION_SERVICE = "memorylayer-authentication-service"
EXT_AUTHORIZATION_SERVICE = 'memorylayer-authorization-service'

# ============================================
# Session
# ============================================
EXT_SESSION_SERVICE = 'memorylayer-session-service'

# ============================================
# Workspace
# ============================================
EXT_WORKSPACE_SERVICE = 'memorylayer-workspace-service'

# ============================================
# Cache
# ============================================
EXT_CACHE_SERVICE = 'memorylayer-cache-service'

# ============================================
# Embedding
# ============================================
EXT_EMBEDDING_PROVIDER = 'embedding-provider'
EXT_EMBEDDING_SERVICE = 'embedding-service'

# ============================================
# LLM
# ============================================
EXT_LLM_PROVIDER = 'memorylayer-llm-provider'
EXT_LLM_SERVICE = 'memorylayer-llm-service'
EXT_LLM_REGISTRY = 'memorylayer-llm-registry'

# ============================================
# Reranker
# ============================================
EXT_RERANKER_PROVIDER = 'reranker-provider'
EXT_RERANKER_SERVICE = 'reranker-service'

# ============================================
# Memory
# ============================================
EXT_MEMORY_SERVICE = 'memorylayer-memory-service'

# ============================================
# Extraction
# ============================================
EXT_EXTRACTION_SERVICE = 'memorylayer-extraction-service'

# ============================================
# Deduplication
# ============================================
EXT_DEDUPLICATION_SERVICE = 'memorylayer-deduplication-service'

# ============================================
# Contradiction
# ============================================
EXT_CONTRADICTION_SERVICE = 'memorylayer-contradiction-service'

# ============================================
# Decay
# ============================================
EXT_DECAY_SERVICE = 'memorylayer-decay-service'

# ============================================
# Semantic Tiering
# ============================================
EXT_SEMANTIC_TIERING_SERVICE = 'memorylayer-tier-generation-service'

# ============================================
# Association
# ============================================
EXT_ASSOCIATION_SERVICE = 'memorylayer-association-service'

# ============================================
# Ontology
# ============================================
EXT_ONTOLOGY_SERVICE = 'memorylayer-ontology-service'

# ============================================
# Reflect
# ============================================
EXT_REFLECT_SERVICE = 'memorylayer-reflect-service'

# ============================================
# Context Environment
# ============================================
EXT_CONTEXT_ENVIRONMENT_SERVICE = 'memorylayer-context-environment-service'

# ============================================
# Tasks
# ============================================
EXT_TASK_SERVICE = 'memorylayer-task-service'
EXT_MULTI_TASK_HANDLERS = 'memorylayer-multi-task-handlers'