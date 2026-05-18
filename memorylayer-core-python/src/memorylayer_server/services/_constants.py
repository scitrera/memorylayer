"""
Centralized extension point constants for all MemoryLayer services.

All EXT_* constants are defined here to avoid circular import issues.
Individual service base modules re-export the relevant constants for
backward compatibility.
"""

# ============================================
# Storage
# ============================================
EXT_STORAGE_BACKEND = "memorylayer-primary-storage"

# ============================================
# Authentication & Authorization
# ============================================
EXT_AUTHENTICATION_SERVICE = "memorylayer-authentication-service"
EXT_AUTHORIZATION_SERVICE = "memorylayer-authorization-service"

# ============================================
# Session
# ============================================
EXT_SESSION_SERVICE = "memorylayer-session-service"

# ============================================
# Workspace
# ============================================
EXT_WORKSPACE_SERVICE = "memorylayer-workspace-service"

# ============================================
# Cache
# ============================================
EXT_CACHE_SERVICE = "memorylayer-cache-service"

# ============================================
# Embedding
# ============================================
EXT_EMBEDDING_PROVIDER = "embedding-provider"
EXT_EMBEDDING_SERVICE = "embedding-service"

# ============================================
# LLM
# ============================================
EXT_LLM_SERVICE = "memorylayer-llm-service"
EXT_LLM_REGISTRY = "memorylayer-llm-registry"

# ============================================
# Reranker
# ============================================
EXT_RERANKER_PROVIDER = "reranker-provider"
EXT_RERANKER_SERVICE = "reranker-service"

# ============================================
# Memory
# ============================================
EXT_MEMORY_SERVICE = "memorylayer-memory-service"

# ============================================
# Extraction
# ============================================
EXT_EXTRACTION_SERVICE = "memorylayer-extraction-service"

# ============================================
# Deduplication
# ============================================
EXT_DEDUPLICATION_SERVICE = "memorylayer-deduplication-service"

# ============================================
# Contradiction
# ============================================
EXT_CONTRADICTION_SERVICE = "memorylayer-contradiction-service"

# ============================================
# Decay
# ============================================
EXT_DECAY_SERVICE = "memorylayer-decay-service"

# ============================================
# Semantic Tiering
# ============================================
EXT_SEMANTIC_TIERING_SERVICE = "memorylayer-tier-generation-service"

# ============================================
# Association
# ============================================
EXT_ASSOCIATION_SERVICE = "memorylayer-association-service"

# ============================================
# Ontology
# ============================================
EXT_ONTOLOGY_SERVICE = "memorylayer-ontology-service"
EXT_MULTI_ONTOLOGY_CONTRIBUTORS = "memorylayer-multi-ontology-contributors"

# ============================================
# Reflect
# ============================================
EXT_REFLECT_SERVICE = "memorylayer-reflect-service"

# ============================================
# Inference (entity insight derivation)
# ============================================
EXT_INFERENCE_SERVICE = "memorylayer-inference-service"

# ============================================
# Context Environment
# ============================================
EXT_CONTEXT_ENVIRONMENT_SERVICE = "memorylayer-context-environment-service"

# ============================================
# Tasks
# ============================================
EXT_TASK_SERVICE = "memorylayer-task-service"
EXT_MULTI_TASK_HANDLERS = "memorylayer-multi-task-handlers"

# ============================================
# Chat History
# ============================================
EXT_CHAT_SERVICE = "memorylayer-chat-service"

# ============================================
# Audit
# ============================================
EXT_AUDIT_SERVICE = "memorylayer-audit-service"

# ============================================
# Rate Limiting
# ============================================
EXT_RATE_LIMIT_SERVICE = "memorylayer-rate-limit-service"

# ============================================
# Metrics / Observability
# ============================================
EXT_METRICS_SERVICE = "memorylayer-metrics-service"

# ============================================
# Document Ingestion
# ============================================
EXT_DOCUMENT_SERVICE = "memorylayer-document-service"

# Embed-server REST client (relocated from enterprise → OSS in Phase 3).
EXT_EMBED_SERVER_CLIENT = "memorylayer-embed-server-client"

# ============================================
# Data Provider
# ============================================
EXT_DATA_PROVIDER_SERVICE = "memorylayer-data-provider-service"

# ============================================
# Skills
# ============================================
EXT_SKILLS_SERVICE = "memorylayer-skills-service"

# ============================================
# Graph Analysis
# ============================================
EXT_GRAPH_ANALYSIS_SERVICE = "memorylayer-graph-analysis-service"

# ============================================
# Knowledgebase
# ============================================
EXT_KNOWLEDGEBASE_SERVICE = "memorylayer-knowledgebase-service"

# ============================================
# Aether Service connection (shared gRPC client)
# ============================================
# Phase 1 (Aether convergence): MemoryLayer's in-process Aether connection
# now registers as a Service principal (sv::memorylayer::*) instead of an
# Agent.  The extension key was renamed to reflect that.  The legacy name
# is kept as an alias for one release so external callers (and any not-yet
# migrated code paths) continue to resolve the same extension instance.
EXT_AETHER_SERVICE_CONNECTION = "memorylayer-aether-service-connection"
# Deprecated alias — points at the same extension instance.  Slated for
# removal once enterprise + cowork are off the old name.
EXT_AETHER_AGENT_SERVICE = EXT_AETHER_SERVICE_CONNECTION
