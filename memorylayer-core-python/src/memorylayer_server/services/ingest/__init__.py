"""Source ingest adapters that normalize external sources onto the shared
``MemoryService.remember()`` / ``enqueue_post_store`` pipeline.

Today this hosts the email adapter. Adapters here are deterministic MAPPINGS (source payload ->
``RememberInput``); they never store or enrich directly — the shared pipeline
owns decompose + enrich + entity accretion.
"""

from .email import email_to_remember_input
from .knowledge_work import (
    KNOWLEDGE_WORK_NORMALIZATION_KEY,
    KnowledgeWorkNormalization,
    connector_to_remember_input,
    normalize_connector_metadata,
)

__all__ = (
    "KNOWLEDGE_WORK_NORMALIZATION_KEY",
    "KnowledgeWorkNormalization",
    "connector_to_remember_input",
    "email_to_remember_input",
    "normalize_connector_metadata",
)
