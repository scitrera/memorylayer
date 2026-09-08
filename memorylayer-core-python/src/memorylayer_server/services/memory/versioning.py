"""Deterministic versioning helpers for native semantic memories.

Operational metadata, embeddings, importance/decay, access counters, and recall
projections are intentionally outside the semantic state. Those fields may be
updated independently without invalidating a refinement compare-and-swap token.
"""

from __future__ import annotations

from typing import Any

from ...models.memory import Memory
from ..skills.versioning import canonical_hash

SEMANTIC_MEMORY_FIELDS = frozenset(
    {
        "content",
        "content_hash",
        "type",
        "subtype",
        "tags",
        "refinement_metadata",
        "pinned",
        "deleted_at",
    }
)


def memory_semantic_state(memory: Memory) -> dict[str, Any]:
    """Return the complete state protected by a memory resource ETag."""

    return {
        "tenant_id": memory.tenant_id,
        "workspace_id": memory.workspace_id,
        "user_id": memory.user_id,
        "logical_key": memory.logical_key,
        "content": memory.content,
        "content_hash": memory.content_hash,
        "type": memory.type.value,
        "subtype": memory.subtype,
        "tags": memory.tags,
        "refinement_metadata": memory.refinement_metadata,
        "pinned": memory.pinned,
        "deleted_at": memory.deleted_at,
    }


def memory_etag(revision: int, memory: Memory) -> str:
    """Build the opaque CAS token for one semantic memory revision."""

    return f'"memory-{revision}-{canonical_hash(memory_semantic_state(memory))}"'


def memory_revision_snapshot(memory: Memory) -> Memory:
    """Remove large/request-local derived fields from an immutable snapshot."""

    return memory.model_copy(
        deep=True,
        update={
            "embedding": None,
            "source_scope": None,
            "relevance_score": None,
            "boosted_score": None,
            "match_signals": None,
            "trust_score": None,
            "trust_signals": None,
            "freshness_score": None,
            "staleness_warning": None,
            "age_days": None,
        },
    )
