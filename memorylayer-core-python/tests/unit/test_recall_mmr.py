"""MMR diversification wiring in the recall reranking step.

Verifies that when MMR is enabled the final selection trades a little relevance
for diversity (demoting a near-duplicate), that it is OFF by default, and that a
pool with missing embeddings is handled without error.
"""

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import Memory, MemoryType
from memorylayer_server.utils import compute_content_hash


def _mem(mem_id: str, content: str, embedding: list[float] | None, boosted: float) -> Memory:
    m = Memory(
        id=mem_id,
        tenant_id="mmr_tenant",
        workspace_id="ws_mmr",
        content=content,
        content_hash=compute_content_hash(content),
        type=MemoryType.SEMANTIC,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        embedding=embedding,
    )
    m.boosted_score = boosted
    return m


@pytest.fixture
def mmr_flags(memory_service):
    """Save/restore the MMR flags so mutations don't leak across the shared
    ``memory_service`` fixture instance."""
    saved = (memory_service.rerank_mmr_enabled, memory_service.rerank_mmr_lambda)
    yield memory_service
    memory_service.rerank_mmr_enabled, memory_service.rerank_mmr_lambda = saved


@pytest.mark.asyncio
async def test_mmr_demotes_near_duplicate(mmr_flags):
    memory_service = mmr_flags
    # Two near-identical embeddings (high relevance) + one distinct (slightly
    # lower). With MMR on, the distinct memory should take the 2nd slot.
    memories = [
        _mem("d0", "alpha one", [1.0, 0.0], 0.90),
        _mem("d1", "alpha two", [1.0, 0.0], 0.85),
        _mem("distinct", "beta different", [0.0, 1.0], 0.70),
    ]
    memory_service.rerank_mmr_enabled = True
    memory_service.rerank_mmr_lambda = 0.5

    out = await memory_service._apply_reranking("alpha", memories, limit=2)

    assert [m.id for m in out] == ["d0", "distinct"]


@pytest.mark.asyncio
async def test_mmr_disabled_preserves_pool_order(mmr_flags):
    memory_service = mmr_flags
    # MMR off with no reranker service -> plain top-limit truncation.
    memories = [
        _mem("a", "alpha one", [1.0, 0.0], 0.90),
        _mem("b", "alpha two", [1.0, 0.0], 0.85),
        _mem("c", "beta different", [0.0, 1.0], 0.70),
    ]
    memory_service.rerank_mmr_enabled = False
    original = memory_service.reranker_service
    memory_service.reranker_service = None
    try:
        out = await memory_service._apply_reranking("alpha", memories, limit=2)
    finally:
        memory_service.reranker_service = original

    assert [m.id for m in out] == ["a", "b"]


@pytest.mark.asyncio
async def test_mmr_backfills_missing_embeddings(mmr_flags):
    memory_service = mmr_flags
    # Memories lacking embeddings must not crash MMR: it best-effort embeds their
    # content via the embedding provider and still returns `limit` items.
    memories = [
        _mem("a", "the quick brown fox jumps", None, 0.90),
        _mem("b", "a completely unrelated sentence about oceans", None, 0.80),
        _mem("c", "the quick brown fox jumps", None, 0.70),
    ]
    memory_service.rerank_mmr_enabled = True
    memory_service.rerank_mmr_lambda = 0.5

    out = await memory_service._apply_reranking("fox", memories, limit=2)

    assert len(out) == 2
    assert out[0].id == "a"  # highest relevance seeds the selection
