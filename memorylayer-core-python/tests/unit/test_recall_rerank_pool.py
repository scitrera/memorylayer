"""Regression test: the reranker receives the over-fetched candidate pool.

Previously `_recall_rag` truncated to `limit` before the reranker ran, so the
`recall_overfetch` pool was discarded and the reranker was starved (often never
invoked). This pins the fixed behaviour: the pool (recall_overfetch x limit)
reaches the reranker, which can promote a tail candidate into the top-k.
"""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memorylayer_server.models.memory import RecallInput, RecallMode, RememberInput
from memorylayer_server.models.workspace import Workspace


class _SpyReranker:
    """Records the candidates it receives and reorders by reversing them."""

    def __init__(self):
        self.received_ids: list[str] | None = None

    async def rerank_objects_adaptive(self, query, objects, content_fn, score_fn, requested_k):
        self.received_ids = [o.id for o in objects]
        reordered = list(reversed(objects))[:requested_k]
        return [SimpleNamespace(document=o) for o in reordered]


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"rerankpool_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="rerank_tenant", name="Rerank Pool Test", created_at=now, updated_at=now)
    )
    return ws


@pytest.mark.asyncio
async def test_reranker_receives_overfetch_pool(memory_service, iso_ws):
    # Nine matching memories; limit=3 with default recall_overfetch=3 -> pool up to 9.
    for i in range(9):
        await memory_service.remember(iso_ws, RememberInput(content=f"alpha record entry number {i}"))

    spy = _SpyReranker()
    original = memory_service.reranker_service
    memory_service.reranker_service = spy
    try:
        result = await memory_service.recall(
            iso_ws,
            RecallInput(
                query="alpha record entry",
                mode=RecallMode.RAG,
                limit=3,
                min_relevance=0.0,
                include_associations=False,  # isolate: no expansion -> single rerank pass
                include_global=False,
                include_global_user=False,
            ),
        )
    finally:
        memory_service.reranker_service = original

    # The reranker saw more than `limit` candidates (the overfetch pool), not just 3.
    assert spy.received_ids is not None
    assert len(spy.received_ids) > 3
    # And recall returned the reranker's order (reversed pool, top-3).
    assert [m.id for m in result.memories] == list(reversed(spy.received_ids))[:3]
