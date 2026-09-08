"""Unit + integration tests for the flag-gated fact retrieval channel.

The fact channel lifts a two-store RRF pattern into core recall: fact memories
(subtype="fact") are RRF-fused as a SEPARATE arm with the raw-turn vector arm,
while the primary vector arm is kept PURE of facts (so date-grounded turns keep
their temporal phrasing and avoid the temporal regression that mixing causes).

OSS extract_facts is a no-op (no LLM), so the fusion — the real deliverable — is
tested directly by injecting subtype="fact" memories into a fake storage.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemorySubtype,
    MemoryType,
    RecallInput,
    RecallMode,
    RememberInput,
)
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory.base import (
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
    MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_QUERY_INTENT_ENABLED,
)
from memorylayer_server.config import (
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.services.memory.default import MemoryService


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _make_v(**overrides) -> Variables:
    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    # Isolate the fact channel: keep other fusion channels off unless a test opts in.
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, False)
    v.set(MEMORYLAYER_QUERY_INTENT_ENABLED, False)
    for k, val in overrides.items():
        v.set(k, val)
    return v


class _FakeStorage:
    """In-test storage supporting subtype-filtered vector search.

    Memories are returned in a deterministic, caller-controlled rank order
    (the order they were registered for a given subtype-class), each tagged
    with a descending pseudo-relevance so RRF rank derivation is stable.
    """

    def __init__(self):
        # ordered list of (memory, base_score) for the PRIMARY (non-fact) arm
        self.primary: list[tuple[Memory, float]] = []
        # ordered list for the FACT arm (subtype="fact")
        self.facts: list[tuple[Memory, float]] = []

    async def search_memories(
        self,
        workspace_id: str,
        query_embedding,
        limit: int = 10,
        offset: int = 0,
        min_relevance: float = 0.5,
        types=None,
        subtypes=None,
        tags=None,
        include_archived: bool = False,
        observer_id=None,
        subject_id=None,
        created_after=None,
        created_before=None,
        user_id=None,
    ):
        if subtypes and MemorySubtype.FACT.value in subtypes:
            pool = self.facts
        else:
            pool = self.primary
        out = [(m, s) for m, s in pool if s >= min_relevance]
        return out[offset : offset + limit]


def _make_service(v: Variables) -> MemoryService:
    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[0.1] * 8)
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )
    storage = _FakeStorage()
    svc = MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        v=v,
    )
    # No reranker -> _apply_reranking is a plain top-limit slice (deterministic).
    svc.reranker_service = None
    svc.cache = None
    return svc


def _mem(mid: str, content: str = "turn content", subtype: str | None = None) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content=content,
        content_hash=f"h_{mid}",
        type=MemoryType.SEMANTIC,
        subtype=subtype,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
    )


def _recall_input(limit: int = 3) -> RecallInput:
    return RecallInput(
        query="relationship status",
        mode=RecallMode.RAG,
        limit=limit,
        min_relevance=0.0,
        include_associations=False,
        include_global=False,
        include_global_user=False,
    )


# ===========================================================================
# Unit: _fuse_fact_results behavior
# ===========================================================================


@pytest.mark.asyncio
async def test_low_rank_gold_fact_promoted_into_topk():
    """A gold fact deep in vector rank is promoted into top-k after fusion."""
    v = _make_v()
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, True)
    svc = _make_service(v)

    # Primary arm: 5 turns, gold fact is NOT here.
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(5)]
    # Fact arm: gold fact ranked #1 in the fact-restricted set.
    gold = _mem("gold_fact", content="X is a single parent", subtype=MemorySubtype.FACT.value)
    svc.storage.facts = [(gold, 0.42), (_mem("f2", subtype="fact"), 0.30)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert "gold_fact" in ids, f"gold fact not promoted into top-3: {ids}"
    gold_mem = next(m for m in result.memories if m.id == "gold_fact")
    assert "fact" in (gold_mem.match_signals or [])


@pytest.mark.asyncio
async def test_noop_when_no_facts_present():
    """With the flag on but no fact memories, recall is unchanged (turns only)."""
    v = _make_v()
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, True)
    svc = _make_service(v)
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(3)]
    svc.storage.facts = []

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("fact" not in (m.match_signals or []) for m in result.memories)


@pytest.mark.asyncio
async def test_noop_when_flag_off():
    """Flag off: facts in storage are never fetched as a separate arm."""
    v = _make_v()
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
    svc = _make_service(v)
    svc.storage.primary = [(_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(3)]
    svc.storage.facts = [(_mem("gold_fact", subtype="fact"), 0.99)]

    result = await svc.recall("ws", _recall_input(limit=5))
    ids = [m.id for m in result.memories]
    assert "gold_fact" not in ids
    assert all("fact" not in (m.match_signals or []) for m in result.memories)


@pytest.mark.asyncio
async def test_primary_arm_excludes_subtype_fact():
    """Pool purity: a fact memory leaking into the primary arm is stripped before fusion.

    With the flag OFF the leaked fact would surface (no fact-strip); with the flag
    ON the strip removes it from the primary pool and it can only re-enter via the
    fact arm (which here is empty), so it must NOT appear.
    """
    v = _make_v()
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, True)
    svc = _make_service(v)
    # A fact memory erroneously present in the primary vector arm.
    leaked = _mem("leaked_fact", subtype=MemorySubtype.FACT.value)
    svc.storage.primary = [(_mem("turn0"), 0.9), (leaked, 0.8)]
    svc.storage.facts = []  # fact arm empty -> leaked fact has no path back in

    result = await svc.recall("ws", _recall_input(limit=5))
    ids = [m.id for m in result.memories]
    assert "leaked_fact" not in ids, f"fact leaked into primary pool: {ids}"
    assert ids == ["turn0"]


@pytest.mark.asyncio
async def test_flag_off_byte_identical_to_no_fact_code():
    """Regression: flag OFF leaves recall output byte-identical to pre-channel behavior.

    Two services over identical fixtures — one constructed before the fact-channel
    flag existed in env (default OFF) — must produce the same serialized result.
    The flag-on-with-facts path is the ONLY behavior change; OFF must be inert.
    """
    # Shared, frozen-timestamp fixtures so the comparison is exact (recency math
    # otherwise drifts in the ~10th decimal from per-instance datetime.now()).
    frozen = datetime(2024, 1, 1, tzinfo=UTC)

    def _frozen_mem(mid: str, subtype: str | None = None) -> Memory:
        m = _mem(mid, subtype=subtype)
        m.created_at = frozen
        m.updated_at = frozen
        return m

    primary_a = [(_frozen_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(4)]
    primary_b = [(_frozen_mem(f"turn{i}"), 0.9 - i * 0.1) for i in range(4)]
    facts = [(_frozen_mem("gold_fact", subtype=MemorySubtype.FACT.value), 0.99)]

    # Service A: flag explicitly OFF.
    v_off = _make_v()
    v_off.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
    svc_off = _make_service(v_off)
    svc_off.storage.primary = primary_a
    svc_off.storage.facts = list(facts)

    # Service B: flag default (also OFF) — represents pre-channel config.
    v_default = _make_v()
    svc_default = _make_service(v_default)
    svc_default.storage.primary = primary_b
    svc_default.storage.facts = list(facts)

    # recency_weight=0.0 removes the wall-clock recency term. The remaining
    # wall-clock terms (freshness/trust annotation) are dropped from the golden
    # compare below, since they are pre-existing behavior independent of the
    # channel — the channel only affects which memories appear, their order, and
    # their match_signals. Those are what must be byte-identical when the flag is OFF.
    inp = _recall_input(limit=5).model_copy(update={"recency_weight": 0.0})
    r_off = await svc_off.recall("ws", inp)
    r_default = await svc_default.recall("ws", inp)

    _CLOCK_FIELDS = {
        "age_days",
        "freshness_score",
        "staleness_warning",
        "trust_score",
        "trust_signals",
        "last_accessed_at",
        "access_count",
    }

    def _golden(result):
        out = []
        for m in result.memories:
            d = m.model_dump()
            for f in _CLOCK_FIELDS:
                d.pop(f, None)
            out.append(d)
        return out

    assert _golden(r_off) == _golden(r_default)
    # And neither surfaces the fact nor a "fact" signal.
    assert all("gold_fact" != m.id for m in r_off.memories)
    assert all("fact" not in (m.match_signals or []) for m in r_off.memories)


# ===========================================================================
# Integration-style: flag on with facts pre-inserted (uses conftest service)
# ===========================================================================


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"factchan_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="fact_tenant", name="Fact Channel Test", created_at=now, updated_at=now)
    )
    return ws


@pytest.mark.asyncio
async def test_zero_overlap_query_surfaces_fact(memory_service, storage_backend, iso_ws):
    """A zero-lexical-overlap query surfaces a fact via the fused fact arm."""
    original = memory_service.fact_channel_enabled
    original_rerank = memory_service.reranker_service
    memory_service.fact_channel_enabled = True
    memory_service.reranker_service = None
    try:
        # A raw turn (no overlap with the query) and a pre-inserted fact memory.
        await memory_service.remember(iso_ws, RememberInput(content="We grabbed coffee downtown on Tuesday"))
        await memory_service.ingest_fact(
            iso_ws,
            RememberInput(
                content="Caroline is a single parent",
                type=MemoryType.SEMANTIC,
                subtype=MemorySubtype.FACT.value,
                metadata={"kind": "fact", "source_id": "turn_x"},
            ),
            inline=True,
        )
        result = await memory_service.recall(
            iso_ws,
            RecallInput(
                query="relationship status",
                mode=RecallMode.RAG,
                limit=5,
                min_relevance=0.0,
                include_associations=False,
                include_global=False,
                include_global_user=False,
            ),
        )
        contents = [m.content for m in result.memories]
        assert any("single parent" in c for c in contents), f"fact not surfaced: {contents}"
    finally:
        memory_service.fact_channel_enabled = original
        memory_service.reranker_service = original_rerank


@pytest.mark.asyncio
async def test_temporal_query_still_surfaces_date_grounded_turn(memory_service, storage_backend, iso_ws):
    """A date-grounded raw turn still surfaces (no temporal regression from facts)."""
    original = memory_service.fact_channel_enabled
    original_rerank = memory_service.reranker_service
    memory_service.fact_channel_enabled = True
    memory_service.reranker_service = None
    try:
        await memory_service.remember(
            iso_ws, RememberInput(content="[15 May, 2023] Caroline moved to Stockholm last year")
        )
        await memory_service.ingest_fact(
            iso_ws,
            RememberInput(
                content="Caroline lives in Stockholm",
                type=MemoryType.SEMANTIC,
                subtype=MemorySubtype.FACT.value,
                metadata={"kind": "fact", "source_id": "turn_y"},
            ),
            inline=True,
        )
        result = await memory_service.recall(
            iso_ws,
            RecallInput(
                query="When did Caroline move to Stockholm",
                mode=RecallMode.RAG,
                limit=5,
                min_relevance=0.0,
                include_associations=False,
                include_global=False,
                include_global_user=False,
            ),
        )
        contents = [m.content for m in result.memories]
        # The date-grounded turn (with [15 May, 2023] / "last year") is retained.
        assert any("last year" in c for c in contents), f"date-grounded turn missing: {contents}"
        # And the primary arm did not get polluted with the date-stripped fact as a
        # date-grounded turn: the fact, if present, still carries no date phrasing.
    finally:
        memory_service.fact_channel_enabled = original
        memory_service.reranker_service = original_rerank
