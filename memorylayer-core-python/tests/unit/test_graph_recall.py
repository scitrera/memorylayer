"""Unit tests for the flag-gated graph-traversal recall channel (P4 channel G).

The graph channel (``_fuse_graph_results``) is the cross-source /
relationship-traversal consumer of the canonical-entity layer: query-entity ->
canonical entity -> a SMALL, BOUNDED neighborhood of MENTIONED memories (via
``GraphQueryService.entity_neighborhood``) -> RRF-fused as an ADDITIVE arm.

These tests are deterministic and mock-free on the graph side (a small fake
graph-query service + fake registry resolver + fake storage), so the fusion and —
critically — the ANTI-FLOOD bound are tested without an LLM or embedding server.

CARDINAL: the killed registry-recall flood (LoCoMo 0.54 -> 0.13) dumped ~all turns
mentioning an entity into the pool. The bounded-pool proof below asserts the arm
requests ``entity_neighborhood`` with the SMALL overfetch-bounded ``memory_limit``,
NOT 100 / unbounded — the guardrail that prevents repeating that regression.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import (
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.models.entity_registry import Entity, EntityResolution, EntityType
from memorylayer_server.models.graph_query import EntityMentionedMemory, EntityNeighborhood
from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType,
    RecallInput,
    RecallMode,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory.base import (
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
    MEMORYLAYER_GRAPH_RECALL_ENABLED,
    MEMORYLAYER_GRAPH_RECALL_POOL,
    MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_QUERY_INTENT_ENABLED,
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
    # Isolate the graph channel: keep the other fusion channels off unless a test
    # opts in.
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, False)
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, False)
    v.set(MEMORYLAYER_QUERY_INTENT_ENABLED, False)
    for k, val in overrides.items():
        v.set(k, val)
    return v


def _mem(mid: str, content: str = "turn content", embedding: list[float] | None = None) -> Memory:
    now = datetime.now(UTC)
    return Memory(
        id=mid,
        workspace_id="ws",
        tenant_id="t",
        context_id="_default",
        content=content,
        content_hash=f"h_{mid}",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
        embedding=embedding,
    )


class _FakeStorage:
    """In-test storage: a deterministic primary vector arm + get_memory by id."""

    def __init__(self):
        self.primary: list[tuple[Memory, float]] = []
        self.by_id: dict[str, Memory] = {}
        self.get_memory_calls: list[tuple[str, bool]] = []

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
        out = [(m, s) for m, s in self.primary if s >= min_relevance]
        return out[offset : offset + limit]

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True):
        self.get_memory_calls.append((memory_id, track_access))
        return self.by_id.get(memory_id)


class _FakeRegistry:
    """Minimal EntityRegistryService stand-in: resolve a query name -> canonical id."""

    def __init__(self):
        self.entities_by_name: dict[str, Entity] = {}
        self.resolve_calls: list[dict] = []

    def _entity(self, name: str) -> Entity:
        now = datetime.now(UTC)
        return Entity(
            id=f"ent_{name.lower()}",
            workspace_id="ws",
            entity_type=EntityType.PERSON,
            canonical_name=name,
            normalized_name=name.lower(),
            created_at=now,
            updated_at=now,
        )

    async def resolve(self, workspace_id, name, entity_type, *, allow_create=True, **kwargs):
        self.resolve_calls.append({"name": name, "entity_type": entity_type, "allow_create": allow_create})
        ent = self.entities_by_name.get(name)
        if ent is not None:
            return EntityResolution(entity=ent, matched_via="exact", score=1.0)
        if not allow_create:
            raise LookupError(f"no entity {name!r} (allow_create=False)")
        raise AssertionError("graph recall channel must never create entities (allow_create=True)")


class _FakeGraphQuery:
    """Minimal GraphQueryService stand-in for the recall channel.

    ``neighborhoods`` maps entity_id -> list[memory_id] the entity MENTIONS.
    ``calls`` records every entity_neighborhood call (entity_id + memory_limit +
    hops) so a test can assert the bounded memory_limit (the anti-flood guard).
    """

    def __init__(self):
        self.neighborhoods: dict[str, list[str]] = {}
        self.calls: list[dict] = []

    async def entity_neighborhood(
        self, workspace_id, entity_id, *, hops=1, memory_limit=100, entity_limit=50
    ) -> EntityNeighborhood:
        self.calls.append({"entity_id": entity_id, "hops": hops, "memory_limit": memory_limit})
        mids = self.neighborhoods.get(entity_id, [])[:memory_limit]
        return EntityNeighborhood(
            entity_id=entity_id,
            memories_for_entity=[EntityMentionedMemory(memory_id=m) for m in mids],
            entities_co_mentioned=[],
        )


def _make_service(v: Variables, registry=None, graph=None) -> MemoryService:
    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[1.0, 0.0])
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )
    storage = _FakeStorage()
    svc = MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        entity_registry_service=registry,
        graph_query_service=graph,
        v=v,
    )
    svc.reranker_service = None  # plain top-limit slice (deterministic)
    svc.cache = None
    return svc


def _recall_input(limit: int = 3, query: str = "What did Caroline say") -> RecallInput:
    return RecallInput(
        query=query,
        mode=RecallMode.RAG,
        limit=limit,
        min_relevance=0.0,
        include_associations=False,
        include_global=False,
        include_global_user=False,
    )


# ===========================================================================
# Unit: _fuse_graph_results behavior
# ===========================================================================


@pytest.mark.asyncio
async def test_graph_neighbor_promoted_into_topk_with_graph_signal():
    """Flag ON + entity resolves: a graph-neighborhood memory is RRF-fused, signal="graph"."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)

    # Primary vector arm: 5 turns, the gold neighbor is NOT here.
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(5)]

    # Gold neighbor: highly aligned to the query embedding [1.0, 0.0] -> ranks #1
    # within the (bounded) neighborhood set.
    gold = _mem("gold_neighbor", content="She moved to Stockholm", embedding=[1.0, 0.0])
    svc.storage.by_id["gold_neighbor"] = gold

    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    graph.neighborhoods[ent.id] = ["gold_neighbor"]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert "gold_neighbor" in ids, f"graph neighbor not promoted into top-3: {ids}"
    gold_out = next(m for m in result.memories if m.id == "gold_neighbor")
    assert "graph" in (gold_out.match_signals or [])


@pytest.mark.asyncio
async def test_bounded_pool_memory_limit_is_overfetch_not_100():
    """ANTI-FLOOD: entity_neighborhood is called with the SMALL overfetch-bounded
    memory_limit, NOT 100 and NOT unbounded.

    limit=3, recall_overfetch=3 -> overfetch budget = 9, capped at
    graph_recall_pool=50 -> memory_limit == 9. The DTO default (100) must NEVER be
    used; that default is exactly the killed-flood magnitude.
    """
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    graph.neighborhoods[ent.id] = []  # neighborhood content irrelevant; assert the call arg

    await svc.recall("ws", _recall_input(limit=3))

    assert graph.calls, "entity_neighborhood was never called"
    for call in graph.calls:
        assert call["memory_limit"] == 9, (
            f"expected bounded memory_limit=9 (limit*overfetch), got {call['memory_limit']}"
        )
        assert call["memory_limit"] != 100, "memory_limit must NEVER be the unbounded 100 default"
        assert call["hops"] == 1


@pytest.mark.asyncio
async def test_pool_hard_capped_below_overfetch():
    """ANTI-FLOOD: graph_recall_pool is a hard ceiling below the overfetch budget.

    limit=50, recall_overfetch=3 -> overfetch budget = 150, but graph_recall_pool=5
    -> memory_limit == 5 (the cap wins). Even a huge requested limit cannot turn the
    graph arm into a member-dump.
    """
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    v.set(MEMORYLAYER_GRAPH_RECALL_POOL, 5)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    # Stuff the neighborhood with many candidates; only the capped slice may surface.
    graph.neighborhoods[ent.id] = [f"n{i}" for i in range(100)]
    for i in range(100):
        svc.storage.by_id[f"n{i}"] = _mem(f"n{i}", embedding=[1.0, 0.0])

    await svc.recall("ws", _recall_input(limit=50))

    assert graph.calls
    for call in graph.calls:
        assert call["memory_limit"] == 5, f"cap not enforced: {call['memory_limit']}"
    # The graph candidate set the arm fetched (the n* neighbors) is bounded by the
    # cap; only count graph-candidate reads (track_access=False), excluding any
    # downstream primary-arm reads, to isolate the arm's own fetch budget.
    graph_fetched = {mid for mid, track in svc.storage.get_memory_calls if mid.startswith("n")}
    assert len(graph_fetched) <= 5, f"more than pool graph candidates fetched: {len(graph_fetched)}"


@pytest.mark.asyncio
async def test_resolve_called_with_allow_create_false():
    """The graph channel must NEVER create an entity from a query string."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    await svc.recall("ws", _recall_input(limit=3))

    assert registry.resolve_calls, "resolve was never called for the query entity"
    assert all(c["allow_create"] is False for c in registry.resolve_calls), registry.resolve_calls


@pytest.mark.asyncio
async def test_noop_when_flag_off_byte_identical_and_no_graph_call():
    """Flag OFF: the graph service is never consulted and neighbors never surface."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, False)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]
    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    graph.neighborhoods[ent.id] = ["gold_neighbor"]
    svc.storage.by_id["gold_neighbor"] = _mem("gold_neighbor", embedding=[1.0, 0.0])

    result = await svc.recall("ws", _recall_input(limit=5))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert "gold_neighbor" not in ids
    assert all("graph" not in (m.match_signals or []) for m in result.memories)
    # Flag-off short-circuits the whole arm: no graph call, no resolve.
    assert graph.calls == []
    assert registry.resolve_calls == []


@pytest.mark.asyncio
async def test_flag_off_default_byte_identical():
    """Regression: flag at default (OFF) leaves recall byte-identical to the
    explicit-off config even when the graph would promote a neighbor."""
    frozen = datetime(2024, 1, 1, tzinfo=UTC)

    def _frozen_mem(mid: str, embedding=None) -> Memory:
        m = _mem(mid, embedding=embedding)
        m.created_at = frozen
        m.updated_at = frozen
        return m

    def _build(flag_value: bool | None):
        v = _make_v()
        if flag_value is not None:
            v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, flag_value)
        registry = _FakeRegistry()
        graph = _FakeGraphQuery()
        ent = registry._entity("Caroline")
        registry.entities_by_name["Caroline"] = ent
        graph.neighborhoods[ent.id] = ["gold_neighbor"]
        svc = _make_service(v, registry=registry, graph=graph)
        svc.storage.primary = [(_frozen_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(4)]
        svc.storage.by_id["gold_neighbor"] = _frozen_mem("gold_neighbor", embedding=[1.0, 0.0])
        return svc

    inp = _recall_input(limit=5).model_copy(update={"recency_weight": 0.0})
    r_off = await _build(False).recall("ws", inp)
    r_default = await _build(None).recall("ws", inp)

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
    assert all(m.id != "gold_neighbor" for m in r_off.memories)


@pytest.mark.asyncio
async def test_noop_when_no_graph_query_service():
    """Flag on but no graph service wired: clean no-op (degrades to vector results)."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry, graph=None)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]
    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("graph" not in (m.match_signals or []) for m in result.memories)
    # No registry resolve attempted either — the missing-service guard short-circuits first.
    assert registry.resolve_calls == []


@pytest.mark.asyncio
async def test_noop_when_no_registry_service():
    """Flag on, graph service wired, but no registry resolver: clean no-op.

    The graph service keys on entity_id; without a resolver we cannot turn the
    query name into an id, so the arm short-circuits with zero graph calls.
    """
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=None, graph=graph)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("graph" not in (m.match_signals or []) for m in result.memories)
    assert graph.calls == []


@pytest.mark.asyncio
async def test_noop_when_query_entity_does_not_resolve():
    """A query entity that resolves to nothing (allow_create=False miss) is a no-op."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()  # empty: every resolve misses -> LookupError
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("graph" not in (m.match_signals or []) for m in result.memories)
    # resolve was attempted (and missed, allow_create=False) but no graph call followed.
    assert registry.resolve_calls
    assert graph.calls == []


@pytest.mark.asyncio
async def test_result_count_le_limit_preserved():
    """The overfetch-rerank contract holds: recall returns <= limit even with the arm on."""
    v = _make_v()
    v.set(MEMORYLAYER_GRAPH_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    graph = _FakeGraphQuery()
    svc = _make_service(v, registry=registry, graph=graph)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.01) for i in range(10)]

    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    graph.neighborhoods[ent.id] = [f"n{i}" for i in range(8)]
    for i in range(8):
        svc.storage.by_id[f"n{i}"] = _mem(f"n{i}", embedding=[1.0, 0.0])

    result = await svc.recall("ws", _recall_input(limit=3))
    assert len(result.memories) <= 3, f"recall returned more than limit: {len(result.memories)}"
