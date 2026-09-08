"""Unit tests for the flag-gated registry-backed entity-expansion recall channel.

The registry recall channel (``_fuse_registry_results``) is the entity registry's
RETRIEVAL consumer (and the future AGE ``Entity``/``MENTIONS`` consumer:
query-entity -> canonical entity -> member-memories). It resolves the query's
entities to canonical registry entities with ``allow_create=False`` (a query
string must NEVER create an entity), pulls each entity's MEMBER memories — which
include memories linked via aliases and fuzzy/LLM-merged surface forms — ranks
them by vector similarity, and RRF-fuses them as an ADDITIVE arm.

The value over the metadata-string entity-anchor arm is exercised directly: a
member memory whose content does NOT contain the query string verbatim (linked
via an alias) is promoted into the fused top-k. The arm is mock-free on the
registry side (a small fake registry + fake storage) so the fusion — the real
deliverable — is tested without an LLM or embedding server.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.models.entity_registry import Entity, EntityResolution, EntityType, Member
from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType,
    RecallInput,
    RecallMode,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.config import (
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.services.memory.base import (
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
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
    # Isolate the registry channel: keep the other fusion channels off unless a
    # test opts in.
    v.set(MEMORYLAYER_HYBRID_SEARCH_ENABLED, False)
    v.set(MEMORYLAYER_ENTITY_ANCHOR_ENABLED, False)
    v.set(MEMORYLAYER_FACT_CHANNEL_ENABLED, False)
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
    """In-test storage: a deterministic primary vector arm + get_memory by id.

    ``primary`` is the (memory, base_score) list returned by the vector arm.
    ``by_id`` maps memory id -> Memory for the registry channel's member fetch
    (``get_memory``). Member memories live ONLY in ``by_id`` (not the primary
    arm) so a promoted member is unambiguously the registry channel's doing.
    """

    def __init__(self):
        self.primary: list[tuple[Memory, float]] = []
        self.by_id: dict[str, Memory] = {}
        self.get_memory_calls: list[tuple[str, str]] = []

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
    """Minimal EntityRegistryService stand-in for the recall channel.

    ``entities_by_name`` maps a (normalized) surface name -> Entity (the canonical
    node it resolves to). ``members`` maps entity_id -> list[Member]. ``resolve``
    records its calls (so a test can assert allow_create=False) and raises
    LookupError on a miss with allow_create=False, mirroring the real default
    service contract.
    """

    def __init__(self):
        self.entities_by_name: dict[str, Entity] = {}
        self.members: dict[str, list[Member]] = {}
        self.resolve_calls: list[dict] = []
        self.list_members_calls: list[dict] = []

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
        raise AssertionError("registry recall channel must never create entities (allow_create=True)")

    async def list_members(self, workspace_id, entity_id, *, role=None, limit=100):
        self.list_members_calls.append({"entity_id": entity_id, "role": role, "limit": limit})
        return self.members.get(entity_id, [])[:limit]


def _make_service(v: Variables, registry=None) -> MemoryService:
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
        v=v,
    )
    # No reranker -> _apply_reranking is a plain top-limit slice (deterministic).
    svc.reranker_service = None
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
# Unit: _fuse_registry_results behavior
# ===========================================================================


@pytest.mark.asyncio
async def test_alias_linked_member_promoted_into_topk():
    """A member whose text lacks the query string verbatim (alias-linked) is promoted.

    This is the value-add over the metadata entity-anchor arm: the gold member's
    content does NOT contain "Caroline" — it was linked to the canonical entity
    via an alias/merge during accretion. The registry channel resolves the query
    entity to that canonical entity and pulls the member regardless of its text.
    """
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry)

    # Primary vector arm: 5 turns, the gold member is NOT here.
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(5)]

    # Gold member: content has NO "Caroline" (linked via alias), highly aligned to
    # the query embedding [1.0, 0.0] so it ranks #1 within the member set.
    gold = _mem("gold_member", content="She moved to Stockholm last spring", embedding=[1.0, 0.0])
    svc.storage.by_id["gold_member"] = gold

    # Registry: "Caroline" resolves to a canonical entity whose members include gold.
    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    registry.members[ent.id] = [Member(entity_id=ent.id, memory_id="gold_member")]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert "gold_member" in ids, f"alias-linked member not promoted into top-3: {ids}"
    gold_out = next(m for m in result.memories if m.id == "gold_member")
    assert "registry" in (gold_out.match_signals or [])
    # The query string never appears in the promoted member's content — proving the
    # promotion came from the registry edge, not metadata/text matching.
    assert "Caroline" not in gold_out.content


@pytest.mark.asyncio
async def test_resolve_called_with_allow_create_false():
    """The channel must NEVER create an entity from a query string."""
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    await svc.recall("ws", _recall_input(limit=3))

    assert registry.resolve_calls, "resolve was never called for the query entity"
    assert all(c["allow_create"] is False for c in registry.resolve_calls), registry.resolve_calls


@pytest.mark.asyncio
async def test_noop_when_query_entity_does_not_resolve():
    """A query entity that resolves to nothing (allow_create=False miss) is a no-op."""
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    registry = _FakeRegistry()  # empty: every resolve misses -> LookupError
    svc = _make_service(v, registry=registry)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("registry" not in (m.match_signals or []) for m in result.memories)
    # resolve was attempted (and missed via LookupError, allow_create=False).
    assert registry.resolve_calls
    assert all(c["allow_create"] is False for c in registry.resolve_calls)


@pytest.mark.asyncio
async def test_noop_when_flag_off():
    """Flag off: the registry is never consulted and members never surface."""
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, False)
    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]
    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    registry.members[ent.id] = [Member(entity_id=ent.id, memory_id="gold_member")]
    svc.storage.by_id["gold_member"] = _mem("gold_member", embedding=[1.0, 0.0])

    result = await svc.recall("ws", _recall_input(limit=5))
    ids = [m.id for m in result.memories]
    assert "gold_member" not in ids
    assert all("registry" not in (m.match_signals or []) for m in result.memories)
    assert registry.resolve_calls == []


@pytest.mark.asyncio
async def test_noop_when_no_registry_service():
    """Flag on but no registry service wired: inert (degrades to vector results)."""
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    svc = _make_service(v, registry=None)
    svc.storage.primary = [(_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(3)]

    result = await svc.recall("ws", _recall_input(limit=3))
    ids = [m.id for m in result.memories]
    assert ids == ["turn0", "turn1", "turn2"]
    assert all("registry" not in (m.match_signals or []) for m in result.memories)


@pytest.mark.asyncio
async def test_flag_off_byte_identical_to_no_registry_code():
    """Regression: flag OFF leaves recall output byte-identical to pre-channel behavior.

    Two services over identical fixtures — one with the flag explicitly OFF, one at
    the default (also OFF, representing pre-channel config) — must produce the same
    serialized result even when the registry holds a member that WOULD be promoted
    with the flag on. Mirrors the fact-channel golden test.
    """
    frozen = datetime(2024, 1, 1, tzinfo=UTC)

    def _frozen_mem(mid: str, content: str = "turn content", embedding=None) -> Memory:
        m = _mem(mid, content=content, embedding=embedding)
        m.created_at = frozen
        m.updated_at = frozen
        return m

    def _build(flag_value: bool | None):
        if flag_value is None:
            v = _make_v()  # default (OFF)
        else:
            v = _make_v()
            v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, flag_value)
        registry = _FakeRegistry()
        ent = registry._entity("Caroline")
        registry.entities_by_name["Caroline"] = ent
        registry.members[ent.id] = [Member(entity_id=ent.id, memory_id="gold_member")]
        svc = _make_service(v, registry=registry)
        svc.storage.primary = [(_frozen_mem(f"turn{i}", embedding=[0.1, 0.9]), 0.9 - i * 0.1) for i in range(4)]
        svc.storage.by_id["gold_member"] = _frozen_mem("gold_member", embedding=[1.0, 0.0])
        return svc

    svc_off = _build(False)
    svc_default = _build(None)

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
    # And neither surfaces the registry member nor a "registry" signal.
    assert all(m.id != "gold_member" for m in r_off.memories)
    assert all("registry" not in (m.match_signals or []) for m in r_off.memories)


@pytest.mark.asyncio
async def test_list_members_called_with_mention_role():
    """list_members is always called with role='mention' (role pinning).

    Accretion only writes 'mention' members today. Pinning the role prevents a
    future role from silently becoming a positive expansion candidate.
    """
    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    ent = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent
    registry.members[ent.id] = [Member(entity_id=ent.id, memory_id="m1")]
    svc.storage.by_id["m1"] = _mem("m1", embedding=[1.0, 0.0])

    await svc.recall("ws", _recall_input(limit=3))

    assert registry.list_members_calls, "list_members was never called"
    assert all(c["role"] == "mention" for c in registry.list_members_calls), (
        f"list_members called without role='mention': {registry.list_members_calls}"
    )


@pytest.mark.asyncio
async def test_per_entity_sub_cap_prevents_flooding():
    """Per-entity sub-cap distributes the pool fairly across resolved entities.

    Two canonical entities (A and B) each have many members. With pool=4 and 2
    entities, per_entity = ceil(4/2) = 2, so each entity contributes at most 2
    members to the candidate set — the overall pool stays at 4 and neither
    entity monopolises the slot budget.
    """
    import math

    v = _make_v()
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED, True)
    # Small pool (4) with two entities -> per_entity = ceil(4/2) = 2.
    from memorylayer_server.config import MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL
    v.set(MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL, 4)

    registry = _FakeRegistry()
    svc = _make_service(v, registry=registry)
    svc.storage.primary = [(_mem("turn0", embedding=[0.1, 0.9]), 0.9)]

    # Entity A: 5 members (only first 2 should be fetched due to sub-cap).
    ent_a = registry._entity("Caroline")
    registry.entities_by_name["Caroline"] = ent_a
    registry.members[ent_a.id] = [
        Member(entity_id=ent_a.id, memory_id=f"a{i}") for i in range(5)
    ]

    # Entity B: 5 members (only first 2 should be fetched due to sub-cap).
    ent_b = registry._entity("Alice")
    registry.entities_by_name["Alice"] = ent_b
    registry.members[ent_b.id] = [
        Member(entity_id=ent_b.id, memory_id=f"b{i}") for i in range(5)
    ]

    for i in range(5):
        svc.storage.by_id[f"a{i}"] = _mem(f"a{i}", embedding=[0.9, 0.1])
        svc.storage.by_id[f"b{i}"] = _mem(f"b{i}", embedding=[0.8, 0.2])

    await svc.recall("ws", _recall_input(limit=10, query="What did Caroline and Alice do"))

    assert registry.list_members_calls, "list_members was never called"
    n_entities = len({c["entity_id"] for c in registry.list_members_calls})
    expected_per_entity = max(1, math.ceil(4 / n_entities))
    for call in registry.list_members_calls:
        assert call["limit"] == expected_per_entity, (
            f"Expected per-entity limit {expected_per_entity}, got {call['limit']} "
            f"(pool=4, n_entities={n_entities})"
        )
