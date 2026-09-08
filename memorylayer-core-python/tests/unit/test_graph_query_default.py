"""Unit tests for the relational (default) GraphQueryService on SQLite (P2 Track A).

DB-free of any graph database: exercises ``RelationalGraphQueryService``
against a real on-disk SQLite backend over a small fixture graph, covering all
five public methods (neighbors depth 1/2, k_hop, shortest_path, typed_pattern,
relationship_rollup). These also lock the OSS "golden" semantics that the
enterprise AGE conformance suite compares against.
"""

import pytest

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.graph_query.default import RelationalGraphQueryService
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend

from datetime import UTC, datetime


@pytest.fixture
async def sqlite_backend(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "test_graph_query.db"))
    await backend.connect()
    yield backend
    await backend.disconnect()


def _svc(storage) -> RelationalGraphQueryService:
    from scitrera_app_framework import Variables
    return RelationalGraphQueryService(storage=storage, v=Variables())


async def _ensure_ws(storage, ws_id: str) -> None:
    if not await storage.get_workspace(ws_id):
        await storage.create_workspace(
            Workspace(
                id=ws_id,
                tenant_id="default_tenant",
                name="GQ test %s" % ws_id,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )


async def _add_mem(storage, ws_id: str, content: str) -> str:
    mem = await storage.create_memory(ws_id, RememberInput(content=content))
    return mem.id


async def _add_assoc(storage, ws_id, src, tgt, rel="related_to", strength=0.7):
    await storage.create_association(
        ws_id, AssociateInput(source_id=src, target_id=tgt, relationship=rel, strength=strength)
    )


async def _build_chain(storage, ws_id: str) -> list[str]:
    """m0 - m1 - m2 - m3 line + a m1->m4 branch.

    Edges (stored direction):
      m0 -related_to-> m1 (0.9)
      m1 -leads_to->   m2 (0.8)
      m2 -leads_to->   m3 (0.7)
      m1 -similar_to-> m4 (0.5)
    """
    await _ensure_ws(storage, ws_id)
    ids = [await _add_mem(storage, ws_id, "node %d" % i) for i in range(5)]
    await _add_assoc(storage, ws_id, ids[0], ids[1], "related_to", 0.9)
    await _add_assoc(storage, ws_id, ids[1], ids[2], "leads_to", 0.8)
    await _add_assoc(storage, ws_id, ids[2], ids[3], "leads_to", 0.7)
    await _add_assoc(storage, ws_id, ids[1], ids[4], "similar_to", 0.5)
    return ids


@pytest.mark.asyncio
async def test_neighbors_depth1(sqlite_backend):
    ws = "gq_neigh1"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.neighbors(ws, ids[1], depth=1)
    node_ids = {n.memory_id for n in res.nodes}
    # m1's direct neighbors: m0, m2, m4 (+ m1 itself).
    assert node_ids == {ids[0], ids[1], ids[2], ids[4]}
    assert res.root_id == ids[1]
    assert res.truncated is False
    # Induced edges among {m0,m1,m2,m4}: m0-m1, m1-m2, m1-m4 (not m2-m3).
    edge_pairs = {(e.source_id, e.target_id) for e in res.edges}
    assert (ids[0], ids[1]) in edge_pairs
    assert (ids[1], ids[2]) in edge_pairs
    assert (ids[1], ids[4]) in edge_pairs
    assert (ids[2], ids[3]) not in edge_pairs


@pytest.mark.asyncio
async def test_neighbors_depth2(sqlite_backend):
    ws = "gq_neigh2"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.neighbors(ws, ids[0], depth=2)
    node_ids = {n.memory_id for n in res.nodes}
    # 2 hops from m0: m0 -> m1 -> {m2, m4}. m3 is 3 hops away, excluded.
    assert node_ids == {ids[0], ids[1], ids[2], ids[4]}


@pytest.mark.asyncio
async def test_neighbors_limit_truncates(sqlite_backend):
    ws = "gq_trunc"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.neighbors(ws, ids[1], depth=1, limit=2)
    assert res.truncated is True
    assert len(res.nodes) == 2
    # Root is always retained.
    assert any(n.memory_id == ids[1] for n in res.nodes)


@pytest.mark.asyncio
async def test_neighbors_relationship_filter(sqlite_backend):
    ws = "gq_relfilter"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.neighbors(ws, ids[1], depth=1, relationship_types=["similar_to"])
    node_ids = {n.memory_id for n in res.nodes}
    # Only the similar_to edge m1-m4 qualifies.
    assert node_ids == {ids[1], ids[4]}
    assert all(e.relationship == "similar_to" for e in res.edges)


@pytest.mark.asyncio
async def test_k_hop_subgraph(sqlite_backend):
    ws = "gq_khop"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.k_hop_subgraph(ws, [ids[0], ids[3]], depth=1)
    node_ids = {n.memory_id for n in res.nodes}
    # 1 hop from m0: m1. 1 hop from m3: m2. Plus roots.
    assert node_ids == {ids[0], ids[1], ids[2], ids[3]}
    assert res.root_ids == sorted({ids[0], ids[3]})


@pytest.mark.asyncio
async def test_shortest_path(sqlite_backend):
    ws = "gq_path"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.shortest_path(ws, ids[0], ids[3], max_hops=5)
    assert res.found is True
    assert res.hops == 3
    assert [n.memory_id for n in res.nodes] == [ids[0], ids[1], ids[2], ids[3]]
    assert len(res.edges) == 3


@pytest.mark.asyncio
async def test_shortest_path_self(sqlite_backend):
    ws = "gq_path_self"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    res = await svc.shortest_path(ws, ids[2], ids[2])
    assert res.found is True
    assert res.hops == 0
    assert len(res.nodes) == 1
    assert res.edges == []


@pytest.mark.asyncio
async def test_shortest_path_unreachable(sqlite_backend):
    ws = "gq_path_none"
    await _ensure_ws(sqlite_backend, ws)
    a = await _add_mem(sqlite_backend, ws, "lonely a")
    b = await _add_mem(sqlite_backend, ws, "lonely b")
    svc = _svc(sqlite_backend)

    res = await svc.shortest_path(ws, a, b, max_hops=3)
    assert res.found is False
    assert res.hops == 0
    assert res.nodes == []


@pytest.mark.asyncio
async def test_typed_pattern(sqlite_backend):
    ws = "gq_typed"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    matches = await svc.typed_pattern(ws, relationship="leads_to")
    pairs = {(m.source_id, m.target_id) for m in matches}
    assert pairs == {(ids[1], ids[2]), (ids[2], ids[3])}
    assert all(m.relationship == "leads_to" for m in matches)


@pytest.mark.asyncio
async def test_typed_pattern_limit(sqlite_backend):
    ws = "gq_typed_limit"
    ids = await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    matches = await svc.typed_pattern(ws, relationship="leads_to", limit=1)
    assert len(matches) == 1


@pytest.mark.asyncio
async def test_relationship_rollup(sqlite_backend):
    ws = "gq_rollup"
    await _build_chain(sqlite_backend, ws)
    svc = _svc(sqlite_backend)

    rollups = await svc.relationship_rollup(ws)
    counts = {r.relationship: r.count for r in rollups}
    assert counts == {"related_to": 1, "leads_to": 2, "similar_to": 1}
    # Sorted by count desc, relationship asc.
    assert rollups[0].relationship == "leads_to"
    assert rollups[0].count == 2


@pytest.mark.asyncio
async def test_storage_count_associations_by_relationship(sqlite_backend):
    """The new storage primitive is a single GROUP BY aggregate."""
    ws = "gq_count"
    await _build_chain(sqlite_backend, ws)
    counts = await sqlite_backend.count_associations_by_relationship(ws)
    assert counts == {"related_to": 1, "leads_to": 2, "similar_to": 1}


# ---------------------------------------------------------------------------
# entity_neighborhood (OSS relational fallback)
# ---------------------------------------------------------------------------


async def _build_registry(storage, ws_id: str) -> dict[str, str]:
    """Three entities + member edges over four memories.

    alice (PERSON): self->m1, mention->m2
    bob   (PERSON): mention->m2          (co-mention with alice via m2)
    proj  (CONCEPT): mention->m1, m3     (co-mention with alice via m1)
    Returns {name: entity_id} plus {"m0".."m3": memory_id}.
    """
    await _ensure_ws(storage, ws_id)
    mems = {f"m{i}": await _add_mem(storage, ws_id, "reg node %d" % i) for i in range(4)}
    alice = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "person", "canonical_name": "Alice", "normalized_name": "alice"}
    )
    bob = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "person", "canonical_name": "Bob", "normalized_name": "bob"}
    )
    proj = await storage.store_entity(
        {"workspace_id": ws_id, "entity_type": "concept", "canonical_name": "Proj", "normalized_name": "proj"}
    )
    await storage.add_entity_member(ws_id, alice["id"], mems["m1"], role="self")
    await storage.add_entity_member(ws_id, alice["id"], mems["m2"], role="mention")
    await storage.add_entity_member(ws_id, bob["id"], mems["m2"], role="mention")
    await storage.add_entity_member(ws_id, proj["id"], mems["m1"], role="mention")
    await storage.add_entity_member(ws_id, proj["id"], mems["m3"], role="mention")
    return {"alice": alice["id"], "bob": bob["id"], "proj": proj["id"], **mems}


@pytest.mark.asyncio
async def test_entity_neighborhood_memories_and_co_mentions(sqlite_backend):
    ws = "gq_en_basic"
    r = await _build_registry(sqlite_backend, ws)
    res = await _svc(sqlite_backend).entity_neighborhood(ws, r["alice"])

    assert res.entity_id == r["alice"]
    # Alice mentions m1 (self) + m2 (mention).
    assert {(m.memory_id, m.role) for m in res.memories_for_entity} == {
        (r["m1"], "self"),
        (r["m2"], "mention"),
    }
    # Co-mentioned: bob (shares m2) + proj (shares m1), each count 1.
    co = {c.entity_id: c for c in res.entities_co_mentioned}
    assert set(co) == {r["bob"], r["proj"]}
    assert co[r["bob"]].shared_memory_count == 1
    assert co[r["proj"]].shared_memory_count == 1
    assert co[r["proj"]].label == "Proj"
    assert co[r["proj"]].entity_type == "concept"
    assert res.truncated is False


@pytest.mark.asyncio
async def test_entity_neighborhood_co_mention_ranking(sqlite_backend):
    """A co-entity sharing TWO memories outranks one sharing a single memory."""
    ws = "gq_en_rank"
    await _ensure_ws(sqlite_backend, ws)
    m = [await _add_mem(sqlite_backend, ws, "rk %d" % i) for i in range(3)]
    a = await sqlite_backend.store_entity(
        {"workspace_id": ws, "entity_type": "person", "canonical_name": "A", "normalized_name": "a"}
    )
    heavy = await sqlite_backend.store_entity(
        {"workspace_id": ws, "entity_type": "person", "canonical_name": "Heavy", "normalized_name": "heavy"}
    )
    light = await sqlite_backend.store_entity(
        {"workspace_id": ws, "entity_type": "person", "canonical_name": "Light", "normalized_name": "light"}
    )
    for mid in (m[0], m[1], m[2]):
        await sqlite_backend.add_entity_member(ws, a["id"], mid, role="mention")
    # heavy shares m0+m1 (2); light shares m2 (1).
    await sqlite_backend.add_entity_member(ws, heavy["id"], m[0], role="mention")
    await sqlite_backend.add_entity_member(ws, heavy["id"], m[1], role="mention")
    await sqlite_backend.add_entity_member(ws, light["id"], m[2], role="mention")

    res = await _svc(sqlite_backend).entity_neighborhood(ws, a["id"])
    assert [c.entity_id for c in res.entities_co_mentioned] == [heavy["id"], light["id"]]
    assert res.entities_co_mentioned[0].shared_memory_count == 2
    assert res.entities_co_mentioned[1].shared_memory_count == 1


@pytest.mark.asyncio
async def test_entity_neighborhood_truncation(sqlite_backend):
    ws = "gq_en_trunc"
    r = await _build_registry(sqlite_backend, ws)
    res = await _svc(sqlite_backend).entity_neighborhood(ws, r["alice"], memory_limit=1)
    assert res.truncated is True
    assert len(res.memories_for_entity) == 1


@pytest.mark.asyncio
async def test_entity_neighborhood_unknown_entity_is_empty(sqlite_backend):
    ws = "gq_en_empty"
    await _build_registry(sqlite_backend, ws)
    res = await _svc(sqlite_backend).entity_neighborhood(ws, "ent_does_not_exist")
    assert res.entity_id == "ent_does_not_exist"
    assert res.memories_for_entity == []
    assert res.entities_co_mentioned == []
    assert res.truncated is False


@pytest.mark.asyncio
async def test_entity_neighborhood_empty_registry_no_error(sqlite_backend):
    """A workspace with NO registry rows yields an empty neighborhood, no error."""
    ws = "gq_en_no_registry"
    await _ensure_ws(sqlite_backend, ws)
    res = await _svc(sqlite_backend).entity_neighborhood(ws, "anything")
    assert res.memories_for_entity == []
    assert res.entities_co_mentioned == []


@pytest.mark.asyncio
async def test_list_workspace_entities_and_members(sqlite_backend):
    """The new workspace-scoped listing primitives enumerate the full registry."""
    ws = "gq_en_list"
    r = await _build_registry(sqlite_backend, ws)
    entities = await sqlite_backend.list_workspace_entities(ws)
    assert {e["id"] for e in entities} == {r["alice"], r["bob"], r["proj"]}
    # Deterministic id ordering.
    assert [e["id"] for e in entities] == sorted(e["id"] for e in entities)

    members = await sqlite_backend.list_workspace_entity_members(ws)
    assert len(members) == 5  # 2 alice + 1 bob + 2 proj
    self_members = await sqlite_backend.list_workspace_entity_members(ws, role="self")
    assert len(self_members) == 1 and self_members[0]["memory_id"] == r["m1"]


@pytest.mark.asyncio
async def test_entity_neighborhood_multi_role_dedup(sqlite_backend):
    """MAJOR-1: when one memory is mentioned under two roles, only the
    lexicographically-first role survives in memories_for_entity.
    The list_workspace_entity_members ORDER BY (entity_id, memory_id, role)
    puts 'mention' before 'self', so 'mention' must win.
    """
    ws = "gq_en_multirole"
    await _ensure_ws(sqlite_backend, ws)
    mem_id = await _add_mem(sqlite_backend, ws, "dual-role node")
    alice = await sqlite_backend.store_entity(
        {"workspace_id": ws, "entity_type": "person",
         "canonical_name": "Alice", "normalized_name": "alice"}
    )
    await sqlite_backend.add_entity_member(ws, alice["id"], mem_id, role="mention")
    await sqlite_backend.add_entity_member(ws, alice["id"], mem_id, role="self")

    res = await _svc(sqlite_backend).entity_neighborhood(ws, alice["id"])
    # Must deduplicate: only one entry per memory_id.
    assert len(res.memories_for_entity) == 1
    # "mention" < "self" lexicographically, so it wins the setdefault.
    assert res.memories_for_entity[0].memory_id == mem_id
    assert res.memories_for_entity[0].role == "mention"


@pytest.mark.asyncio
async def test_entity_neighborhood_members_cap_sets_truncated(sqlite_backend):
    """MINOR-3: truncated=True when list_workspace_entity_members returns exactly
    its limit (meaning more rows exist that were silently omitted).

    The production cap is 100000. We simulate hitting it by mocking the storage
    method to return a list whose length equals the cap value it was called with,
    which is exactly the condition the production code tests.
    """
    from unittest.mock import AsyncMock, patch

    ws = "gq_en_cap"
    await _ensure_ws(sqlite_backend, ws)
    alice = await sqlite_backend.store_entity(
        {"workspace_id": ws, "entity_type": "person",
         "canonical_name": "Alice", "normalized_name": "alice"}
    )
    mid = await _add_mem(sqlite_backend, ws, "cap node")
    await sqlite_backend.add_entity_member(ws, alice["id"], mid, role="mention")

    # The production cap is 100000; mock returns exactly that many rows
    # (filled with the one real row padded — but length is what matters here).
    real_row = {"entity_id": alice["id"], "memory_id": mid, "role": "mention", "confidence": 1.0}
    cap = 100000
    fake_members = [real_row] * cap  # length == cap → truncated fires

    with patch.object(
        sqlite_backend, "list_workspace_entity_members",
        new=AsyncMock(return_value=fake_members),
    ):
        res = await _svc(sqlite_backend).entity_neighborhood(ws, alice["id"])

    assert res.truncated is True


# ---------------------------------------------------------------------------
# fragments_for_memory — OSS relational fallback (parity golden)
# ---------------------------------------------------------------------------


async def _add_fact(storage, ws_id: str, content: str, source_id: str) -> str:
    """Store a decomposed-fact memory (subtype="fact", metadata source_id)."""
    from memorylayer_server.models.memory import MemorySubtype, MemoryType

    mem = await storage.create_memory(
        ws_id,
        RememberInput(
            content=content,
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.FACT.value,
            metadata={"kind": "fact", "source_id": source_id},
        ),
    )
    return mem.id


@pytest.mark.asyncio
async def test_fragments_for_memory_returns_derived_facts(sqlite_backend):
    ws = "gq_frag_basic"
    await _ensure_ws(sqlite_backend, ws)
    src1 = await _add_mem(sqlite_backend, ws, "source one")
    src2 = await _add_mem(sqlite_backend, ws, "source two")
    f1 = await _add_fact(sqlite_backend, ws, "fact a of src1", src1)
    f2 = await _add_fact(sqlite_backend, ws, "fact b of src1", src1)
    await _add_fact(sqlite_backend, ws, "fact a of src2", src2)

    res = await _svc(sqlite_backend).fragments_for_memory(ws, src1)
    assert res.source_id == src1
    assert {f.fragment_id for f in res.fragments} == {f1, f2}
    # source_id on every returned fragment is the queried memory.
    assert all(f.source_id == src1 for f in res.fragments)
    # Deterministic order by fragment_id asc.
    assert [f.fragment_id for f in res.fragments] == sorted([f1, f2])
    assert res.truncated is False


@pytest.mark.asyncio
async def test_fragments_for_memory_empty_when_no_facts(sqlite_backend):
    ws = "gq_frag_empty"
    await _ensure_ws(sqlite_backend, ws)
    src = await _add_mem(sqlite_backend, ws, "no facts here")

    res = await _svc(sqlite_backend).fragments_for_memory(ws, src)
    assert res.fragments == []
    assert res.truncated is False


@pytest.mark.asyncio
async def test_fragments_for_memory_truncates(sqlite_backend):
    ws = "gq_frag_trunc"
    await _ensure_ws(sqlite_backend, ws)
    src = await _add_mem(sqlite_backend, ws, "source")
    for i in range(5):
        await _add_fact(sqlite_backend, ws, "fact %d" % i, src)

    res = await _svc(sqlite_backend).fragments_for_memory(ws, src, limit=3)
    assert len(res.fragments) == 3
    assert res.truncated is True
