"""Unit tests for entity relatedness (co-occurrence over shared member memories)."""

import types
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.entity_registry import Entity
from memorylayer_server.services.entity_registry.default import DefaultEntityRegistryService


def _entity(eid, status="active"):
    return Entity(
        id=eid, workspace_id="ws", entity_type="org", canonical_name=eid, normalized_name=eid,
        status=status, created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )


def _registry(rows, statuses=None):
    statuses = statuses or {}
    reg = DefaultEntityRegistryService.__new__(DefaultEntityRegistryService)

    async def list_workspace_entity_members(ws, role=None, limit=100000):
        return rows

    async def get(ws, eid):
        return _entity(eid, status=statuses.get(eid, "active"))

    reg._storage = types.SimpleNamespace(list_workspace_entity_members=list_workspace_entity_members)
    reg.get = get
    return reg


# e1 in m1,m2,m3; e2 shares m1,m2; e3 shares m3; e4 no overlap; e5 shares m1 but merged
_ROWS = [
    {"entity_id": "e1", "memory_id": "m1"}, {"entity_id": "e1", "memory_id": "m2"}, {"entity_id": "e1", "memory_id": "m3"},
    {"entity_id": "e2", "memory_id": "m1"}, {"entity_id": "e2", "memory_id": "m2"},
    {"entity_id": "e3", "memory_id": "m3"}, {"entity_id": "e4", "memory_id": "m9"}, {"entity_id": "e5", "memory_id": "m1"},
]


@pytest.mark.asyncio
async def test_related_ranks_by_overlap_and_scores():
    reg = _registry(_ROWS, statuses={"e5": "merged"})
    res = await reg.related_entities("ws", "e1", limit=10)
    assert [(r.entity.id, r.shared_memories, r.score) for r in res] == [
        ("e2", 2, round(2 / 3, 3)),
        ("e3", 1, round(1 / 3, 3)),
    ]  # e4 (no overlap) + e5 (merged) excluded


@pytest.mark.asyncio
async def test_related_min_shared_filter_and_limit():
    reg = _registry(_ROWS, statuses={"e5": "merged"})
    assert [r.entity.id for r in await reg.related_entities("ws", "e1", min_shared=2)] == ["e2"]
    assert len(await reg.related_entities("ws", "e1", limit=1)) == 1


@pytest.mark.asyncio
async def test_related_empty_when_no_members():
    reg = _registry(_ROWS)
    assert await reg.related_entities("ws", "zzz") == []


@pytest.mark.asyncio
async def test_cooccurrence_map_pairs_and_counts():
    # e1&e2 co-occur in m1,m2 (2); e1&e3 in m3 (1); e2&e5 in m1 (1); e1&e5 in m1 (1)
    reg = _registry(_ROWS)
    m = await reg.cooccurrence_map("ws")
    assert m["e1"]["e2"] == 2 and m["e2"]["e1"] == 2  # symmetric
    assert m["e1"]["e3"] == 1
    assert "e4" not in m  # m9 is a singleton memory → no pairs
    assert set(m["e1"]) == {"e2", "e3", "e5"}


@pytest.mark.asyncio
async def test_cooccurrence_map_min_shared_and_memory_cap():
    reg = _registry(_ROWS)
    strong = await reg.cooccurrence_map("ws", min_shared=2)
    assert strong == {"e1": {"e2": 2}, "e2": {"e1": 2}}  # only the m1+m2 pair survives
    # a memory over the cap contributes no pairs
    assert await reg.cooccurrence_map("ws", max_entities_per_memory=1) == {}
