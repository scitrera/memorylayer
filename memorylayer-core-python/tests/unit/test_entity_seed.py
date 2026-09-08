"""Unit tests for EntityRegistryService.seed_entities (curated catalog seeding)."""

import types

import pytest

from memorylayer_server.models.entity_registry import SeedEntity
from memorylayer_server.services.entity_registry.base import EntityRegistryService


class _SpyRegistry(EntityRegistryService):
    """Minimal registry that records upsert calls (seed_entities is concrete on the base)."""

    def __init__(self):
        self.calls = []

    async def resolve(self, *a, **k): ...
    async def get(self, *a, **k): ...
    async def find_by_alias(self, *a, **k): ...
    async def list_members(self, *a, **k): ...
    async def add_member(self, *a, **k): ...
    async def merge(self, *a, **k): ...

    async def upsert(self, ws, name, entity_type, *, aliases=None, confidence=1.0, provenance=None,
                     representative_memory_id=None):
        if name == "BOOM":
            raise RuntimeError("simulated upsert failure")
        self.calls.append(
            {"name": name, "entity_type": entity_type, "aliases": aliases,
             "confidence": confidence, "provenance": provenance}
        )
        return types.SimpleNamespace(id="e_" + name)


@pytest.mark.asyncio
async def test_seed_entities_batches_upsert_with_seed_provenance():
    reg = _SpyRegistry()
    seeds = [
        SeedEntity(name="Acme", entity_type="org", aliases=["ACME"], description="a firm",
                   external_ids={"wikidata": "Q1"}, confidence=0.9),
        SeedEntity(name="Widget", entity_type="project"),
    ]
    result = await reg.seed_entities("ws", seeds)
    assert result == {"seeded": 2, "failed": []}
    acme = next(c for c in reg.calls if c["name"] == "Acme")
    assert acme["entity_type"] == "org" and acme["aliases"] == ["ACME"] and acme["confidence"] == 0.9
    assert acme["provenance"] == {
        "activity": "entity.seed",
        "agent": "memorylayer",
        "source": "seed",
        "description": "a firm",
        "external_ids": {"wikidata": "Q1"},
    }


@pytest.mark.asyncio
async def test_seed_entities_isolates_per_row_failures():
    reg = _SpyRegistry()
    seeds = [
        SeedEntity(name="Ok1", entity_type="concept"),
        SeedEntity(name="BOOM", entity_type="person"),
        SeedEntity(name="Ok2", entity_type="concept"),
    ]
    result = await reg.seed_entities("ws", seeds)
    assert result == {"seeded": 2, "failed": ["BOOM"]}
    assert {c["name"] for c in reg.calls} == {"Ok1", "Ok2"}


@pytest.mark.asyncio
async def test_seed_entities_minimal_provenance_when_no_extras():
    reg = _SpyRegistry()
    await reg.seed_entities("ws", [SeedEntity(name="Bare", entity_type="concept")])
    assert reg.calls[0]["provenance"] == {"activity": "entity.seed", "agent": "memorylayer", "source": "seed"}
