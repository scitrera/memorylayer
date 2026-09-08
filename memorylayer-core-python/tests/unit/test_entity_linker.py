"""Unit tests for the entity linker seam + registry enrichment."""

import types

import pytest

from memorylayer_server.models.entity_registry import ExternalEntityLink
from memorylayer_server.services.entity_linker.default import DefaultEntityLinkerService
from memorylayer_server.services.entity_registry.default import DefaultEntityRegistryService


@pytest.mark.asyncio
async def test_default_linker_is_noop():
    assert await DefaultEntityLinkerService().link("Acme", "org") is None


def _entity(eid, name, etype, provenance=None):
    return types.SimpleNamespace(
        id=eid, canonical_name=name,
        entity_type=types.SimpleNamespace(value=etype),
        provenance=provenance or {},
    )


def _registry(entities, updates_sink):
    reg = DefaultEntityRegistryService.__new__(DefaultEntityRegistryService)
    reg.logger = types.SimpleNamespace(debug=lambda *a, **k: None, info=lambda *a, **k: None)

    async def list_entities(ws, status="active", limit=100):
        return entities

    class _Store:
        async def update_entity(self, ws, eid, **kw):
            updates_sink[eid] = kw

    reg.list_entities = list_entities
    reg._storage = _Store()
    return reg


class _Linker:
    async def link(self, name, entity_type):
        if name == "Google":
            return ExternalEntityLink(
                source="wikidata", external_id="Q95", label="Google",
                description="tech company", url="u", score=1.0,
            )
        return None


@pytest.mark.asyncio
async def test_enrich_links_eligible_dedupes_and_writes_provenance():
    updates = {}
    entities = [
        _entity("e1", "Google", "org"),
        _entity("e2", "Widget", "project"),  # not eligible
        _entity("e3", "Acme", "org", {"external_ids": {"wikidata": "Q1"}}),  # already linked
    ]
    reg = _registry(entities, updates)
    result = await reg.enrich_entities(
        "ws", _Linker(), eligible_types={"org", "person", "place"}, source="wikidata"
    )
    assert result == {"checked": 2, "enriched": 1, "skipped": 1}
    prov = updates["e1"]["provenance"]
    assert prov["external_ids"]["wikidata"] == "Q95"
    assert prov["external_links"]["wikidata"]["label"] == "Google"
    assert "e2" not in updates and "e3" not in updates


@pytest.mark.asyncio
async def test_enrich_none_linker_is_noop():
    reg = _registry([_entity("e1", "Google", "org")], {})
    assert await reg.enrich_entities("ws", None) == {"checked": 0, "enriched": 0, "skipped": 0}


@pytest.mark.asyncio
async def test_enrich_overwrite_relinks_existing():
    updates = {}
    reg = _registry([_entity("e1", "Google", "org", {"external_ids": {"wikidata": "Qold"}})], updates)
    result = await reg.enrich_entities("ws", _Linker(), eligible_types={"org"}, overwrite=True)
    assert result == {"checked": 1, "enriched": 1, "skipped": 0}
    assert updates["e1"]["provenance"]["external_ids"]["wikidata"] == "Q95"
