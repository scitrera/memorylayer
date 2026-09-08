"""Unit tests for entity provenance/lineage summarization."""

from datetime import UTC, datetime

from memorylayer_server.models.entity_registry import Entity
from memorylayer_server.services.entity_registry.provenance import summarize_entity_provenance


def _entity(provenance, representative_memory_id=None):
    return Entity(
        id="e1", workspace_id="ws", entity_type="org", canonical_name="Acme", normalized_name="acme",
        provenance=provenance, representative_memory_id=representative_memory_id,
        created_at=datetime.now(UTC), updated_at=datetime.now(UTC),
    )


def test_created_and_enriched():
    p = summarize_entity_provenance(_entity(
        {
            "matched_via": "created",
            "created_at": "2026-01-01T00:00:00Z",
            "source_memory_id": "m1",
            "external_ids": {"wikidata": "Q95"},
            "external_links": {"wikidata": {"id": "Q95", "label": "Google", "description": "tech co", "url": "u", "score": 1.0}},
        },
        representative_memory_id="m9",
    ))
    assert p.origin == "created"
    assert p.source_memory_ids == ["m1", "m9"]
    assert p.external_links["wikidata"]["id"] == "Q95"
    assert p.description == "tech co"  # falls back to the external link's description
    assert p.generated_at == "2026-01-01T00:00:00Z"


def test_seeded_prefers_seed_description():
    p = summarize_entity_provenance(_entity({"source": "seed", "description": "a firm", "external_ids": {"wikidata": "Q1"}}))
    assert p.origin == "seeded" and p.description == "a firm"


def test_promoted():
    p = summarize_entity_provenance(_entity({"promoted_from": "concept", "promoted_source_memory_id": "m3"}))
    assert p.origin == "promoted" and p.source_memory_ids == ["m3"]


def test_matched_alias():
    assert summarize_entity_provenance(_entity({"matched_via": "alias"})).origin == "matched"


def test_empty_provenance_degrades():
    p = summarize_entity_provenance(_entity({}))
    assert p.origin == "unknown" and p.external_links == {} and p.source_memory_ids == []
    assert p.description is None and p.raw == {}
