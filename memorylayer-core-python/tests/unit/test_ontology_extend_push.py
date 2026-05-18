"""Tests for the push (extend_ontology) escape hatch on DefaultOntologyService."""

import logging

import pytest

from memorylayer_server.services.ontology.base import BASE_ONTOLOGY
from memorylayer_server.services.ontology.default import DefaultOntologyService


def _meta(description: str, category: str = "test_cat") -> dict:
    return {
        "description": description,
        "symmetric": False,
        "transitive": False,
        "inverse": None,
        "category": category,
    }


def test_extend_ontology_basic_visibility():
    service = DefaultOntologyService(v=None)
    service.extend_ontology({"my_type": _meta("a test type")}, source="test")

    assert service.validate_relationship("my_type", "_default") is True
    info = service.get_relationship_info("my_type", "_default")
    assert info["description"] == "a test type"
    assert info["category"] == "test_cat"


def test_extend_ontology_lists_contributors_and_categories():
    service = DefaultOntologyService(v=None)
    service.extend_ontology({"my_type": _meta("desc")}, source="test")

    contributors = service.list_contributors()
    assert {"type_name": "my_type", "kind": "relationship", "source": "test"} in contributors

    categories = service.list_categories("_default")
    assert "test_cat" in categories


def test_extend_ontology_missing_field_raises():
    service = DefaultOntologyService(v=None)
    bad = {"bad_type": {"description": "x", "symmetric": False, "transitive": False, "inverse": None}}
    with pytest.raises(ValueError) as exc:
        service.extend_ontology(bad, source="test")
    assert "bad_type" in str(exc.value)
    assert "category" in str(exc.value)


def test_extend_ontology_collision_with_base_logs_warning(caplog):
    service = DefaultOntologyService(v=None)
    assert "parent_of" in BASE_ONTOLOGY

    new_meta = _meta("REPLACED parent_of", category="hierarchical")

    with caplog.at_level(logging.WARNING):
        service.extend_ontology({"parent_of": new_meta}, source="test_override")

    # WARNING was emitted referencing the override
    assert any("parent_of" in rec.message and "test_override" in rec.message for rec in caplog.records)

    # Merged ontology returns the new metadata, not the base value
    info = service.get_relationship_info("parent_of", "_default")
    assert info["description"] == "REPLACED parent_of"


def test_extend_ontology_empty_or_none_is_noop():
    service = DefaultOntologyService(v=None)
    service.extend_ontology(None, source="test")
    service.extend_ontology({}, source="test")
    assert service.list_contributors() == []
