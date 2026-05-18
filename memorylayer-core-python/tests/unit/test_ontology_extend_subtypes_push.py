"""Tests for the subtype push (extend_subtypes) escape hatch on DefaultOntologyService."""

import logging

import pytest

from memorylayer_server.services.ontology.default import DefaultOntologyService


def test_extend_subtypes_basic_visibility():
    service = DefaultOntologyService(v=None)
    service.extend_subtypes({"semantic": {"my_subtype"}}, source="test")

    assert service.validate_subtype("semantic", "my_subtype") is True
    assert service.validate_subtype("episodic", "my_subtype") is False
    assert "my_subtype" in service.list_subtypes("semantic")


def test_extend_subtypes_wildcard_applies_to_all_memory_types():
    service = DefaultOntologyService(v=None)
    service.extend_subtypes({"*": {"universal_subtype"}}, source="test")

    assert service.validate_subtype("semantic", "universal_subtype") is True
    assert service.validate_subtype("episodic", "universal_subtype") is True
    assert service.validate_subtype("procedural", "universal_subtype") is True


def test_extend_subtypes_lists_in_contributors():
    service = DefaultOntologyService(v=None)
    service.extend_subtypes({"semantic": {"alpha", "beta"}}, source="my_source")

    contributors = service.list_contributors()
    subtype_entries = [c for c in contributors if c.get("kind") == "subtype"]
    assert len(subtype_entries) == 2
    assert {"memory_type": "semantic", "subtype": "alpha", "kind": "subtype", "source": "my_source"} in subtype_entries
    assert {"memory_type": "semantic", "subtype": "beta", "kind": "subtype", "source": "my_source"} in subtype_entries


def test_list_subtypes_includes_oss_known():
    service = DefaultOntologyService(v=None)
    # OSS-known subtypes are visible without any contributions.
    all_subtypes = service.list_subtypes()
    assert "solution" in all_subtypes
    assert "preference" in all_subtypes
    assert "inference" in all_subtypes


def test_extend_subtypes_collision_with_oss_known_logs_warning(caplog):
    service = DefaultOntologyService(v=None)
    with caplog.at_level(logging.WARNING):
        service.extend_subtypes({"semantic": {"preference"}}, source="test_override")

    assert any("preference" in rec.message and "test_override" in rec.message for rec in caplog.records)


def test_extend_subtypes_validates_input():
    service = DefaultOntologyService(v=None)
    with pytest.raises(ValueError):
        service.extend_subtypes({"": {"x"}}, source="test")
    with pytest.raises(ValueError):
        service.extend_subtypes({"semantic": "not_a_set"}, source="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        service.extend_subtypes({"semantic": {""}}, source="test")


def test_extend_subtypes_empty_or_none_is_noop():
    service = DefaultOntologyService(v=None)
    service.extend_subtypes(None, source="test")
    service.extend_subtypes({}, source="test")
    # No subtype contributors recorded
    contributors = service.list_contributors()
    assert [c for c in contributors if c.get("kind") == "subtype"] == []
