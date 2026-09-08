"""Unit tests for the ontology entity-type vocabulary (the third dimension)."""

from memorylayer_server.services.ontology.default import DefaultOntologyService


class _FakeV:
    """Minimal Variables stand-in exposing get() + environ() for env-vocab tests."""

    def __init__(self, env: dict):
        self._env = env

    def get(self, key, default=None):
        return default

    def environ(self, key, default=None, **_):
        return self._env.get(key, default)


def test_base_vocabulary_and_ner_labels():
    svc = DefaultOntologyService(v=None)
    types = set(svc.list_entity_types())
    assert {"person", "org", "project", "place", "event", "concept"}.issubset(types)
    labels = svc.get_ner_labels()
    # core NER labels; concept has ner_label=None so it is NOT a NER label
    assert "organization" in labels and "person" in labels and "location" in labels
    assert "concept" not in labels
    rev = svc.ner_label_to_entity_type()
    assert rev["organization"] == "org" and rev["location"] == "place"


def test_validate_and_info():
    svc = DefaultOntologyService(v=None)
    assert svc.validate_entity_type("person") and not svc.validate_entity_type("nope")
    assert svc.get_entity_type_info("org")["ner_label"] == "organization"
    assert svc.get_entity_type_info("nope") is None


def test_extend_entity_types_push():
    svc = DefaultOntologyService(v=None)
    svc.extend_entity_types(
        {"equipment": {"ner_label": "equipment", "description": "a device"}}, source="test"
    )
    assert "equipment" in svc.list_entity_types()
    assert "equipment" in svc.get_ner_labels()
    assert svc.ner_label_to_entity_type()["equipment"] == "equipment"


def test_extend_entity_types_validates_metadata():
    svc = DefaultOntologyService(v=None)
    import pytest

    with pytest.raises(ValueError):
        svc.extend_entity_types({"bad": {"description": "no ner_label field"}})


def test_env_vocabulary_loading():
    # bare "type" -> ner_label = type; "type:label" -> explicit; "type:" -> non-NER
    v = _FakeV({"MEMORYLAYER_ENTITY_TYPES": "equipment,chemical:chemical,standard:standard,acronym:"})
    svc = DefaultOntologyService(v=v)
    types = set(svc.list_entity_types())
    assert {"equipment", "chemical", "standard", "acronym"}.issubset(types)
    labels = svc.get_ner_labels()
    assert "equipment" in labels and "chemical" in labels and "standard" in labels
    assert "acronym" not in labels  # empty ner_label -> not a NER label
