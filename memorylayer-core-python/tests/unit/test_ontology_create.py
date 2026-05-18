"""Tests for DefaultOntologyService.create_ontology and the layered merge."""

import pytest

from memorylayer_server.services.ontology.base import BASE_ONTOLOGY
from memorylayer_server.services.ontology.default import DefaultOntologyService


def _meta(description: str, category: str = "custom") -> dict:
    return {
        "description": description,
        "symmetric": False,
        "transitive": False,
        "inverse": None,
        "category": category,
    }


def test_create_ontology_tenant_scoped_isolation():
    service = DefaultOntologyService(v=None)
    service.create_ontology(
        tenant_id="t1",
        name="custom1",
        relationships={"my_type": _meta("tenant1 type")},
    )

    assert service.validate_relationship("my_type", "t1") is True
    with pytest.raises(ValueError):
        service.validate_relationship("my_type", "t2")


def test_create_ontology_workspace_scoped_isolation():
    service = DefaultOntologyService(v=None)
    service.create_ontology(
        tenant_id="t1",
        workspace_id="w1",
        name="ws1",
        relationships={"ws_type": _meta("ws1 type")},
    )

    assert service.validate_relationship("ws_type", "t1", "w1") is True
    with pytest.raises(ValueError):
        service.validate_relationship("ws_type", "t1", None)
    with pytest.raises(ValueError):
        service.validate_relationship("ws_type", "t1", "w2")


def test_create_ontology_validation_errors():
    service = DefaultOntologyService(v=None)

    with pytest.raises(ValueError):
        service.create_ontology(tenant_id="t1", name="", relationships={"x": _meta("x")})

    bad = {"x": {"description": "x", "symmetric": False, "transitive": False, "inverse": None}}
    with pytest.raises(ValueError):
        service.create_ontology(tenant_id="t1", name="bad", relationships=bad)


def test_create_ontology_returns_summary():
    service = DefaultOntologyService(v=None)
    result = service.create_ontology(
        tenant_id="t1",
        name="custom1",
        relationships={"a": _meta("a"), "b": _meta("b")},
    )
    assert result == {
        "name": "custom1",
        "tenant_id": "t1",
        "workspace_id": None,
        "relationship_count": 2,
    }


def test_layered_priority_base_contributor_tenant_workspace():
    """Layered merge order: base < contributions < tenant < workspace."""
    service = DefaultOntologyService(v=None)
    key = "parent_of"
    assert key in BASE_ONTOLOGY
    base_desc = BASE_ONTOLOGY[key]["description"]

    # Layer 2: contributor pushes parent_of
    service.extend_ontology(
        {key: _meta("from contributor", category="hierarchical")},
        source="contrib",
    )

    # Layer 3: tenant ontology for t1
    service.create_ontology(
        tenant_id="t1",
        name="t1_ontology",
        relationships={key: _meta("from tenant", category="hierarchical")},
    )

    # Layer 4: workspace ontology for (t1, w1)
    service.create_ontology(
        tenant_id="t1",
        workspace_id="w1",
        name="t1w1_ontology",
        relationships={key: _meta("from workspace", category="hierarchical")},
    )

    # (t1, w1) -> workspace wins
    assert service.get_relationship_info(key, "t1", "w1")["description"] == "from workspace"
    # (t1, None) -> tenant wins
    assert service.get_relationship_info(key, "t1", None)["description"] == "from tenant"
    # (t2, None) -> contributor wins (no tenant ontology for t2)
    assert service.get_relationship_info(key, "t2", None)["description"] == "from contributor"

    # Clear contributions and (t2, None) falls back to base
    service._contributions.clear()
    service._contribution_sources.clear()
    assert service.get_relationship_info(key, "t2", None)["description"] == base_desc
