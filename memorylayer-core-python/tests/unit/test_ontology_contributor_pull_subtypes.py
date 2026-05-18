"""Tests for the subtype pull (OntologyContributorPlugin.get_subtypes) path."""

from memorylayer_server.services.ontology.contributor import OntologyContributorPlugin
from memorylayer_server.services.ontology.default import DefaultOntologyService


class _SubtypeOnlyContributor(OntologyContributorPlugin):
    def get_relationship_types(self) -> dict[str, dict]:
        return {}

    def get_subtypes(self) -> dict[str, set[str]]:
        return {"*": {"plugin_kind_a", "plugin_kind_b"}}


class _BothContributor(OntologyContributorPlugin):
    def get_relationship_types(self) -> dict[str, dict]:
        return {
            "plugin_rel": {
                "description": "a relationship from a plugin",
                "symmetric": False,
                "transitive": False,
                "inverse": None,
                "category": "plugin_cat",
            }
        }

    def get_subtypes(self) -> dict[str, set[str]]:
        return {"semantic": {"plugin_semantic_subtype"}}


def _simulate_async_ready(service: DefaultOntologyService, contributors):
    for c in contributors:
        rel_types = c.get_relationship_types()
        if rel_types:
            service.extend_ontology(rel_types, source=c.name())
        subtypes = c.get_subtypes()
        if subtypes:
            service.extend_subtypes(subtypes, source=c.name())


def test_subtype_only_contributor_visible_after_collection():
    service = DefaultOntologyService(v=None)
    _simulate_async_ready(service, [_SubtypeOnlyContributor()])

    assert service.validate_subtype("semantic", "plugin_kind_a") is True
    assert service.validate_subtype("episodic", "plugin_kind_b") is True

    listing = service.list_subtypes("semantic")
    assert "plugin_kind_a" in listing
    assert "plugin_kind_b" in listing
    # OSS-known subtypes still appear too
    assert "solution" in listing


def test_combined_relationship_and_subtype_contributor():
    service = DefaultOntologyService(v=None)
    _simulate_async_ready(service, [_BothContributor()])

    # Relationship side
    assert service.validate_relationship("plugin_rel", "_default") is True
    # Subtype side
    assert service.validate_subtype("semantic", "plugin_semantic_subtype") is True
    # Subtype is scoped to semantic, not episodic (no "*" wildcard used)
    assert service.validate_subtype("episodic", "plugin_semantic_subtype") is False

    contributors = service.list_contributors()
    rel_entries = [c for c in contributors if c.get("kind") == "relationship"]
    sub_entries = [c for c in contributors if c.get("kind") == "subtype"]
    assert any(c["type_name"] == "plugin_rel" for c in rel_entries)
    assert any(c["subtype"] == "plugin_semantic_subtype" for c in sub_entries)
