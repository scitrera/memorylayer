"""Tests for the pull (OntologyContributorPlugin) contribution path.

These tests simulate the contributor lifecycle without exercising the
full scitrera-app-framework plugin discovery cycle. The DefaultOntologyServicePlugin's
``async_ready`` hook calls ``get_extensions(EXT_MULTI_ONTOLOGY_CONTRIBUTORS, v)``
and then funnels every contributor through ``service.extend_ontology(...)``.
We exercise the same code path manually here -- the framework registration
machinery is covered by other test suites in the framework itself.
"""

import pytest

from memorylayer_server.services.ontology.contributor import OntologyContributorPlugin
from memorylayer_server.services.ontology.default import DefaultOntologyService


def _meta(description: str, category: str = "pull_cat") -> dict:
    return {
        "description": description,
        "symmetric": False,
        "transitive": False,
        "inverse": None,
        "category": category,
    }


class _FakeContributorA(OntologyContributorPlugin):
    def get_relationship_types(self) -> dict[str, dict]:
        return {
            "pull_alpha": _meta("alpha pull type", category="pull_cat_a"),
            "pull_beta": _meta("beta pull type", category="pull_cat_a"),
        }


class _FakeContributorB(OntologyContributorPlugin):
    def get_relationship_types(self) -> dict[str, dict]:
        return {
            "pull_gamma": _meta("gamma pull type", category="pull_cat_b"),
        }


def _simulate_async_ready(service: DefaultOntologyService, contributors):
    """Mirror DefaultOntologyServicePlugin.async_ready without the framework."""
    for c in contributors:
        service.extend_ontology(c.get_relationship_types(), source=c.name())


def test_single_contributor_visible_after_collection():
    service = DefaultOntologyService(v=None)
    contributor = _FakeContributorA()
    _simulate_async_ready(service, [contributor])

    assert service.validate_relationship("pull_alpha", "_default") is True
    assert service.validate_relationship("pull_beta", "_default") is True
    merged = service.get_merged_ontology("_default")
    assert merged["pull_alpha"]["description"] == "alpha pull type"

    sources = {c["type_name"]: c["source"] for c in service.list_contributors()}
    assert sources["pull_alpha"] == "_FakeContributorA"
    assert sources["pull_beta"] == "_FakeContributorA"


def test_two_disjoint_contributors_both_visible():
    service = DefaultOntologyService(v=None)
    _simulate_async_ready(service, [_FakeContributorA(), _FakeContributorB()])

    assert service.validate_relationship("pull_alpha", "_default")
    assert service.validate_relationship("pull_gamma", "_default")

    cats = service.list_categories("_default")
    assert "pull_cat_a" in cats
    assert "pull_cat_b" in cats


def test_contributor_failure_does_not_break_others(caplog):
    """A broken contributor is logged but doesn't abort collection.

    Mirrors the try/except in DefaultOntologyServicePlugin.async_ready.
    """

    class _Broken(OntologyContributorPlugin):
        def name(self) -> str:
            return "broken"

        def get_relationship_types(self) -> dict[str, dict]:
            raise RuntimeError("boom")

    service = DefaultOntologyService(v=None)
    contributors = [_Broken(), _FakeContributorB()]

    # Replicate the protective wrapper from async_ready manually here.
    import logging

    logger = logging.getLogger("test")
    for c in contributors:
        try:
            service.extend_ontology(c.get_relationship_types(), source=c.name())
        except Exception:
            logger.exception("Ontology contributor %s failed", c.name())

    # The good contributor still landed
    assert service.validate_relationship("pull_gamma", "_default")
    # The broken one did not
    with pytest.raises(ValueError):
        service.validate_relationship("nonexistent_type", "_default")
