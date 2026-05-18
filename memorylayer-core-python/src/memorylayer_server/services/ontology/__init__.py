"""
Ontology Service - Provides relationship type definitions and validation.

OSS base ontology includes 63 relationship types across 11 categories.
Plugin contributors (e.g. RPG) may extend the ontology at startup via OntologyContributorPlugin.
Enterprise version supports custom tenant/workspace-scoped ontologies.
"""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_MULTI_ONTOLOGY_CONTRIBUTORS
from .base import (
    BASE_ONTOLOGY,
    EXT_ONTOLOGY_SERVICE,
    FeatureRequiresUpgradeError,
    OntologyService,
    OntologyServicePluginBase,
)
from .contributor import OntologyContributorPlugin


def get_ontology_service(v: Variables = None) -> OntologyService:
    """Get the ontology service instance."""
    return get_extension(EXT_ONTOLOGY_SERVICE, v)


__all__ = (
    "OntologyService",
    "OntologyServicePluginBase",
    "OntologyContributorPlugin",
    "get_ontology_service",
    "EXT_ONTOLOGY_SERVICE",
    "EXT_MULTI_ONTOLOGY_CONTRIBUTORS",
    "FeatureRequiresUpgradeError",
    "BASE_ONTOLOGY",
)
