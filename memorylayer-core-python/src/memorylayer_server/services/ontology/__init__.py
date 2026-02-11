"""
Ontology Service - Provides relationship type definitions and validation.

OSS version includes unified ontology with 65 relationship types across 11 categories.
Enterprise version supports custom ontologies.
"""
from .base import (
    OntologyService,
    OntologyServicePluginBase,
    EXT_ONTOLOGY_SERVICE,
    FeatureRequiresUpgradeError,
    BASE_ONTOLOGY,
    RELATIONSHIP_CATEGORIES,
)

from scitrera_app_framework import Variables, get_extension


def get_ontology_service(v: Variables = None) -> OntologyService:
    """Get the ontology service instance."""
    return get_extension(EXT_ONTOLOGY_SERVICE, v)


__all__ = (
    'OntologyService',
    'OntologyServicePluginBase',
    'get_ontology_service',
    'EXT_ONTOLOGY_SERVICE',
    'FeatureRequiresUpgradeError',
    'BASE_ONTOLOGY',
    'RELATIONSHIP_CATEGORIES',
)
