"""Entity registry service package (entity registry slice 1).

The canonical-entity layer over the workspace memory set: stable entity nodes,
aliases, and member accretion, with deterministic exact+alias resolution. OSS
ships the relational ``default`` provider; the enterprise package mirrors the
same storage methods in PostgreSQL behind the same plugin pattern. Selection via
``MEMORYLAYER_ENTITY_REGISTRY_PROVIDER`` (default ``"default"``).

Ships DARK: ``MEMORYLAYER_ENTITY_REGISTRY_ENABLED`` defaults to ``False`` so the
service can be registered without affecting recall/ingest until flipped.
"""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_ENTITY_REGISTRY_SERVICE
from ._normalize import normalize_entity_name
from .base import EntityRegistryService, EntityRegistryServicePluginBase


def get_entity_registry_service(v: Variables = None) -> EntityRegistryService:
    """Get the entity registry service instance."""
    return get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)


__all__ = (
    "EntityRegistryService",
    "EntityRegistryServicePluginBase",
    "get_entity_registry_service",
    "normalize_entity_name",
    "EXT_ENTITY_REGISTRY_SERVICE",
)
