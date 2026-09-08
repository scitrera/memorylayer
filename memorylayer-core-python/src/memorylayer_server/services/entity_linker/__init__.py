"""Entity linker service package (canonical entity -> external KB)."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EntityLinkerService,
    EntityLinkerServicePluginBase,
    EXT_ENTITY_LINKER_SERVICE,
)


def get_entity_linker_service(v: Variables = None) -> EntityLinkerService:
    """Get the entity linker service instance."""
    return get_extension(EXT_ENTITY_LINKER_SERVICE, v)


__all__ = (
    "EntityLinkerService",
    "EntityLinkerServicePluginBase",
    "get_entity_linker_service",
    "EXT_ENTITY_LINKER_SERVICE",
)
