"""Memory service package."""
from .base import (
    MemoryServicePluginBase,
    EXT_MEMORY_SERVICE,
)
from .default import DefaultMemoryServicePlugin, MemoryService

from scitrera_app_framework import Variables, get_extension


def get_memory_service(v: Variables = None) -> MemoryService:
    """Get the memory service instance."""
    return get_extension(EXT_MEMORY_SERVICE, v)


__all__ = (
    'MemoryService',
    'MemoryServicePluginBase',
    'get_memory_service',
    'EXT_MEMORY_SERVICE',
    'DefaultMemoryServicePlugin',
)
