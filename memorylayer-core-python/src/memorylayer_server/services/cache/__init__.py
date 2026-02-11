"""Cache service package."""
from .base import (
    CacheService,
    CacheServicePluginBase,
    EXT_CACHE_SERVICE,
)

from scitrera_app_framework import Variables, get_extension


def get_cache_service(v: Variables = None) -> CacheService:
    """Get the cache service instance."""
    return get_extension(EXT_CACHE_SERVICE, v)


__all__ = (
    'CacheService',
    'CacheServicePluginBase',
    'get_cache_service',
    'EXT_CACHE_SERVICE',
)
