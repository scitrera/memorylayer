"""Cache service package."""

from scitrera_app_framework import Variables, get_extension

from .base import (
    EXT_CACHE_SERVICE,
    CacheService,
    CacheServicePluginBase,
)


def get_cache_service(v: Variables = None) -> CacheService:
    """Get the cache service instance."""
    return get_extension(EXT_CACHE_SERVICE, v)


__all__ = (
    "CacheService",
    "CacheServicePluginBase",
    "get_cache_service",
    "EXT_CACHE_SERVICE",
)
