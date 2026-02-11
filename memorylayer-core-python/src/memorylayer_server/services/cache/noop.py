"""No-op cache service - always returns None (OSS default)."""
from logging import Logger
from typing import Optional, Any

from scitrera_app_framework.api import Variables

from .base import CacheService, CacheServicePluginBase


class NoOpCacheService(CacheService):
    """No-op cache service."""

    async def get(self, key: str) -> Optional[Any]:
        return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        return False

    async def delete(self, key: str) -> bool:
        return False

    async def exists(self, key: str) -> bool:
        return False

    async def clear_prefix(self, prefix: str) -> int:
        return 0

    async def get_or_set(self, key: str, factory, ttl_seconds: Optional[int] = None) -> Any:
        value = await factory()
        return value


class NoOpCacheServicePlugin(CacheServicePluginBase):
    """Plugin for no cache service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> Optional[CacheService]:
        return NoOpCacheService()
