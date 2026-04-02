"""No-op cache service - always returns None (OSS default)."""

from logging import Logger
from typing import Any

from scitrera_app_framework.api import Variables

from .base import CacheService, CacheServicePluginBase


class NoOpCacheService(CacheService):
    """No-op cache service."""

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None):
        return False

    async def delete(self, key: str) -> bool:
        return False

    async def exists(self, key: str) -> bool:
        return False

    async def clear_prefix(self, prefix: str) -> int:
        return 0

    async def get_or_set(self, key: str, factory, ttl_seconds: int | None = None) -> Any:
        value = await factory()
        return value


class NoOpCacheServicePlugin(CacheServicePluginBase):
    """Plugin for no cache service."""

    PROVIDER_NAME = "noop"

    def initialize(self, v: Variables, logger: Logger) -> CacheService | None:
        return NoOpCacheService()
