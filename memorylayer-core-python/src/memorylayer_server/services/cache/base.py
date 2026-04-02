"""Cache Service - Pluggable caching interface."""

from abc import ABC, abstractmethod
from typing import Any

from ...config import DEFAULT_MEMORYLAYER_CACHE_SERVICE, MEMORYLAYER_CACHE_SERVICE
from .._constants import EXT_CACHE_SERVICE
from .._plugin_factory import make_service_plugin_base


class CacheService(ABC):
    """Abstract cache service interface.

    Provides a simple key-value cache interface that can be implemented
    by different backends (no-op, in-memory, Redis, etc.).
    """

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl_seconds: Time-to-live in seconds (None = no expiry)

        Returns:
            True if successfully cached
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if key was deleted, False if not found
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists and not expired
        """
        pass

    @abstractmethod
    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of keys deleted
        """
        pass

    async def get_or_set(
        self,
        key: str,
        factory,
        ttl_seconds: int | None = None,
    ) -> Any:
        """Get from cache or compute and cache.

        Args:
            key: Cache key
            factory: Async callable to produce value if not cached
            ttl_seconds: TTL for cached value

        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value

        value = await factory()
        await self.set(key, value, ttl_seconds)
        return value


# noinspection PyAbstractClass
CacheServicePluginBase = make_service_plugin_base(
    ext_name=EXT_CACHE_SERVICE,
    config_key=MEMORYLAYER_CACHE_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_CACHE_SERVICE,
)
