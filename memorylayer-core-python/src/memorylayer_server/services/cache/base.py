"""Cache Service - Pluggable caching interface."""
from abc import ABC, abstractmethod
from typing import Optional, Any

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_CACHE_SERVICE, DEFAULT_MEMORYLAYER_CACHE_SERVICE

EXT_CACHE_SERVICE = 'memorylayer-cache-service'


class CacheService(ABC):
    """Abstract cache service interface.

    Provides a simple key-value cache interface that can be implemented
    by different backends (no-op, in-memory, Redis, etc.).
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        pass

    @abstractmethod
    async def set(
            self,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> bool:
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
            ttl_seconds: Optional[int] = None,
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
class CacheServicePluginBase(Plugin):
    """Base plugin for cache service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CACHE_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CACHE_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_CACHE_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        # Set a default value for MEMORYLAYER_CACHE_SERVICE; defaults are lower priority than .set(...) values
        v.set_default_value(MEMORYLAYER_CACHE_SERVICE, DEFAULT_MEMORYLAYER_CACHE_SERVICE)
