"""In-memory LRU cache service."""
import time
from logging import Logger
from typing import Optional, Any

from scitrera_app_framework import Variables, get_logger

from .base import CacheService, CacheServicePluginBase

# Environment variable constants (specific to this implementation)
MEMORYLAYER_CACHE_LRU_MAXSIZE = 'MEMORYLAYER_CACHE_LRU_MAXSIZE'
DEFAULT_MEMORYLAYER_CACHE_LRU_MAXSIZE = 4096


class LRUCacheService(CacheService):
    """In-memory LRU cache service with optional TTL support.

    Uses cachetools.LRUCache for O(1) lookups with configurable max size.
    Supports TTL (time-to-live) expiration via timestamp tracking.
    """

    def __init__(
            self,
            v: Variables = None,
            logger: Logger = None,
            maxsize: int = DEFAULT_MEMORYLAYER_CACHE_LRU_MAXSIZE,
    ):
        from cachetools import LRUCache
        self._v = v
        self._logger = logger or get_logger(v, name=self.__class__.__name__)
        self._cache: LRUCache = LRUCache(maxsize=maxsize)
        self._timestamps: dict = {}
        self._maxsize = maxsize
        self._logger.info("Initialized LRUCacheService with maxsize=%s", maxsize)

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired based on TTL."""
        if key not in self._timestamps:
            return True
        timestamp, ttl_seconds = self._timestamps[key]
        if ttl_seconds is None:
            return False
        age = time.monotonic() - timestamp
        return age > ttl_seconds

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking TTL expiration."""
        if key not in self._cache:
            return None
        # Check TTL and clean up if expired
        if self._is_expired(key):
            await self.delete(key)
            return None
        return self._cache.get(key)

    async def set(
            self,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        self._cache[key] = value
        self._timestamps[key] = (time.monotonic(), ttl_seconds)
        self._logger.debug("Cache set: key=%s, ttl=%s", key, ttl_seconds)
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            self._logger.debug("Cache delete: key=%s", key)
            return True
        return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache and is not expired."""
        if key not in self._cache:
            return False
        if self._is_expired(key):
            await self.delete(key)
            return False
        return True

    async def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix."""
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]
            self._timestamps.pop(key, None)
        if keys_to_delete:
            self._logger.debug("Cache clear_prefix: prefix=%s, deleted=%s", prefix, len(keys_to_delete))
        return len(keys_to_delete)

    async def get_or_set(
            self,
            key: str,
            factory,
            ttl_seconds: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and cache."""
        value = await self.get(key)
        # Check if value exists and is not expired
        if await self.exists(key):
            return value

        value = await factory()
        await self.set(key, value, ttl_seconds)
        return value


class LRUCacheServicePlugin(CacheServicePluginBase):
    """Plugin for LRU cache service."""
    PROVIDER_NAME = 'lru'

    def initialize(self, v: Variables, logger: Logger) -> Optional[LRUCacheService]:
        maxsize = v.environ(
            MEMORYLAYER_CACHE_LRU_MAXSIZE,
            default=DEFAULT_MEMORYLAYER_CACHE_LRU_MAXSIZE,
            type_fn=int,
        )
        return LRUCacheService(v=v, logger=logger, maxsize=maxsize)
