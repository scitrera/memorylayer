"""Rate Limit Service - Pluggable rate limiting interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...config import MEMORYLAYER_RATE_LIMIT_SERVICE, DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE

from .._constants import EXT_RATE_LIMIT_SERVICE
from .._plugin_factory import make_service_plugin_base

# Re-export for convenience
__all__ = (
    'RateLimitResult',
    'RateLimitService',
    'RateLimitServicePluginBase',
    'EXT_RATE_LIMIT_SERVICE',
)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    limit: int        # max requests per window
    remaining: int    # requests remaining in current window
    reset_at: float   # unix timestamp when window resets


class RateLimitService(ABC):
    """Abstract rate limit service interface.

    Provides a pluggable interface for rate limiting that can be implemented
    by different backends (no-op, in-memory, Redis, Aether KV, etc.).
    """

    @abstractmethod
    async def check_rate_limit(
        self,
        key: str,
        limit: int = 0,
        window_seconds: int = 0,
    ) -> RateLimitResult:
        """Check and increment the rate limit for a key.

        Args:
            key: Rate limit key (e.g. ``"user:user-123"`` or ``"workspace:prod"``).
            limit: Maximum requests per window.  0 = use default from config.
            window_seconds: Window size in seconds.  0 = use default from config.

        Returns:
            RateLimitResult with allowed status and current counters.
        """
        pass

    @abstractmethod
    async def get_usage(self, key: str) -> tuple[int, int]:
        """Return current usage for a key.

        Args:
            key: Rate limit key.

        Returns:
            Tuple of ``(current_count, max_limit)``.
        """
        pass


# noinspection PyAbstractClass
RateLimitServicePluginBase = make_service_plugin_base(
    ext_name=EXT_RATE_LIMIT_SERVICE,
    config_key=MEMORYLAYER_RATE_LIMIT_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_RATE_LIMIT_SERVICE,
)
