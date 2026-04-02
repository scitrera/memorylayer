"""No-op rate limit service - always allows requests (OSS default)."""

import time
from logging import Logger

from scitrera_app_framework.api import Variables

from .base import RateLimitResult, RateLimitService, RateLimitServicePluginBase


class NoopRateLimitService(RateLimitService):
    """No-op rate limit service that always allows all requests."""

    async def check_rate_limit(
        self,
        key: str,
        limit: int = 0,
        window_seconds: int = 0,
    ) -> RateLimitResult:
        effective_limit = limit if limit > 0 else 0
        return RateLimitResult(
            allowed=True,
            limit=effective_limit,
            remaining=effective_limit,
            reset_at=time.time(),
        )

    async def get_usage(self, key: str) -> tuple[int, int]:
        return 0, 0


class NoopRateLimitServicePlugin(RateLimitServicePluginBase):
    """Plugin for no-op rate limit service."""

    PROVIDER_NAME = "noop"

    def initialize(self, v: Variables, logger: Logger) -> RateLimitService | None:
        return NoopRateLimitService()
