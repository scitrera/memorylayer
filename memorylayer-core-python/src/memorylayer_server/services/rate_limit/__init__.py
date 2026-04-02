"""Rate limit service package."""
from .base import (
    RateLimitResult,
    RateLimitService,
    RateLimitServicePluginBase,
    EXT_RATE_LIMIT_SERVICE,
)

from scitrera_app_framework import Variables, get_extension


def get_rate_limit_service(v: Variables = None) -> RateLimitService:
    """Get the rate limit service instance."""
    return get_extension(EXT_RATE_LIMIT_SERVICE, v)


__all__ = (
    'RateLimitResult',
    'RateLimitService',
    'RateLimitServicePluginBase',
    'get_rate_limit_service',
    'EXT_RATE_LIMIT_SERVICE',
)
