"""
Unit tests for the rate limit service — RateLimitResult dataclass and NoopRateLimitService.
"""
import time

import pytest

from memorylayer_server.services.rate_limit.base import RateLimitResult
from memorylayer_server.services.rate_limit.noop import NoopRateLimitService


# ============================================================================
# RateLimitResult dataclass tests
# ============================================================================


class TestRateLimitResult:
    """Test RateLimitResult dataclass field presence and types."""

    def test_fields_are_stored_as_supplied(self):
        """All four fields are stored exactly as provided."""
        reset_ts = time.time() + 60
        result = RateLimitResult(allowed=True, limit=100, remaining=99, reset_at=reset_ts)

        assert result.allowed is True
        assert result.limit == 100
        assert result.remaining == 99
        assert result.reset_at == reset_ts

    def test_allowed_false_is_stored(self):
        """allowed=False is stored without coercion."""
        result = RateLimitResult(allowed=False, limit=10, remaining=0, reset_at=time.time())
        assert result.allowed is False

    def test_limit_field_is_int(self):
        """limit field accepts and stores an integer."""
        result = RateLimitResult(allowed=True, limit=500, remaining=499, reset_at=0.0)
        assert isinstance(result.limit, int)
        assert result.limit == 500

    def test_remaining_field_is_int(self):
        """remaining field accepts and stores an integer."""
        result = RateLimitResult(allowed=True, limit=10, remaining=3, reset_at=0.0)
        assert isinstance(result.remaining, int)
        assert result.remaining == 3

    def test_reset_at_field_is_float(self):
        """reset_at field accepts and stores a float unix timestamp."""
        ts = 1_700_000_000.0
        result = RateLimitResult(allowed=True, limit=10, remaining=10, reset_at=ts)
        assert isinstance(result.reset_at, float)
        assert result.reset_at == ts

    def test_zero_limit_and_remaining(self):
        """limit and remaining can be zero (noop / unlimited semantics)."""
        result = RateLimitResult(allowed=True, limit=0, remaining=0, reset_at=0.0)
        assert result.limit == 0
        assert result.remaining == 0


# ============================================================================
# NoopRateLimitService tests
# ============================================================================


class TestNoopRateLimitService:
    """Test NoopRateLimitService — always allows, usage always zero."""

    @pytest.mark.asyncio
    async def test_check_rate_limit_returns_allowed_true(self):
        """check_rate_limit() always returns allowed=True."""
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="user:123")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_with_explicit_limit_returns_allowed_true(self):
        """check_rate_limit() with an explicit limit still returns allowed=True."""
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="workspace:prod", limit=60, window_seconds=60)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_check_rate_limit_returns_rate_limit_result(self):
        """check_rate_limit() returns a RateLimitResult instance."""
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="user:abc")
        assert isinstance(result, RateLimitResult)

    @pytest.mark.asyncio
    async def test_check_rate_limit_limit_matches_supplied_value(self):
        """When a positive limit is supplied, result.limit and result.remaining equal it."""
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="user:abc", limit=200)
        assert result.limit == 200
        assert result.remaining == 200

    @pytest.mark.asyncio
    async def test_check_rate_limit_zero_limit_gives_zero_limit_and_remaining(self):
        """When limit=0 (use default), result.limit and result.remaining are both 0."""
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="user:abc", limit=0)
        assert result.limit == 0
        assert result.remaining == 0

    @pytest.mark.asyncio
    async def test_check_rate_limit_reset_at_is_recent_timestamp(self):
        """reset_at is a float unix timestamp close to the current time."""
        before = time.time()
        service = NoopRateLimitService()
        result = await service.check_rate_limit(key="tenant:x")
        after = time.time()

        assert isinstance(result.reset_at, float)
        assert before <= result.reset_at <= after

    @pytest.mark.asyncio
    async def test_check_rate_limit_always_allows_repeated_calls(self):
        """Repeated calls for the same key always return allowed=True."""
        service = NoopRateLimitService()
        for _ in range(5):
            result = await service.check_rate_limit(key="user:heavy", limit=2, window_seconds=60)
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_usage_returns_zero_current_count(self):
        """get_usage() returns 0 as the current count."""
        service = NoopRateLimitService()
        current, max_limit = await service.get_usage(key="user:123")
        assert current == 0

    @pytest.mark.asyncio
    async def test_get_usage_returns_zero_max_limit(self):
        """get_usage() returns 0 as the max limit (no limit configured)."""
        service = NoopRateLimitService()
        current, max_limit = await service.get_usage(key="user:123")
        assert max_limit == 0

    @pytest.mark.asyncio
    async def test_get_usage_returns_tuple(self):
        """get_usage() return value is a 2-tuple of ints."""
        service = NoopRateLimitService()
        result = await service.get_usage(key="some:key")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_usage_after_check_still_returns_zero(self):
        """Usage remains zero even after check_rate_limit calls (no state stored)."""
        service = NoopRateLimitService()
        await service.check_rate_limit(key="user:xyz", limit=10)
        current, max_limit = await service.get_usage(key="user:xyz")
        assert current == 0
        assert max_limit == 0
