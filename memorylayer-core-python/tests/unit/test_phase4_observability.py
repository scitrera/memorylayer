"""Phase 4 tests: post-store failure observability + configurable tolerance floors.

Covers:
- TASK 2: swallowed post-store sub-step failures increment an observable metrics
  counter without failing the store path.
- TASK 3: per-tolerance relevance floors are server-configurable (defaults match
  the historical hardcoded values; overrides are honored).
"""

from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import (
    MEMORYLAYER_TOLERANCE_FLOOR_LOOSE,
    MEMORYLAYER_TOLERANCE_FLOOR_MODERATE,
    MEMORYLAYER_TOLERANCE_FLOOR_STRICT,
)
from memorylayer_server.models.memory import Memory, MemoryStatus, MemoryType, SearchTolerance
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory.base import MEMORYLAYER_MEMORY_RECALL_OVERFETCH
from memorylayer_server.services.memory.default import MemoryService


class _FakeMetrics:
    """Minimal metrics double recording counter increments."""

    def __init__(self) -> None:
        self.counters: list[tuple[str, dict]] = []

    def counter(self, name: str, value: float = 1, labels: dict | None = None) -> None:
        self.counters.append((name, labels or {}))

    def histogram(self, name: str, value: float, labels: dict | None = None) -> None:  # pragma: no cover - unused
        pass

    def gauge(self, name: str, value: float, labels: dict | None = None) -> None:  # pragma: no cover - unused
        pass


def _make_v(**overrides) -> Variables:
    v = Variables()
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    for k, val in overrides.items():
        v.set(k, val)
    return v


def _make_service(v: Variables = None, **kwargs) -> MemoryService:
    if v is None:
        v = _make_v()

    storage = AsyncMock()
    storage.search_memories = AsyncMock(return_value=[])
    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[0.1] * 384)
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="New unique memory")
    )

    return MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        v=v,
        **kwargs,
    )


def _make_memory() -> Memory:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return Memory(
        id="mem_obs_1",
        workspace_id="ws_obs",
        tenant_id="t1",
        context_id="_default",
        content="observability test memory",
        content_hash="hash_obs_1",
        type=MemoryType.SEMANTIC,
        created_at=now,
        updated_at=now,
        status=MemoryStatus.ACTIVE,
    )


@pytest.mark.asyncio
class TestPostStoreFailureObservability:
    """TASK 2: failing post-store sub-steps increment a counter, store still succeeds."""

    async def test_failing_tier_generation_increments_counter(self):
        service = _make_service()
        metrics = _FakeMetrics()
        service.metrics = metrics

        # A tier-generation service whose generate_tiers raises.
        tier = AsyncMock()
        tier.generate_tiers = AsyncMock(side_effect=RuntimeError("boom"))
        service.tier_generation_service = tier
        service.contradiction_service = None  # isolate the failing step

        memory = _make_memory()

        # Must NOT raise even though tier generation fails.
        await service._post_store_pipeline("ws_obs", memory, [0.1] * 384, inline=True)

        names = [n for n, _ in metrics.counters]
        assert "memorylayer_post_store_step_failures_total" in names
        steps = [labels.get("step") for _, labels in metrics.counters]
        assert "tier_generation" in steps

    async def test_failing_contradiction_check_increments_counter(self):
        service = _make_service()
        metrics = _FakeMetrics()
        service.metrics = metrics

        service.tier_generation_service = None
        contradiction = AsyncMock()
        contradiction.check_new_memory = AsyncMock(side_effect=RuntimeError("boom"))
        service.contradiction_service = contradiction

        memory = _make_memory()
        await service._post_store_pipeline("ws_obs", memory, [0.1] * 384, inline=True)

        steps = [labels.get("step") for _, labels in metrics.counters]
        assert "contradiction_check" in steps

    async def test_no_metrics_handle_does_not_raise(self):
        """With no metrics wired, a failing sub-step still degrades silently."""
        service = _make_service()
        service.metrics = None  # simulate unwired metrics extension

        tier = AsyncMock()
        tier.generate_tiers = AsyncMock(side_effect=RuntimeError("boom"))
        service.tier_generation_service = tier
        service.contradiction_service = None

        memory = _make_memory()
        # Should not raise.
        await service._post_store_pipeline("ws_obs", memory, [0.1] * 384, inline=True)


class TestConfigurableToleranceFloors:
    """TASK 3: tolerance floors come from config; defaults preserve behavior."""

    def test_defaults_match_historical_values(self):
        service = _make_service()
        # min_relevance=None -> returns the tolerance floor (server default).
        assert service._get_relevance_threshold(SearchTolerance.STRICT, None) == pytest.approx(0.6)
        assert service._get_relevance_threshold(SearchTolerance.MODERATE, None) == pytest.approx(0.3)
        assert service._get_relevance_threshold(SearchTolerance.LOOSE, None) == pytest.approx(0.15)

    def test_overridden_config_is_honored(self):
        v = _make_v(
            **{
                MEMORYLAYER_TOLERANCE_FLOOR_STRICT: 0.8,
                MEMORYLAYER_TOLERANCE_FLOOR_MODERATE: 0.5,
                MEMORYLAYER_TOLERANCE_FLOOR_LOOSE: 0.05,
            }
        )
        service = _make_service(v=v)
        assert service._get_relevance_threshold(SearchTolerance.STRICT, None) == pytest.approx(0.8)
        assert service._get_relevance_threshold(SearchTolerance.MODERATE, None) == pytest.approx(0.5)
        assert service._get_relevance_threshold(SearchTolerance.LOOSE, None) == pytest.approx(0.05)

    def test_override_enforced_as_floor_on_explicit_value(self):
        """An explicit min_relevance below the (overridden) floor is raised to it."""
        v = _make_v(**{MEMORYLAYER_TOLERANCE_FLOOR_MODERATE: 0.5})
        service = _make_service(v=v)
        # Explicit 0.2 is below the overridden 0.5 floor -> clamped up.
        assert service._get_relevance_threshold(SearchTolerance.MODERATE, 0.2) == pytest.approx(0.5)
        # Explicit value above floor is respected.
        assert service._get_relevance_threshold(SearchTolerance.MODERATE, 0.7) == pytest.approx(0.7)
