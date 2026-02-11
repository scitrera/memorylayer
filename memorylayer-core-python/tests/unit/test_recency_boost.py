"""
Unit tests for recency boost in memory recall scoring.

Tests the apply_recency_boost() method which applies time-based decay
to memory scores based on their updated_at timestamp.
"""
import math
import pytest
from datetime import datetime, timezone, timedelta
from memorylayer_server.services.memory.default import MemoryService
from memorylayer_server.models.memory import Memory, MemoryType
from memorylayer_server.config import DEFAULT_RECENCY_HALF_LIFE_HOURS, DEFAULT_RECENCY_WEIGHT


def create_test_memory(
    memory_id: str,
    boosted_score: float,
    updated_at: datetime,
    content: str = "Test memory content",
) -> Memory:
    """
    Create a Memory object for testing recency boost.

    Args:
        memory_id: Unique memory identifier
        boosted_score: Initial boosted score before recency adjustment
        updated_at: Timestamp when memory was last updated
        content: Memory content (defaults to test string)

    Returns:
        Memory object with all required fields populated
    """
    return Memory(
        id=memory_id,
        workspace_id="test_workspace",
        tenant_id="test_tenant",
        context_id="_default",
        content=content,
        content_hash=f"hash_{memory_id}",
        type=MemoryType.SEMANTIC,
        boosted_score=boosted_score,
        updated_at=updated_at,
        created_at=updated_at,
    )


class TestRecencyBoost:
    """Tests for apply_recency_boost() method."""

    def test_fresh_memory_minimal_decay(self):
        """
        Test that a memory updated 1 minute ago has minimal score decay.

        A very recent memory should have recency_factor ~1.0, meaning
        the boosted_score remains nearly unchanged.
        """
        # Create a service instance (we'll call the method directly)
        # Since apply_recency_boost doesn't use instance state, we can
        # create a minimal instance
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)

        memories = [
            create_test_memory("mem_1", boosted_score=0.9, updated_at=one_minute_ago)
        ]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=DEFAULT_RECENCY_WEIGHT,  # 0.2
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,  # 168
        )

        # After 1 minute, decay should be negligible
        # recency_factor ≈ 1.0, so adjusted_score ≈ original boosted_score
        assert len(result) == 1
        assert result[0].boosted_score == pytest.approx(0.9, abs=0.01)

    def test_old_memory_significant_decay(self):
        """
        Test that a memory updated 30 days ago has significant score decay.

        An old memory should have lower recency_factor, reducing its
        boosted_score substantially.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        memories = [
            create_test_memory("mem_1", boosted_score=0.9, updated_at=thirty_days_ago)
        ]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=DEFAULT_RECENCY_WEIGHT,  # 0.2
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,  # 168 (7 days)
        )

        # After 30 days (720 hours), with half_life=168 hours:
        # age_hours = 720
        # decay_lambda = ln(2) / 168 ≈ 0.004124
        # recency_factor = exp(-0.004124 * 720) ≈ 0.051
        # adjusted_score = 0.9 * (1.0 - 0.2 + 0.2 * 0.051)
        #                = 0.9 * (0.8 + 0.0102)
        #                = 0.9 * 0.8102 ≈ 0.729

        assert len(result) == 1
        # Score should be noticeably reduced (but not to zero due to weight=0.2)
        assert result[0].boosted_score < 0.8
        assert result[0].boosted_score > 0.7
        assert result[0].boosted_score == pytest.approx(0.729, abs=0.01)

    def test_recency_weight_zero_no_effect(self):
        """
        Test that with recency_weight=0.0, scores remain unchanged.

        When recency_weight is 0, the method should return immediately
        without modifying any scores.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        old_memory = create_test_memory("mem_1", boosted_score=0.9, updated_at=now - timedelta(days=30))
        recent_memory = create_test_memory("mem_2", boosted_score=0.8, updated_at=now - timedelta(minutes=1))

        memories = [old_memory, recent_memory]
        original_scores = [m.boosted_score for m in memories]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=0.0,
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # All scores should remain unchanged
        assert len(result) == 2
        assert result[0].boosted_score == original_scores[0]
        assert result[1].boosted_score == original_scores[1]

    def test_recency_weight_one_full_effect(self):
        """
        Test that with recency_weight=1.0, old memories are heavily penalized.

        At weight=1.0, the full decay effect applies:
        adjusted_score = boosted_score * recency_factor
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        thirty_days_ago = now - timedelta(days=30)

        memories = [
            create_test_memory("mem_1", boosted_score=0.9, updated_at=thirty_days_ago)
        ]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=1.0,
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # With weight=1.0:
        # adjusted_score = 0.9 * recency_factor
        # recency_factor ≈ 0.051 (from previous calculation)
        # adjusted_score ≈ 0.9 * 0.051 ≈ 0.046

        assert len(result) == 1
        assert result[0].boosted_score < 0.1  # Heavily penalized
        assert result[0].boosted_score == pytest.approx(0.046, abs=0.01)

    def test_recency_reorders_equal_scores(self):
        """
        Test that two memories with same boosted_score are reordered by recency.

        When two memories have identical scores, the newer one should
        rank higher after recency boost is applied.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        old_memory = create_test_memory(
            "mem_old",
            boosted_score=0.8,
            updated_at=now - timedelta(days=14),
            content="Old memory"
        )
        recent_memory = create_test_memory(
            "mem_recent",
            boosted_score=0.8,
            updated_at=now - timedelta(hours=1),
            content="Recent memory"
        )

        # Start with old memory first
        memories = [old_memory, recent_memory]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=0.3,  # Use higher weight to ensure reordering
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # Recent memory should now rank higher
        assert len(result) == 2
        assert result[0].id == "mem_recent"
        assert result[1].id == "mem_old"
        assert result[0].boosted_score > result[1].boosted_score

    def test_half_life_correctness(self):
        """
        Test that a memory exactly half_life_hours old has recency_factor ≈ 0.5.

        This verifies the exponential decay formula is correctly implemented:
        recency_factor = exp(-ln(2) * age_hours / half_life_hours)

        At age = half_life, this should equal 0.5.
        """
        service = object.__new__(MemoryService)

        half_life_hours = 168.0  # 7 days
        now = datetime.now(timezone.utc)
        half_life_ago = now - timedelta(hours=half_life_hours)

        memories = [
            create_test_memory("mem_1", boosted_score=1.0, updated_at=half_life_ago)
        ]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=1.0,  # Full effect to test recency_factor directly
            half_life_hours=half_life_hours,
        )

        # With weight=1.0 and age=half_life:
        # recency_factor = exp(-ln(2) * 1) = exp(-0.693) = 0.5
        # adjusted_score = 1.0 * 0.5 = 0.5

        assert len(result) == 1
        assert result[0].boosted_score == pytest.approx(0.5, abs=0.01)

    def test_empty_list_returns_empty(self):
        """
        Test that an empty list input returns an empty list.

        Edge case: the method should handle empty input gracefully.
        """
        service = object.__new__(MemoryService)

        result = service.apply_recency_boost(
            memories=[],
            recency_weight=DEFAULT_RECENCY_WEIGHT,
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        assert result == []

    def test_multiple_memories_correct_ordering(self):
        """
        Test that multiple memories are correctly ordered after recency boost.

        Verifies that the method correctly sorts by adjusted scores in
        descending order.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)

        # Create memories with varying scores and ages
        memories = [
            create_test_memory("mem_1", boosted_score=0.7, updated_at=now - timedelta(days=1)),
            create_test_memory("mem_2", boosted_score=0.9, updated_at=now - timedelta(days=30)),
            create_test_memory("mem_3", boosted_score=0.6, updated_at=now - timedelta(hours=1)),
            create_test_memory("mem_4", boosted_score=0.8, updated_at=now - timedelta(days=7)),
        ]

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=0.3,
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # Verify all memories returned
        assert len(result) == 4

        # Verify descending order of boosted_score
        for i in range(len(result) - 1):
            assert result[i].boosted_score >= result[i + 1].boosted_score

    def test_negative_recency_weight_treated_as_zero(self):
        """
        Test that negative recency_weight is treated as zero (no effect).

        The method checks for recency_weight <= 0.0 and returns early.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        memories = [
            create_test_memory("mem_1", boosted_score=0.9, updated_at=now - timedelta(days=30))
        ]
        original_score = memories[0].boosted_score

        result = service.apply_recency_boost(
            memories=memories,
            recency_weight=-0.5,  # Negative weight
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # Score should remain unchanged
        assert len(result) == 1
        assert result[0].boosted_score == original_score

    def test_custom_half_life(self):
        """
        Test that custom half_life_hours parameter works correctly.

        Verifies that the decay rate changes with different half-life values.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        one_week_ago = now - timedelta(days=7)

        # Test with 7-day half-life (default)
        memories_default = [
            create_test_memory("mem_1", boosted_score=1.0, updated_at=one_week_ago)
        ]
        result_default = service.apply_recency_boost(
            memories=memories_default,
            recency_weight=1.0,
            half_life_hours=168.0,  # 7 days
        )

        # Test with 14-day half-life (slower decay)
        memories_slow = [
            create_test_memory("mem_2", boosted_score=1.0, updated_at=one_week_ago)
        ]
        result_slow = service.apply_recency_boost(
            memories=memories_slow,
            recency_weight=1.0,
            half_life_hours=336.0,  # 14 days
        )

        # With 7-day half-life, score at 7 days should be ~0.5
        # With 14-day half-life, score at 7 days should be ~0.707
        assert result_default[0].boosted_score == pytest.approx(0.5, abs=0.01)
        assert result_slow[0].boosted_score == pytest.approx(0.707, abs=0.01)
        assert result_slow[0].boosted_score > result_default[0].boosted_score

    def test_preserves_other_memory_fields(self):
        """
        Test that apply_recency_boost only modifies boosted_score.

        All other memory fields should remain unchanged.
        """
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        memory = create_test_memory(
            "mem_1",
            boosted_score=0.8,
            updated_at=now - timedelta(days=7),
            content="Test content"
        )

        # Store original values
        original_id = memory.id
        original_content = memory.content
        original_type = memory.type
        original_updated_at = memory.updated_at

        result = service.apply_recency_boost(
            memories=[memory],
            recency_weight=0.5,
            half_life_hours=DEFAULT_RECENCY_HALF_LIFE_HOURS,
        )

        # Verify only boosted_score changed
        assert result[0].id == original_id
        assert result[0].content == original_content
        assert result[0].type == original_type
        assert result[0].updated_at == original_updated_at
        assert result[0].boosted_score != 0.8  # This should have changed
