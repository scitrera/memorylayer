"""
Unit tests for Phase 1 quick wins:
  1a. Memory freshness/staleness annotations (_annotate_freshness)
  1b. Already-surfaced filtering (exclude_ids in _recall_rag)
  1c. Configurable scope/recency boosts (ScopeBoosts read from config)
"""
import math
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from scitrera_app_framework import Variables

from memorylayer_server.models.memory import Memory, MemoryType, MemoryStatus, RecallInput
from memorylayer_server.services.memory.default import MemoryService, ScopeBoosts
from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
    MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
    MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
    DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
    MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
    DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.memory.base import MEMORYLAYER_MEMORY_RECALL_OVERFETCH
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_v(**overrides) -> Variables:
    """Create a Variables instance with required defaults for MemoryService."""
    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, True)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    for k, val in overrides.items():
        v.set(k, val)
    return v


def _make_service(v: Variables = None) -> MemoryService:
    """Create a MemoryService with all dependencies mocked."""
    if v is None:
        v = _make_v()

    storage = AsyncMock()
    storage.search_memories = AsyncMock(return_value=[])
    embedding = AsyncMock()
    embedding.embed = AsyncMock(return_value=[0.1] * 384)
    dedup = AsyncMock()
    dedup.check_duplicate = AsyncMock(return_value=DeduplicationResult(
        action=DeduplicationAction.CREATE,
        reason="New unique memory",
    ))

    return MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        v=v,
    )


def _make_memory(
    memory_id: str = "mem_1",
    workspace_id: str = "ws_test",
    context_id: str = "_default",
    created_at: datetime = None,
    last_accessed_at: datetime = None,
    boosted_score: float = 0.8,
) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=memory_id,
        workspace_id=workspace_id,
        tenant_id="test_tenant",
        context_id=context_id,
        content="Test memory content",
        content_hash=f"hash_{memory_id}",
        type=MemoryType.SEMANTIC,
        boosted_score=boosted_score,
        created_at=created_at or now,
        updated_at=created_at or now,
        last_accessed_at=last_accessed_at,
        status=MemoryStatus.ACTIVE,
    )


# ===========================================================================
# 1a. Freshness annotation tests
# ===========================================================================

class TestAnnotateFreshness:
    """Tests for MemoryService._annotate_freshness()."""

    def test_empty_list_returns_empty(self):
        service = _make_service()
        result = service._annotate_freshness([])
        assert result == []

    def test_fresh_memory_score_near_one(self):
        """A memory created moments ago should have freshness ~1.0."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(minutes=5))
        result = service._annotate_freshness([memory])
        assert result[0].freshness_score is not None
        assert result[0].freshness_score > 0.99
        assert result[0].staleness_warning == "none"
        assert result[0].age_days < 0.01

    def test_1_day_old_staleness_mild(self):
        """A memory 2 days old should have 'mild' staleness warning."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=2))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "mild"

    def test_7_day_old_staleness_moderate(self):
        """A memory 10 days old should have 'moderate' staleness warning."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=10))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "moderate"

    def test_30_day_old_staleness_severe(self):
        """A memory 31 days old should have 'severe' staleness warning."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=31))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "severe"

    def test_half_life_7_days_at_7_days(self):
        """At exactly half_life_days old, freshness_score should be ~0.5 (weight=1)."""
        service = _make_service()
        # Use exactly 7 days age with 7 days half-life
        half_life = 7.0
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=7))
        result = service._annotate_freshness([memory], half_life_days=half_life)
        assert result[0].freshness_score == pytest.approx(0.5, abs=0.02)

    def test_exponential_decay_formula(self):
        """Verify the exponential decay formula: score = exp(-ln(2) * age / half_life)."""
        service = _make_service()
        half_life = 7.0
        age_days = 14.0  # Two half-lives
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=age_days))
        result = service._annotate_freshness([memory], half_life_days=half_life)
        expected = math.exp(-math.log(2) / half_life * age_days)  # ~0.25
        assert result[0].freshness_score == pytest.approx(expected, abs=0.02)

    def test_access_recency_bonus(self):
        """Memory accessed in the last 24h gets a +0.05 freshness bonus."""
        service = _make_service()
        half_life = 7.0
        # Memory created 5 days ago, accessed 1 hour ago
        created = datetime.now(timezone.utc) - timedelta(days=5)
        last_accessed = datetime.now(timezone.utc) - timedelta(hours=1)
        memory_with_access = _make_memory(created_at=created, last_accessed_at=last_accessed)
        memory_no_access = _make_memory(memory_id="mem_2", created_at=created)

        result_with = service._annotate_freshness([memory_with_access], half_life_days=half_life)
        result_without = service._annotate_freshness([memory_no_access], half_life_days=half_life)

        # The accessed memory should have a higher freshness score (by ~0.05)
        assert result_with[0].freshness_score > result_without[0].freshness_score
        assert result_with[0].freshness_score - result_without[0].freshness_score == pytest.approx(0.05, abs=0.01)

    def test_access_recency_bonus_capped_at_one(self):
        """Freshness bonus should not push score above 1.0."""
        service = _make_service()
        # Very fresh memory (score ~1.0) + access bonus should cap at 1.0
        just_created = datetime.now(timezone.utc) - timedelta(seconds=10)
        last_accessed = datetime.now(timezone.utc) - timedelta(hours=1)
        memory = _make_memory(created_at=just_created, last_accessed_at=last_accessed)
        result = service._annotate_freshness([memory])
        assert result[0].freshness_score <= 1.0

    def test_old_access_no_bonus(self):
        """Memory accessed more than 24h ago should NOT get the access bonus."""
        service = _make_service()
        created = datetime.now(timezone.utc) - timedelta(days=3)
        last_accessed = datetime.now(timezone.utc) - timedelta(hours=25)  # >24h ago
        memory = _make_memory(created_at=created, last_accessed_at=last_accessed)
        result = service._annotate_freshness([memory])

        # Score should match plain formula (no bonus)
        half_life = service.freshness_half_life_days
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
        expected = math.exp(-math.log(2) / half_life * age_days)
        assert result[0].freshness_score == pytest.approx(expected, abs=0.02)

    def test_age_days_populated(self):
        """age_days field should reflect the memory's age."""
        service = _make_service()
        created = datetime.now(timezone.utc) - timedelta(days=5)
        memory = _make_memory(created_at=created)
        result = service._annotate_freshness([memory])
        assert result[0].age_days == pytest.approx(5.0, abs=0.05)

    def test_configurable_half_life_from_init(self):
        """MemoryService should use MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS from config."""
        # Use 14-day half-life (slower decay)
        v = _make_v(**{MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS: 14.0})
        service = _make_service(v)
        assert service.freshness_half_life_days == 14.0

        # At 7 days age with 14 day half-life, score should be ~0.707
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=7))
        result = service._annotate_freshness([memory])
        assert result[0].freshness_score == pytest.approx(0.707, abs=0.02)

    def test_multiple_memories_annotated(self):
        """All memories in the list should be annotated."""
        service = _make_service()
        memories = [
            _make_memory("m1", created_at=datetime.now(timezone.utc) - timedelta(hours=1)),
            _make_memory("m2", created_at=datetime.now(timezone.utc) - timedelta(days=5)),
            _make_memory("m3", created_at=datetime.now(timezone.utc) - timedelta(days=35)),
        ]
        result = service._annotate_freshness(memories)
        assert all(m.freshness_score is not None for m in result)
        assert all(m.staleness_warning is not None for m in result)
        assert all(m.age_days is not None for m in result)
        # Verify descending freshness order
        assert result[0].freshness_score > result[1].freshness_score > result[2].freshness_score

    def test_staleness_boundary_exactly_1_day(self):
        """Memory exactly 1 day old should be 'mild' (boundary >= 1.0)."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=1, seconds=1))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "mild"

    def test_staleness_boundary_exactly_7_days(self):
        """Memory exactly 7 days old should be 'moderate' (boundary >= 7.0)."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=7, seconds=1))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "moderate"

    def test_staleness_boundary_exactly_30_days(self):
        """Memory exactly 30 days old should be 'severe' (boundary >= 30.0)."""
        service = _make_service()
        memory = _make_memory(created_at=datetime.now(timezone.utc) - timedelta(days=30, seconds=1))
        result = service._annotate_freshness([memory])
        assert result[0].staleness_warning == "severe"


# ===========================================================================
# 1b. exclude_ids filtering tests
# ===========================================================================

class TestExcludeIds:
    """Tests for exclude_ids field on RecallInput and filtering in _recall_rag."""

    def test_exclude_ids_field_default_empty(self):
        """RecallInput.exclude_ids should default to an empty list."""
        input = RecallInput(query="test query")
        assert input.exclude_ids == []

    def test_exclude_ids_field_accepts_list(self):
        """RecallInput.exclude_ids should accept a list of strings."""
        input = RecallInput(query="test query", exclude_ids=["id1", "id2"])
        assert input.exclude_ids == ["id1", "id2"]

    @pytest.mark.asyncio
    async def test_recall_rag_filters_excluded_ids(self):
        """_recall_rag should filter out memories whose IDs are in exclude_ids."""
        service = _make_service()

        now = datetime.now(timezone.utc)
        mem_a = _make_memory("mem_a", created_at=now)
        mem_b = _make_memory("mem_b", created_at=now)
        mem_c = _make_memory("mem_c", created_at=now)

        # Storage returns all three memories
        service.storage.search_memories = AsyncMock(return_value=[
            (mem_a, 0.9), (mem_b, 0.85), (mem_c, 0.8),
        ])

        input = RecallInput(
            query="test query",
            limit=10,
            include_global=False,
            exclude_ids=["mem_b"],  # Exclude mem_b
        )

        result = await service._recall_rag(
            workspace_id="ws_test",
            input=input,
            relevance_threshold=0.0,
        )

        ids = [m.id for m in result.memories]
        assert "mem_b" not in ids
        assert "mem_a" in ids
        assert "mem_c" in ids

    @pytest.mark.asyncio
    async def test_recall_rag_no_exclude_ids_returns_all(self):
        """_recall_rag should return all results when exclude_ids is empty."""
        service = _make_service()

        now = datetime.now(timezone.utc)
        mem_a = _make_memory("mem_a", created_at=now)
        mem_b = _make_memory("mem_b", created_at=now)

        service.storage.search_memories = AsyncMock(return_value=[
            (mem_a, 0.9), (mem_b, 0.85),
        ])

        input = RecallInput(
            query="test query",
            limit=10,
            include_global=False,
            exclude_ids=[],  # Empty exclusion list
        )

        result = await service._recall_rag(
            workspace_id="ws_test",
            input=input,
            relevance_threshold=0.0,
        )

        ids = [m.id for m in result.memories]
        assert "mem_a" in ids
        assert "mem_b" in ids

    @pytest.mark.asyncio
    async def test_recall_rag_exclude_multiple_ids(self):
        """_recall_rag should filter out all IDs in exclude_ids."""
        service = _make_service()

        now = datetime.now(timezone.utc)
        memories = [(_make_memory(f"mem_{i}", created_at=now), 0.9 - i * 0.05) for i in range(5)]
        service.storage.search_memories = AsyncMock(return_value=memories)

        input = RecallInput(
            query="test",
            limit=10,
            include_global=False,
            exclude_ids=["mem_1", "mem_3"],
        )

        result = await service._recall_rag(
            workspace_id="ws_test",
            input=input,
            relevance_threshold=0.0,
        )

        ids = [m.id for m in result.memories]
        assert "mem_1" not in ids
        assert "mem_3" not in ids
        assert "mem_0" in ids
        assert "mem_2" in ids
        assert "mem_4" in ids

    @pytest.mark.asyncio
    async def test_recall_rag_exclude_all_returns_empty(self):
        """_recall_rag should return empty when all results are excluded."""
        service = _make_service()

        now = datetime.now(timezone.utc)
        service.storage.search_memories = AsyncMock(return_value=[
            (_make_memory("mem_a", created_at=now), 0.9),
        ])

        input = RecallInput(
            query="test",
            limit=10,
            include_global=False,
            exclude_ids=["mem_a"],
        )

        result = await service._recall_rag(
            workspace_id="ws_test",
            input=input,
            relevance_threshold=0.0,
        )

        assert result.memories == []


# ===========================================================================
# 1c. Configurable scope boosts tests
# ===========================================================================

class TestConfigurableScopeBoosts:
    """Tests for configurable scope boosts read from config in MemoryService.__init__."""

    def test_default_scope_boosts_from_defaults(self):
        """MemoryService should create ScopeBoosts with default values."""
        service = _make_service()
        assert service.default_scope_boosts.same_context == DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT
        assert service.default_scope_boosts.same_workspace == DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE
        assert service.default_scope_boosts.global_workspace == 1.0

    def test_custom_same_context_boost(self):
        """MemoryService should read MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT from config."""
        v = _make_v(**{MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT: 2.0})
        service = _make_service(v)
        assert service.default_scope_boosts.same_context == 2.0

    def test_custom_same_workspace_boost(self):
        """MemoryService should read MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE from config."""
        v = _make_v(**{MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE: 1.8})
        service = _make_service(v)
        assert service.default_scope_boosts.same_workspace == 1.8

    def test_apply_scope_boosts_uses_instance_defaults(self):
        """apply_scope_boosts(boosts=None) should use self.default_scope_boosts."""
        v = _make_v(**{
            MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT: 2.0,
            MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE: 1.5,
        })
        service = _make_service(v)

        now = datetime.now(timezone.utc)
        memory = _make_memory("mem_1", workspace_id="ws_test", context_id="ctx_test", created_at=now)
        memories = [(memory, 0.8)]

        result = service.apply_scope_boosts(
            memories,
            query_context_id="ctx_test",
            query_workspace_id="ws_test",
            boosts=None,  # Should use instance defaults
        )

        assert len(result) == 1
        # same_context boost = 2.0, so boosted_score = 0.8 * 2.0 = 1.6
        assert result[0].boosted_score == pytest.approx(1.6, abs=0.01)

    def test_apply_scope_boosts_explicit_overrides_instance(self):
        """Explicitly passed boosts should override instance defaults."""
        v = _make_v(**{MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT: 2.0})
        service = _make_service(v)

        now = datetime.now(timezone.utc)
        memory = _make_memory("mem_1", workspace_id="ws_test", context_id="ctx_test", created_at=now)
        memories = [(memory, 0.8)]

        explicit_boosts = ScopeBoosts(same_context=3.0, same_workspace=1.2)
        result = service.apply_scope_boosts(
            memories,
            query_context_id="ctx_test",
            query_workspace_id="ws_test",
            boosts=explicit_boosts,
        )

        # Explicit boost 3.0 overrides instance default 2.0
        assert result[0].boosted_score == pytest.approx(2.4, abs=0.01)

    def test_scope_boost_same_workspace(self):
        """Memory from same workspace but different context gets workspace boost."""
        v = _make_v(**{
            MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT: 1.5,
            MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE: 1.3,
        })
        service = _make_service(v)

        now = datetime.now(timezone.utc)
        memory = _make_memory("mem_1", workspace_id="ws_test", context_id="ctx_other", created_at=now)
        memories = [(memory, 0.8)]

        result = service.apply_scope_boosts(
            memories,
            query_context_id="ctx_query",  # Different context
            query_workspace_id="ws_test",  # Same workspace
            boosts=None,
        )

        # same_workspace boost = 1.3
        assert result[0].boosted_score == pytest.approx(0.8 * 1.3, abs=0.01)
        assert result[0].source_scope == "same_workspace"

    def test_scope_boost_no_boost_for_other(self):
        """Memory from different workspace and context gets no boost (factor 1.0)."""
        service = _make_service()

        now = datetime.now(timezone.utc)
        memory = _make_memory("mem_1", workspace_id="ws_other", context_id="ctx_other", created_at=now)
        memories = [(memory, 0.8)]

        result = service.apply_scope_boosts(
            memories,
            query_context_id="ctx_query",
            query_workspace_id="ws_test",
            boosts=None,
        )

        assert result[0].boosted_score == pytest.approx(0.8, abs=0.01)
        assert result[0].source_scope == "other"

    def test_apply_scope_boosts_on_service_without_init(self):
        """apply_scope_boosts should fall back to ScopeBoosts() when no instance default."""
        # Create service without going through __init__ (edge case for legacy code)
        service = object.__new__(MemoryService)

        now = datetime.now(timezone.utc)
        memory = _make_memory("mem_1", workspace_id="ws_test", context_id="ctx_test", created_at=now)
        memories = [(memory, 0.8)]

        # Should not raise, falls back to ScopeBoosts() with hardcoded defaults
        result = service.apply_scope_boosts(
            memories,
            query_context_id="ctx_test",
            query_workspace_id="ws_test",
            boosts=None,
        )

        assert len(result) == 1
        # Default ScopeBoosts.same_context = 1.5
        assert result[0].boosted_score == pytest.approx(0.8 * 1.5, abs=0.01)
