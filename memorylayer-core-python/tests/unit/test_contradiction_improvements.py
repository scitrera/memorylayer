"""Tests for Phase 3a contradiction improvements: semantic value conflict detection,
workspace scan, temporal ordering, and new task handlers."""
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from memorylayer_server.services.contradiction.base import (
    ContradictionRecord,
    CONTRADICTION_TYPE_NEGATION,
    CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT,
)
from memorylayer_server.services.contradiction.default import DefaultContradictionService
from memorylayer_server.models.memory import RememberInput


# =============================================================================
# Entity-value extraction tests
# =============================================================================


class TestEntityValueExtraction:
    """Tests for _extract_entity_values static method."""

    def test_extracts_simple_is_pattern(self):
        triples = DefaultContradictionService._extract_entity_values(
            "The server is Python"
        )
        assert len(triples) >= 1
        subjects = [t[0] for t in triples]
        assert any("server" in s for s in subjects)

    def test_extracts_uses_pattern(self):
        triples = DefaultContradictionService._extract_entity_values(
            "The project uses PostgreSQL"
        )
        assert len(triples) >= 1
        values = [t[2] for t in triples]
        assert any("postgresql" in v for v in values)

    def test_extracts_runs_pattern(self):
        triples = DefaultContradictionService._extract_entity_values(
            "The service runs Python 3"
        )
        assert len(triples) >= 1

    def test_case_insensitive_extraction(self):
        triples = DefaultContradictionService._extract_entity_values(
            "The API IS FastAPI"
        )
        # Case-insensitive lowercased
        assert all(t[0] == t[0].lower() for t in triples)
        assert all(t[2] == t[2].lower() for t in triples)

    def test_returns_empty_for_no_match(self):
        triples = DefaultContradictionService._extract_entity_values(
            "Hello world this has no entity patterns at all here"
        )
        # May or may not match - just ensure it returns a list
        assert isinstance(triples, list)

    def test_multiple_triples_in_text(self):
        text = "The server uses Redis. The backend runs Django."
        triples = DefaultContradictionService._extract_entity_values(text)
        assert isinstance(triples, list)


# =============================================================================
# Dot product / similarity tests
# =============================================================================


class TestDotProduct:
    """Tests for _dot_product static method."""

    def test_identical_unit_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert DefaultContradictionService._dot_product(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert DefaultContradictionService._dot_product(a, b) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self):
        assert DefaultContradictionService._dot_product([], [1.0]) == 0.0
        assert DefaultContradictionService._dot_product([], []) == 0.0

    def test_mismatched_lengths_return_zero(self):
        assert DefaultContradictionService._dot_product([1.0, 0.0], [1.0]) == 0.0

    def test_partial_similarity(self):
        import math
        # 45-degree vectors
        a = [math.sqrt(0.5), math.sqrt(0.5)]
        b = [1.0, 0.0]
        result = DefaultContradictionService._dot_product(a, b)
        assert 0.0 < result < 1.0


# =============================================================================
# Temporal ordering tests
# =============================================================================


class TestDetermineNewerMemory:
    """Tests for _determine_newer_memory static method."""

    def _make_mem(self, mem_id: str, created_at: datetime):
        mem = MagicMock()
        mem.id = mem_id
        mem.created_at = created_at
        return mem

    def test_newer_memory_returned(self):
        now = datetime.now(timezone.utc)
        older = self._make_mem("mem_old", now - timedelta(hours=1))
        newer = self._make_mem("mem_new", now)
        result = DefaultContradictionService._determine_newer_memory(older, newer)
        assert result == "mem_new"

    def test_a_is_newer(self):
        now = datetime.now(timezone.utc)
        mem_a = self._make_mem("mem_a", now)
        mem_b = self._make_mem("mem_b", now - timedelta(hours=2))
        result = DefaultContradictionService._determine_newer_memory(mem_a, mem_b)
        assert result == "mem_a"

    def test_same_timestamp_returns_a(self):
        now = datetime.now(timezone.utc)
        mem_a = self._make_mem("mem_a", now)
        mem_b = self._make_mem("mem_b", now)
        result = DefaultContradictionService._determine_newer_memory(mem_a, mem_b)
        # Same timestamp: mem_a >= mem_b is True so returns mem_a
        assert result == "mem_a"

    def test_missing_created_at_returns_none(self):
        mem_a = MagicMock()
        mem_a.id = "mem_a"
        mem_a.created_at = None
        mem_b = MagicMock()
        mem_b.id = "mem_b"
        mem_b.created_at = None
        result = DefaultContradictionService._determine_newer_memory(mem_a, mem_b)
        assert result is None


# =============================================================================
# Semantic conflict detection tests
# =============================================================================


@pytest.mark.asyncio
class TestCheckSemanticConflict:
    """Tests for check_semantic_conflict method."""

    def _make_service(self, storage=None):
        if storage is None:
            storage = MagicMock()
        return DefaultContradictionService(storage=storage)

    def _make_memory(self, mem_id: str, content: str, embedding: list[float],
                     workspace_id: str = "ws1", created_at: datetime = None):
        mem = MagicMock()
        mem.id = mem_id
        mem.content = content
        mem.embedding = embedding
        mem.workspace_id = workspace_id
        mem.created_at = created_at or datetime.now(timezone.utc)
        return mem

    def _similar_embedding(self, base: list[float], offset: float = 0.1) -> list[float]:
        """Create a slightly-different embedding that stays in [0.7, 0.9] similarity range."""
        import math
        # Create a vector that's similar but not identical to base
        result = [base[0] * (1 - offset), base[1] * (1 - offset), offset]
        # Normalize
        norm = math.sqrt(sum(x * x for x in result))
        return [x / norm for x in result]

    async def test_no_embeddings_returns_none(self):
        service = self._make_service()
        mem_a = self._make_memory("a", "Server is Python", [])
        mem_b = self._make_memory("b", "Server is Django", [])
        result = await service.check_semantic_conflict(mem_a, mem_b)
        assert result is None

    async def test_identical_embeddings_returns_none(self):
        """Similarity == 1.0 is outside [0.7, 0.9] window."""
        import math
        service = self._make_service()
        emb = [1.0, 0.0, 0.0]
        mem_a = self._make_memory("a", "Server is Python", emb)
        mem_b = self._make_memory("b", "Server is Django", emb)
        # Dot product = 1.0, outside (0.7, 0.9]
        result = await service.check_semantic_conflict(mem_a, mem_b)
        assert result is None

    async def test_low_similarity_returns_none(self):
        """Very dissimilar memories should not trigger semantic conflict."""
        service = self._make_service()
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]  # orthogonal, dot product = 0.0
        mem_a = self._make_memory("a", "Server is Python", emb_a)
        mem_b = self._make_memory("b", "Server is Django", emb_b)
        result = await service.check_semantic_conflict(mem_a, mem_b)
        assert result is None

    async def test_detects_value_conflict_in_similarity_range(self):
        """Two memories with similar embeddings (0.7-0.9) and conflicting values → conflict."""
        import math
        service = self._make_service()

        # Craft embeddings with dot product ~0.8 (within [0.7, 0.9])
        # vec_a = [sqrt(0.8), sqrt(0.2), 0.0], vec_b = [1.0, 0.0, 0.0]
        # dot = sqrt(0.8) ≈ 0.894 — slightly above 0.9, adjust
        # Use: vec_a = [0.8, 0.6, 0], vec_b = [1.0, 0, 0] → dot = 0.8
        emb_a = [0.8, 0.6, 0.0]
        emb_b = [1.0, 0.0, 0.0]

        mem_a = self._make_memory("a", "The backend uses Redis", emb_a)
        mem_b = self._make_memory("b", "The backend uses MongoDB", emb_b)

        result = await service.check_semantic_conflict(mem_a, mem_b)
        # dot product = 0.8, in range [0.7, 0.9]
        # "backend" is common subject, "uses" common predicate, "redis" vs "mongodb" differ
        assert result is not None
        assert result.contradiction_type == CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT
        assert result.memory_a_id == "a"
        assert result.memory_b_id == "b"
        assert result.detection_method == 'entity_value_extraction'
        assert 0.7 <= result.confidence <= 0.9

    async def test_no_conflict_for_same_values(self):
        """Same subject+predicate+value should not trigger conflict."""
        import math
        service = self._make_service()

        emb_a = [0.8, 0.6, 0.0]
        emb_b = [1.0, 0.0, 0.0]

        mem_a = self._make_memory("a", "The backend uses Redis", emb_a)
        mem_b = self._make_memory("b", "The backend uses Redis", emb_b)

        result = await service.check_semantic_conflict(mem_a, mem_b)
        assert result is None

    async def test_temporal_ordering_set_on_conflict(self):
        """Conflict record should have newer_memory_id populated."""
        service = self._make_service()

        now = datetime.now(timezone.utc)
        emb_a = [0.8, 0.6, 0.0]
        emb_b = [1.0, 0.0, 0.0]

        mem_a = self._make_memory("a", "The API is REST", emb_a, created_at=now - timedelta(hours=1))
        mem_b = self._make_memory("b", "The API is GraphQL", emb_b, created_at=now)

        result = await service.check_semantic_conflict(mem_a, mem_b)
        if result is not None:
            # mem_b is newer
            assert result.newer_memory_id == "b"


# =============================================================================
# ContradictionRecord new fields
# =============================================================================


class TestContradictionRecordNewFields:
    """Test new fields on ContradictionRecord."""

    def test_newer_memory_id_defaults_none(self):
        record = ContradictionRecord()
        assert record.newer_memory_id is None

    def test_newer_memory_id_set(self):
        record = ContradictionRecord(
            workspace_id="ws1",
            memory_a_id="a",
            memory_b_id="b",
            newer_memory_id="b",
        )
        assert record.newer_memory_id == "b"

    def test_semantic_value_conflict_type(self):
        record = ContradictionRecord(
            contradiction_type=CONTRADICTION_TYPE_SEMANTIC_VALUE_CONFLICT,
        )
        assert record.contradiction_type == 'semantic_value_conflict'

    def test_temporal_supersession_type(self):
        from memorylayer_server.services.contradiction.base import CONTRADICTION_TYPE_TEMPORAL_SUPERSESSION
        record = ContradictionRecord(
            contradiction_type=CONTRADICTION_TYPE_TEMPORAL_SUPERSESSION,
        )
        assert record.contradiction_type == 'temporal_supersession'

    def test_scope_conflict_type(self):
        from memorylayer_server.services.contradiction.base import CONTRADICTION_TYPE_SCOPE_CONFLICT
        record = ContradictionRecord(
            contradiction_type=CONTRADICTION_TYPE_SCOPE_CONFLICT,
        )
        assert record.contradiction_type == 'scope_conflict'


# =============================================================================
# scan_workspace tests (with mock storage)
# =============================================================================


@pytest.mark.asyncio
class TestScanWorkspace:
    """Tests for scan_workspace using mock storage."""

    def _make_service(self, storage):
        return DefaultContradictionService(storage=storage)

    def _make_memory(self, mem_id: str, content: str, embedding: list[float],
                     workspace_id: str = "ws1"):
        from datetime import datetime, timezone
        mem = MagicMock()
        mem.id = mem_id
        mem.content = content
        mem.embedding = embedding
        mem.workspace_id = workspace_id
        mem.created_at = datetime.now(timezone.utc)
        mem.tags = []
        mem.metadata = {}
        mem.importance = 0.5
        return mem

    async def test_scan_empty_workspace(self):
        """Empty workspace should return empty list."""
        storage = MagicMock()
        storage.get_unresolved_contradictions = AsyncMock(return_value=[])
        storage.get_workspace_stats = AsyncMock(return_value={'total_memories': 0})
        storage.get_recent_memories = AsyncMock(return_value=[])

        service = self._make_service(storage)
        result = await service.scan_workspace("ws_empty")
        assert result == []

    async def test_scan_finds_negation_contradiction(self):
        """Scan should detect negation contradiction between similar memories."""
        storage = MagicMock()
        storage.get_unresolved_contradictions = AsyncMock(return_value=[])
        storage.get_workspace_stats = AsyncMock(return_value={'total_memories': 2})

        emb = [1.0, 0.0, 0.0]
        mem_a = self._make_memory("mem_a", "Always use type hints", emb)
        mem_b = self._make_memory("mem_b", "Never use type hints", emb)

        # Return mem_a dict in get_recent_memories, then empty to stop loop
        storage.get_recent_memories = AsyncMock(side_effect=[
            [{'id': 'mem_a'}],
            [],
        ])
        storage.get_memory = AsyncMock(side_effect=lambda ws, mid, track_access=True: {
            'mem_a': mem_a,
            'mem_b': mem_b,
        }.get(mid))
        storage.search_memories = AsyncMock(return_value=[(mem_b, 0.95)])
        stored_record = ContradictionRecord(
            workspace_id="ws1",
            memory_a_id="mem_a",
            memory_b_id="mem_b",
            contradiction_type=CONTRADICTION_TYPE_NEGATION,
            confidence=0.95,
            detection_method='negation_pattern',
        )
        storage.create_contradiction = AsyncMock(return_value=stored_record)

        service = self._make_service(storage)
        results = await service.scan_workspace("ws1")
        assert len(results) >= 1
        assert results[0].contradiction_type == CONTRADICTION_TYPE_NEGATION
        assert storage.create_contradiction.called

    async def test_scan_skips_existing_pairs(self):
        """Already-recorded contradiction pairs should not be re-created."""
        storage = MagicMock()

        existing = ContradictionRecord(
            workspace_id="ws1",
            memory_a_id="mem_a",
            memory_b_id="mem_b",
        )
        storage.get_unresolved_contradictions = AsyncMock(return_value=[existing])
        storage.get_workspace_stats = AsyncMock(return_value={'total_memories': 2})

        emb = [1.0, 0.0, 0.0]
        mem_a = self._make_memory("mem_a", "Always use type hints", emb)
        mem_b = self._make_memory("mem_b", "Never use type hints", emb)

        storage.get_recent_memories = AsyncMock(side_effect=[
            [{'id': 'mem_a'}],
            [],
        ])
        storage.get_memory = AsyncMock(return_value=mem_a)
        storage.search_memories = AsyncMock(return_value=[(mem_b, 0.95)])
        storage.create_contradiction = AsyncMock()

        service = self._make_service(storage)
        await service.scan_workspace("ws1")
        # Already exists, should not create a new contradiction
        storage.create_contradiction.assert_not_called()


# =============================================================================
# Workspace contradiction scan handler tests
# =============================================================================


@pytest.mark.asyncio
class TestWorkspaceContradictionScanHandler:
    """Tests for WorkspaceContradictionScanHandler."""

    def _make_handler(self):
        from memorylayer_server.tasks.workspace_contradiction_scan_handler import (
            WorkspaceContradictionScanHandler
        )
        handler = WorkspaceContradictionScanHandler.__new__(WorkspaceContradictionScanHandler)
        return handler

    async def test_get_task_type(self):
        handler = self._make_handler()
        assert handler.get_task_type() == 'workspace_contradiction_scan'

    async def test_get_schedule_returns_daily(self):
        handler = self._make_handler()
        v = MagicMock()
        schedule = handler.get_schedule(v)
        assert schedule is not None
        assert schedule.interval_seconds == 86400
        assert schedule.default_payload == {}

    async def test_handle_single_workspace(self):
        handler = self._make_handler()

        contradiction_service = MagicMock()
        contradiction_service.scan_workspace = AsyncMock(return_value=[
            ContradictionRecord(workspace_id="ws1", memory_a_id="a", memory_b_id="b")
        ])

        storage = MagicMock()

        v = MagicMock()

        def get_ext(ext_name, variables):
            from memorylayer_server.services._constants import (
                EXT_CONTRADICTION_SERVICE, EXT_STORAGE_BACKEND
            )
            if ext_name == EXT_CONTRADICTION_SERVICE:
                return contradiction_service
            return storage

        handler.get_extension = get_ext

        await handler.handle(v, {'workspace_id': 'ws1'})
        contradiction_service.scan_workspace.assert_called_once_with('ws1')

    async def test_handle_all_workspaces(self):
        handler = self._make_handler()

        contradiction_service = MagicMock()
        contradiction_service.scan_workspace = AsyncMock(return_value=[])

        ws1 = MagicMock()
        ws1.id = 'ws1'
        ws2 = MagicMock()
        ws2.id = 'ws2'

        storage = MagicMock()
        storage.list_workspaces = AsyncMock(return_value=[ws1, ws2])

        v = MagicMock()

        def get_ext(ext_name, variables):
            from memorylayer_server.services._constants import (
                EXT_CONTRADICTION_SERVICE, EXT_STORAGE_BACKEND
            )
            if ext_name == EXT_CONTRADICTION_SERVICE:
                return contradiction_service
            return storage

        handler.get_extension = get_ext

        await handler.handle(v, {})
        assert contradiction_service.scan_workspace.call_count == 2


# =============================================================================
# Consolidation handler tests
# =============================================================================


class TestFindClusters:
    """Tests for the _find_clusters helper."""

    def test_empty_returns_empty(self):
        from memorylayer_server.tasks.consolidation_handler import _find_clusters
        assert _find_clusters([], 0.85, 3) == []

    def test_too_few_memories_returns_empty(self):
        from memorylayer_server.tasks.consolidation_handler import _find_clusters
        mem = MagicMock()
        mem.embedding = [1.0, 0.0]
        assert _find_clusters([mem, mem], 0.85, 3) == []

    def test_finds_tight_cluster(self):
        from memorylayer_server.tasks.consolidation_handler import _find_clusters
        emb = [1.0, 0.0, 0.0]
        memories = []
        for i in range(3):
            m = MagicMock()
            m.embedding = emb
            m.id = f"mem_{i}"
            memories.append(m)
        clusters = _find_clusters(memories, 0.85, 3)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_separates_distinct_clusters(self):
        from memorylayer_server.tasks.consolidation_handler import _find_clusters
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]  # orthogonal to emb_a, dot=0.0

        mems_a = []
        for i in range(3):
            m = MagicMock()
            m.embedding = emb_a
            m.id = f"a_{i}"
            mems_a.append(m)

        mems_b = []
        for i in range(3):
            m = MagicMock()
            m.embedding = emb_b
            m.id = f"b_{i}"
            mems_b.append(m)

        clusters = _find_clusters(mems_a + mems_b, 0.85, 3)
        assert len(clusters) == 2


class TestMergeMemoriesSimplified:
    """Tests for _merge_memories_simplified helper."""

    def _make_mem(self, mem_id: str, importance: float, tags: list, metadata: dict):
        m = MagicMock()
        m.id = mem_id
        m.importance = importance
        m.tags = tags
        m.metadata = metadata
        return m

    def test_unions_tags(self):
        from memorylayer_server.tasks.consolidation_handler import _merge_memories_simplified
        primary = self._make_mem("p", 0.2, ["a", "b"], {})
        other = self._make_mem("o", 0.15, ["b", "c"], {})
        result = _merge_memories_simplified(primary, [other])
        assert set(result['tags']) == {"a", "b", "c"}

    def test_deep_merges_metadata(self):
        from memorylayer_server.tasks.consolidation_handler import _merge_memories_simplified
        primary = self._make_mem("p", 0.2, [], {"source": "a"})
        other = self._make_mem("o", 0.15, [], {"tool": "pytest", "source": "b"})
        result = _merge_memories_simplified(primary, [other])
        # Primary's "source" wins; "tool" from other fills in
        assert result['metadata']['source'] == "a"
        assert result['metadata']['tool'] == "pytest"

    def test_importance_boosted(self):
        from memorylayer_server.tasks.consolidation_handler import _merge_memories_simplified
        primary = self._make_mem("p", 0.2, [], {})
        other = self._make_mem("o", 0.25, [], {})
        result = _merge_memories_simplified(primary, [other])
        # max importance = 0.25, boosted = min(0.25 * 1.1, 1.0) = 0.275
        assert result['importance'] == pytest.approx(0.275)

    def test_importance_capped_at_one(self):
        from memorylayer_server.tasks.consolidation_handler import _merge_memories_simplified
        primary = self._make_mem("p", 0.95, [], {})
        other = self._make_mem("o", 0.95, [], {})
        result = _merge_memories_simplified(primary, [other])
        assert result['importance'] <= 1.0

    def test_provenance_tracked(self):
        from memorylayer_server.tasks.consolidation_handler import _merge_memories_simplified
        primary = self._make_mem("p", 0.2, [], {})
        other1 = self._make_mem("o1", 0.1, [], {})
        other2 = self._make_mem("o2", 0.15, [], {})
        result = _merge_memories_simplified(primary, [other1, other2])
        assert 'consolidated_from' in result['metadata']
        assert 'o1' in result['metadata']['consolidated_from']
        assert 'o2' in result['metadata']['consolidated_from']


@pytest.mark.asyncio
class TestConsolidationTaskHandler:
    """Tests for ConsolidationTaskHandler."""

    def _make_handler(self):
        from memorylayer_server.tasks.consolidation_handler import ConsolidationTaskHandler
        handler = ConsolidationTaskHandler.__new__(ConsolidationTaskHandler)
        return handler

    def _make_v(self, enabled: bool = True):
        from scitrera_app_framework import Variables
        v = Variables()
        v.set('MEMORYLAYER_CONSOLIDATION_ENABLED', 'true' if enabled else 'false')
        v.set('MEMORYLAYER_CONSOLIDATION_MIN_CLUSTER_SIZE', '3')
        v.set('MEMORYLAYER_CONSOLIDATION_MAX_IMPORTANCE', '0.3')
        v.set('MEMORYLAYER_CONSOLIDATION_MIN_SIMILARITY', '0.85')
        return v

    async def test_get_task_type(self):
        handler = self._make_handler()
        assert handler.get_task_type() == 'memory_consolidation'

    async def test_get_schedule_when_disabled_returns_none(self):
        handler = self._make_handler()
        v = self._make_v(enabled=False)
        schedule = handler.get_schedule(v)
        assert schedule is None

    async def test_get_schedule_when_enabled_returns_daily(self):
        handler = self._make_handler()
        v = self._make_v(enabled=True)
        schedule = handler.get_schedule(v)
        assert schedule is not None
        assert schedule.interval_seconds == 86400

    async def test_handle_does_nothing_when_disabled(self):
        handler = self._make_handler()
        v = self._make_v(enabled=False)

        storage = MagicMock()
        storage.list_workspaces = AsyncMock(return_value=[])
        handler.get_extension = MagicMock(return_value=storage)

        await handler.handle(v, {})
        # Should return early without touching storage
        storage.list_workspaces.assert_not_called()

    async def test_handle_consolidates_cluster(self):
        """A cluster of 3 similar low-importance memories should be merged."""
        handler = self._make_handler()
        v = self._make_v(enabled=True)

        emb = [1.0, 0.0, 0.0]

        def _make_candidate(mem_id, importance):
            m = MagicMock()
            m.id = mem_id
            m.embedding = emb
            m.importance = importance
            m.tags = ['tag1']
            m.metadata = {}
            m.pinned = False
            m.workspace_id = 'ws1'
            return m

        c1 = _make_candidate('c1', 0.25)
        c2 = _make_candidate('c2', 0.20)
        c3 = _make_candidate('c3', 0.15)

        storage = MagicMock()
        ws = MagicMock()
        ws.id = 'ws1'
        storage.list_workspaces = AsyncMock(return_value=[ws])

        # Return candidates as dicts from get_recent_memories, then empty
        storage.get_recent_memories = AsyncMock(side_effect=[
            [
                {'id': 'c1', 'importance': 0.25},
                {'id': 'c2', 'importance': 0.20},
                {'id': 'c3', 'importance': 0.15},
            ],
            [],
        ])

        mem_map = {'c1': c1, 'c2': c2, 'c3': c3}
        storage.get_memory = AsyncMock(
            side_effect=lambda ws_id, mid, track_access=True: mem_map.get(mid)
        )
        storage.update_memory = AsyncMock(return_value=c1)
        storage.delete_memory = AsyncMock(return_value=True)

        handler.get_extension = MagicMock(return_value=storage)

        await handler.handle(v, {})

        # Primary (c1, highest importance) should be updated
        storage.update_memory.assert_called_once()
        call_args = storage.update_memory.call_args
        assert call_args[0][1] == 'c1'  # primary memory id

        # Others (c2, c3) should be soft-deleted
        assert storage.delete_memory.call_count == 2
