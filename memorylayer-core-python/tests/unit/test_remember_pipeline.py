"""
Tests for the refactored remember() pipeline.

Verifies:
- _post_store_pipeline() runs association, contradiction, tier gen
- ingest_fact() deduplicates, stores, and runs post-store pipeline
- remember() conditionally routes to decomposition vs post-store pipeline
- _decompose_and_process_inline() decomposes and processes facts inline
- FactDecompositionTaskHandler uses ingest_fact() for per-fact pipeline
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from scitrera_app_framework import Variables

from memorylayer_server.models.memory import (
    RememberInput, MemoryType, MemoryStatus, Memory,
)
from memorylayer_server.services.memory import MemoryService
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_v():
    """Provide a Variables instance for test construction."""
    from memorylayer_server.config import (
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    )
    from memorylayer_server.services.association.base import (
        MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    )
    from memorylayer_server.services.memory.base import (
        MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    )
    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, True)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    return v


@pytest.fixture
def mock_storage():
    """Mock storage backend."""
    storage = AsyncMock()
    storage.create_memory = AsyncMock()
    storage.update_memory = AsyncMock()
    storage.get_memory = AsyncMock()
    storage.search_memories = AsyncMock(return_value=[])
    storage.create_association = AsyncMock()
    return storage


@pytest.fixture
def mock_embedding():
    """Mock embedding service."""
    service = AsyncMock()
    service.embed = AsyncMock(return_value=[0.1] * 384)
    return service


@pytest.fixture
def mock_dedup():
    """Mock deduplication service that always returns CREATE."""
    service = AsyncMock()
    service.check_duplicate = AsyncMock(return_value=DeduplicationResult(
        action=DeduplicationAction.CREATE,
        reason="New unique memory",
    ))
    return service


@pytest.fixture
def mock_task_service():
    """Mock task service."""
    service = AsyncMock()
    service.schedule_task = AsyncMock(return_value="task_123")
    return service


@pytest.fixture
def mock_tier_gen():
    """Mock tier generation service."""
    service = AsyncMock()
    service.request_tier_generation = AsyncMock(return_value=None)
    service.generate_tiers = AsyncMock(return_value=None)
    return service


@pytest.fixture
def mock_contradiction():
    """Mock contradiction service."""
    service = AsyncMock()
    service.check_new_memory = AsyncMock(return_value=[])
    return service


@pytest.fixture
def mock_association():
    """Mock association service."""
    service = AsyncMock()
    service.auto_enrich = AsyncMock()
    return service


def _make_memory(memory_id="mem_test", content="test content", **kwargs):
    """Create a Memory object for testing."""
    defaults = dict(
        id=memory_id,
        content=content,
        type=MemoryType.SEMANTIC,
        workspace_id="ws_test",
        tenant_id="default_tenant",
        content_hash="testhash",
        importance=0.5,
        tags=[],
        metadata={},
        embedding=[0.1] * 384,
        status=MemoryStatus.ACTIVE,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.fixture
def memory_service_unit(
    mock_v, mock_storage, mock_embedding, mock_dedup,
    mock_task_service, mock_tier_gen, mock_contradiction, mock_association,
):
    """Construct a MemoryService with all mocked dependencies."""
    mem = _make_memory()
    mock_storage.create_memory = AsyncMock(return_value=mem)
    mock_storage.update_memory = AsyncMock(return_value=mem)

    return MemoryService(
        storage=mock_storage,
        embedding_service=mock_embedding,
        deduplication_service=mock_dedup,
        association_service=mock_association,
        cache=None,
        v=mock_v,
        tier_generation_service=mock_tier_gen,
        contradiction_service=mock_contradiction,
        task_service=mock_task_service,
    )


# ---------------------------------------------------------------------------
# _post_store_pipeline tests
# ---------------------------------------------------------------------------

class TestPostStorePipeline:
    """Tests for _post_store_pipeline()."""

    @pytest.mark.asyncio
    async def test_schedules_auto_association_background(self, memory_service_unit, mock_task_service):
        """Background mode should schedule auto_enrich task."""
        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=False)

        mock_task_service.schedule_task.assert_called_once()
        call_args = mock_task_service.schedule_task.call_args
        assert call_args[0][0] == 'auto_enrich'

    @pytest.mark.asyncio
    async def test_runs_inline_association(self, memory_service_unit, mock_task_service, mock_storage):
        """Inline mode should call _inline_auto_enrich directly."""
        mock_task_service.schedule_task = AsyncMock()  # Should not be called
        mock_storage.search_memories = AsyncMock(return_value=[])

        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=True)

        # Should NOT schedule auto_enrich task
        for call in mock_task_service.schedule_task.call_args_list:
            assert call[0][0] != 'auto_enrich'

    @pytest.mark.asyncio
    async def test_calls_tier_generation(self, memory_service_unit, mock_tier_gen):
        """Should call request_tier_generation in background mode."""
        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=False)

        mock_tier_gen.request_tier_generation.assert_called_once_with(mem.id, "ws_test")

    @pytest.mark.asyncio
    async def test_calls_generate_tiers_inline(self, memory_service_unit, mock_tier_gen):
        """Should call generate_tiers directly in inline mode."""
        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=True)

        mock_tier_gen.generate_tiers.assert_called_once_with(mem.id, "ws_test")

    @pytest.mark.asyncio
    async def test_calls_contradiction_check(self, memory_service_unit, mock_contradiction):
        """Should call contradiction check."""
        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=False)

        mock_contradiction.check_new_memory.assert_called_once_with("ws_test", mem.id)

    @pytest.mark.asyncio
    async def test_handles_tier_gen_error_gracefully(self, memory_service_unit, mock_tier_gen, mock_contradiction):
        """Tier gen failure should not prevent other pipeline steps."""
        mock_tier_gen.request_tier_generation = AsyncMock(side_effect=RuntimeError("boom"))

        mem = _make_memory()
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=False)

        # Contradiction check should still run
        mock_contradiction.check_new_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_association_schedule_failure(
        self, memory_service_unit, mock_task_service, mock_storage,
    ):
        """If scheduling auto_enrich fails, should fall back to inline."""
        mock_task_service.schedule_task = AsyncMock(side_effect=RuntimeError("task service down"))
        mock_storage.search_memories = AsyncMock(return_value=[])

        mem = _make_memory()
        # Should not raise
        await memory_service_unit._post_store_pipeline("ws_test", mem, [0.1] * 384, inline=False)


# ---------------------------------------------------------------------------
# ingest_fact tests
# ---------------------------------------------------------------------------

class TestIngestFact:
    """Tests for ingest_fact()."""

    @pytest.mark.asyncio
    async def test_creates_new_fact(self, memory_service_unit, mock_storage, mock_dedup):
        """Should store a new fact when dedup returns CREATE."""
        fact_mem = _make_memory(memory_id="mem_fact1", content="atomic fact")
        mock_storage.create_memory = AsyncMock(return_value=fact_mem)
        mock_storage.update_memory = AsyncMock(return_value=fact_mem)

        input_data = RememberInput(content="atomic fact", type=MemoryType.SEMANTIC)
        result = await memory_service_unit.ingest_fact("ws_test", input_data)

        assert result is not None
        assert result.id == "mem_fact1"
        mock_storage.create_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_skip(self, memory_service_unit, mock_dedup):
        """Should return None when dedup returns SKIP."""
        mock_dedup.check_duplicate = AsyncMock(return_value=DeduplicationResult(
            action=DeduplicationAction.SKIP,
            existing_memory_id="mem_existing",
            reason="Exact duplicate",
        ))

        input_data = RememberInput(content="duplicate", type=MemoryType.SEMANTIC)
        result = await memory_service_unit.ingest_fact("ws_test", input_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_on_dedup_update(self, memory_service_unit, mock_dedup, mock_storage):
        """Should update existing memory when dedup returns UPDATE."""
        updated_mem = _make_memory(memory_id="mem_existing", content="updated")
        mock_dedup.check_duplicate = AsyncMock(return_value=DeduplicationResult(
            action=DeduplicationAction.UPDATE,
            existing_memory_id="mem_existing",
            similarity_score=0.96,
            reason="Semantic duplicate",
        ))
        mock_storage.update_memory = AsyncMock(return_value=updated_mem)

        input_data = RememberInput(content="updated content", type=MemoryType.SEMANTIC)
        result = await memory_service_unit.ingest_fact("ws_test", input_data)

        assert result is not None
        assert result.id == "mem_existing"
        mock_storage.update_memory.assert_called()

    @pytest.mark.asyncio
    async def test_sets_source_memory_id(self, memory_service_unit, mock_storage):
        """Should set source_memory_id when provided."""
        fact_mem = _make_memory(memory_id="mem_fact")
        mock_storage.create_memory = AsyncMock(return_value=fact_mem)
        mock_storage.update_memory = AsyncMock(return_value=fact_mem)

        input_data = RememberInput(content="decomposed fact", type=MemoryType.SEMANTIC)
        await memory_service_unit.ingest_fact(
            "ws_test", input_data, source_memory_id="mem_parent",
        )

        # Should have an update_memory call with source_memory_id
        update_calls = mock_storage.update_memory.call_args_list
        source_calls = [c for c in update_calls if c.kwargs.get('source_memory_id') == 'mem_parent']
        assert len(source_calls) >= 1

    @pytest.mark.asyncio
    async def test_runs_post_store_pipeline(
        self, memory_service_unit, mock_tier_gen, mock_contradiction, mock_storage,
    ):
        """Should run post-store pipeline after storing fact."""
        fact_mem = _make_memory(memory_id="mem_fact")
        mock_storage.create_memory = AsyncMock(return_value=fact_mem)
        mock_storage.update_memory = AsyncMock(return_value=fact_mem)

        input_data = RememberInput(content="fact for pipeline", type=MemoryType.SEMANTIC)
        await memory_service_unit.ingest_fact("ws_test", input_data)

        mock_tier_gen.request_tier_generation.assert_called_once()
        mock_contradiction.check_new_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_generates_embedding_when_not_provided(
        self, memory_service_unit, mock_embedding, mock_storage,
    ):
        """Should generate embedding if not provided."""
        fact_mem = _make_memory(memory_id="mem_fact")
        mock_storage.create_memory = AsyncMock(return_value=fact_mem)
        mock_storage.update_memory = AsyncMock(return_value=fact_mem)

        input_data = RememberInput(content="fact needing embedding", type=MemoryType.SEMANTIC)
        await memory_service_unit.ingest_fact("ws_test", input_data, embedding=None)

        mock_embedding.embed.assert_called_once_with("fact needing embedding")

    @pytest.mark.asyncio
    async def test_uses_provided_embedding(
        self, memory_service_unit, mock_embedding, mock_storage,
    ):
        """Should use provided embedding without generating a new one."""
        fact_mem = _make_memory(memory_id="mem_fact")
        mock_storage.create_memory = AsyncMock(return_value=fact_mem)
        mock_storage.update_memory = AsyncMock(return_value=fact_mem)

        pre_computed = [0.5] * 384
        input_data = RememberInput(content="fact with embedding", type=MemoryType.SEMANTIC)
        await memory_service_unit.ingest_fact("ws_test", input_data, embedding=pre_computed)

        mock_embedding.embed.assert_not_called()


# ---------------------------------------------------------------------------
# remember() conditional pipeline tests
# ---------------------------------------------------------------------------

class TestRememberConditionalPipeline:
    """Tests verifying remember() routes correctly based on decomposition."""

    @pytest.mark.asyncio
    async def test_non_decomposable_runs_post_store(
        self, memory_service_unit, mock_tier_gen, mock_contradiction,
    ):
        """Non-decomposable memory should run post-store pipeline directly."""
        input_data = RememberInput(content="short", type=MemoryType.SEMANTIC)
        await memory_service_unit.remember("ws_test", input_data)

        # Post-store pipeline should run (tier gen + contradiction)
        mock_tier_gen.request_tier_generation.assert_called_once()
        mock_contradiction.check_new_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_decomposable_schedules_decomposition(
        self, memory_service_unit, mock_storage, mock_task_service, mock_tier_gen, mock_contradiction,
    ):
        """Decomposable memory should schedule decompose_facts, not run post-store."""
        # Content that qualifies for decomposition (multiple sentences, long enough)
        content = "Drew likes Python for backend development. He also prefers dark mode for all editors."
        input_data = RememberInput(content=content, type=MemoryType.SEMANTIC)

        # Override mock to return memory with decomposable content
        decomposable_mem = _make_memory(content=content)
        mock_storage.create_memory = AsyncMock(return_value=decomposable_mem)
        mock_storage.update_memory = AsyncMock(return_value=decomposable_mem)

        await memory_service_unit.remember("ws_test", input_data)

        # Should have scheduled decompose_facts
        decompose_calls = [
            c for c in mock_task_service.schedule_task.call_args_list
            if c[0][0] == 'decompose_facts'
        ]
        assert len(decompose_calls) == 1

        # Should NOT have run post-store pipeline on the composite
        mock_tier_gen.request_tier_generation.assert_not_called()
        mock_contradiction.check_new_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_decomposable_no_auto_association_on_composite(
        self, memory_service_unit, mock_storage, mock_task_service,
    ):
        """Decomposable memory should NOT schedule auto_enrich on the composite."""
        content = "First sentence here. Second sentence here. Third for good measure."
        input_data = RememberInput(content=content, type=MemoryType.SEMANTIC)

        # Override mock to return memory with decomposable content
        decomposable_mem = _make_memory(content=content)
        mock_storage.create_memory = AsyncMock(return_value=decomposable_mem)
        mock_storage.update_memory = AsyncMock(return_value=decomposable_mem)

        await memory_service_unit.remember("ws_test", input_data)

        # auto_enrich should NOT be scheduled
        auto_assoc_calls = [
            c for c in mock_task_service.schedule_task.call_args_list
            if c[0][0] == 'auto_enrich'
        ]
        assert len(auto_assoc_calls) == 0

    @pytest.mark.asyncio
    async def test_decompose_failure_falls_back_to_post_store(
        self, memory_service_unit, mock_task_service, mock_tier_gen, mock_contradiction,
    ):
        """If scheduling decomposition fails, should fall back to post-store pipeline."""
        content = "First sentence for decomp. Second sentence for decomp."
        input_data = RememberInput(content=content, type=MemoryType.SEMANTIC)

        # Make schedule_task fail for decompose_facts
        async def selective_fail(task_type, payload, **kwargs):
            if task_type == 'decompose_facts':
                raise RuntimeError("task service down")
            return "task_123"

        mock_task_service.schedule_task = AsyncMock(side_effect=selective_fail)

        await memory_service_unit.remember("ws_test", input_data)

        # Should have fallen back to post-store pipeline
        mock_tier_gen.request_tier_generation.assert_called_once()
        mock_contradiction.check_new_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_remember_inline_parameter_accepted(self, memory_service_unit):
        """remember() should accept inline parameter without error."""
        input_data = RememberInput(content="short content", type=MemoryType.SEMANTIC)
        result = await memory_service_unit.remember("ws_test", input_data, inline=False)
        assert result is not None

    @pytest.mark.asyncio
    async def test_working_memory_not_decomposed(
        self, memory_service_unit, mock_task_service, mock_tier_gen,
    ):
        """WORKING type memories should never be decomposed."""
        content = "Currently working on this task. Making good progress on it."
        input_data = RememberInput(content=content, type=MemoryType.WORKING)

        await memory_service_unit.remember("ws_test", input_data)

        # Should NOT schedule decompose_facts
        decompose_calls = [
            c for c in mock_task_service.schedule_task.call_args_list
            if c[0][0] == 'decompose_facts'
        ]
        assert len(decompose_calls) == 0

        # Should have run post-store pipeline instead
        mock_tier_gen.request_tier_generation.assert_called_once()


# ---------------------------------------------------------------------------
# FactDecompositionTaskHandler integration tests
# ---------------------------------------------------------------------------

class TestFactDecompositionHandlerIntegration:
    """Tests for FactDecompositionTaskHandler using ingest_fact()."""

    @pytest.mark.asyncio
    async def test_handler_uses_ingest_fact(self):
        """Handler should route each fact through memory_service.ingest_fact()."""
        from memorylayer_server.services.extraction.fact_decomposition_handler import (
            FactDecompositionTaskHandler,
        )

        handler = FactDecompositionTaskHandler()
        mock_v = MagicMock()
        handler._v = mock_v

        parent = _make_memory(
            memory_id="mem_parent",
            content="Drew likes Python. He uses vim.",
            metadata={},
        )

        mock_storage = AsyncMock()
        mock_storage.get_memory = AsyncMock(return_value=parent)
        mock_storage.update_memory = AsyncMock()
        mock_storage.create_association = AsyncMock()

        mock_extraction = AsyncMock()
        mock_extraction.decompose_to_facts = AsyncMock(return_value=[
            {"content": "Drew likes Python"},
            {"content": "Drew uses vim"},
        ])

        mock_memory_service = AsyncMock()
        fact1 = _make_memory(memory_id="mem_fact1", content="Drew likes Python")
        fact2 = _make_memory(memory_id="mem_fact2", content="Drew uses vim")
        mock_memory_service.ingest_fact = AsyncMock(side_effect=[fact1, fact2])

        def get_ext(name, v):
            from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
            from memorylayer_server.services.extraction.base import EXT_EXTRACTION_SERVICE
            from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
            if name == EXT_STORAGE_BACKEND:
                return mock_storage
            elif name == EXT_EXTRACTION_SERVICE:
                return mock_extraction
            elif name == EXT_MEMORY_SERVICE:
                return mock_memory_service
            return MagicMock()

        with patch.object(handler, 'get_extension', side_effect=get_ext):
            await handler.handle({
                'memory_id': 'mem_parent',
                'workspace_id': 'ws_test',
            })

        # ingest_fact should have been called twice (once per fact)
        assert mock_memory_service.ingest_fact.call_count == 2

        # First call
        first_call = mock_memory_service.ingest_fact.call_args_list[0]
        assert first_call.kwargs['workspace_id'] == 'ws_test'
        assert first_call.kwargs['source_memory_id'] == 'mem_parent'
        assert first_call.kwargs['input'].content == 'Drew likes Python'

        # PART_OF associations should be created
        assert mock_storage.create_association.call_count == 2

        # Parent should be archived
        mock_storage.update_memory.assert_called_once_with(
            'ws_test', 'mem_parent', status=MemoryStatus.ARCHIVED.value,
        )

    @pytest.mark.asyncio
    async def test_handler_skips_atomic_memory(self):
        """Handler should skip decomposition for atomic content."""
        from memorylayer_server.services.extraction.fact_decomposition_handler import (
            FactDecompositionTaskHandler,
        )

        handler = FactDecompositionTaskHandler()
        mock_v = MagicMock()
        handler._v = mock_v

        parent = _make_memory(memory_id="mem_atomic", content="Single fact.")

        mock_storage = AsyncMock()
        mock_storage.get_memory = AsyncMock(return_value=parent)

        mock_extraction = AsyncMock()
        mock_extraction.decompose_to_facts = AsyncMock(return_value=[
            {"content": "Single fact."},
        ])

        mock_memory_service = AsyncMock()

        def get_ext(name, v):
            from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
            from memorylayer_server.services.extraction.base import EXT_EXTRACTION_SERVICE
            from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
            if name == EXT_STORAGE_BACKEND:
                return mock_storage
            elif name == EXT_EXTRACTION_SERVICE:
                return mock_extraction
            elif name == EXT_MEMORY_SERVICE:
                return mock_memory_service
            return MagicMock()

        with patch.object(handler, 'get_extension', side_effect=get_ext):
            await handler.handle({
                'memory_id': 'mem_atomic',
                'workspace_id': 'ws_test',
            })

        # ingest_fact should NOT have been called
        mock_memory_service.ingest_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_handles_dedup_skip_in_fact(self):
        """Handler should handle facts that are deduplicated (return None)."""
        from memorylayer_server.services.extraction.fact_decomposition_handler import (
            FactDecompositionTaskHandler,
        )

        handler = FactDecompositionTaskHandler()
        mock_v = MagicMock()
        handler._v = mock_v

        parent = _make_memory(
            memory_id="mem_parent2",
            content="Drew likes Python. Drew likes Python.",
            metadata={},
        )

        mock_storage = AsyncMock()
        mock_storage.get_memory = AsyncMock(return_value=parent)
        mock_storage.update_memory = AsyncMock()
        mock_storage.create_association = AsyncMock()

        mock_extraction = AsyncMock()
        mock_extraction.decompose_to_facts = AsyncMock(return_value=[
            {"content": "Drew likes Python"},
            {"content": "Drew likes Python"},  # Duplicate fact
        ])

        mock_memory_service = AsyncMock()
        fact1 = _make_memory(memory_id="mem_f1", content="Drew likes Python")
        # First fact creates, second is deduped (returns None)
        mock_memory_service.ingest_fact = AsyncMock(side_effect=[fact1, None])

        def get_ext(name, v):
            from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
            from memorylayer_server.services.extraction.base import EXT_EXTRACTION_SERVICE
            from memorylayer_server.services.memory import EXT_MEMORY_SERVICE
            if name == EXT_STORAGE_BACKEND:
                return mock_storage
            elif name == EXT_EXTRACTION_SERVICE:
                return mock_extraction
            elif name == EXT_MEMORY_SERVICE:
                return mock_memory_service
            return MagicMock()

        with patch.object(handler, 'get_extension', side_effect=get_ext):
            await handler.handle({
                'memory_id': 'mem_parent2',
                'workspace_id': 'ws_test',
            })

        # Only 1 PART_OF association (the non-None fact)
        assert mock_storage.create_association.call_count == 1

        # Parent still archived
        mock_storage.update_memory.assert_called_once()


# ---------------------------------------------------------------------------
# classify_type flag tests
# ---------------------------------------------------------------------------

class TestClassifyTypeFlag:
    """Tests for classify_type flag in the auto-enrich pipeline."""

    @pytest.mark.asyncio
    async def test_classify_type_true_in_payload_when_type_auto(
        self, memory_service_unit, mock_task_service,
    ):
        """When input.type is None, classify_type should be True in the task payload."""
        input_data = RememberInput(content="short", type=None)
        await memory_service_unit.remember("ws_test", input_data)

        enrich_calls = [
            c for c in mock_task_service.schedule_task.call_args_list
            if c[0][0] == 'auto_enrich'
        ]
        assert len(enrich_calls) == 1
        payload = enrich_calls[0][0][1]
        assert payload['classify_type'] is True

    @pytest.mark.asyncio
    async def test_classify_type_false_in_payload_when_type_explicit(
        self, memory_service_unit, mock_task_service,
    ):
        """When input.type is explicitly set, classify_type should be False."""
        input_data = RememberInput(content="short", type=MemoryType.EPISODIC)
        await memory_service_unit.remember("ws_test", input_data)

        enrich_calls = [
            c for c in mock_task_service.schedule_task.call_args_list
            if c[0][0] == 'auto_enrich'
        ]
        assert len(enrich_calls) == 1
        payload = enrich_calls[0][0][1]
        assert payload['classify_type'] is False

    @pytest.mark.asyncio
    async def test_inline_classify_type_calls_extraction_service(
        self, memory_service_unit, mock_storage,
    ):
        """Inline auto-enrich with classify_type=True should call extraction_service.classify_content."""
        mock_extraction = AsyncMock()
        mock_extraction.classify_content = AsyncMock(
            return_value=(MemoryType.PROCEDURAL, None),
        )
        memory_service_unit.extraction_service = mock_extraction
        memory_service_unit.task_service = None  # Force inline path

        mock_storage.search_memories = AsyncMock(return_value=[])

        mem = _make_memory(type=MemoryType.SEMANTIC)
        await memory_service_unit._post_store_pipeline(
            "ws_test", mem, [0.1] * 384, inline=True, classify_type=True,
        )

        mock_extraction.classify_content.assert_called_once_with(mem.content)

    @pytest.mark.asyncio
    async def test_inline_classify_type_false_skips_extraction(
        self, memory_service_unit, mock_storage,
    ):
        """Inline auto-enrich with classify_type=False should NOT call extraction service."""
        mock_extraction = AsyncMock()
        mock_extraction.classify_content = AsyncMock()
        memory_service_unit.extraction_service = mock_extraction
        memory_service_unit.task_service = None

        mock_storage.search_memories = AsyncMock(return_value=[])

        mem = _make_memory()
        await memory_service_unit._post_store_pipeline(
            "ws_test", mem, [0.1] * 384, inline=True, classify_type=False,
        )

        mock_extraction.classify_content.assert_not_called()

    @pytest.mark.asyncio
    async def test_inline_classify_type_updates_when_different(
        self, memory_service_unit, mock_storage,
    ):
        """When LLM classifies a different type, memory should be updated."""
        mock_extraction = AsyncMock()
        mock_extraction.classify_content = AsyncMock(
            return_value=(MemoryType.PROCEDURAL, None),
        )
        memory_service_unit.extraction_service = mock_extraction
        memory_service_unit.task_service = None

        mock_storage.search_memories = AsyncMock(return_value=[])
        mock_storage.update_memory = AsyncMock(return_value=_make_memory())

        mem = _make_memory(type=MemoryType.SEMANTIC)
        await memory_service_unit._post_store_pipeline(
            "ws_test", mem, [0.1] * 384, inline=True, classify_type=True,
        )

        # Should have called update_memory with the new type
        update_calls = mock_storage.update_memory.call_args_list
        type_updates = [
            c for c in update_calls
            if c.kwargs.get('type') == MemoryType.PROCEDURAL.value
        ]
        assert len(type_updates) == 1

    @pytest.mark.asyncio
    async def test_inline_classify_type_graceful_without_extraction_service(
        self, memory_service_unit, mock_storage,
    ):
        """classify_type=True should not fail when extraction_service is None."""
        memory_service_unit.extraction_service = None
        memory_service_unit.task_service = None

        mock_storage.search_memories = AsyncMock(return_value=[])

        mem = _make_memory()
        # Should not raise
        await memory_service_unit._post_store_pipeline(
            "ws_test", mem, [0.1] * 384, inline=True, classify_type=True,
        )
