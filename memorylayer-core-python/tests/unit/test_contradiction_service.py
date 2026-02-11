"""Tests for ContradictionService - negation detection, contradiction creation, resolution logic."""
import pytest
import pytest_asyncio

from memorylayer_server.services.contradiction.base import ContradictionRecord
from memorylayer_server.services.contradiction.default import (
    DefaultContradictionService,
    NEGATION_PAIRS,
)
from memorylayer_server.models.memory import RememberInput, MemoryType


# =============================================================================
# Pure unit tests: negation pattern detection (no fixtures needed)
# =============================================================================


class TestNegationPatternDetection:
    """Tests for _has_negation_pattern static method."""

    def test_detects_use_vs_dont_use(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Use React for the frontend",
            "Don't use React for the frontend"
        )

    def test_detects_enable_vs_disable(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Enable dark mode by default",
            "Disable dark mode by default"
        )

    def test_detects_always_vs_never(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Always use type hints",
            "Never use type hints"
        )

    def test_detects_true_vs_false(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Set debug to true",
            "Set debug to false"
        )

    def test_detects_should_vs_should_not(self):
        assert DefaultContradictionService._has_negation_pattern(
            "You should use async",
            "You should not use async"
        )

    def test_detects_prefer_vs_avoid(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Prefer composition over inheritance",
            "Avoid composition, use inheritance"
        )

    def test_detects_include_vs_exclude(self):
        assert DefaultContradictionService._has_negation_pattern(
            "Include logging in all services",
            "Exclude logging from services"
        )

    def test_detects_bidirectional(self):
        """Negation should be detected regardless of which text has the positive term."""
        assert DefaultContradictionService._has_negation_pattern(
            "Don't use tabs",
            "Use tabs for indentation"
        )

    def test_case_insensitive(self):
        assert DefaultContradictionService._has_negation_pattern(
            "ALWAYS run tests",
            "Never run tests"
        )

    def test_no_negation_for_unrelated_texts(self):
        assert not DefaultContradictionService._has_negation_pattern(
            "The sky is blue",
            "Python is a programming language"
        )

    def test_no_negation_for_agreeing_texts(self):
        assert not DefaultContradictionService._has_negation_pattern(
            "Use Python for backend",
            "Use Python for data science"
        )

    def test_negation_pairs_list_is_populated(self):
        """Ensure NEGATION_PAIRS has meaningful entries."""
        assert len(NEGATION_PAIRS) > 10
        for pair in NEGATION_PAIRS:
            assert len(pair) == 2
            assert pair[0] != pair[1]


# =============================================================================
# ContradictionRecord tests
# =============================================================================


class TestContradictionRecord:
    """Tests for the ContradictionRecord dataclass."""

    def test_default_id_generated(self):
        record = ContradictionRecord(
            workspace_id="ws1",
            memory_a_id="mem_a",
            memory_b_id="mem_b",
        )
        assert record.id.startswith("contra_")

    def test_default_confidence_is_zero(self):
        record = ContradictionRecord()
        assert record.confidence == 0.0

    def test_fields_stored(self):
        record = ContradictionRecord(
            workspace_id="ws1",
            memory_a_id="mem_a",
            memory_b_id="mem_b",
            contradiction_type="negation",
            confidence=0.85,
            detection_method="negation_pattern",
        )
        assert record.workspace_id == "ws1"
        assert record.memory_a_id == "mem_a"
        assert record.memory_b_id == "mem_b"
        assert record.contradiction_type == "negation"
        assert record.confidence == 0.85
        assert record.detection_method == "negation_pattern"
        assert record.resolved_at is None
        assert record.resolution is None


# =============================================================================
# Integration tests: contradiction detection with storage backend
# =============================================================================


async def _create_memory_with_embedding(storage_backend, embedding_service, workspace_id, content, embedding_text=None):
    """Helper: create a memory and assign an embedding.

    Args:
        embedding_text: Text to embed. If None, uses content. Pass the same value
                        for two memories to guarantee they appear as similar in
                        vector search (mock embedder is hash-based).
    """
    emb_source = embedding_text if embedding_text is not None else content
    embedding = (await embedding_service.embed_batch([emb_source]))[0]
    mem_input = RememberInput(content=content, importance=0.8)
    mem = await storage_backend.create_memory(workspace_id, mem_input)
    mem = await storage_backend.update_memory(workspace_id, mem.id, embedding=embedding)
    return mem


@pytest.mark.asyncio
class TestContradictionDetection:
    """Test contradiction detection against real storage."""

    async def test_check_new_memory_finds_negation(
        self, storage_backend, embedding_service, workspace_id
    ):
        """When two memories share an embedding but have negating content, detect contradiction."""
        service = DefaultContradictionService(storage=storage_backend)

        # Use the SAME embedding text so mock embedder produces identical vectors
        # (mock embedder is hash-based, different strings => unrelated vectors)
        shared_embedding_text = "tabs for indentation"

        mem1 = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="Use tabs for indentation",
            embedding_text=shared_embedding_text,
        )
        mem2 = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="Don't use tabs for indentation",
            embedding_text=shared_embedding_text,
        )

        contradictions = await service.check_new_memory(workspace_id, mem2.id)

        assert len(contradictions) >= 1
        found = any(
            c.memory_a_id == mem2.id and c.memory_b_id == mem1.id
            for c in contradictions
        )
        assert found, f"Expected contradiction between {mem2.id} and {mem1.id}"

    async def test_check_new_memory_no_contradiction_for_unrelated(
        self, storage_backend, embedding_service, workspace_id
    ):
        """Unrelated memories should not trigger contradictions."""
        service = DefaultContradictionService(storage=storage_backend)

        mem1 = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="Python is great for data science",
        )
        mem2 = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="The weather is sunny today",
        )

        contradictions = await service.check_new_memory(workspace_id, mem2.id)
        related = [c for c in contradictions if c.memory_b_id == mem1.id]
        assert len(related) == 0

    async def test_check_new_memory_skips_without_embedding(
        self, storage_backend, workspace_id
    ):
        """Memory without embedding should return empty contradictions list."""
        service = DefaultContradictionService(storage=storage_backend)

        input1 = RememberInput(content="No embedding here", importance=0.5)
        mem1 = await storage_backend.create_memory(workspace_id, input1)

        contradictions = await service.check_new_memory(workspace_id, mem1.id)
        assert contradictions == []

    async def test_check_new_memory_nonexistent_returns_empty(
        self, storage_backend, workspace_id
    ):
        """Nonexistent memory ID should return empty list."""
        service = DefaultContradictionService(storage=storage_backend)
        contradictions = await service.check_new_memory(workspace_id, "mem_nonexistent")
        assert contradictions == []


# =============================================================================
# Resolution tests
# =============================================================================


@pytest.mark.asyncio
class TestContradictionResolution:
    """Test contradiction resolution logic."""

    async def test_resolve_keep_a_soft_deletes_b(
        self, storage_backend, embedding_service, workspace_id
    ):
        """Resolving with keep_a should soft-delete the contradiction's memory_b.

        check_new_memory sets memory_a_id=new_memory, memory_b_id=existing_memory.
        keep_a keeps memory_a (the new memory) and soft-deletes memory_b (the existing one).
        """
        service = DefaultContradictionService(storage=storage_backend)

        shared_embedding_text = "caching for services"

        existing_mem = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="Enable caching for all services",
            embedding_text=shared_embedding_text,
        )
        new_mem = await _create_memory_with_embedding(
            storage_backend, embedding_service, workspace_id,
            content="Disable caching for all services",
            embedding_text=shared_embedding_text,
        )

        # Detect contradiction: memory_a_id=new_mem.id, memory_b_id=existing_mem.id
        contradictions = await service.check_new_memory(workspace_id, new_mem.id)
        assert len(contradictions) >= 1
        contradiction = contradictions[0]
        assert contradiction.memory_a_id == new_mem.id
        assert contradiction.memory_b_id == existing_mem.id

        # Resolve: keep A (new_mem) → soft-delete B (existing_mem)
        resolved = await service.resolve(
            workspace_id, contradiction.id, "keep_a"
        )
        assert resolved is not None
        assert resolved.resolution == "keep_a"

        # existing_mem (memory_b) should be soft-deleted
        existing_after = await storage_backend.get_memory(workspace_id, existing_mem.id)
        assert existing_after is None  # get_memory filters deleted

        # new_mem (memory_a) should still be accessible
        new_after = await storage_backend.get_memory(workspace_id, new_mem.id)
        assert new_after is not None

    async def test_resolve_keep_both(
        self, storage_backend, workspace_id
    ):
        """Resolving with keep_both should keep both memories intact."""
        service = DefaultContradictionService(storage=storage_backend)

        # Create real memories to satisfy FK constraints
        input_a = RememberInput(content="Memory A for keep_both test", importance=0.5)
        input_b = RememberInput(content="Memory B for keep_both test", importance=0.5)
        mem_a = await storage_backend.create_memory(workspace_id, input_a)
        mem_b = await storage_backend.create_memory(workspace_id, input_b)

        record = ContradictionRecord(
            workspace_id=workspace_id,
            memory_a_id=mem_a.id,
            memory_b_id=mem_b.id,
            contradiction_type="negation",
            confidence=0.9,
            detection_method="negation_pattern",
        )
        stored = await storage_backend.create_contradiction(record)

        resolved = await service.resolve(workspace_id, stored.id, "keep_both")
        assert resolved is not None
        assert resolved.resolution == "keep_both"

    async def test_resolve_nonexistent_returns_none(
        self, storage_backend, workspace_id
    ):
        """Resolving a nonexistent contradiction should return None."""
        service = DefaultContradictionService(storage=storage_backend)
        result = await service.resolve(workspace_id, "contra_nonexistent", "keep_a")
        assert result is None


# =============================================================================
# Get unresolved tests
# =============================================================================


@pytest.mark.asyncio
class TestGetUnresolved:
    """Test retrieving unresolved contradictions."""

    async def test_get_unresolved_returns_pending(
        self, storage_backend, workspace_id
    ):
        """Unresolved contradictions should be returned."""
        service = DefaultContradictionService(storage=storage_backend)

        # Create real memories for FK constraints
        input_a = RememberInput(content="Memory for unresolved test A", importance=0.5)
        input_b = RememberInput(content="Memory for unresolved test B", importance=0.5)
        mem_a = await storage_backend.create_memory(workspace_id, input_a)
        mem_b = await storage_backend.create_memory(workspace_id, input_b)

        record = ContradictionRecord(
            workspace_id=workspace_id,
            memory_a_id=mem_a.id,
            memory_b_id=mem_b.id,
            contradiction_type="negation",
            confidence=0.75,
            detection_method="negation_pattern",
        )
        await storage_backend.create_contradiction(record)

        unresolved = await service.get_unresolved(workspace_id, limit=50)
        assert len(unresolved) >= 1
        ids = [c.id for c in unresolved]
        assert record.id in ids

    async def test_get_unresolved_excludes_resolved(
        self, storage_backend, workspace_id
    ):
        """Resolved contradictions should not appear in unresolved list."""
        service = DefaultContradictionService(storage=storage_backend)

        # Create real memories for FK constraints
        input_a = RememberInput(content="Memory for resolved test A", importance=0.5)
        input_b = RememberInput(content="Memory for resolved test B", importance=0.5)
        mem_a = await storage_backend.create_memory(workspace_id, input_a)
        mem_b = await storage_backend.create_memory(workspace_id, input_b)

        record = ContradictionRecord(
            workspace_id=workspace_id,
            memory_a_id=mem_a.id,
            memory_b_id=mem_b.id,
            contradiction_type="negation",
            confidence=0.8,
            detection_method="negation_pattern",
        )
        stored = await storage_backend.create_contradiction(record)
        await service.resolve(workspace_id, stored.id, "keep_both")

        unresolved = await service.get_unresolved(workspace_id, limit=50)
        ids = [c.id for c in unresolved]
        assert stored.id not in ids

    async def test_get_unresolved_respects_limit(
        self, storage_backend, workspace_id
    ):
        """Limit parameter should be respected."""
        service = DefaultContradictionService(storage=storage_backend)
        unresolved = await service.get_unresolved(workspace_id, limit=1)
        assert len(unresolved) <= 1
