"""
Unit tests for the perspective seed feature (observer_id / subject_id auto-population).

Tests:
- Speaker-attributed content gets observer_id set from the speaker name.
- Self-report: subject_id remains None (not auto-set to observer_id) by design.
- Explicit observer_id from caller is never overwritten.
- Content with no identifiable speaker leaves observer_id None (no error).
- Fact memory created via _store_fact_memories carries the source turn's observer_id.
- _seed_perspective_ids is tolerant of extraction_service failures.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemorySubtype,
    MemoryType,
    RememberInput,
)
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory import MemoryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_v():
    """Minimal Variables for MemoryService construction."""
    from memorylayer_server.config import (
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    )
    from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
    from memorylayer_server.services.memory.base import MEMORYLAYER_MEMORY_RECALL_OVERFETCH

    v = Variables()
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, False)  # disable decomp to keep tests simple
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 9999)
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    return v


def _make_memory(memory_id="mem_test", content="test content", **kwargs):
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
        embedding=[0.1] * 8,
        status=MemoryStatus.ACTIVE,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


@pytest.fixture
def mock_storage():
    storage = AsyncMock()
    mem = _make_memory()
    storage.create_memory = AsyncMock(return_value=mem)
    storage.update_memory = AsyncMock(return_value=mem)
    storage.get_memory = AsyncMock(return_value=mem)
    storage.search_memories = AsyncMock(return_value=[])
    storage.create_association = AsyncMock()
    return storage


@pytest.fixture
def mock_embedding():
    svc = AsyncMock()
    svc.embed = AsyncMock(return_value=[0.1] * 8)
    return svc


@pytest.fixture
def mock_dedup():
    svc = AsyncMock()
    svc.check_duplicate = AsyncMock(
        return_value=DeduplicationResult(action=DeduplicationAction.CREATE, reason="new")
    )
    return svc


@pytest.fixture
def memory_service(mock_v, mock_storage, mock_embedding, mock_dedup):
    """MemoryService with real _seed_perspective_ids, no task service (inline path)."""
    return MemoryService(
        storage=mock_storage,
        embedding_service=mock_embedding,
        deduplication_service=mock_dedup,
        association_service=None,
        cache=None,
        v=mock_v,
        task_service=None,
    )


# ---------------------------------------------------------------------------
# Unit tests for _seed_perspective_ids
# ---------------------------------------------------------------------------


class TestSeedPerspectiveIds:
    """Direct unit tests for _seed_perspective_ids()."""

    def test_speaker_content_sets_observer_id(self, memory_service):
        """Content with a dialogue speaker prefix yields observer_id = speaker."""
        content = "[2024-01-01 10:00] Alice: I prefer Python over Java."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs == "Alice"
        assert subj is None

    def test_explicit_observer_id_not_overwritten(self, memory_service):
        """Caller-provided observer_id is never overwritten, even with a speaker present."""
        content = "[2024-01-01 10:00] Alice: I prefer Python over Java."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id="explicit-agent", subject_id=None
        )
        assert obs == "explicit-agent"  # caller value preserved
        assert subj is None

    def test_no_speaker_leaves_observer_id_none(self, memory_service):
        """Content without a speaker prefix leaves observer_id as None."""
        content = "The database schema needs refactoring."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs is None
        assert subj is None

    def test_empty_content_returns_none(self, memory_service):
        """Empty content returns (None, None) without error."""
        obs, subj = memory_service._seed_perspective_ids(
            content="", observer_id=None, subject_id=None
        )
        assert obs is None
        assert subj is None

    def test_subject_id_not_auto_set_to_observer(self, memory_service):
        """subject_id is NOT automatically set to observer_id (self-report is caller's job)."""
        content = "[2024-01-01 10:00] Bob: I use Vim exclusively."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs == "Bob"
        assert subj is None  # not auto-mirrored

    def test_explicit_subject_id_preserved(self, memory_service):
        """An explicit subject_id is passed through unchanged."""
        content = "[2024-01-01 10:00] Alice: Bob prefers dark mode."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id="user-bob"
        )
        assert obs == "Alice"
        assert subj == "user-bob"

    def test_extraction_service_failure_is_tolerated(self, memory_service):
        """If extraction_service.extract_entities raises, seed returns (None, None) without error."""
        broken_svc = MagicMock()
        broken_svc.extract_entities = MagicMock(side_effect=RuntimeError("NER exploded"))
        memory_service.extraction_service = broken_svc

        content = "[2024-01-01 10:00] Alice: Hello."
        obs, subj = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs is None
        assert subj is None

    def test_uses_extraction_service_when_wired(self, memory_service):
        """When extraction_service is present, _seed_perspective_ids delegates to it."""
        mock_svc = MagicMock()
        mock_svc.extract_entities = MagicMock(
            return_value={"speaker": "Charlie", "entities": ["Charlie"]}
        )
        memory_service.extraction_service = mock_svc

        content = "[2024-01-01 10:00] Charlie: Hello world."
        obs, _ = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs == "Charlie"
        mock_svc.extract_entities.assert_called_once_with(content)

    def test_falls_back_to_regex_without_extraction_service(self, memory_service):
        """Without extraction_service, falls back to the shared regex extractor."""
        memory_service.extraction_service = None
        content = "[2024-01-01 10:00] Diana: Testing fallback."
        obs, _ = memory_service._seed_perspective_ids(
            content=content, observer_id=None, subject_id=None
        )
        assert obs == "Diana"


# ---------------------------------------------------------------------------
# Integration tests via remember() — persistence guarantee
# ---------------------------------------------------------------------------


class TestPerspectiveSeedViaRemember:
    """Tests that remember() stores observer_id/subject_id correctly via the seed."""

    @pytest.mark.asyncio
    async def test_speaker_content_persists_observer_id(self, memory_service, mock_storage):
        """A memory stored with speaker-attributed content gets observer_id persisted."""
        speaker_content = "[2024-01-01 10:00] Alice: I prefer statically typed languages."

        stored_mem = _make_memory(content=speaker_content, observer_id="Alice")
        mock_storage.create_memory = AsyncMock(return_value=stored_mem)
        mock_storage.update_memory = AsyncMock(return_value=stored_mem)

        memory = await memory_service.remember(
            "ws_test",
            RememberInput(content=speaker_content, type=MemoryType.SEMANTIC),
        )

        # Verify create_memory was called with observer_id="Alice"
        call_kwargs = mock_storage.create_memory.call_args
        passed_input: RememberInput = call_kwargs.args[1]
        assert passed_input.observer_id == "Alice"
        assert passed_input.subject_id is None

    @pytest.mark.asyncio
    async def test_explicit_observer_id_not_overwritten_via_remember(self, memory_service, mock_storage):
        """Explicit observer_id on RememberInput is never overwritten by the seed."""
        content = "[2024-01-01 10:00] Alice: Some observation."
        stored_mem = _make_memory(content=content, observer_id="caller-provided-agent")
        mock_storage.create_memory = AsyncMock(return_value=stored_mem)
        mock_storage.update_memory = AsyncMock(return_value=stored_mem)

        await memory_service.remember(
            "ws_test",
            RememberInput(
                content=content,
                type=MemoryType.SEMANTIC,
                observer_id="caller-provided-agent",
            ),
        )

        call_kwargs = mock_storage.create_memory.call_args
        passed_input: RememberInput = call_kwargs.args[1]
        assert passed_input.observer_id == "caller-provided-agent"  # not overwritten to "Alice"

    @pytest.mark.asyncio
    async def test_no_speaker_content_leaves_observer_id_none(self, memory_service, mock_storage):
        """Plain content without a speaker prefix results in observer_id=None."""
        content = "The service mesh should use mTLS for all internal communication."
        stored_mem = _make_memory(content=content)
        mock_storage.create_memory = AsyncMock(return_value=stored_mem)
        mock_storage.update_memory = AsyncMock(return_value=stored_mem)

        await memory_service.remember(
            "ws_test",
            RememberInput(content=content, type=MemoryType.SEMANTIC),
        )

        call_kwargs = mock_storage.create_memory.call_args
        passed_input: RememberInput = call_kwargs.args[1]
        assert passed_input.observer_id is None
        assert passed_input.subject_id is None


# ---------------------------------------------------------------------------
# Fact memory observer_id propagation
# ---------------------------------------------------------------------------


class TestFactObserverPropagation:
    """observer_id propagates from source memory to fact memories in _store_fact_memories."""

    @pytest.mark.asyncio
    async def test_fact_memory_inherits_source_observer_id(self, memory_service, mock_storage, mock_dedup):
        """Facts created from a source memory carry the source turn's observer_id."""
        source_mem = _make_memory(
            memory_id="mem_source",
            content="[2024-01-01 10:00] Eve: I enjoy graph databases.",
            observer_id="Eve",
        )

        ingested_facts = []

        async def capture_ingest_fact(workspace_id, input_data, *, source_memory_id=None, inline=True):
            ingested_facts.append(input_data)
            return _make_memory(
                memory_id=f"fact_{len(ingested_facts)}",
                content=input_data.content,
                observer_id=input_data.observer_id,
            )

        memory_service.ingest_fact = capture_ingest_fact

        facts = [
            {"subject": "Eve", "relation": "enjoys", "object": "graph databases"},
        ]
        await memory_service._store_fact_memories("ws_test", source_mem, facts)

        assert len(ingested_facts) == 1
        assert ingested_facts[0].observer_id == "Eve"
        assert ingested_facts[0].subject_id is None  # enterprise fills this; OSS leaves None

    @pytest.mark.asyncio
    async def test_fact_memory_observer_id_none_when_source_has_none(
        self, memory_service, mock_storage, mock_dedup
    ):
        """When source memory has no observer, fact inherits None (no error)."""
        source_mem = _make_memory(
            memory_id="mem_source_anon",
            content="The sky is blue.",
            observer_id=None,
        )

        ingested_facts = []

        async def capture_ingest_fact(workspace_id, input_data, *, source_memory_id=None, inline=True):
            ingested_facts.append(input_data)
            return _make_memory(content=input_data.content)

        memory_service.ingest_fact = capture_ingest_fact

        facts = [{"subject": "sky", "relation": "is", "object": "blue"}]
        await memory_service._store_fact_memories("ws_test", source_mem, facts)

        assert len(ingested_facts) == 1
        assert ingested_facts[0].observer_id is None
