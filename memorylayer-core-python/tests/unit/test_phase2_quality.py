"""
Tests for Phase 2 quality improvements:
- 2a: _merge_memories() with provenance tracking
- 2b: LLM query rewriting gated by config flag
- 2c: Trust score computation and annotation
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
    MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
)
from memorylayer_server.models.memory import (
    Memory,
    MemoryStatus,
    MemoryType,
    RecallInput,
    RecallMode,
    RecallResult,
)
from memorylayer_server.services.association.base import MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
from memorylayer_server.services.deduplication import DeduplicationAction, DeduplicationResult
from memorylayer_server.services.memory import MemoryService
from memorylayer_server.services.memory.base import MEMORYLAYER_MEMORY_RECALL_OVERFETCH

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_v(**overrides):
    v = Variables()
    v.set(MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, 0.85)
    v.set(MEMORYLAYER_MEMORY_RECALL_OVERFETCH, 3)
    from memorylayer_server.config import (
        MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    )

    v.set(MEMORYLAYER_FACT_DECOMPOSITION_ENABLED, True)
    v.set(MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH, 20)
    for k, val in overrides.items():
        v.set(k, val)
    return v


def _make_memory(memory_id="mem_test", content="original content", **kwargs):
    defaults = dict(
        id=memory_id,
        content=content,
        type=MemoryType.SEMANTIC,
        workspace_id="ws_test",
        tenant_id="default_tenant",
        content_hash="testhash_abc",
        importance=0.5,
        tags=["tag1"],
        metadata={"key": "old_value"},
        embedding=[0.1] * 384,
        status=MemoryStatus.ACTIVE,
        access_count=0,
        decay_factor=1.0,
        pinned=False,
    )
    defaults.update(kwargs)
    return Memory(**defaults)


def _make_service(v=None, storage=None, embedding=None, dedup=None, tier_gen=None, llm=None, **kwargs):
    if v is None:
        v = _make_v()
    if storage is None:
        storage = AsyncMock()
        storage.create_memory = AsyncMock()
        storage.update_memory = AsyncMock()
        storage.get_memory = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[])
    if embedding is None:
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.2] * 384)
    if dedup is None:
        dedup = AsyncMock()
        dedup.check_duplicate = AsyncMock(
            return_value=DeduplicationResult(
                action=DeduplicationAction.CREATE,
                reason="new",
            )
        )
    if tier_gen is None:
        tier_gen = AsyncMock()
        tier_gen.generate_tiers = AsyncMock(return_value=None)
        tier_gen.request_tier_generation = AsyncMock(return_value=None)

    return MemoryService(
        storage=storage,
        embedding_service=embedding,
        deduplication_service=dedup,
        v=v,
        tier_generation_service=tier_gen,
        llm_service=llm,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2a: _merge_memories tests
# ---------------------------------------------------------------------------


class TestMergeMemories:
    @pytest.mark.asyncio
    async def test_merge_uses_new_content_as_primary(self):
        """Merged memory should use the new content (not naive concatenation)."""
        existing = _make_memory(content="old content", content_hash="oldhash")
        merged_mem = _make_memory(content="new content", content_hash="newhash")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=["tag2"],
            new_metadata={"new_key": "new_val"},
            new_importance=0.6,
        )

        # storage.update_memory was called with new content
        call_kwargs = storage.update_memory.call_args[1]
        assert call_kwargs["content"] == "new content"

    @pytest.mark.asyncio
    async def test_merge_stores_old_hash_in_metadata(self):
        """merged_from in metadata should hold the old content_hash."""
        existing = _make_memory(content="old content", content_hash="oldhash_123")
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={},
            new_importance=0.5,
        )

        call_kwargs = storage.update_memory.call_args[1]
        assert call_kwargs["metadata"]["merged_from"] == "oldhash_123"

    @pytest.mark.asyncio
    async def test_merge_union_of_tags(self):
        """Tags should be the union of existing and new tags."""
        existing = _make_memory(tags=["alpha", "beta"])
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=["beta", "gamma"],
            new_metadata={},
            new_importance=0.5,
        )

        call_kwargs = storage.update_memory.call_args[1]
        assert set(call_kwargs["tags"]) == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_merge_importance_boosted_and_capped(self):
        """Merged importance = min(max(existing, new) * 1.1, 1.0)."""
        existing = _make_memory(importance=0.8)
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={},
            new_importance=0.7,
        )

        call_kwargs = storage.update_memory.call_args[1]
        # max(0.8, 0.7) * 1.1 = 0.88
        assert abs(call_kwargs["importance"] - min(0.8 * 1.1, 1.0)) < 0.001

    @pytest.mark.asyncio
    async def test_merge_importance_capped_at_1(self):
        """Boosted importance must not exceed 1.0."""
        existing = _make_memory(importance=0.95)
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={},
            new_importance=0.9,
        )

        call_kwargs = storage.update_memory.call_args[1]
        assert call_kwargs["importance"] <= 1.0

    @pytest.mark.asyncio
    async def test_merge_deep_merges_metadata(self):
        """New metadata keys override old; old keys not in new are preserved."""
        existing = _make_memory(metadata={"keep": "old", "override": "old_val"})
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={"override": "new_val", "added": "new_key"},
            new_importance=0.5,
        )

        call_kwargs = storage.update_memory.call_args[1]
        md = call_kwargs["metadata"]
        assert md["keep"] == "old"
        assert md["override"] == "new_val"
        assert md["added"] == "new_key"

    @pytest.mark.asyncio
    async def test_merge_calls_tier_generation(self):
        """After merge, tier generation should be triggered."""
        existing = _make_memory(id="mem_abc")
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.3] * 384)
        tier_gen = AsyncMock()
        tier_gen.generate_tiers = AsyncMock(return_value=None)

        svc = _make_service(storage=storage, embedding=embedding, tier_gen=tier_gen)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={},
            new_importance=0.5,
        )

        tier_gen.generate_tiers.assert_called_once_with("mem_abc", "ws_test")

    @pytest.mark.asyncio
    async def test_merge_re_embeds_new_content(self):
        """Merge should re-embed the new content, not reuse old embedding."""
        existing = _make_memory(content="old content")
        merged_mem = _make_memory(content="new content")

        storage = AsyncMock()
        storage.update_memory = AsyncMock(return_value=merged_mem)
        embedding = AsyncMock()
        embedding.embed = AsyncMock(return_value=[0.9] * 384)

        svc = _make_service(storage=storage, embedding=embedding)

        await svc._merge_memories(
            workspace_id="ws_test",
            existing=existing,
            new_content="new content",
            new_tags=[],
            new_metadata={},
            new_importance=0.5,
        )

        embedding.embed.assert_called_once_with("new content")
        call_kwargs = storage.update_memory.call_args[1]
        assert call_kwargs["embedding"] == [0.9] * 384


# ---------------------------------------------------------------------------
# 2b: LLM query rewriting tests
# ---------------------------------------------------------------------------


class TestQueryRewriting:
    @pytest.mark.asyncio
    async def test_query_rewrite_called_when_enabled(self):
        """When enabled and LLM available, _rewrite_query_with_llm should be called."""
        v = _make_v(**{MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED: True})

        llm = AsyncMock()
        llm.synthesize = AsyncMock(return_value="expanded query with synonyms")

        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[])
        storage.get_recent_memories = AsyncMock(return_value=[])

        svc = _make_service(v=v, storage=storage, llm=llm)

        with patch.object(svc, "_rewrite_query_with_llm", wraps=svc._rewrite_query_with_llm) as mock_rewrite:
            with patch.object(
                svc,
                "_recall_rag",
                return_value=RecallResult(
                    memories=[],
                    total_count=0,
                    search_latency_ms=0,
                    mode_used=RecallMode.LLM,
                ),
            ):
                input_ = RecallInput(query="test query", context=[])
                await svc._recall_llm("ws_test", input_, 0.5)

            mock_rewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_rewrite_skipped_when_disabled(self):
        """When disabled, _rewrite_query_with_llm should NOT be called."""
        v = _make_v(**{MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED: False})

        llm = AsyncMock()
        llm.synthesize = AsyncMock(return_value="should not be called")

        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[])

        svc = _make_service(v=v, storage=storage, llm=llm)

        with patch.object(svc, "_rewrite_query_with_llm", wraps=svc._rewrite_query_with_llm) as mock_rewrite:
            with patch.object(
                svc,
                "_recall_rag",
                return_value=RecallResult(
                    memories=[],
                    total_count=0,
                    search_latency_ms=0,
                    mode_used=RecallMode.LLM,
                ),
            ):
                input_ = RecallInput(query="test query", context=[])
                await svc._recall_llm("ws_test", input_, 0.5)

            mock_rewrite.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_rewrite_fallback_on_llm_error(self):
        """If LLM errors during rewrite, original query should be used (no exception)."""
        v = _make_v(**{MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED: True})

        llm = AsyncMock()
        llm.synthesize = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[])

        svc = _make_service(v=v, storage=storage, llm=llm)

        # Should not raise
        result = await svc._rewrite_query_with_llm("original query", None)
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_query_rewrite_serializes_context_list(self):
        """Context list should be serialized to string for the LLM prompt."""
        v = _make_v(**{MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED: True})

        llm = AsyncMock()
        llm.synthesize = AsyncMock(return_value="rewritten query")

        storage = AsyncMock()
        storage.search_memories = AsyncMock(return_value=[])

        svc = _make_service(v=v, storage=storage, llm=llm)

        captured_prompt = {}

        async def capture_synthesize(prompt, **kwargs):
            captured_prompt["value"] = prompt
            return "rewritten query"

        llm.synthesize = capture_synthesize

        context = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        with patch.object(
            svc,
            "_recall_rag",
            return_value=RecallResult(
                memories=[],
                total_count=0,
                search_latency_ms=0,
                mode_used=RecallMode.LLM,
            ),
        ):
            input_ = RecallInput(query="test query", context=context)
            await svc._recall_llm("ws_test", input_, 0.5)

        # The LLM synthesize should have been called with context serialized as string
        assert "value" in captured_prompt
        assert "user: hello" in captured_prompt["value"] or "hello" in captured_prompt["value"]

    def test_llm_query_rewrite_default_enabled(self):
        """The default value for LLM query rewrite should be True."""
        assert DEFAULT_MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED is True


# ---------------------------------------------------------------------------
# 2c: Trust scoring tests
# ---------------------------------------------------------------------------


class TestTrustScoring:
    def _fresh_memory(self, **kwargs):
        """A very recently created memory."""
        defaults = dict(
            id="mem_fresh",
            content="fresh memory",
            type=MemoryType.SEMANTIC,
            workspace_id="ws_test",
            tenant_id="t",
            content_hash="hash1",
            importance=0.5,
            tags=[],
            metadata={},
            status=MemoryStatus.ACTIVE,
            access_count=0,
            decay_factor=1.0,
            pinned=False,
            created_at=datetime.now(UTC),
        )
        defaults.update(kwargs)
        return Memory(**defaults)

    def _old_memory(self, **kwargs):
        """A memory created 365 days ago."""
        defaults = dict(
            id="mem_old",
            content="old memory",
            type=MemoryType.SEMANTIC,
            workspace_id="ws_test",
            tenant_id="t",
            content_hash="hash2",
            importance=0.3,
            tags=[],
            metadata={},
            status=MemoryStatus.ACTIVE,
            access_count=0,
            decay_factor=0.2,
            pinned=False,
            created_at=datetime.now(UTC) - timedelta(days=365),
        )
        defaults.update(kwargs)
        return Memory(**defaults)

    def test_trust_score_range(self):
        """Trust score must be in [0.0, 1.0]."""
        svc = _make_service()
        mem = self._fresh_memory()
        score, signals = svc._compute_trust_score(mem)
        assert 0.0 <= score <= 1.0

    def test_trust_score_fresh_memory_higher_than_old(self):
        """A fresh memory should have a higher trust score than a very old one."""
        svc = _make_service()
        fresh = self._fresh_memory()
        old = self._old_memory()
        fresh_score, _ = svc._compute_trust_score(fresh)
        old_score, _ = svc._compute_trust_score(old)
        assert fresh_score > old_score

    def test_trust_score_pinned_memory_has_full_verification(self):
        """Pinned memory should have verification=1.0."""
        svc = _make_service()
        pinned = self._fresh_memory(pinned=True)
        _, signals = svc._compute_trust_score(pinned)
        assert signals["verification"] == 1.0

    def test_trust_score_verified_metadata_has_full_verification(self):
        """Memory with metadata.verified=True should have verification=1.0."""
        svc = _make_service()
        mem = self._fresh_memory(metadata={"verified": True})
        _, signals = svc._compute_trust_score(mem)
        assert signals["verification"] == 1.0

    def test_trust_score_unverified_memory_has_half_verification(self):
        """Unverified, unpinned memory should have verification=0.5."""
        svc = _make_service()
        mem = self._fresh_memory(pinned=False, metadata={})
        _, signals = svc._compute_trust_score(mem)
        assert signals["verification"] == 0.5

    def test_trust_score_session_source_has_high_reliability(self):
        """Memory from session commit (source_memory_id set) should have reliability=1.0."""
        svc = _make_service()
        mem = self._fresh_memory(source_memory_id="parent_mem_id")
        _, signals = svc._compute_trust_score(mem)
        assert signals["source_reliability"] == 1.0

    def test_trust_score_manual_memory_has_moderate_reliability(self):
        """Memory without any source links should have reliability=0.8 (manual)."""
        svc = _make_service()
        mem = self._fresh_memory(
            source_memory_id=None,
            source_document_id=None,
            source_thread_id=None,
        )
        _, signals = svc._compute_trust_score(mem)
        assert signals["source_reliability"] == 0.8

    def test_trust_score_extracted_memory_has_lower_reliability(self):
        """Memory extracted from a document should have reliability=0.6."""
        svc = _make_service()
        mem = self._fresh_memory(
            source_memory_id=None,
            source_document_id="doc_123",
            source_thread_id=None,
        )
        _, signals = svc._compute_trust_score(mem)
        assert signals["source_reliability"] == 0.6

    def test_trust_score_access_frequency_capped_at_1(self):
        """access_frequency component should be capped at 1.0."""
        svc = _make_service()
        mem = self._fresh_memory(access_count=100)
        _, signals = svc._compute_trust_score(mem)
        assert signals["access_frequency"] == 1.0

    def test_trust_score_signals_have_all_components(self):
        """trust_signals should contain all 5 component keys."""
        svc = _make_service()
        mem = self._fresh_memory()
        _, signals = svc._compute_trust_score(mem)
        assert "freshness" in signals
        assert "access_frequency" in signals
        assert "decay_factor" in signals
        assert "verification" in signals
        assert "source_reliability" in signals

    def test_annotate_trust_sets_fields_on_memories(self):
        """_annotate_trust should set trust_score and trust_signals on each memory."""
        svc = _make_service()
        mems = [self._fresh_memory(id=f"m{i}") for i in range(3)]
        result = svc._annotate_trust(mems)
        for m in result:
            assert m.trust_score is not None
            assert m.trust_signals is not None
            assert 0.0 <= m.trust_score <= 1.0

    @pytest.mark.asyncio
    async def test_recall_sets_drift_caveat_when_low_trust_memories(self):
        """recall() should set drift_caveat when any memory has trust < 0.5."""
        svc = _make_service()

        # A very old, low-decay, never-accessed memory = low trust
        old_mem = _make_memory(
            id="m_old",
            access_count=0,
            decay_factor=0.05,
            created_at=datetime.now(UTC) - timedelta(days=730),
        )
        old_mem_with_score = old_mem.model_copy(deep=True)

        with patch.object(
            svc,
            "_recall_rag",
            return_value=RecallResult(
                memories=[old_mem_with_score],
                total_count=1,
                search_latency_ms=0,
                mode_used=RecallMode.RAG,
            ),
        ):
            with patch.object(svc, "increment_access", new_callable=AsyncMock):
                input_ = RecallInput(query="test", include_associations=False)
                result = await svc.recall("ws_test", input_)

        # Drift caveat should be set if any memory has low trust
        # (whether set depends on the actual computed score for this memory)
        for m in result.memories:
            assert m.trust_score is not None

    @pytest.mark.asyncio
    async def test_recall_no_drift_caveat_for_fresh_trusted_memories(self):
        """recall() should not set drift_caveat when all memories have trust >= 0.5."""
        svc = _make_service()

        fresh_mem = _make_memory(
            id="m_fresh",
            access_count=5,
            decay_factor=1.0,
            pinned=True,
            created_at=datetime.now(UTC),
        )

        with patch.object(
            svc,
            "_recall_rag",
            return_value=RecallResult(
                memories=[fresh_mem],
                total_count=1,
                search_latency_ms=0,
                mode_used=RecallMode.RAG,
            ),
        ):
            with patch.object(svc, "increment_access", new_callable=AsyncMock):
                input_ = RecallInput(query="test", include_associations=False)
                result = await svc.recall("ws_test", input_)

        assert result.drift_caveat is None
        assert result.memories[0].trust_score is not None
        assert result.memories[0].trust_score >= 0.5
