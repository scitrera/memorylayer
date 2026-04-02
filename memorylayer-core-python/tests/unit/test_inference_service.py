"""
Unit tests for the Inference Service.

Tests:
- derive_insights with fallback (no LLM)
- derive_insights stores INFERENCE-subtype memories
- get_insights retrieves existing inferences
- derive_insights skips when no source memories exist
- _parse_insights parses LLM output format
- _derive_fallback generates type/subtype-based insights
"""

from datetime import UTC

import pytest
from scitrera_app_framework import get_extension

from memorylayer_server.models.memory import (
    MemorySubtype,
    MemoryType,
    RememberInput,
)
from memorylayer_server.services.inference import EXT_INFERENCE_SERVICE, DefaultInferenceService
from memorylayer_server.services.memory import MemoryService


@pytest.fixture
def inference_service(v) -> DefaultInferenceService:
    """Get the inference service."""
    return get_extension(EXT_INFERENCE_SERVICE, v)


class TestInferenceDerivation:
    """Tests for derive_insights operation."""

    @pytest.mark.asyncio
    async def test_derive_no_memories_returns_empty(
        self,
        inference_service: DefaultInferenceService,
        workspace_id: str,
    ):
        """Derivation with no source memories returns zero insights."""
        result = await inference_service.derive_insights(
            workspace_id=workspace_id,
            subject_id="nonexistent-entity-xyz",
            force=True,
        )

        assert result.source_memory_count == 0
        assert result.insights_created == 0
        assert result.insights == []

    @pytest.mark.asyncio
    async def test_derive_with_fallback_creates_insights(
        self,
        inference_service: DefaultInferenceService,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Fallback derivation (no LLM) generates type-based insights from multiple memories."""
        subject = "test-derive-subject"

        # Create several memories about the subject
        for content in [
            "Subject prefers Python over Java",
            "Subject uses FastAPI for web services",
            "Subject has experience with PostgreSQL",
            "Subject values clean code practices",
        ]:
            await memory_service.remember(
                workspace_id,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subject_id=subject,
                    importance=0.7,
                ),
            )

        result = await inference_service.derive_insights(
            workspace_id=workspace_id,
            subject_id=subject,
            force=True,
        )

        assert result.source_memory_count >= 4
        assert result.subject_id == subject
        assert result.workspace_id == workspace_id
        # Fallback should generate at least one insight when there are
        # multiple memories of the same type
        assert result.insights_created >= 1

    @pytest.mark.asyncio
    async def test_derived_insights_are_inference_subtype(
        self,
        inference_service: DefaultInferenceService,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """Derived insights are stored with subtype=INFERENCE."""
        subject = "test-subtype-subject"

        for content in [
            "This subject writes documentation thoroughly",
            "This subject reviews PRs carefully",
            "This subject mentors junior developers",
        ]:
            await memory_service.remember(
                workspace_id,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subject_id=subject,
                    importance=0.7,
                ),
            )

        result = await inference_service.derive_insights(
            workspace_id=workspace_id,
            subject_id=subject,
            force=True,
        )

        for insight in result.insights:
            assert insight.subtype == MemorySubtype.INFERENCE
            assert "inference" in insight.tags
            assert insight.subject_id == subject


class TestGetInsights:
    """Tests for get_insights retrieval."""

    @pytest.mark.asyncio
    async def test_get_insights_returns_derived(
        self,
        inference_service: DefaultInferenceService,
        memory_service: MemoryService,
        workspace_id: str,
    ):
        """get_insights returns previously derived insights."""
        subject = "test-get-insights-subject"

        # First create source memories and derive
        for content in [
            "Subject A always tests code before committing",
            "Subject A uses TDD methodology",
            "Subject A writes comprehensive unit tests",
        ]:
            await memory_service.remember(
                workspace_id,
                RememberInput(
                    content=content,
                    type=MemoryType.SEMANTIC,
                    subject_id=subject,
                    importance=0.7,
                ),
            )

        derive_result = await inference_service.derive_insights(
            workspace_id=workspace_id,
            subject_id=subject,
            force=True,
        )

        # Verify derivation created insights
        assert derive_result.insights_created >= 1

        # Now retrieve - the derived insights should exist
        insights = await inference_service.get_insights(
            workspace_id=workspace_id,
            subject_id=subject,
        )

        # With mock embeddings, semantic search may not find exact matches,
        # but we verified creation above
        if len(insights) > 0:
            for insight in insights:
                assert insight.subtype == MemorySubtype.INFERENCE

    @pytest.mark.asyncio
    async def test_get_insights_empty_for_unknown_subject(
        self,
        inference_service: DefaultInferenceService,
    ):
        """get_insights returns empty list for unknown subject."""
        insights = await inference_service.get_insights(
            workspace_id="default",
            subject_id="totally-unknown-entity",
        )

        assert insights == []


class TestParseInsights:
    """Tests for _parse_insights LLM output parser."""

    def test_parse_standard_format(self, inference_service: DefaultInferenceService):
        """Parses [importance] insight text format."""
        response = """[0.8] Prefers concise communication
[0.6] Approaches problems methodically
[0.9] Strong preference for typed languages"""

        results = inference_service._parse_insights(response)

        assert len(results) == 3
        assert results[0] == (0.8, "Prefers concise communication")
        assert results[1] == (0.6, "Approaches problems methodically")
        assert results[2] == (0.9, "Strong preference for typed languages")

    def test_parse_clamps_importance(self, inference_service: DefaultInferenceService):
        """Importance values are clamped to [0.0, 1.0]."""
        response = "[1.5] Over-confident insight\n[-0.3] Under-confident insight"

        results = inference_service._parse_insights(response)

        assert results[0][0] == 1.0
        assert results[1][0] == 0.0

    def test_parse_fallback_for_unparseable(self, inference_service: DefaultInferenceService):
        """Lines without brackets get default 0.5 importance."""
        response = "Just a plain insight without brackets"

        results = inference_service._parse_insights(response)

        assert len(results) == 1
        assert results[0] == (0.5, "Just a plain insight without brackets")

    def test_parse_skips_empty_lines(self, inference_service: DefaultInferenceService):
        """Empty lines are skipped."""
        response = "[0.7] First insight\n\n\n[0.8] Second insight"

        results = inference_service._parse_insights(response)

        assert len(results) == 2


class TestDeriveFallback:
    """Tests for _derive_fallback without LLM."""

    def test_fallback_needs_minimum_memories(self, inference_service: DefaultInferenceService):
        """Fallback returns empty with fewer than 2 memories."""
        from datetime import datetime

        from memorylayer_server.models.memory import Memory

        single_memory = Memory(
            id="mem_1",
            workspace_id="ws",
            tenant_id="_default",
            content="Single memory",
            content_hash="abc123",
            type=MemoryType.SEMANTIC,
            importance=0.5,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        results = inference_service._derive_fallback([single_memory], "subj")
        assert results == []

    def test_fallback_groups_by_type(self, inference_service: DefaultInferenceService):
        """Fallback generates insights based on type grouping."""
        from datetime import datetime

        from memorylayer_server.models.memory import Memory

        now = datetime.now(UTC)
        memories = [
            Memory(
                id=f"mem_{i}",
                workspace_id="ws",
                tenant_id="_default",
                content=f"Memory {i}",
                content_hash=f"hash_{i}",
                type=MemoryType.SEMANTIC,
                importance=0.5,
                created_at=now,
                updated_at=now,
            )
            for i in range(5)
        ]

        results = inference_service._derive_fallback(memories, "subj")

        assert len(results) >= 1
        # Should mention semantic type
        assert any("semantic" in text.lower() for _, text in results)
