"""Unit tests for DefaultTierGenerationService."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.models.llm import LLMRole
from memorylayer_server.services.semantic_tiering.default import DefaultSemanticTieringService


def _make_service(
    llm_response: str = "A summary.",
    abstract_skip_chars: int = 0,
    overview_skip_chars: int = 0,
) -> DefaultSemanticTieringService:
    """Create a DefaultTierGenerationService with a mocked LLM service.

    Short-content skipping is DISABLED by default here so these tests exercise the LLM
    path with short fixture strings. Skip behaviour has its own tests below, which set
    the thresholds explicitly.
    """
    mock_llm_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_llm_service.complete.return_value = mock_response

    return DefaultSemanticTieringService(
        llm_service=mock_llm_service,
        storage=None,
        enabled=True,
        abstract_skip_chars=abstract_skip_chars,
        overview_skip_chars=overview_skip_chars,
    )


class TestGenerateAbstract:
    """Tests for abstract generation."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """LLM response content should be returned stripped."""
        service = _make_service("  A brief summary.  ")
        result = await service.generate_abstract("Some long content here.")
        assert result == "A brief summary."

    @pytest.mark.asyncio
    async def test_uses_system_and_user_messages(self):
        """Request should have system + user messages, not a single user message."""
        service = _make_service("Summary.")
        await service.generate_abstract("Test content")

        call_args = service.llm_service.complete.call_args
        request = call_args[0][0]
        assert len(request.messages) == 2
        assert request.messages[0].role == LLMRole.SYSTEM
        assert request.messages[1].role == LLMRole.USER
        assert "summarization assistant" in request.messages[0].content.lower()
        assert "Test content" in request.messages[1].content

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        """Default max_tokens should be 500 (Gemini uses this as length signal)."""
        service = _make_service("Summary.")
        await service.generate_abstract("Test content")

        request = service.llm_service.complete.call_args[0][0]
        assert request.max_tokens == 500

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """Should fall back to truncation when LLM fails."""
        service = _make_service()
        service.llm_service.complete.side_effect = RuntimeError("API error")

        content = "x" * 200
        result = await service.generate_abstract(content)
        assert result == content[:100] + "..."

    @pytest.mark.asyncio
    async def test_fallback_short_content(self):
        """Fallback should return short content unchanged."""
        service = _make_service()
        service.llm_service.complete.side_effect = RuntimeError("API error")

        result = await service.generate_abstract("Short content.")
        assert result == "Short content."


class TestGenerateOverview:
    """Tests for overview generation."""

    @pytest.mark.asyncio
    async def test_returns_llm_response(self):
        """LLM response content should be returned stripped."""
        service = _make_service("  A detailed overview.  ")
        result = await service.generate_overview("Some long content here.")
        assert result == "A detailed overview."

    @pytest.mark.asyncio
    async def test_uses_system_and_user_messages(self):
        """Request should have system + user messages."""
        service = _make_service("Overview.")
        await service.generate_overview("Test content")

        request = service.llm_service.complete.call_args[0][0]
        assert len(request.messages) == 2
        assert request.messages[0].role == LLMRole.SYSTEM
        assert request.messages[1].role == LLMRole.USER
        assert "overview" in request.messages[0].content.lower()
        assert "Test content" in request.messages[1].content

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        """Default max_tokens should be 500."""
        service = _make_service("Overview.")
        await service.generate_overview("Test content")

        request = service.llm_service.complete.call_args[0][0]
        assert request.max_tokens == 500

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """Should fall back to truncation when LLM fails."""
        service = _make_service()
        service.llm_service.complete.side_effect = RuntimeError("API error")

        content = "x" * 600
        result = await service.generate_overview(content)
        assert result == content[:500] + "..."


class TestGenerateTiersForContent:
    """Tests for generate_tiers_for_content."""

    @pytest.mark.asyncio
    async def test_returns_abstract_and_overview(self):
        """Should return both abstract and overview."""
        service = _make_service("A summary.")
        abstract, overview = await service.generate_tiers_for_content("Test content")
        assert abstract == "A summary."
        assert overview == "A summary."
        assert service.llm_service.complete.call_count == 2


class TestShortContentSkip:
    """Skip the LLM when the content is already at or below the tier's target length.

    Summarizing text shorter than its own summary spends a call to produce something no
    more compact than the input, and risks dropping a detail. On a conversational corpus
    (median memory 145 chars) this skips 76% of abstract calls and 100% of overview calls.
    """

    @pytest.mark.asyncio
    async def test_short_content_skips_abstract_llm_call(self):
        service = _make_service("Should not be used.", abstract_skip_chars=200)
        content = "I moved to Seattle."

        result = await service.generate_abstract(content)

        assert result == content, "short content should pass through verbatim"
        service.llm_service.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_content_skips_overview_llm_call(self):
        service = _make_service("Should not be used.", overview_skip_chars=500)
        content = "I moved to Seattle last month for a new job."

        result = await service.generate_overview(content)

        assert result == content
        service.llm_service.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_content_still_calls_llm(self):
        """The skip must not swallow content that genuinely needs summarizing."""
        service = _make_service("A summary.", abstract_skip_chars=200, overview_skip_chars=500)
        content = "x" * 600

        assert await service.generate_overview(content) == "A summary."
        assert await service.generate_abstract(content) == "A summary."
        assert service.llm_service.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_threshold_is_inclusive(self):
        """Content exactly at the threshold is 'already short enough'."""
        service = _make_service("Should not be used.", abstract_skip_chars=20)

        assert await service.generate_abstract("x" * 20) == "x" * 20
        service.llm_service.complete.assert_not_called()

        assert await service.generate_abstract("x" * 21) == "Should not be used."
        assert service.llm_service.complete.call_count == 1

    @pytest.mark.asyncio
    async def test_zero_threshold_disables_skipping(self):
        """0 is the documented escape hatch: always call the LLM."""
        service = _make_service("A summary.", abstract_skip_chars=0, overview_skip_chars=0)

        assert await service.generate_abstract("tiny") == "A summary."
        assert await service.generate_overview("tiny") == "A summary."
        assert service.llm_service.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_tiers_for_content_makes_no_calls_for_short_input(self):
        """The end-to-end saving: a short memory costs zero LLM calls, not two."""
        service = _make_service("Should not be used.", abstract_skip_chars=200, overview_skip_chars=500)
        content = "I have two cats named Milo and Otis."

        abstract, overview = await service.generate_tiers_for_content(content)

        assert abstract == content
        assert overview == content
        service.llm_service.complete.assert_not_called()
