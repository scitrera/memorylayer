"""Unit tests for DefaultTierGenerationService."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from memorylayer_server.models.llm import LLMRole
from memorylayer_server.services.semantic_tiering.default import DefaultSemanticTieringService


def _make_service(llm_response: str = "A summary.") -> DefaultSemanticTieringService:
    """Create a DefaultTierGenerationService with a mocked LLM service."""
    mock_llm_service = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = llm_response
    mock_llm_service.complete.return_value = mock_response

    return DefaultSemanticTieringService(
        llm_service=mock_llm_service,
        storage=None,
        enabled=True,
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
