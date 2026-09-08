"""Unit tests for write-time temporal normalization in fact decomposition.

Covers ``DefaultExtractionService.decompose_to_facts`` resolving relative dates
to an absolute ``event_time`` using a supplied ``reference_time`` (adopted from
Microsoft Memora's write-time normalization).
"""

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.llm import LLMResponse, LLMRole
from memorylayer_server.services.extraction.default import DefaultExtractionService


class FakeLLMService:
    """Minimal LLM stub that returns a canned response and captures the request.

    Mirrors the ``complete(request, profile=...)`` contract the extraction
    service relies on. The last request is stored so tests can assert on the
    system prompt that was sent.
    """

    def __init__(self, response_content: str):
        self._response_content = response_content
        self.last_request = None
        self.last_system_prompt = None

    async def complete(self, request, profile: str = "default", **_generation_metadata) -> LLMResponse:
        self.last_request = request
        for message in request.messages:
            if message.role == LLMRole.SYSTEM:
                self.last_system_prompt = message.content
        return LLMResponse(
            content=self._response_content,
            model="fake",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            finish_reason="stop",
        )


class TestDecomposeTemporalNormalization:
    """Tests for reference_time -> event_time resolution."""

    def _service(self, llm_service):
        return DefaultExtractionService(
            llm_service=llm_service,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
        )

    @pytest.mark.asyncio
    async def test_event_time_parsed_to_datetime(self):
        """A canned event_time string is parsed to a datetime equal to that date."""
        llm = FakeLLMService('[{"content": "Team shipped v2", "event_time": "2026-06-30"}]')
        service = self._service(llm)
        reference_time = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

        result = await service.decompose_to_facts("The team shipped v2 yesterday", reference_time=reference_time)

        assert len(result) == 1
        assert result[0]["content"] == "Team shipped v2"
        assert result[0]["event_time"] == datetime(2026, 6, 30)

    @pytest.mark.asyncio
    async def test_reference_time_appears_in_prompt(self):
        """The reference time ISO string is injected into the system prompt."""
        llm = FakeLLMService('[{"content": "Team shipped v2", "event_time": "2026-06-30"}]')
        service = self._service(llm)
        reference_time = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

        await service.decompose_to_facts("The team shipped v2 yesterday", reference_time=reference_time)

        assert llm.last_system_prompt is not None
        assert reference_time.isoformat() in llm.last_system_prompt
        # The temporal-normalization instruction is present.
        assert "Resolve any relative dates" in llm.last_system_prompt
        assert "event_time" in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_prompt_unchanged_without_reference_time(self):
        """Without reference_time the prompt has no temporal block (byte-stable)."""
        llm = FakeLLMService('[{"content": "Team shipped v2"}]')
        service = self._service(llm)

        await service.decompose_to_facts("The team shipped v2")

        assert llm.last_system_prompt is not None
        assert "reference time" not in llm.last_system_prompt.lower()
        assert "Resolve any relative dates" not in llm.last_system_prompt
        assert "event_time" not in llm.last_system_prompt

    @pytest.mark.asyncio
    async def test_backward_compat_no_llm_returns_single_fact(self):
        """No LLM + no reference_time still returns the original as a single fact."""
        service = self._service(None)

        result = await service.decompose_to_facts("Some composite content")

        assert result == [{"content": "Some composite content"}]
        # Fallback dict has no event_time key (callers treat as absent).
        assert "event_time" not in result[0]

    @pytest.mark.asyncio
    async def test_unparseable_event_time_yields_none(self):
        """A malformed event_time string degrades to None, not an exception."""
        llm = FakeLLMService('[{"content": "Team shipped v2", "event_time": "not-a-date"}]')
        service = self._service(llm)
        reference_time = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)

        result = await service.decompose_to_facts("The team shipped v2", reference_time=reference_time)

        assert len(result) == 1
        assert result[0]["event_time"] is None

    @pytest.mark.asyncio
    async def test_event_time_with_trailing_z(self):
        """A trailing 'Z' is normalized so the timestamp parses to UTC-aware dt."""
        llm = FakeLLMService('[{"content": "Meeting happened", "event_time": "2026-01-15T14:00:00Z"}]')
        service = self._service(llm)
        reference_time = datetime(2026, 1, 20, tzinfo=UTC)

        result = await service.decompose_to_facts("The meeting happened last week", reference_time=reference_time)

        assert result[0]["event_time"] == datetime(2026, 1, 15, 14, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_missing_event_time_key_yields_none(self):
        """A fact without event_time gets event_time=None when reference_time set."""
        llm = FakeLLMService('[{"content": "General fact about Python"}]')
        service = self._service(llm)
        reference_time = datetime(2026, 7, 1, tzinfo=UTC)

        result = await service.decompose_to_facts("Python is a language", reference_time=reference_time)

        assert result[0]["event_time"] is None


class TestDecomposeReasoningEffort:
    """Decomposition is a transformation, not multi-step inference.

    Thinking tokens are drawn from the SAME completion budget as the JSON fact
    array, so reasoning does not add deliberation on top of the output -- it
    takes the output's place. That is why the token cap is set so high and why
    the parser carries a truncated-array recovery path.
    """

    def _service(self, llm_service, v=None):
        return DefaultExtractionService(
            llm_service=llm_service,
            storage=None,
            deduplication_service=None,
            embedding_service=None,
            v=v,
        )

    @pytest.mark.asyncio
    async def test_reasoning_is_disabled_by_default(self):
        llm = FakeLLMService('[{"content": "Python is a language"}]')
        service = self._service(llm)

        await service.decompose_to_facts("Python is a language")

        assert llm.last_request.reasoning_effort == "none"

    @pytest.mark.asyncio
    async def test_effort_is_configurable_for_when_quality_needs_it(self):
        from unittest.mock import MagicMock

        v = MagicMock()
        v.get = MagicMock(
            side_effect=lambda key, default=None, **kw: (
                "medium" if "REASONING_EFFORT" in key else default
            )
        )
        llm = FakeLLMService('[{"content": "Python is a language"}]')
        service = self._service(llm, v=v)

        await service.decompose_to_facts("Python is a language")

        assert llm.last_request.reasoning_effort == "medium"

    @pytest.mark.asyncio
    async def test_the_visual_path_also_disables_reasoning(self):
        # Reading facts off a page image is still extraction, and it competes
        # for the same budget.
        llm = FakeLLMService('[{"content": "Invoice total is $1,250"}]')
        service = self._service(llm)

        await service.decompose_to_facts("[Visual page 3]", images=["data:image/png;base64,AAA"])

        assert llm.last_request.reasoning_effort == "none"
