"""Unit tests for Anthropic and Google GenAI LLM providers."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_server.models.llm import (
    LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk,
)


# ============================================
# Shared fixtures
# ============================================

def _make_request(
        messages=None,
        model=None,
        max_tokens=512,
        temperature=0.7,
        stop=None,
        stream=False,
):
    if messages is None:
        messages = [
            LLMMessage(role=LLMRole.USER, content="Hello"),
        ]
    return LLMRequest(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        stream=stream,
    )


def _make_system_request():
    return _make_request(messages=[
        LLMMessage(role=LLMRole.SYSTEM, content="You are helpful."),
        LLMMessage(role=LLMRole.USER, content="Hi"),
        LLMMessage(role=LLMRole.ASSISTANT, content="Hello!"),
        LLMMessage(role=LLMRole.USER, content="How are you?"),
    ])


# ============================================
# Anthropic Provider Tests
# ============================================

class TestAnthropicLLMProvider:
    """Tests for AnthropicLLMProvider."""

    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.llm.anthropic import AnthropicLLMProvider
        return AnthropicLLMProvider(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
        )

    def test_default_model(self, provider):
        assert provider.default_model == "claude-sonnet-4-20250514"

    def test_supports_streaming(self, provider):
        assert provider.supports_streaming is True

    def test_prepare_messages_simple(self, provider):
        request = _make_request()
        system_text, messages = provider._prepare_messages(request)

        assert system_text is None
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    def test_prepare_messages_with_system(self, provider):
        request = _make_system_request()
        system_text, messages = provider._prepare_messages(request)

        assert system_text == "You are helpful."
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hi"}
        assert messages[1] == {"role": "assistant", "content": "Hello!"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_prepare_messages_multiple_system(self, provider):
        """Multiple system messages are concatenated."""
        request = _make_request(messages=[
            LLMMessage(role=LLMRole.SYSTEM, content="First system."),
            LLMMessage(role=LLMRole.SYSTEM, content="Second system."),
            LLMMessage(role=LLMRole.USER, content="Hi"),
        ])
        system_text, messages = provider._prepare_messages(request)

        assert system_text == "First system.\nSecond system."
        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_complete(self, provider):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello there!")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_request()
        response = await provider.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello there!"
        assert response.model == "claude-sonnet-4-20250514"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15
        assert response.finish_reason == "stop"

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, provider):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I'm fine!")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 3
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_system_request()
        response = await provider.complete(request)

        assert response.content == "I'm fine!"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."
        assert len(call_kwargs["messages"]) == 3

    @pytest.mark.asyncio
    async def test_complete_with_stop_sequences(self, provider):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="result")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1
        mock_response.stop_reason = "stop_sequence"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_request(stop=["END", "DONE"])
        response = await provider.complete(request)

        assert response.finish_reason == "stop"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["stop_sequences"] == ["END", "DONE"]

    @pytest.mark.asyncio
    async def test_complete_max_tokens_finish(self, provider):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="truncated")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 100
        mock_response.stop_reason = "max_tokens"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_request()
        response = await provider.complete(request)

        assert response.finish_reason == "length"

    @pytest.mark.asyncio
    async def test_complete_model_override(self, provider):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hi")]
        mock_response.model = "claude-haiku-4-20250514"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_request(model="claude-haiku-4-20250514")
        response = await provider.complete(request)

        assert response.model == "claude-haiku-4-20250514"
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-20250514"

    @pytest.mark.asyncio
    async def test_complete_no_temperature(self, provider):
        """Temperature=None should not be passed to the API."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hi")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1
        mock_response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        request = _make_request(temperature=None)
        await provider.complete(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "temperature" not in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_stream(self, provider):
        mock_final = MagicMock()
        mock_final.stop_reason = "end_turn"

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)

        async def mock_text_stream():
            yield "Hello"
            yield " world"

        mock_stream.text_stream = mock_text_stream()
        mock_stream.get_final_message = AsyncMock(return_value=mock_final)

        mock_client = AsyncMock()
        mock_client.messages.stream = MagicMock(return_value=mock_stream)
        provider._client = mock_client

        request = _make_request(stream=True)
        chunks = []
        async for chunk in provider.complete_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].content == " world"
        assert chunks[1].is_final is False
        assert chunks[2].content == ""
        assert chunks[2].is_final is True
        assert chunks[2].finish_reason == "stop"

    def test_lazy_client_import_error(self, provider):
        with patch.dict('sys.modules', {'anthropic': None}):
            provider._client = None
            with pytest.raises(ImportError, match="anthropic package not installed"):
                provider._get_client()


# ============================================
# Google GenAI Provider Tests
# ============================================

class TestGoogleLLMProvider:
    """Tests for GoogleLLMProvider."""

    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.llm.google import GoogleLLMProvider
        return GoogleLLMProvider(
            api_key="test-key",
            model="gemini-3-flash-preview",
        )

    def test_default_model(self, provider):
        assert provider.default_model == "gemini-3-flash-preview"

    def test_supports_streaming(self, provider):
        assert provider.supports_streaming is True

    def test_extract_messages_simple(self, provider):
        request = _make_request()
        system_text, messages = provider._extract_messages(request)

        assert system_text is None
        assert len(messages) == 1
        assert messages[0] == ("user", "Hello")

    def test_extract_messages_with_system(self, provider):
        request = _make_system_request()
        system_text, messages = provider._extract_messages(request)

        assert system_text == "You are helpful."
        # System message extracted; 3 remaining messages
        assert len(messages) == 3
        # "assistant" mapped to "model"
        assert messages[0] == ("user", "Hi")
        assert messages[1] == ("model", "Hello!")
        assert messages[2] == ("user", "How are you?")

    def test_extract_messages_multiple_system(self, provider):
        """Multiple system messages are concatenated."""
        request = _make_request(messages=[
            LLMMessage(role=LLMRole.SYSTEM, content="First."),
            LLMMessage(role=LLMRole.SYSTEM, content="Second."),
            LLMMessage(role=LLMRole.USER, content="Hi"),
        ])
        system_text, messages = provider._extract_messages(request)

        assert system_text == "First.\nSecond."
        assert len(messages) == 1

    def test_map_finish_reason(self, provider):
        assert provider._map_finish_reason("STOP") == "stop"
        assert provider._map_finish_reason("MAX_TOKENS") == "length"
        assert provider._map_finish_reason("SAFETY") == "content_filter"
        assert provider._map_finish_reason("RECITATION") == "content_filter"
        assert provider._map_finish_reason(None) == "stop"
        assert provider._map_finish_reason("UNKNOWN") == "stop"

    @pytest.fixture
    def _mock_build(self, provider):
        """Mock _build_request to avoid google-genai import in unit tests."""
        mock_contents = MagicMock()
        mock_config = MagicMock()
        with patch.object(
            provider.__class__, '_build_request',
            return_value=(mock_contents, mock_config),
        ) as mock_build:
            mock_build.mock_contents = mock_contents
            mock_build.mock_config = mock_config
            yield mock_build

    @pytest.mark.asyncio
    async def test_complete(self, provider, _mock_build):
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 8
        mock_usage.candidates_token_count = 4
        mock_usage.total_token_count = 12

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request()
        response = await provider.complete(request)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello there!"
        assert response.model == "gemini-3-flash-preview"
        assert response.prompt_tokens == 8
        assert response.completion_tokens == 4
        assert response.total_tokens == 12
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_model_override(self, provider, _mock_build):
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 2
        mock_usage.total_token_count = 7

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.text = "hi"
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request(model="gemini-2.0-flash")
        response = await provider.complete(request)

        assert response.model == "gemini-2.0-flash"
        call_kwargs = mock_aio_models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_complete_no_usage_metadata(self, provider, _mock_build):
        """Handle missing usage_metadata gracefully."""
        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.text = "response"
        mock_response.usage_metadata = None
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request()
        response = await provider.complete(request)

        assert response.prompt_tokens == 0
        assert response.completion_tokens == 0
        assert response.total_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_content_filter(self, provider, _mock_build):
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 0
        mock_usage.total_token_count = 5

        mock_candidate = MagicMock()
        mock_candidate.finish_reason = "SAFETY"

        mock_response = MagicMock()
        mock_response.text = ""
        mock_response.usage_metadata = mock_usage
        mock_response.candidates = [mock_candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request()
        response = await provider.complete(request)

        assert response.finish_reason == "content_filter"

    @pytest.mark.asyncio
    async def test_complete_stream(self, provider, _mock_build):
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"

        async def mock_stream():
            yield chunk1
            yield chunk2

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content_stream = AsyncMock(return_value=mock_stream())

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request(stream=True)
        chunks = []
        async for chunk in provider.complete_stream(request):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].content == " world"
        assert chunks[1].is_final is False
        assert chunks[2].content == ""
        assert chunks[2].is_final is True
        assert chunks[2].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_stream_skips_empty_chunks(self, provider, _mock_build):
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = ""
        chunk3 = MagicMock()
        chunk3.text = " world"

        async def mock_stream():
            yield chunk1
            yield chunk2
            yield chunk3

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content_stream = AsyncMock(return_value=mock_stream())

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = _make_request(stream=True)
        chunks = []
        async for chunk in provider.complete_stream(request):
            chunks.append(chunk)

        # Empty chunk should be skipped, only Hello, world, and final
        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " world"
        assert chunks[2].is_final is True

    def test_lazy_client_import_error(self, provider):
        with patch.dict('sys.modules', {'google': None, 'google.genai': None}):
            provider._client = None
            with pytest.raises(ImportError, match="google-genai package not installed"):
                provider._get_client()


