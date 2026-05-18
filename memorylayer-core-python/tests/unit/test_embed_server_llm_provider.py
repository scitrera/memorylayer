"""Unit tests for :class:`EmbedServerLLMProvider`.

We don't spin up a real embed-server peer. Instead, we plug a fake
``EmbedServerClient`` (with ``chat_completions`` stubbed for both
non-streaming and streaming) into ``v.get(EXT_EMBED_SERVER_CLIENT, ...)``
and assert that:

  * Canonical ``LLMRequest`` → OpenAI-shape payload is wired correctly
    (tools, tool_choice, response_format, reasoning_effort,
    max_completion_tokens, extra_body all forwarded).
  * Non-streaming responses are re-materialized into ``LLMResponse``
    including ``tool_calls`` and ``reasoning_content``.
  * Streaming chunks (SSE-encoded bytes) are parsed into
    ``LLMStreamChunk`` events with delta text, tool_calls_delta, and a
    terminal ``is_final`` chunk.
  * Per-profile URL/transport/aether-target/timeout overrides build a
    dedicated client rather than using the shared singleton.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from memorylayer_server.models.llm import LLMMessage, LLMRequest, LLMRole


class _FakeEmbedClient:
    """Stand-in for ``EmbedServerClient`` recording calls + scripting responses."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[dict, bool]] = []
        self.chat_response: dict | list[bytes] = {
            "id": "chatcmpl-test",
            "model": "qwen",
            "choices": [
                {
                    "message": {"content": "ok", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }

    async def chat_completions(self, payload, *, stream=False):
        self.chat_calls.append((payload, stream))
        if stream:
            chunks = list(self.chat_response)  # type: ignore[arg-type]

            async def _aiter():
                for c in chunks:
                    yield c

            return _aiter()
        return self.chat_response


def _build_provider(*, with_overrides: bool = False, fake_client=None):
    """Return an EmbedServerLLMProvider whose ``_get_client`` resolves to ``fake_client``."""
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider

    if with_overrides:
        provider = EmbedServerLLMProvider(
            model="qwen",
            embed_server_url="http://embed-peer:61051",
            embed_server_transport="http",
            embed_server_timeout=120,
        )
        provider._dedicated_client = fake_client
        provider._dedicated_needs_connect = False
        return provider

    provider = EmbedServerLLMProvider(model="qwen")
    # Patch get_extension to hand back the fake client.
    provider._get_client = AsyncMock(return_value=fake_client)
    return provider


def _user(text: str) -> LLMMessage:
    return LLMMessage(role=LLMRole.USER, content=text)


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_forwards_canonical_fields():
    fake = _FakeEmbedClient()
    provider = _build_provider(fake_client=fake)

    request = LLMRequest(
        messages=[_user("weather?")],
        tools=[
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": {}},
            }
        ],
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning_effort="medium",
        max_completion_tokens=128,
        temperature=0.3,
    )
    await provider.complete(request)
    payload, stream = fake.chat_calls[0]
    assert stream is False
    assert payload["model"] == "qwen"
    assert payload["messages"][0]["content"] == "weather?"
    assert payload["tools"][0]["function"]["name"] == "get_weather"
    assert payload["tool_choice"] == "auto"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "medium"
    assert payload["max_completion_tokens"] == 128
    assert payload["temperature"] == 0.3
    # `stream` is False, so payload must NOT include the streaming flag.
    assert "stream" not in payload


@pytest.mark.asyncio
async def test_complete_materializes_response_with_tool_calls():
    fake = _FakeEmbedClient()
    fake.chat_response = {
        "id": "chatcmpl-1",
        "model": "qwen-7b",
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"loc": "Boston"}'},
                        }
                    ],
                    "reasoning_content": "let me think",
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }
    provider = _build_provider(fake_client=fake)
    result = await provider.complete(LLMRequest(messages=[_user("weather?")]))
    assert result.content == ""
    assert result.model == "qwen-7b"
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls is not None
    assert result.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.reasoning_content == "let me think"
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.total_tokens == 18


@pytest.mark.asyncio
async def test_extra_body_merged_into_payload():
    fake = _FakeEmbedClient()
    provider = _build_provider(fake_client=fake)
    await provider.complete(
        LLMRequest(
            messages=[_user("hi")],
            extra_body={"vllm_custom_field": True, "metadata": {"trace_id": "abc"}},
        )
    )
    payload = fake.chat_calls[0][0]
    assert payload["vllm_custom_field"] is True
    assert payload["metadata"] == {"trace_id": "abc"}


@pytest.mark.asyncio
async def test_message_with_tool_calls_serialized_correctly():
    fake = _FakeEmbedClient()
    provider = _build_provider(fake_client=fake)
    await provider.complete(
        LLMRequest(
            messages=[
                _user("weather in Boston?"),
                LLMMessage(
                    role=LLMRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"loc": "Boston"}'},
                        }
                    ],
                ),
                LLMMessage(
                    role=LLMRole.TOOL,
                    content='{"temp_f": 65}',
                    tool_call_id="call_a",
                ),
            ]
        )
    )
    sent_messages = fake.chat_calls[0][0]["messages"]
    assert sent_messages[1]["role"] == "assistant"
    assert sent_messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert sent_messages[1]["content"] is None
    assert sent_messages[2]["role"] == "tool"
    assert sent_messages[2]["tool_call_id"] == "call_a"
    assert sent_messages[2]["content"] == '{"temp_f": 65}'


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _sse_chunk(data: dict) -> bytes:
    return f"data: {json.dumps(data)}\n\n".encode()


@pytest.mark.asyncio
async def test_complete_stream_yields_text_deltas():
    fake = _FakeEmbedClient()
    fake.chat_response = [
        _sse_chunk({"choices": [{"delta": {"content": "Hello"}}]}),
        _sse_chunk({"choices": [{"delta": {"content": " world"}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    provider = _build_provider(fake_client=fake)
    request = LLMRequest(messages=[_user("hi")], stream=True)
    chunks = [c async for c in provider.complete_stream(request)]

    # Two content deltas + one finish_reason chunk + one terminal-from-parser chunk.
    text_chunks = [c for c in chunks if c.content]
    assert [c.content for c in text_chunks] == ["Hello", " world"]
    finals = [c for c in chunks if c.is_final]
    assert finals  # at least one terminal
    assert finals[0].finish_reason == "stop"

    # Streaming payload had stream=True merged in.
    payload, stream = fake.chat_calls[0]
    assert stream is True
    assert payload["stream"] is True


@pytest.mark.asyncio
async def test_complete_stream_extracts_tool_call_deltas():
    fake = _FakeEmbedClient()
    fake.chat_response = [
        _sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_weather"}}],
                        }
                    }
                ]
            }
        ),
        _sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}],
                        }
                    }
                ]
            }
        ),
        _sse_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '": "Boston"}'}}],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        ),
        b"data: [DONE]\n\n",
    ]
    provider = _build_provider(fake_client=fake)
    request = LLMRequest(messages=[_user("weather?")], stream=True)
    chunks = [c async for c in provider.complete_stream(request)]

    tc_chunks = [c for c in chunks if c.tool_calls_delta]
    assert len(tc_chunks) == 3
    assert tc_chunks[0].tool_calls_delta[0]["id"] == "call_a"
    # Final chunk reports tool_calls finish_reason.
    finals = [c for c in chunks if c.is_final]
    assert any(c.finish_reason == "tool_calls" for c in finals)


@pytest.mark.asyncio
async def test_complete_stream_extracts_reasoning_content_delta():
    fake = _FakeEmbedClient()
    fake.chat_response = [
        _sse_chunk({"choices": [{"delta": {"reasoning_content": "let me think"}}]}),
        _sse_chunk({"choices": [{"delta": {"content": "the answer is 42"}}]}),
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    provider = _build_provider(fake_client=fake)
    chunks = [
        c
        async for c in provider.complete_stream(
            LLMRequest(messages=[_user("hard q")], stream=True, reasoning_effort="high"),
        )
    ]
    reasoning = [c for c in chunks if c.reasoning_content_delta]
    text = [c for c in chunks if c.content]
    assert reasoning[0].reasoning_content_delta == "let me think"
    assert text[0].content == "the answer is 42"
    # reasoning_effort forwarded into payload.
    assert fake.chat_calls[0][0]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_complete_stream_handles_multi_chunk_split_records():
    """SSE records that arrive across multiple byte chunks reassemble correctly."""
    fake = _FakeEmbedClient()
    sse_full = _sse_chunk({"choices": [{"delta": {"content": "Hello world"}}]})
    # Split mid-record on a byte boundary that's NOT \n\n.
    split = len(sse_full) // 2
    fake.chat_response = [
        sse_full[:split],
        sse_full[split:],
        _sse_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    provider = _build_provider(fake_client=fake)
    chunks = [
        c
        async for c in provider.complete_stream(
            LLMRequest(messages=[_user("hi")], stream=True),
        )
    ]
    text = [c.content for c in chunks if c.content]
    assert text == ["Hello world"]


# ---------------------------------------------------------------------------
# Per-profile overrides
# ---------------------------------------------------------------------------


def test_per_profile_overrides_build_dedicated_client():
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider

    provider = EmbedServerLLMProvider(
        model="qwen",
        embed_server_url="http://other-peer:61051",
        embed_server_transport="http",
        embed_server_timeout=120,
    )
    assert provider._dedicated_client is not None
    assert provider._dedicated_client._base_url == "http://other-peer:61051"
    assert provider._dedicated_client._timeout == 120.0
    assert provider._dedicated_client._transport == "http"


def test_aether_transport_override_requires_extension():
    """Constructing with transport=aether but no Aether extension wired raises."""
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider

    # No Variables, so get_extension returns None → constructor must reject.
    with patch(
        "memorylayer_server.services.llm.embed_server.get_extension",
        return_value=None,
    ):
        with pytest.raises(RuntimeError, match="EXT_AETHER_SERVICE_CONNECTION"):
            EmbedServerLLMProvider(
                model="qwen",
                embed_server_url="http://does-not-matter",
                embed_server_transport="aether",
                embed_server_aether_target="sv::memorylayer-embed::region-a",
            )


def test_no_overrides_uses_shared_singleton():
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider

    provider = EmbedServerLLMProvider(model="qwen")
    assert provider._dedicated_client is None


@pytest.mark.asyncio
async def test_get_client_returns_shared_singleton_when_no_override():
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider

    fake_singleton = _FakeEmbedClient()
    with patch(
        "memorylayer_server.services.llm.embed_server.get_extension",
        return_value=fake_singleton,
    ):
        provider = EmbedServerLLMProvider(model="qwen")
        client = await provider._get_client()
        assert client is fake_singleton


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_create_provider_from_config_dispatches_embed_server():
    from memorylayer_server.services.llm.embed_server import EmbedServerLLMProvider
    from memorylayer_server.services.llm.registry import create_provider_from_config

    provider = create_provider_from_config(
        name="inference",
        provider_type="embed_server",
        model="qwen",
        embed_server_url="http://peer-a:61051",
    )
    assert isinstance(provider, EmbedServerLLMProvider)
    assert provider.model == "qwen"
    assert provider._dedicated_client is not None
