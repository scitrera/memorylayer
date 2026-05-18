"""Tool calling / reasoning forwarding across LLM providers.

The OSS LLM abstraction lets callers send OpenAI-shape tools, tool_choice,
response_format, and reasoning_effort. Each provider forwards or
translates them to its native SDK call. These tests assert:

* OpenAI: forwards canonical fields untouched, surfaces ``tool_calls`` on the response.
* Anthropic: translates OpenAI-shape tools/messages to Anthropic content blocks
  + ``tool_use`` schema, maps ``reasoning_effort`` to ``thinking`` budget.
* Google: forwards ``tools`` as-is (caller supplies Google-shape), maps
  ``reasoning_effort`` to ``thinking_config`` budget tokens, extracts
  ``function_call`` parts into canonical tool_calls.
"""
import json

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memorylayer_server.models.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
)


_WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    },
}


def _user(text: str) -> LLMMessage:
    return LLMMessage(role=LLMRole.USER, content=text)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIToolCalling:
    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.llm.openai import OpenAILLMProvider

        return OpenAILLMProvider(api_key="test-key", model="gpt-5-nano")

    @pytest.mark.asyncio
    async def test_tools_and_tool_choice_forwarded(self, provider):
        # Tool call response shape mirrors openai SDK.
        tc = MagicMock()
        tc.id = "call_abc"
        tc.type = "function"
        tc.function.name = "get_weather"
        tc.function.arguments = '{"location": "Boston"}'

        message = MagicMock()
        message.content = None
        message.tool_calls = [tc]

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "tool_calls"

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-5-nano"
        response.usage = MagicMock(prompt_tokens=20, completion_tokens=8, total_tokens=28)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(
            messages=[_user("What's the weather in Boston?")],
            tools=[_WEATHER_TOOL_OPENAI],
            tool_choice="auto",
            response_format={"type": "json_object"},
            reasoning_effort="medium",
            max_completion_tokens=128,
        )
        result = await provider.complete(request)

        assert result.tool_calls == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Boston"}',
                },
            }
        ]
        assert result.finish_reason == "tool_calls"

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"] == [_WEATHER_TOOL_OPENAI]
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["max_completion_tokens"] == 128

    @pytest.mark.asyncio
    async def test_tool_call_round_trip_message_serialization(self, provider):
        """Assistant + tool-result messages serialize correctly when round-tripping."""
        message = MagicMock()
        message.content = "It's 65°F in Boston."
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "stop"

        response = MagicMock()
        response.choices = [choice]
        response.model = "gpt-5-nano"
        response.usage = MagicMock(prompt_tokens=30, completion_tokens=10, total_tokens=40)

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(
            messages=[
                _user("What's the weather in Boston?"),
                LLMMessage(
                    role=LLMRole.ASSISTANT,
                    content="",
                    tool_calls=[{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Boston"}',
                        },
                    }],
                ),
                LLMMessage(
                    role=LLMRole.TOOL,
                    content='{"temp_f": 65}',
                    tool_call_id="call_abc",
                ),
            ],
        )
        await provider.complete(request)

        sent_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "user", "content": "What's the weather in Boston?"}
        # Assistant tool-call message has tool_calls list and null content.
        assert sent_messages[1]["role"] == "assistant"
        assert sent_messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert sent_messages[1]["content"] is None
        # Tool result message includes tool_call_id.
        assert sent_messages[2]["role"] == "tool"
        assert sent_messages[2]["tool_call_id"] == "call_abc"
        assert sent_messages[2]["content"] == '{"temp_f": 65}'

    @pytest.mark.asyncio
    async def test_extra_body_forwarded(self, provider):
        message = MagicMock(content="ok", tool_calls=None)
        choice = MagicMock(message=message, finish_reason="stop")
        response = MagicMock(
            choices=[choice], model="gpt-5-nano",
            usage=MagicMock(prompt_tokens=5, completion_tokens=1, total_tokens=6),
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(messages=[_user("hi")], extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        await provider.complete(request)
        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    @pytest.mark.asyncio
    async def test_streaming_tool_call_deltas_emitted(self, provider):
        """delta.tool_calls becomes an LLMStreamChunk with tool_calls_delta."""
        # First chunk: tool call id + name.
        delta1 = MagicMock()
        delta1.content = None
        tc1 = MagicMock(index=0, id="call_xyz", type="function")
        tc1.function = MagicMock(name="get_weather", arguments=None)
        delta1.tool_calls = [tc1]
        choice1 = MagicMock(delta=delta1, finish_reason=None)
        chunk1 = MagicMock(choices=[choice1])

        # Second chunk: arguments fragment.
        delta2 = MagicMock()
        delta2.content = None
        tc2 = MagicMock(index=0, id=None, type=None)
        tc2.function = MagicMock(name=None, arguments='{"loc')
        delta2.tool_calls = [tc2]
        choice2 = MagicMock(delta=delta2, finish_reason=None)
        chunk2 = MagicMock(choices=[choice2])

        # Final chunk: finish_reason.
        delta3 = MagicMock()
        delta3.content = None
        delta3.tool_calls = None
        choice3 = MagicMock(delta=delta3, finish_reason="tool_calls")
        chunk3 = MagicMock(choices=[choice3])

        async def fake_stream():
            for c in (chunk1, chunk2, chunk3):
                yield c

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())
        provider._client = mock_client

        request = LLMRequest(
            messages=[_user("weather?")],
            tools=[_WEATHER_TOOL_OPENAI],
            stream=True,
        )
        chunks = [c async for c in provider.complete_stream(request)]

        # Expect two tool_calls_delta chunks + one final.
        tc_chunks = [c for c in chunks if c.tool_calls_delta]
        assert len(tc_chunks) == 2
        # First delta has the id.
        assert tc_chunks[0].tool_calls_delta[0]["index"] == 0
        assert tc_chunks[0].tool_calls_delta[0]["id"] == "call_xyz"
        # Final chunk reports tool_calls finish reason.
        assert any(c.is_final and c.finish_reason == "tool_calls" for c in chunks)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicToolCalling:
    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.llm.anthropic import AnthropicLLMProvider

        return AnthropicLLMProvider(api_key="test-key", model="claude-sonnet-4-20250514")

    @pytest.mark.asyncio
    async def test_tools_translated_to_anthropic_shape(self, provider):
        text_block = MagicMock(type="text", text="Sure thing.")
        tool_block = MagicMock(type="tool_use", id="toolu_001", input={"location": "Boston"})
        # ``name=`` is a reserved MagicMock kwarg (the mock's repr name) —
        # assign the attribute post-construction so getattr resolves properly.
        tool_block.name = "get_weather"

        response = MagicMock()
        response.content = [text_block, tool_block]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=15, output_tokens=8)
        response.stop_reason = "tool_use"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(
            messages=[_user("Weather in Boston?")],
            tools=[_WEATHER_TOOL_OPENAI],
            tool_choice="auto",
            reasoning_effort="medium",
        )
        result = await provider.complete(request)

        # Tools translated.
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["tools"] == [{
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "input_schema": _WEATHER_TOOL_OPENAI["function"]["parameters"],
        }]
        assert kwargs["tool_choice"] == {"type": "auto"}
        # reasoning_effort → thinking budget.
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}

        # tool_use block extracted into canonical tool_calls.
        assert result.tool_calls == [{
            "id": "toolu_001",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            },
        }]
        assert result.content == "Sure thing."
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_assistant_tool_call_and_tool_result_translated(self, provider):
        response = MagicMock()
        response.content = [MagicMock(type="text", text="Got it.")]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=20, output_tokens=2)
        response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(messages=[
            _user("Weather?"),
            LLMMessage(
                role=LLMRole.ASSISTANT,
                content="",
                tool_calls=[{
                    "id": "toolu_001",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}',
                    },
                }],
            ),
            LLMMessage(
                role=LLMRole.TOOL,
                content='{"temp_f": 65}',
                tool_call_id="toolu_001",
            ),
        ])
        await provider.complete(request)

        sent_messages = mock_client.messages.create.call_args.kwargs["messages"]
        # 3 messages → assistant (with tool_use content blocks) + tool result (as user) — system is None.
        assert len(sent_messages) == 3
        assistant_msg = sent_messages[1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"][0]["type"] == "tool_use"
        assert assistant_msg["content"][0]["id"] == "toolu_001"
        assert assistant_msg["content"][0]["name"] == "get_weather"
        assert assistant_msg["content"][0]["input"] == {"location": "Boston"}
        tool_msg = sent_messages[2]
        assert tool_msg["role"] == "user"
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "toolu_001"
        assert tool_msg["content"][0]["content"] == '{"temp_f": 65}'

    @pytest.mark.asyncio
    async def test_thinking_block_extracted_to_reasoning_content(self, provider):
        thinking_block = MagicMock(type="thinking", thinking="Let me think about this...")
        text_block = MagicMock(type="text", text="The answer is 42.")

        response = MagicMock()
        response.content = [thinking_block, text_block]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=10, output_tokens=20)
        response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(messages=[_user("hard question")], reasoning_effort="high")
        result = await provider.complete(request)

        assert result.reasoning_content == "Let me think about this..."
        assert result.content == "The answer is 42."
        # high → 4096 token budget.
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    @pytest.mark.asyncio
    async def test_extra_body_merged_into_kwargs(self, provider):
        response = MagicMock()
        response.content = [MagicMock(type="text", text="ok")]
        response.model = "claude-sonnet-4-20250514"
        response.usage = MagicMock(input_tokens=1, output_tokens=1)
        response.stop_reason = "end_turn"

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=response)
        provider._client = mock_client

        request = LLMRequest(messages=[_user("hi")], extra_body={"metadata": {"user_id": "u-1"}})
        await provider.complete(request)
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["metadata"] == {"user_id": "u-1"}


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


class TestGoogleToolCalling:
    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.llm.google import GoogleLLMProvider

        return GoogleLLMProvider(api_key="test-key", model="gemini-3-flash-preview")

    @pytest.fixture
    def _mock_build(self, provider):
        mock_contents = MagicMock()
        mock_config = MagicMock()
        with patch.object(
            provider.__class__,
            "_build_request",
            return_value=(mock_contents, mock_config),
        ) as mock_build:
            yield mock_build

    @pytest.mark.asyncio
    async def test_function_call_extracted_to_tool_calls(self, provider, _mock_build):
        function_call = MagicMock()
        function_call.name = "get_weather"
        function_call.args = {"location": "Boston"}
        function_call.id = "call_g_1"

        part = MagicMock(function_call=function_call)
        content = MagicMock(parts=[part])
        candidate = MagicMock(content=content, finish_reason="STOP")

        response = MagicMock()
        response.text = ""
        response.usage_metadata = MagicMock(
            prompt_token_count=12, candidates_token_count=4, total_token_count=16,
        )
        response.candidates = [candidate]

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content = AsyncMock(return_value=response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        request = LLMRequest(messages=[_user("Weather?")], tools=[_WEATHER_TOOL_OPENAI])
        result = await provider.complete(request)

        assert result.tool_calls == [{
            "id": "call_g_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
            },
        }]
        # If model emits tool_calls with otherwise STOP finish, we surface as tool_calls.
        assert result.finish_reason == "tool_calls"

    def test_reasoning_effort_maps_to_thinking_config(self, provider):
        """High effort budget reaches Gemini's ThinkingConfig (when SDK exposes it)."""
        # We can't easily mock the SDK types module fully here without
        # stubbing the whole google.genai package — just verify the
        # _REASONING_EFFORT_TO_BUDGET mapping is in place.
        from memorylayer_server.services.llm.google import _REASONING_EFFORT_TO_BUDGET

        assert _REASONING_EFFORT_TO_BUDGET["minimal"] < _REASONING_EFFORT_TO_BUDGET["high"]
        assert _REASONING_EFFORT_TO_BUDGET["high"] >= 4096
