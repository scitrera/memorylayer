"""OpenAI-compatible LLM provider."""

from collections.abc import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from .base import LLMProvider

DEFAULT_LLM_OPENAI_MODEL = "gpt-5-nano"


def _role_str(role) -> str:
    return role.value if isinstance(role, LLMRole) else str(role)


def _message_to_openai_dict(msg: LLMMessage) -> dict:
    """Serialize ``LLMMessage`` to an OpenAI chat-completions message dict.

    Handles assistant messages with ``tool_calls`` (content may be empty)
    and tool-result messages (role=tool with ``tool_call_id``).
    """
    role = _role_str(msg.role)
    out: dict = {"role": role}
    if msg.tool_calls is not None:
        out["tool_calls"] = msg.tool_calls
        # OpenAI accepts content=null on assistant messages that only
        # request tool calls; sending "" is also accepted.
        out["content"] = msg.content if msg.content else None
    else:
        out["content"] = msg.content or ""
    if msg.tool_call_id is not None:
        out["tool_call_id"] = msg.tool_call_id
    if msg.name is not None:
        out["name"] = msg.name
    return out


def _openai_tool_call_to_dict(tc) -> dict:
    """Convert an OpenAI SDK ChatCompletionMessageToolCall to a plain dict."""
    return {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }


def _openai_tool_call_delta_to_dict(tc) -> dict:
    """Serialize a streaming tool-call delta. Any field may be ``None`` mid-stream."""
    out: dict = {"index": tc.index}
    if getattr(tc, "id", None) is not None:
        out["id"] = tc.id
    if getattr(tc, "type", None) is not None:
        out["type"] = tc.type
    fn = getattr(tc, "function", None)
    if fn is not None:
        fn_out: dict = {}
        if getattr(fn, "name", None) is not None:
            fn_out["name"] = fn.name
        if getattr(fn, "arguments", None) is not None:
            fn_out["arguments"] = fn.arguments
        if fn_out:
            out["function"] = fn_out
    return out


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with OpenAI API, Azure OpenAI, Ollama, vLLM, and any
    OpenAI-compatible endpoint by configuring the base URL.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = None,
        model: str = DEFAULT_LLM_OPENAI_MODEL,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        v: Variables = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._client = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized OpenAILLMProvider: base_url=%s, model=%s", base_url, model)

    def _get_client(self):
        """Lazy-load OpenAI async client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client

    def _build_kwargs(self, request: LLMRequest, *, stream: bool) -> dict:
        """Assemble kwargs for ``chat.completions.create()``."""
        messages = [_message_to_openai_dict(msg) for msg in request.messages]
        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        kwargs: dict = {"model": model, "messages": messages}
        if stream:
            kwargs["stream"] = True
        if request.stop is not None:
            kwargs["stop"] = request.stop
        effective_max = request.max_completion_tokens if request.max_completion_tokens is not None else max_tokens
        if effective_max is not None:
            kwargs["max_completion_tokens"] = effective_max
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.tools is not None:
            kwargs["tools"] = request.tools
        if request.tool_choice is not None:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format is not None:
            kwargs["response_format"] = request.response_format
        if request.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if request.extra_body is not None:
            kwargs["extra_body"] = request.extra_body
        return kwargs

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        client = self._get_client()
        kwargs = self._build_kwargs(request, stream=False)
        self.logger.debug(
            "LLM request: model=%s, messages=%d, tools=%s",
            kwargs["model"],
            len(kwargs["messages"]),
            (len(request.tools) if request.tools else 0),
        )

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        message = choice.message
        usage = response.usage

        tool_calls = None
        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            tool_calls = [_openai_tool_call_to_dict(tc) for tc in raw_tool_calls]

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            finish_reason=choice.finish_reason or "stop",
            tool_calls=tool_calls,
        )

    async def complete_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using OpenAI API."""
        client = self._get_client()
        kwargs = self._build_kwargs(request, stream=True)

        stream = await client.chat.completions.create(**kwargs)

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if getattr(delta, "content", None):
                yield LLMStreamChunk(
                    content=delta.content,
                    is_final=False,
                )

            raw_tc_deltas = getattr(delta, "tool_calls", None)
            if raw_tc_deltas:
                yield LLMStreamChunk(
                    content="",
                    is_final=False,
                    tool_calls_delta=[_openai_tool_call_delta_to_dict(tc) for tc in raw_tc_deltas],
                )

            if choice.finish_reason:
                yield LLMStreamChunk(
                    content="",
                    is_final=True,
                    finish_reason=choice.finish_reason,
                )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
