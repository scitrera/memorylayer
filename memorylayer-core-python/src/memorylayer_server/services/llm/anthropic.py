"""Anthropic Claude LLM provider."""

import json as _json
from collections.abc import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from ..api_key_store import resolve_api_key_ref
from .base import LLMProvider

DEFAULT_LLM_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

# Anthropic stop_reason -> our finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}

# Map OpenAI o-series "reasoning_effort" levels to Anthropic thinking budgets.
_REASONING_EFFORT_TO_BUDGET = {
    "minimal": 1024,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


def _role_str(role) -> str:
    return role.value if isinstance(role, LLMRole) else str(role)


def _translate_message_for_anthropic(msg: LLMMessage) -> dict:
    """Convert an ``LLMMessage`` (OpenAI canonical) → Anthropic API message dict.

    - ``role="tool"`` becomes a user message with a ``tool_result`` content block.
    - ``role="assistant"`` with ``tool_calls`` becomes content blocks combining
      ``text`` + ``tool_use``.
    - Everything else passes through as plain ``{"role", "content"}``.
    """
    role = _role_str(msg.role)

    if role == "tool":
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
            ],
        }

    if role == "assistant" and msg.tool_calls:
        blocks: list[dict] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            fn = tc.get("function", {})
            args_field = fn.get("arguments", "{}")
            if isinstance(args_field, str):
                try:
                    args = _json.loads(args_field) if args_field else {}
                except _json.JSONDecodeError:
                    args = {}
            else:
                args = args_field or {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn.get("name", ""),
                    "input": args,
                }
            )
        return {"role": "assistant", "content": blocks}

    return {"role": role, "content": msg.content}


def _translate_tools_for_anthropic(tools: list[dict]) -> list[dict]:
    """OpenAI-shape tools → Anthropic-shape. Tools already in Anthropic shape pass through."""
    out: list[dict] = []
    for t in tools:
        if t.get("type") == "function" and "function" in t:
            fn = t["function"]
            out.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters",
                        {"type": "object", "properties": {}},
                    ),
                }
            )
        else:
            out.append(t)
    return out


def _translate_tool_choice_for_anthropic(tc):
    if isinstance(tc, str):
        if tc in ("auto", "any"):
            return {"type": tc}
        if tc == "required":
            return {"type": "any"}
        if tc == "none":
            return None
        return {"type": "tool", "name": tc}
    if isinstance(tc, dict):
        if tc.get("type") == "function" and "function" in tc:
            return {"type": "tool", "name": tc["function"]["name"]}
        return tc
    return tc


def _build_thinking_config(reasoning_effort: str | None) -> dict | None:
    if reasoning_effort is None:
        return None
    budget = _REASONING_EFFORT_TO_BUDGET.get(reasoning_effort, 2048)
    return {"type": "enabled", "budget_tokens": budget}


def _extract_anthropic_response(response):
    """Anthropic Message → (text, tool_calls, reasoning_content). Last two may be None."""
    texts: list[str] = []
    tool_calls: list[dict] = []
    thinking_texts: list[str] = []
    if response.content:
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                texts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(block, "name", ""),
                            "arguments": _json.dumps(getattr(block, "input", {}) or {}),
                        },
                    }
                )
            elif block_type == "thinking":
                thinking_texts.append(getattr(block, "thinking", "") or "")
    return (
        "".join(texts),
        (tool_calls or None),
        ("".join(thinking_texts) or None),
    )


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude LLM provider.

    Uses the Anthropic Messages API for completions and streaming.
    """

    # Default key name resolved from the API key store when no literal
    # ``api_key`` is given and no per-profile override is configured.
    DEFAULT_API_KEY_NAME = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_LLM_ANTHROPIC_MODEL,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        api_key_name: str | None = None,
        api_key_store=None,
        v: Variables = None,
    ):
        # ``api_key`` (literal) wins and skips the store; otherwise resolve the
        # key by name through the store on each use (picks up rotations).
        self.api_key, self._key_ref = resolve_api_key_ref(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            default_name=self.DEFAULT_API_KEY_NAME,
        )
        self.model = model
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._client = None
        self._client_key = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized AnthropicLLMProvider: model=%s", model)

    def _build_client(self, api_key):
        """Construct an Anthropic async client for ``api_key``."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        return AsyncAnthropic(api_key=api_key)

    def _get_client(self):
        """Lazy-load Anthropic async client from the static key (no store)."""
        if self._client is None:
            self._client = self._build_client(self.api_key)
            self._client_key = self.api_key
        return self._client

    async def _ensure_client(self):
        """Return a client built with the currently-resolved key.

        Rebuilds the cached client only when the store-resolved key changes.
        """
        if self._key_ref is None:
            return self._get_client()
        key = await self._key_ref.resolve()
        if self._client is None or key != self._client_key:
            self._client = self._build_client(key)
            self._client_key = key
        return self._client

    @staticmethod
    def _prepare_messages(request: LLMRequest):
        """Extract system message and translate the rest to Anthropic API shape.

        Anthropic requires system messages as a separate parameter,
        not within the messages list. Tool calls / tool results are
        translated to Anthropic's content-block format.

        Returns:
            Tuple of (system_message_text or None, formatted_messages_list)
        """
        system_text = None
        messages: list[dict] = []
        for msg in request.messages:
            role_str = _role_str(msg.role)
            if role_str == "system":
                if system_text is None:
                    system_text = msg.content
                else:
                    system_text += "\n" + msg.content
            else:
                messages.append(_translate_message_for_anthropic(msg))
        return system_text, messages

    def _build_kwargs(self, request: LLMRequest) -> dict:
        system_text, messages = self._prepare_messages(request)
        model = request.model or self.model
        max_tokens, temperature = self.resolve_params(request)

        effective_max = request.max_completion_tokens if request.max_completion_tokens is not None else max_tokens

        kwargs: dict = {"model": model, "messages": messages}
        if effective_max is not None:
            kwargs["max_tokens"] = effective_max
        if system_text is not None:
            kwargs["system"] = system_text
        if temperature is not None:
            kwargs["temperature"] = temperature
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        if request.tools is not None:
            kwargs["tools"] = _translate_tools_for_anthropic(request.tools)
        if request.tool_choice is not None:
            translated_tc = _translate_tool_choice_for_anthropic(request.tool_choice)
            if translated_tc is not None:
                kwargs["tool_choice"] = translated_tc
        thinking = _build_thinking_config(request.reasoning_effort)
        if thinking is not None:
            kwargs["thinking"] = thinking
        if request.extra_body is not None:
            # Anthropic SDK has no extra_body kwarg; merge into top-level kwargs.
            kwargs.update(request.extra_body)
        return kwargs

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic Messages API."""
        client = await self._ensure_client()
        kwargs = self._build_kwargs(request)

        self.logger.debug(
            "LLM request: model=%s, messages=%d, tools=%s",
            kwargs["model"],
            len(kwargs["messages"]),
            (len(request.tools) if request.tools else 0),
        )

        response = await client.messages.create(**kwargs)

        content, tool_calls, reasoning_content = _extract_anthropic_response(response)

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=_STOP_REASON_MAP.get(response.stop_reason, response.stop_reason or "stop"),
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    async def complete_stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming completion using Anthropic Messages API.

        Text deltas stream incrementally via ``stream.text_stream``. Tool
        calls and reasoning content are extracted from the final message
        and emitted on the terminal chunk — Anthropic's SDK exposes them
        post-aggregation rather than as raw incremental deltas in the
        same loop.
        """
        client = await self._ensure_client()
        kwargs = self._build_kwargs(request)

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield LLMStreamChunk(
                    content=text,
                    is_final=False,
                )

            message = await stream.get_final_message()
            _, tool_calls, reasoning_content = _extract_anthropic_response(message)

            if tool_calls:
                yield LLMStreamChunk(
                    content="",
                    is_final=False,
                    tool_calls_delta=tool_calls,
                )
            if reasoning_content:
                yield LLMStreamChunk(
                    content="",
                    is_final=False,
                    reasoning_content_delta=reasoning_content,
                )
            yield LLMStreamChunk(
                content="",
                is_final=True,
                finish_reason=_STOP_REASON_MAP.get(message.stop_reason, message.stop_reason or "stop"),
            )

    @property
    def default_model(self) -> str:
        return self.model

    @property
    def supports_streaming(self) -> bool:
        return True
