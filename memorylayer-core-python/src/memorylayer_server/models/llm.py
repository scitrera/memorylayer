from dataclasses import dataclass
from enum import Enum


class LLMRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    # Tool result message. The ``content`` is the function's return value
    # and ``tool_call_id`` references the assistant's tool_call that produced it.
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Single message in conversation.

    ``tool_calls`` / ``tool_call_id`` / ``name`` are the OpenAI canonical
    fields for function calling. Anthropic and Google providers translate
    them to their native shapes internally so callers can write a single
    transport-agnostic conversation.

    ``content`` is permitted to be an empty string for assistant messages
    that consist only of ``tool_calls``; providers serialize this in
    whatever way their wire format requires.
    """

    role: LLMRole
    content: str = ""
    # OpenAI-shape tool calls on an assistant message:
    # ``[{"id": "...", "type": "function",
    #     "function": {"name": "...", "arguments": "<json-string>"}}]``
    tool_calls: list[dict] | None = None
    # Set on a tool-result message; matches an earlier tool_call's id.
    tool_call_id: str | None = None
    # Legacy ``role=function`` name field; rarely needed in modern usage.
    name: str | None = None


@dataclass
class LLMRequest:
    """Request to LLM provider.

    Temperature resolution (applied by providers):
        1. Explicit ``temperature`` wins if set.
        2. ``temperature_factor * provider.default_temperature`` if factor is set.
        3. ``provider.default_temperature`` as the baseline fallback.

    ``max_tokens`` resolution: explicit value wins, else ``provider.default_max_tokens``.
    ``max_completion_tokens``, if set, takes precedence over ``max_tokens``
    for OpenAI o-series reasoning models.
    """

    messages: list[LLMMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    temperature_factor: float | None = None
    stop: list[str] | None = None
    stream: bool = False
    # Tool calling (OpenAI-shape canonical):
    # ``[{"type": "function",
    #     "function": {"name": "...", "description": "...", "parameters": {...}}}]``
    # Anthropic / Google providers translate into their native tools schemas.
    tools: list[dict] | None = None
    # ``"auto" | "none" | "required" | {"type": "function", "function": {"name": "..."}}``.
    tool_choice: str | dict | None = None
    # Structured output: ``{"type": "json_object"}`` or
    # ``{"type": "json_schema", "json_schema": {...}}``.
    response_format: dict | None = None
    # OpenAI o-series reasoning effort: ``"minimal"|"low"|"medium"|"high"``.
    # Anthropic maps to ``thinking`` budget; Google to ``thinking_config``.
    reasoning_effort: str | None = None
    # When set, overrides ``max_tokens`` for OpenAI o-series.
    max_completion_tokens: int | None = None
    # Provider-specific escape hatch. Caller owns the shape — these kwargs
    # are merged into the provider's SDK call directly.
    extra_body: dict | None = None


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str  # "stop", "length", "content_filter", "tool_calls"
    # OpenAI-shape tool calls produced by the assistant. ``None`` when the
    # model did not request any tool invocations.
    tool_calls: list[dict] | None = None
    # Reasoning / thinking summary text emitted by reasoning-capable models
    # (OpenAI o-series, Anthropic extended thinking, Gemini thinking).
    reasoning_content: str | None = None


@dataclass
class LLMStreamChunk:
    """Streaming response chunk.

    Either ``content`` is set (a text delta) or ``tool_calls_delta``
    is set (an incremental update to one or more tool calls under
    construction), not both — providers emit them in separate chunks.
    """

    content: str
    is_final: bool = False
    finish_reason: str | None = None
    # List of incremental tool-call updates. Shape mirrors OpenAI's
    # streaming tool_call deltas:
    # ``[{"index": 0, "id": "?", "type": "function",
    #     "function": {"name": "?", "arguments": "?"}}]``
    # with any field optional except ``index``.
    tool_calls_delta: list[dict] | None = None
    # Incremental reasoning / thinking text.
    reasoning_content_delta: str | None = None
