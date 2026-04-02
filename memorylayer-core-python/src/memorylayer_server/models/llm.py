from dataclasses import dataclass
from enum import Enum


class LLMRole(str, Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """Single message in conversation."""

    role: LLMRole
    content: str


@dataclass
class LLMRequest:
    """Request to LLM provider.

    Temperature resolution (applied by providers):
        1. Explicit ``temperature`` wins if set.
        2. ``temperature_factor * provider.default_temperature`` if factor is set.
        3. ``provider.default_temperature`` as the baseline fallback.

    ``max_tokens`` resolution: explicit value wins, else ``provider.default_max_tokens``.
    """

    messages: list[LLMMessage]
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    temperature_factor: float | None = None
    stop: list[str] | None = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str  # "stop", "length", "content_filter"


@dataclass
class LLMStreamChunk:
    """Streaming response chunk."""

    content: str
    is_final: bool = False
    finish_reason: str | None = None
