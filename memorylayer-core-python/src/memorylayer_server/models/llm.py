from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


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
    messages: List[LLMMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    temperature_factor: Optional[float] = None
    stop: Optional[List[str]] = None
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
    finish_reason: Optional[str] = None
