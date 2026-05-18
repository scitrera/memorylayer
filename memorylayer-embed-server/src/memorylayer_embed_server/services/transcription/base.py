"""Base transcription provider interface."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from logging import Logger

from scitrera_app_framework import Variables, get_logger


@dataclass
class TranscriptionAttempt:
    """Record of a single transcription attempt."""

    model: str
    provider: str
    success: bool = False
    content: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "unknown"
    error: str | None = None


@dataclass
class PageTranscription:
    """Result of transcribing a single page."""

    page_index: int
    content: str = ""
    success: bool = False
    model_used: str | None = None
    provider_used: str | None = None
    attempts: list[TranscriptionAttempt] = field(default_factory=list)


# Finish reasons that indicate content problems - should try next model
REJECTED_FINISH_REASONS = frozenset(
    {
        "recitation",
        "content_filter",
        "safety",
    }
)

# Finish reasons that indicate token limit hit
LENGTH_FINISH_REASONS = frozenset(
    {
        "length",
        "max_length",
        "max_tokens",
    }
)


def strip_thinking_tokens(content: str) -> str:
    """Strip thinking tokens from model output."""
    # Standard <think>...</think>
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # GLM/Kimi style ◁think▷...◁/think▷
    content = re.sub(r"◁think▷.*?◁/think▷", "", content, flags=re.DOTALL)
    return content.strip()


def strip_markdown_wrapper(content: str) -> str:
    """Strip markdown code-block wrapper if present."""
    content = content.strip()
    if content.startswith("```markdown") and content.endswith("```"):
        content = content[len("```markdown") : -(len("```"))].strip()
    elif content.startswith("```md") and content.endswith("```"):
        content = content[len("```md") : -(len("```"))].strip()
    elif content.startswith("```") and content.endswith("```"):
        # Only strip if first line is just ```
        lines = content.split("\n", 1)
        if lines[0].strip() == "```":
            content = content[len("```") : -(len("```"))].strip()
    return content


def clean_transcription_output(content: str) -> str:
    """Clean model output: strip thinking tokens and markdown wrappers."""
    content = strip_thinking_tokens(content)
    content = strip_markdown_wrapper(content)
    return content.strip()


class TranscriptionProvider(ABC):
    """Abstract transcription provider."""

    PROVIDER_NAME: str = ""

    def __init__(self, v: Variables = None):
        self.logger: Logger = get_logger(v, name=self.__class__.__name__)

    async def preload(self):
        """Optionally preload model resources."""
        return

    @abstractmethod
    async def transcribe_page(
        self,
        image_data: bytes,
        system_prompt: str,
        max_tokens: int = 16384,
    ) -> TranscriptionAttempt:
        """
        Transcribe a single page image to markdown.

        Args:
            image_data: Raw image bytes (PNG/JPEG)
            system_prompt: System prompt for the transcription
            max_tokens: Maximum output tokens

        Returns:
            TranscriptionAttempt with results
        """
        pass
