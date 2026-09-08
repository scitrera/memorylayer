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


# Grounding markup emitted by the DeepSeek-OCR lineage (DeepSeek-OCR-2, Baidu
# Unlimited-OCR). Each recognized region is rendered as a text span followed by
# its bounding box:
#     <|ref|>Section 1<|/ref|><|det|>[[12, 34, 56, 78]]<|/det|>
# The text belongs in the markdown; the coordinates do not.
_GROUNDING_REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|>", re.DOTALL)
_GROUNDING_DET_RE = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
# Whatever markers survive the pass above — an unpaired opener from output
# truncated at max_tokens, or a special token left in because the request ran
# with skip_special_tokens=False (which the Unlimited-OCR recipe requires).
# Both the ASCII bar and the fullwidth bar the DeepSeek tokenizers use are
# matched. Bounded length so a stray "<|" in real document text can't eat a
# paragraph.
_RESIDUAL_SPECIAL_TOKEN_RE = re.compile(r"<[|｜][^<>\n]{0,64}?[|｜]>")


def strip_grounding_tokens(content: str) -> str:
    """Unwrap ``<|ref|>`` spans and drop ``<|det|>`` coordinate boxes.

    Turns the grounded OCR transcript into plain markdown: the referenced text
    is kept in place, the bounding boxes are discarded, and any residual
    special-token markers are removed.
    """
    content = _GROUNDING_DET_RE.sub("", content)
    content = _GROUNDING_REF_RE.sub(lambda m: m.group(1), content)
    return _RESIDUAL_SPECIAL_TOKEN_RE.sub("", content)


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
