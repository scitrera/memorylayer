"""No-op LLM provider - raises NotConfigured (OSS default)."""
from typing import AsyncIterator

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import LLMProvider
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk


class LLMNotConfiguredError(Exception):
    """Raised when LLM is used but not configured."""
    pass


class NoOpLLMProvider(LLMProvider):
    """Default LLM provider that raises when called.

    OSS default - LLM features require explicit configuration.
    Set MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER and
    MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL to enable LLM features.
    """

    def __init__(self, v: Variables = None):
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized NoOpLLMProvider - LLM calls will raise NotConfigured. "
            "Set MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER and "
            "MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL to enable LLM features."
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise LLMNotConfiguredError(
            "LLM provider not configured. Set MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER "
            "and MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL to enable LLM features."
        )

    async def complete_stream(
        self,
        request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        raise LLMNotConfiguredError(
            "LLM provider not configured. Set MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER "
            "and MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL to enable LLM features."
        )
        # Yield is needed for type hints even though we raise
        yield  # pragma: no cover

    @property
    def default_model(self) -> str:
        return "not-configured"

    @property
    def supports_streaming(self) -> bool:
        return False
