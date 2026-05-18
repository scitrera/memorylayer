"""Cascade transcription service - tries models in priority order."""

from logging import Logger

from scitrera_app_framework import Variables, get_logger

from .base import TranscriptionProvider, PageTranscription
from ...config import (
    EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT, DEFAULT_EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT,
)


class CascadeTranscriber:
    """
    Orchestrates transcription through a prioritized cascade of providers.

    Tries each provider in order until one succeeds. Collects statistics
    from all attempts for reporting.
    """

    def __init__(
            self,
            v: Variables = None,
            providers: list[TranscriptionProvider] = None,
    ):
        self.logger: Logger = get_logger(v, name=self.__class__.__name__)
        self.providers: list[TranscriptionProvider] = providers or []
        self._v = v
        self.logger.info(
            "Initialized CascadeTranscriber with %d providers: %s",
            len(self.providers),
            [p.PROVIDER_NAME for p in self.providers]
        )

    def get_system_prompt(self, override: str = None) -> str:
        """Get system prompt, using override or configured default."""
        if override:
            return override
        if self._v:
            return self._v.environ(
                EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT,
                default=DEFAULT_EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT,
            )
        return DEFAULT_EMBED_SERVER_TRANSCRIPTION_SYSTEM_PROMPT

    async def transcribe_page(
            self,
            image_data: bytes,
            page_index: int = 0,
            system_prompt: str = None,
            max_tokens: int = None,
    ) -> PageTranscription:
        """
        Transcribe a single page through the cascade.

        Tries each provider in priority order. Returns the first
        successful result or a failure with all attempts recorded.
        """
        system_prompt = self.get_system_prompt(system_prompt)
        result = PageTranscription(page_index=page_index)

        for provider in self.providers:
            self.logger.debug(
                "Trying provider %s for page %d",
                provider.PROVIDER_NAME, page_index
            )

            attempt = await provider.transcribe_page(
                image_data=image_data,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            result.attempts.append(attempt)

            if attempt.success:
                result.content = attempt.content
                result.success = True
                result.model_used = attempt.model
                result.provider_used = attempt.provider
                self.logger.info(
                    "Page %d transcribed by %s (%s) in %.1fms",
                    page_index, attempt.provider, attempt.model, attempt.latency_ms
                )
                break
            else:
                self.logger.info(
                    "Provider %s failed for page %d: %s",
                    provider.PROVIDER_NAME, page_index, attempt.error
                )

        if not result.success:
            self.logger.warning("All providers failed for page %d", page_index)
            result.content = "**Transcription Failed for this page**"

        return result

    async def transcribe_pages(
            self,
            images: list[bytes],
            system_prompt: str = None,
            max_tokens: int = None,
    ) -> list[PageTranscription]:
        """
        Transcribe multiple pages sequentially through the cascade.

        Each page is processed independently through the full cascade.
        """
        results = []
        for idx, image_data in enumerate(images):
            result = await self.transcribe_page(
                image_data=image_data,
                page_index=idx,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            results.append(result)

        return results

    async def preload(self):
        """Preload all providers."""
        for provider in self.providers:
            try:
                await provider.preload()
            except Exception as e:
                self.logger.warning(
                    "Failed to preload provider %s: %s",
                    provider.PROVIDER_NAME, e
                )
                import traceback
                traceback.print_exc()
