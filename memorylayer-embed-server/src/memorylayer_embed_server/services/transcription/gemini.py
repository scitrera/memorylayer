"""Gemini Flash transcription provider (external API fallback)."""

import time

from scitrera_app_framework import Variables

from .base import (
    TranscriptionProvider, TranscriptionAttempt, clean_transcription_output,
    REJECTED_FINISH_REASONS, LENGTH_FINISH_REASONS,
)
from ...config import (
    EMBED_SERVER_GEMINI_MODEL, DEFAULT_EMBED_SERVER_GEMINI_MODEL,
    EMBED_SERVER_GEMINI_MAX_TOKENS, DEFAULT_EMBED_SERVER_GEMINI_MAX_TOKENS,
)


class GeminiProvider(TranscriptionProvider):
    """
    Google Gemini Flash transcription provider.

    Uses the google-genai SDK for multimodal document transcription.
    Serves as a fallback when local GLM-OCR fails.
    """

    PROVIDER_NAME = "gemini"

    # TODO: add api key as an input here -- so that we can feed it in via plugin+custom env variable
    def __init__(
            self,
            v: Variables = None,
            model_name: str = DEFAULT_EMBED_SERVER_GEMINI_MODEL,
            max_tokens: int = DEFAULT_EMBED_SERVER_GEMINI_MAX_TOKENS,
    ):
        super().__init__(v)
        self.model_name = model_name
        self.default_max_tokens = max_tokens
        self._client = None
        self.logger.info("Initialized GeminiProvider with model: %s", model_name)

    def _get_client(self):
        """Lazy initialize Google GenAI client."""
        if self._client is None:
            from google import genai

            self._client = genai.Client()
            self.logger.info("Google GenAI client initialized")
        return self._client

    async def transcribe_page(
            self,
            image_data: bytes,
            system_prompt: str,
            max_tokens: int = None,
    ) -> TranscriptionAttempt:
        """Transcribe a page image using Gemini Flash."""
        import asyncio

        max_tokens = max_tokens or self.default_max_tokens
        start_time = time.monotonic()

        attempt = TranscriptionAttempt(
            model=self.model_name,
            provider=self.PROVIDER_NAME,
        )

        try:
            client = self._get_client()
            from google.genai import types

            # Build multimodal content with image
            image_part = types.Part.from_bytes(
                data=image_data,
                mime_type="image/png",
            )

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.3,
            )

            # Make API call in thread to not block event loop
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model_name,
                contents=[image_part, "Please transcribe this document page to clean markdown."],
                config=config,
            )

            if response and response.text:
                raw_content = response.text

                # Extract usage metadata
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    attempt.tokens_in = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    attempt.tokens_out = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                # Extract finish reason
                finish_reason = 'unknown'
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason).lower()
                        # google-genai uses enum values like STOP, MAX_TOKENS, SAFETY, RECITATION
                        finish_reason = finish_reason.replace('finishreason.', '').lower()

                attempt.finish_reason = finish_reason

                # Check for rejected finish reasons
                if finish_reason in REJECTED_FINISH_REASONS:
                    attempt.error = f"Rejected finish reason: {finish_reason}"
                    self.logger.info("Gemini rejected: finish_reason=%s", finish_reason)
                elif finish_reason in LENGTH_FINISH_REASONS or finish_reason == 'max_tokens':
                    attempt.error = f"Token limit reached: {finish_reason}"
                    self.logger.info("Gemini token limit: finish_reason=%s", finish_reason)
                else:
                    content = clean_transcription_output(raw_content)
                    if content:
                        attempt.content = content
                        attempt.success = True
                    else:
                        attempt.error = "Empty content after cleaning"
            else:
                attempt.error = "Empty response from Gemini"

        except Exception as e:
            attempt.error = str(e)
            self.logger.warning("Gemini transcription failed: %s", e)

        attempt.latency_ms = (time.monotonic() - start_time) * 1000
        return attempt
