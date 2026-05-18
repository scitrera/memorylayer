"""GLM-OCR transcription provider using HuggingFace Transformers.

Loads the GLM-OCR model locally via AutoModelForImageTextToText and
runs inference on-device. No external server or cloud API required.
"""

import asyncio
import io
import time

from PIL import Image
from scitrera_app_framework import Variables

from .base import (
    TranscriptionProvider, TranscriptionAttempt, clean_transcription_output,
    LENGTH_FINISH_REASONS,
)
from ...config import (
    EMBED_SERVER_GLM_OCR_MODEL, DEFAULT_EMBED_SERVER_GLM_OCR_MODEL,
    EMBED_SERVER_GLM_OCR_MAX_TOKENS, DEFAULT_EMBED_SERVER_GLM_OCR_MAX_TOKENS,
)


class GLMOCRProvider(TranscriptionProvider):
    """
    GLM-OCR transcription provider using HuggingFace Transformers.

    Loads the model locally with device_map="auto" for automatic GPU placement.
    Uses AutoProcessor for chat-template-based image+text input formatting.
    """

    PROVIDER_NAME = "glm-ocr"

    def __init__(
            self,
            v: Variables = None,
            model_name: str = DEFAULT_EMBED_SERVER_GLM_OCR_MODEL,
            max_tokens: int = DEFAULT_EMBED_SERVER_GLM_OCR_MAX_TOKENS,
    ):
        super().__init__(v)
        self.model_name = model_name
        self.default_max_tokens = max_tokens
        self._processor = None
        self._model = None
        self.logger.info(
            "Initialized GLMOCRProvider with model: %s, max_tokens: %d",
            model_name, max_tokens
        )

    def _load_model(self):
        """Lazy load the model and processor."""
        if self._model is None:
            from transformers import AutoProcessor, AutoModelForImageTextToText

            self.logger.info("Loading GLM-OCR processor: %s", self.model_name)
            self._processor = AutoProcessor.from_pretrained(self.model_name)

            self.logger.info("Loading GLM-OCR model: %s", self.model_name)
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                torch_dtype="auto",
                device_map="auto",
            )
            self.logger.info("GLM-OCR model loaded successfully")

    async def preload(self):
        """Preload the model and processor onto GPU."""
        self.logger.info("Preloading GLM-OCR model")
        await asyncio.to_thread(self._load_model)

    async def transcribe_page(
            self,
            image_data: bytes,
            system_prompt: str,
            max_tokens: int = None,
    ) -> TranscriptionAttempt:
        """Transcribe a page image using local GLM-OCR model inference."""
        max_tokens = max_tokens or self.default_max_tokens
        start_time = time.monotonic()

        attempt = TranscriptionAttempt(
            model=self.model_name,
            provider=self.PROVIDER_NAME,
        )

        try:
            self._load_model()

            # Convert raw bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))

            # Build chat messages following GLM-OCR format
            prompt = f"{system_prompt}\n\nPlease transcribe the document in the image to markdown."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Run model inference in thread to avoid blocking the event loop
            def _generate():
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self._model.device)
                inputs.pop("token_type_ids", None)

                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                )

                # Decode only the newly generated tokens (exclude input)
                input_len = inputs["input_ids"].shape[1]
                output_ids = generated_ids[0][input_len:]
                text = self._processor.decode(output_ids, skip_special_tokens=True)

                return text, input_len, len(output_ids)

            raw_content, prompt_tokens, completion_tokens = await asyncio.to_thread(_generate)

            attempt.tokens_in = prompt_tokens
            attempt.tokens_out = completion_tokens
            attempt.finish_reason = "length" if completion_tokens >= max_tokens else "stop"

            if attempt.finish_reason in LENGTH_FINISH_REASONS:
                attempt.error = f"Token limit reached: {attempt.finish_reason}"
                self.logger.info(
                    "GLM-OCR token limit: finish_reason=%s, tokens=%d",
                    attempt.finish_reason, completion_tokens
                )
            else:
                # Clean and validate content
                content = clean_transcription_output(raw_content)
                if content:
                    attempt.content = content
                    attempt.success = True
                else:
                    attempt.error = "Empty content after cleaning"

        except Exception as e:
            attempt.error = str(e)
            self.logger.warning("GLM-OCR transcription failed: %s", e)

        attempt.latency_ms = (time.monotonic() - start_time) * 1000
        return attempt
