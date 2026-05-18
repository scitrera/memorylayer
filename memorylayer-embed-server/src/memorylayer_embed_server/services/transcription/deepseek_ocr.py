"""DeepSeek-OCR-2 transcription provider using HuggingFace Transformers.

Loads the DeepSeek-OCR-2 model locally via AutoModel with trust_remote_code
and runs inference on-device using the model's built-in infer() method.
No external server or cloud API required.
"""

import asyncio
import io
import os
import tempfile
import time

from PIL import Image
from scitrera_app_framework import Variables

from .base import (
    TranscriptionProvider, TranscriptionAttempt, clean_transcription_output,
)
from ...config import (
    EMBED_SERVER_DEEPSEEK_OCR_MODEL, DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MODEL,
    EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS, DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS,
)


class DeepSeekOCRProvider(TranscriptionProvider):
    """
    DeepSeek-OCR-2 transcription provider using HuggingFace Transformers.

    Uses AutoModel with trust_remote_code=True and the model's built-in
    infer() method for document OCR. The model uses special prompt tokens
    (<image>, <|grounding|>) rather than standard chat templates.
    """

    PROVIDER_NAME = "deepseek-ocr"

    def __init__(
            self,
            v: Variables = None,
            model_name: str = DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MODEL,
            max_tokens: int = DEFAULT_EMBED_SERVER_DEEPSEEK_OCR_MAX_TOKENS,
    ):
        super().__init__(v)
        self.model_name = model_name
        self.default_max_tokens = max_tokens
        self._tokenizer = None
        self._model = None
        self.logger.info(
            "Initialized DeepSeekOCRProvider with model: %s, max_tokens: %d",
            model_name, max_tokens
        )

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is None:
            import torch

            model_name = self.model_name
            self.logger.info("Loading DeepSeek-OCR-2 model: %s", model_name)

            def _init():
                from transformers import AutoTokenizer, AutoModel

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True,
                )
                model = AutoModel.from_pretrained(
                    model_name,
                    use_mla=True,
                    _attn_implementation='flash_attention_2',
                    trust_remote_code=True,
                    use_safetensors=True,
                )
                model = model.eval().cuda().to(torch.bfloat16)
                self._model = model
                self._tokenizer = tokenizer

            def _patch_and_init():
                """Patch missing imports for newer transformers versions."""
                import sys
                # LlamaFlashAttention2 was removed in newer transformers
                llama_mod = sys.modules.get("transformers.models.llama.modeling_llama")
                if llama_mod and not hasattr(llama_mod, "LlamaFlashAttention2"):
                    setattr(llama_mod, "LlamaFlashAttention2", object())
                # is_torch_fx_available was removed in newer transformers
                import_mod = sys.modules.get("transformers.utils.import_utils")
                if import_mod and not hasattr(import_mod, "is_torch_fx_available"):
                    setattr(import_mod, "is_torch_fx_available", lambda: True)
                _init()

            try:
                _init()
            except (ImportError, AttributeError):
                self.logger.info("Applying transformers compatibility patches")
                _patch_and_init()

            self.logger.info("DeepSeek-OCR-2 model loaded successfully")

    async def preload(self):
        """Preload the model and tokenizer onto GPU."""
        self.logger.info("Preloading DeepSeek-OCR-2 model")
        await asyncio.to_thread(self._load_model)

    async def transcribe_page(
            self,
            image_data: bytes,
            system_prompt: str,
            max_tokens: int = None,
    ) -> TranscriptionAttempt:
        """Transcribe a page image using local DeepSeek-OCR-2 model inference.

        Uses the model's built-in infer() method which handles image
        preprocessing, cropping, and OCR internally. The system_prompt
        parameter is accepted for interface compatibility but not used;
        DeepSeek-OCR-2 uses its own fixed prompt tokens.
        """
        start_time = time.monotonic()

        attempt = TranscriptionAttempt(
            model=self.model_name,
            provider=self.PROVIDER_NAME,
        )

        try:
            self._load_model()

            def _generate():
                # DeepSeek-OCR-2 infer() requires a file path, not raw bytes
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    pil_image = Image.open(io.BytesIO(image_data))
                    pil_image.save(tmp, format="PNG")

                try:
                    # DeepSeek-OCR-2 uses special prompt tokens, not chat templates
                    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

                    res = self._model.infer(
                        self._tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        base_size=1024,
                        image_size=768,
                        crop_mode=True,
                        save_results=False,
                    )
                    return res
                finally:
                    os.unlink(tmp_path)

            raw_content = await asyncio.to_thread(_generate)

            if raw_content:
                content = clean_transcription_output(str(raw_content))
                if content:
                    attempt.content = content
                    attempt.success = True
                    attempt.finish_reason = "stop"
                else:
                    attempt.error = "Empty content after cleaning"
            else:
                attempt.error = "Empty response from DeepSeek-OCR-2"

        except Exception as e:
            attempt.error = str(e)
            self.logger.warning("DeepSeek-OCR-2 transcription failed: %s", e)

        attempt.latency_ms = (time.monotonic() - start_time) * 1000
        return attempt
