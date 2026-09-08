"""Multimodal transcription via vLLM ``/v1/chat/completions``.

Replaces the in-process HuggingFace path for GLM-OCR and DeepSeek-OCR-2
with an out-of-process ``vllm serve`` subprocess that exposes the
OpenAI-compatible chat API. The transcription request is sent as a
chat-completion with a single user message containing the page image
(base64 data URL) and the transcription prompt.

Why this provider exists:

* The HF in-process DeepSeek-OCR-2 path is essentially broken — the
  arch doesn't support Flash Attention 2 or SDPA, only eager attention,
  which is slow and memory-hungry. vLLM has a documented serving recipe.
* GLM-OCR has its own vLLM recipe (with MTP speculative decoding) that
  gives meaningful throughput gains over in-process HF.
* The rest of the embed-server already runs vLLM subprocesses for
  single-vec embedding and multi-vector ColPali. Doing the same for OCR
  unifies the GPU isolation, memory accounting, and ops surface.

Each cascade member (GLM-OCR, DeepSeek-OCR-2) gets its own subprocess
on its own loopback port, with model-specific launch flags supplied via
``extra_args``. The provider class itself is model-agnostic — the
caller picks the right ``model_name`` + ``extra_args`` + ``provider_name``
combination.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Callable
from logging import Logger

from scitrera_app_framework import Variables, get_logger

from .._vllm_runner import VLLMSubprocessRunner
from .base import (
    LENGTH_FINISH_REASONS,
    REJECTED_FINISH_REASONS,
    TranscriptionAttempt,
    TranscriptionProvider,
    clean_transcription_output,
    strip_grounding_tokens,
)

# Defaults shared across both vLLM-served OCR profiles. Operators override
# per-profile via the wiring in ``dependencies._setup_transcription_cascade``.
DEFAULT_VLLM_OCR_HOST = "127.0.0.1"
DEFAULT_VLLM_OCR_DTYPE = "bfloat16"
# Conservative because OCR subprocesses share the GPU with the embedding
# and multi-vector subprocesses. vLLM measures utilization against TOTAL
# memory, not free memory — bumping above this risks "free < desired"
# startup failures on a half-used GPU. Production boxes on a dedicated
# inference GPU can safely raise this.
DEFAULT_VLLM_OCR_GPU_MEMORY_UTIL = 0.15
DEFAULT_VLLM_OCR_STARTUP_TIMEOUT_SEC = 600.0
DEFAULT_VLLM_OCR_CMD = "vllm"

# Unlimited-OCR is trained for one prompt recipe and returns empty output for
# anything else. The literal "<image>" prefix is part of the contract, not a
# formatting accident — see build_unlimited_ocr_vllm_provider.
UNLIMITED_OCR_PROMPT = "<image>document parsing."
# Per-request arguments for the model's n-gram logits processor, from the
# upstream card. window_size 128 is the single-image setting; multi-page /
# PDF input wants 1024.
UNLIMITED_OCR_NGRAM_SIZE = 35
UNLIMITED_OCR_WINDOW_SIZE = 128


class VLLMTranscriptionProvider(TranscriptionProvider):
    """OCR transcription served by a child ``vllm serve`` over chat-completions.

    Parameters
    ----------
    provider_name
        Cascade-attribution string (``"glm-ocr"`` / ``"deepseek-ocr"``).
        Kept stable across HF and vLLM backends so downstream consumers
        looking at ``provider_used`` see the same name.
    model_name
        HF repo id passed to ``vllm serve``.
    port
        Loopback port for the child vLLM API server. The provider
        connects to it via the OpenAI Python SDK.
    extra_args
        Model-specific ``vllm serve`` flags — e.g. ``--logits_processors``,
        ``--speculative-config.method``, ``--no-enable-prefix-caching``.
        Appended verbatim to the runner argv.
    prompt_override
        Exact user text to send instead of the generic
        ``"<system prompt>\\n\\nPlease transcribe …"`` phrasing. Set this for
        models trained against one fixed prompt recipe (Unlimited-OCR), where
        anything else degrades or empties the output. When set, the caller's
        ``system_prompt`` is ignored.
    text_first
        Put the text block ahead of the image block in the user message.
        Needed when the prompt itself carries the ``<image>`` placeholder and
        its position in the sequence is load-bearing. Default (``False``)
        keeps the image-then-text order the chat-templated models expect.
    extra_body
        Non-OpenAI request fields forwarded verbatim to the vLLM server —
        e.g. ``skip_special_tokens`` and per-request ``vllm_xargs`` for a
        custom logits processor.
    postprocess
        Applied to the raw completion text before the shared cleaning pass.
        Use it to unwrap model-specific markup (grounding tokens) that the
        generic cleaner knows nothing about.
    """

    PROVIDER_NAME: str = "vllm-transcription"  # overridden per-instance

    def __init__(
        self,
        *,
        provider_name: str,
        model_name: str,
        port: int,
        max_tokens: int,
        v: Variables = None,
        host: str = DEFAULT_VLLM_OCR_HOST,
        dtype: str = DEFAULT_VLLM_OCR_DTYPE,
        max_model_len: int | None = None,
        gpu_memory_utilization: float = DEFAULT_VLLM_OCR_GPU_MEMORY_UTIL,
        enforce_eager: bool = False,
        extra_args: list[str] | None = None,
        startup_timeout_sec: float = DEFAULT_VLLM_OCR_STARTUP_TIMEOUT_SEC,
        cmd: str = DEFAULT_VLLM_OCR_CMD,
        max_concurrent: int | None = None,
        oversubscribe_factor: float = 1.0,
        prompt_override: str | None = None,
        text_first: bool = False,
        extra_body: dict | None = None,
        postprocess: Callable[[str], str] | None = None,
    ):
        super().__init__(v)
        # Make PROVIDER_NAME instance-level so cascade attribution lines up
        # with whatever the operator chose ("glm-ocr", "deepseek-ocr", ...).
        self.PROVIDER_NAME = provider_name
        self.model_name = model_name
        self.default_max_tokens = max_tokens
        self.host = host
        self.port = int(port)
        self.prompt_override = prompt_override
        self.text_first = text_first
        self.extra_body = dict(extra_body) if extra_body else None
        self.postprocess = postprocess
        self.logger = get_logger(v, name=f"{self.__class__.__name__}[{provider_name}]")

        self._runner = VLLMSubprocessRunner(
            role="llm",  # generative, chat-completions endpoint
            model_name=model_name,
            host=host,
            port=int(port),
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            extra_args=extra_args,
            cmd=cmd,
            startup_timeout_sec=float(startup_timeout_sec),
            max_concurrent=max_concurrent,
            oversubscribe_factor=oversubscribe_factor,
            logger=self.logger,
        )

        self._client = None
        self._start_lock = asyncio.Lock()
        self._ready: bool = False
        self._skip_subprocess: bool = False  # test-harness hook

        self.logger.info(
            "Initialized VLLMTranscriptionProvider: provider=%s, model=%s, port=%d, gpu_memory_utilization=%.2f, extra_args=%s",
            provider_name,
            model_name,
            port,
            gpu_memory_utilization,
            extra_args or [],
        )

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._runner.base_url

    async def _ensure_started(self):
        if self._ready:
            return self._client
        async with self._start_lock:
            if self._ready:
                return self._client
            if not self._skip_subprocess:
                await self._runner.start()
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._runner.base_url, api_key="x")
            self._ready = True
            return self._client

    async def preload(self) -> None:
        await self._ensure_started()

    async def shutdown(self) -> None:
        await self._runner.shutdown()
        self._ready = False

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_data_url(image_data: bytes) -> str:
        """Encode page bytes as a ``data:image/png;base64,...`` URL.

        vLLM's multimodal pipeline accepts any common image format here;
        we declare PNG because that's what the route layer guarantees.
        """
        return f"data:image/png;base64,{base64.b64encode(image_data).decode('ascii')}"

    async def transcribe_page(
        self,
        image_data: bytes,
        system_prompt: str,
        max_tokens: int | None = None,
    ) -> TranscriptionAttempt:
        """Transcribe a single page image via vLLM chat-completions.

        The system prompt is folded into the single user message because
        not every model's chat_template handles the ``system`` role
        identically (DeepSeek-OCR-2 in particular uses a custom template
        with grounding tokens). One user message is the universal shape.

        Providers configured with a ``prompt_override`` ignore the caller's
        system prompt entirely — see the constructor docs.
        """
        max_tokens = max_tokens or self.default_max_tokens
        start_time = time.monotonic()

        attempt = TranscriptionAttempt(
            model=self.model_name,
            provider=self.PROVIDER_NAME,
        )

        try:
            client = await self._ensure_started()
            data_url = self._image_to_data_url(image_data)
            if self.prompt_override is not None:
                if system_prompt:
                    self.logger.debug(
                        "%s ignores the caller system prompt; using its fixed prompt recipe",
                        self.PROVIDER_NAME,
                    )
                user_text = self.prompt_override
            elif system_prompt:
                user_text = f"{system_prompt}\n\nPlease transcribe the document in the image to markdown."
            else:
                user_text = "Please transcribe the document in the image to markdown."

            text_part = {"type": "text", "text": user_text}
            image_part = {"type": "image_url", "image_url": {"url": data_url}}
            parts = [text_part, image_part] if self.text_first else [image_part, text_part]
            messages = [{"role": "user", "content": parts}]

            request_kwargs = {}
            if self.extra_body:
                request_kwargs["extra_body"] = self.extra_body

            async with self._runner.concurrency_slot():
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    **request_kwargs,
                )

            choice = response.choices[0]
            raw_content = choice.message.content or ""
            attempt.finish_reason = (choice.finish_reason or "unknown").lower()

            usage = getattr(response, "usage", None)
            if usage is not None:
                attempt.tokens_in = getattr(usage, "prompt_tokens", 0) or 0
                attempt.tokens_out = getattr(usage, "completion_tokens", 0) or 0

            if attempt.finish_reason in REJECTED_FINISH_REASONS:
                attempt.error = f"Rejected finish reason: {attempt.finish_reason}"
                self.logger.info(
                    "%s rejected: finish_reason=%s",
                    self.PROVIDER_NAME,
                    attempt.finish_reason,
                )
            elif attempt.finish_reason in LENGTH_FINISH_REASONS:
                attempt.error = f"Token limit reached: {attempt.finish_reason}"
                self.logger.info(
                    "%s token limit: finish_reason=%s, tokens=%d",
                    self.PROVIDER_NAME,
                    attempt.finish_reason,
                    attempt.tokens_out,
                )
            else:
                if self.postprocess is not None:
                    raw_content = self.postprocess(raw_content)
                content = clean_transcription_output(raw_content)
                if content:
                    attempt.content = content
                    attempt.success = True
                else:
                    attempt.error = "Empty content after cleaning"

        except Exception as e:  # noqa: BLE001 - cascade collects errors
            attempt.error = str(e)
            self.logger.warning("%s transcription failed: %s", self.PROVIDER_NAME, e)

        attempt.latency_ms = (time.monotonic() - start_time) * 1000
        return attempt


def build_glm_ocr_vllm_provider(
    *,
    v: Variables,
    logger: Logger,
    model_name: str,
    max_tokens: int,
    port: int,
    gpu_memory_utilization: float,
    startup_timeout_sec: float,
    cmd: str,
    enforce_eager: bool = True,
    max_concurrent: int | None = None,
    oversubscribe_factor: float = 1.0,
) -> VLLMTranscriptionProvider:
    """GLM-OCR via vLLM with MTP speculative decoding.

    From the upstream recipe::

        vllm serve zai-org/GLM-OCR \\
          --tensor-parallel-size 1 \\
          --speculative-config.method mtp \\
          --speculative-config.num_speculative_tokens 1

    ``--trust-remote-code`` is always added by VLLMSubprocessRunner.

    ``enforce_eager`` defaults to ``True`` because the CUDA-graph capture
    phase otherwise transiently allocates several extra GiB of GPU memory
    during boot — enough to trip earlyoom/cgroup OOM on a shared GPU.
    Operators on a dedicated inference card can flip it off for a 10-20%
    latency win.
    """
    del logger  # included for symmetry with build_deepseek_ocr_vllm_provider
    extra_args = [
        "--speculative-config.method",
        "mtp",
        "--speculative-config.num_speculative_tokens",
        "1",
    ]
    return VLLMTranscriptionProvider(
        v=v,
        provider_name="glm-ocr",
        model_name=model_name,
        port=port,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        extra_args=extra_args,
        startup_timeout_sec=startup_timeout_sec,
        cmd=cmd,
        max_concurrent=max_concurrent,
        oversubscribe_factor=oversubscribe_factor,
    )


def build_deepseek_ocr_vllm_provider(
    *,
    v: Variables,
    logger: Logger,
    model_name: str,
    max_tokens: int,
    port: int,
    gpu_memory_utilization: float,
    startup_timeout_sec: float,
    cmd: str,
    enforce_eager: bool = True,
    max_concurrent: int | None = None,
    oversubscribe_factor: float = 1.0,
) -> VLLMTranscriptionProvider:
    """DeepSeek-OCR-2 via vLLM with the recipe-mandated flags.

    From the upstream recipe::

        vllm serve deepseek-ai/DeepSeek-OCR-2 \\
          --trust-remote-code \\
          --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \\
          --no-enable-prefix-caching \\
          --mm-processor-cache-gb 0 \\
          --tensor-parallel-size 1

    ``--trust-remote-code`` is always added by VLLMSubprocessRunner.
    The logits processor + cache settings are model-specific and required.
    ``enforce_eager`` defaults to ``True`` — same reasoning as
    ``build_glm_ocr_vllm_provider``.
    """
    del logger  # included for symmetry with build_glm_ocr_vllm_provider
    extra_args = [
        "--logits_processors",
        "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        # Faster weight load than the default safetensors path. Requires
        # the ``instanttensor`` package — declared in the [vllm] extra.
        "--load-format",
        "instanttensor",
        # Skip vLLM's multimodal memory profiling — for DeepSeek-OCR-2
        # the default profile runs a forward pass with 2 max-feature-size
        # images, which produces a single transient memory peak large
        # enough to push the engine past its gpu_memory_utilization
        # budget on memory-constrained / shared GPUs (where SIGKILL
        # follows). Skipping this profile means vLLM measures only the
        # language backbone for KV-cache sizing; per-request multimodal
        # memory is bounded by ``--mm-processor-cache-gb 0`` anyway.
        "--skip-mm-profiling",
    ]
    return VLLMTranscriptionProvider(
        v=v,
        provider_name="deepseek-ocr",
        model_name=model_name,
        port=port,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        extra_args=extra_args,
        startup_timeout_sec=startup_timeout_sec,
        cmd=cmd,
        max_concurrent=max_concurrent,
        oversubscribe_factor=oversubscribe_factor,
    )


def build_unlimited_ocr_vllm_provider(
    *,
    v: Variables,
    logger: Logger,
    model_name: str,
    max_tokens: int,
    port: int,
    gpu_memory_utilization: float,
    startup_timeout_sec: float,
    cmd: str,
    enforce_eager: bool = True,
    max_concurrent: int | None = None,
    oversubscribe_factor: float = 1.0,
    prompt: str = UNLIMITED_OCR_PROMPT,
    ngram_size: int = UNLIMITED_OCR_NGRAM_SIZE,
    window_size: int = UNLIMITED_OCR_WINDOW_SIZE,
) -> VLLMTranscriptionProvider:
    """Baidu Unlimited-OCR via vLLM, wired to its mandated serving recipe.

    Unlimited-OCR is DeepSeek-OCR lineage (same *gundam* SAM-ViT-B + CLIP-L
    DeepEncoder vision stack) and inherits its n-gram logits processor, so the
    server flags mirror ``build_deepseek_ocr_vllm_provider``::

        vllm serve baidu/Unlimited-OCR \\
          --trust-remote-code \\
          --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \\
          --no-enable-prefix-caching \\
          --mm-processor-cache-gb 0

    The model ships **no chat template** and is trained against one exact
    prompt/decode recipe. Deviating from any part of it yields empty or
    looping output, so all four pieces are pinned here rather than left to
    the caller:

    * the user text is exactly ``"<image>document parsing."`` — the literal
      ``<image>`` placeholder leads the prompt, which is why the text block is
      sent ahead of the image block (``text_first=True``);
    * ``skip_special_tokens=False``, because the grounding markup the model
      emits is made of special tokens and dropping them at decode time leaves
      an empty string;
    * ``ngram_size`` / ``window_size`` ride along as ``vllm_xargs`` for the
      logits processor. Raise ``window_size`` to 1024 for multi-page input,
      which falls back to non-crop (base) mode;
    * the raw transcript is grounded markup — ``<|ref|>text<|/ref|>`` spans
      followed by ``<|det|>`` coordinate boxes — so ``strip_grounding_tokens``
      unwraps it back to markdown before the shared cleaning pass.

    ``--trust-remote-code`` is always added by VLLMSubprocessRunner.
    ``enforce_eager`` defaults to ``True`` — same reasoning as
    ``build_glm_ocr_vllm_provider``.
    """
    del logger  # included for symmetry with the sibling builders
    extra_args = [
        "--logits_processors",
        "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor",
        "--no-enable-prefix-caching",
        "--mm-processor-cache-gb",
        "0",
        # Same transient-peak problem as DeepSeek-OCR-2: vLLM's multimodal
        # profiling pass runs a forward at max feature size, which on a shared
        # GPU overshoots the utilization budget during engine init.
        "--skip-mm-profiling",
    ]
    return VLLMTranscriptionProvider(
        v=v,
        provider_name="unlimited-ocr",
        model_name=model_name,
        port=port,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        extra_args=extra_args,
        startup_timeout_sec=startup_timeout_sec,
        cmd=cmd,
        max_concurrent=max_concurrent,
        oversubscribe_factor=oversubscribe_factor,
        prompt_override=prompt,
        text_first=True,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": int(ngram_size), "window_size": int(window_size)},
        },
        postprocess=strip_grounding_tokens,
    )
