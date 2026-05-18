"""Out-of-process vLLM embedding provider.

Spawns ``vllm serve`` (OpenAI-compatible API) as a child process and
proxies requests over HTTP via the ``openai`` async client. The vLLM
engine runs in its own process with its own asyncio loop and CUDA
context — so its memory accounting, scheduler, and crashes are
isolated from the embed-server's FastAPI loop.

Same configuration surface as the in-process ``VLLMEmbeddingProvider``
plus a few subprocess-specific knobs (host / port / startup timeout).

Selected on the embed-server with::

    MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_subprocess

Multimodal: ``embed_image`` and ``embed_multimodal(text+image)`` POST
a chat-style ``messages`` payload to ``/v1/embeddings`` (vLLM's
extension for vision-language embedding models like
``Qwen/Qwen3-VL-Embedding-2B``). Image bytes / file paths are
base64-encoded into a ``data:image/...`` URL; ``http(s)://`` and
pre-encoded ``data:`` URLs pass through.

Subprocess lifecycle (spawn / health-poll / SIGTERM-then-SIGKILL) is
delegated to :class:`memorylayer_embed_server.services._vllm_runner.VLLMSubprocessRunner`,
which is shared with the LLM-side ``vllm_subprocess`` provider.
"""

from __future__ import annotations

import asyncio
from logging import Logger
from pathlib import Path

from memorylayer_server.config import MEMORYLAYER_EMBEDDING_DIMENSIONS, MEMORYLAYER_EMBEDDING_MODEL
from memorylayer_server.services.embedding.base import (
    EmbeddingProviderPluginBase,
    MultimodalEmbeddingProvider,
)
from scitrera_app_framework import Variables, get_logger

from .._vllm_runner import VLLMSubprocessRunner

# Reuse the in-process vLLM env-var names where they overlap so operators
# only learn one config surface.
from .vllm import (
    DEFAULT_DTYPE,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_ENFORCE_EAGER,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_LENGTH,
    MEMORYLAYER_EMBEDDING_VLLM_DTYPE,
    MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER,
    MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL,
    MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH,
    MEMORYLAYER_EMBEDDING_VLLM_MODEL,
)

# Subprocess-specific env vars.
MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST = "MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST"
MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT = "MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT"
MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC = "MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC"
MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_CMD = "MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_CMD"

DEFAULT_VLLM_SUBPROCESS_HOST = "127.0.0.1"
DEFAULT_VLLM_SUBPROCESS_PORT = 18000
DEFAULT_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC = 600.0  # cold model load can take minutes
DEFAULT_VLLM_SUBPROCESS_CMD = "vllm"  # binary on PATH; allow override (e.g. full path)

PROVIDER_NAME_VLLM_SUBPROCESS = "vllm_subprocess"


class VLLMSubprocessEmbeddingProvider(MultimodalEmbeddingProvider):
    """Talks to a child ``vllm serve`` process over OpenAI-compatible HTTP."""

    def __init__(
        self,
        v: Variables = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        dtype: str = DEFAULT_DTYPE,
        max_model_len: int = DEFAULT_MAX_LENGTH,
        output_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
        host: str = DEFAULT_VLLM_SUBPROCESS_HOST,
        port: int = DEFAULT_VLLM_SUBPROCESS_PORT,
        startup_timeout_sec: float = DEFAULT_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
        cmd: str = DEFAULT_VLLM_SUBPROCESS_CMD,
    ):
        super().__init__(v, output_dimensions=output_dimensions)
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.output_dimensions = output_dimensions
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.host = host
        self.port = int(port)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.cmd = cmd
        self._dimensions = output_dimensions

        self.logger = get_logger(v, name=self.__class__.__name__)

        self._runner = VLLMSubprocessRunner(
            role="embedding",
            model_name=model_name,
            host=host,
            port=int(port),
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            cmd=cmd,
            startup_timeout_sec=float(startup_timeout_sec),
            logger=self.logger,
        )

        self._client = None
        self._mm_http = None  # lazy httpx.AsyncClient for multimodal calls
        self._start_lock = asyncio.Lock()
        self._ready: bool = False

        # Allow the test harness to skip starting the subprocess and
        # talk straight to a pre-existing endpoint (e.g. compose-managed
        # vllm sidecar). When this flag is true, ``_ensure_started``
        # just opens the HTTP client.
        self._skip_subprocess: bool = False

        self.logger.info(
            "Initialized VLLMSubprocessEmbeddingProvider: model=%s, dtype=%s, "
            "host=%s, port=%d, gpu_memory_utilization=%.2f, enforce_eager=%s, "
            "startup_timeout_sec=%.0f",
            model_name,
            dtype,
            host,
            port,
            gpu_memory_utilization,
            enforce_eager,
            startup_timeout_sec,
        )

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._runner.base_url

    @property
    def health_url(self) -> str:
        return self._runner.health_url

    def _build_vllm_argv(self) -> list[str]:
        return self._runner.build_argv()

    async def _ensure_started(self):
        if self._ready:
            return self._client
        async with self._start_lock:
            if self._ready:
                return self._client

            if not self._skip_subprocess:
                await self._runner.start()

            # Build the OpenAI-compat client. API key is required by the
            # SDK but vllm serve doesn't enforce it.
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=self._runner.base_url, api_key="x")
            self._ready = True
            return self._client

    async def preload(self) -> None:
        """Start the subprocess and wait for it to be healthy."""
        await self._ensure_started()

    async def shutdown(self) -> None:
        """Terminate the child process tree. Safe to call multiple times."""
        await self._runner.shutdown()
        if self._mm_http is not None:
            try:
                await self._mm_http.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            self._mm_http = None
        self._ready = False

    # ------------------------------------------------------------------
    # Embedding methods
    # ------------------------------------------------------------------

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = await self._ensure_started()
        response = await client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        # OpenAI client preserves request order.
        out: list[list[float]] = []
        for item in response.data:
            emb = list(item.embedding)
            if len(emb) > self.output_dimensions:
                emb = emb[: self.output_dimensions]
            out.append(emb)
        return out

    async def _embed_chat_messages(self, messages: list[dict]) -> list[float]:
        """POST a chat-style messages payload to ``/v1/embeddings``.

        vLLM's OpenAI-compatible server accepts an alternate ``messages``
        field on ``/v1/embeddings`` for vision-language embedding models
        (e.g. ``Qwen/Qwen3-VL-Embedding-2B``). The OpenAI Python SDK
        doesn't type this field, so we bypass it and call the endpoint
        directly via ``httpx``.
        """
        import httpx

        # Reuse one AsyncClient instance for multimodal traffic so the
        # connection pool is shared across calls.
        if self._mm_http is None:
            self._mm_http = httpx.AsyncClient(
                base_url=self._runner.base_url,
                timeout=300.0,
            )
        await self._ensure_started()
        resp = await self._mm_http.post(
            "/embeddings",
            json={"model": self.model_name, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            embedding = list(data["data"][0]["embedding"])
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected vllm /v1/embeddings response shape: {data!r}") from e
        if len(embedding) > self.output_dimensions:
            embedding = embedding[: self.output_dimensions]
        return embedding

    @staticmethod
    def _image_to_data_url(image: str | bytes | Path) -> str:
        """Coerce supported image inputs to an ``image_url`` value.

        Accepted: pre-encoded ``data:`` URLs (pass through), ``http(s)://``
        URLs (pass through so vllm can fetch them server-side), raw bytes,
        ``Path``, and strings that name a readable file or are a raw
        base64 blob. Bytes / file contents are base64-encoded as a
        ``data:image/jpeg;base64,...`` URL.
        """
        import base64

        if isinstance(image, str):
            if image.startswith("data:image"):
                return image
            if image.startswith(("http://", "https://")):
                return image
            p = Path(image)
            if len(image) <= 500 or p.exists():
                try:
                    raw = p.read_bytes()
                    b64 = base64.b64encode(raw).decode("ascii")
                    return f"data:image/jpeg;base64,{b64}"
                except (OSError, ValueError):
                    pass
            # Treat as a raw base64 string.
            return f"data:image/jpeg;base64,{image}"
        if isinstance(image, bytes):
            b64 = base64.b64encode(image).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        if isinstance(image, Path):
            b64 = base64.b64encode(image.read_bytes()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    @classmethod
    def _build_chat_messages(
        cls,
        text: str | None,
        image: str | bytes | Path | None,
    ) -> list[dict]:
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        if image is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": cls._image_to_data_url(image)},
                }
            )
        return [{"role": "user", "content": content}]

    async def embed(self, text: str) -> list[float]:
        self.logger.debug("vllm-subprocess embed(): %d chars", len(text))
        results = await self._embed_texts([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.logger.debug("vllm-subprocess embed_batch(): %d texts", len(texts))
        return await self._embed_texts(texts)

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        """Image-only embedding via vLLM's chat-style ``/v1/embeddings`` payload.

        Requires a vision-language embedding model on the subprocess side
        (e.g. ``Qwen/Qwen3-VL-Embedding-2B``). Text-only models will
        respond with an error which propagates to the caller.
        """
        self.logger.debug("vllm-subprocess embed_image()")
        messages = self._build_chat_messages(text=None, image=image)
        return await self._embed_chat_messages(messages)

    async def embed_multimodal(
        self,
        text: str | None = None,
        image: str | bytes | Path | None = None,
    ) -> list[float]:
        """Combined text + image embedding via vLLM's chat-style payload."""
        if image is None and text is None:
            raise ValueError("At least one of text or image must be provided")
        if image is None:
            # Text-only — the cheap /v1/embeddings ``input`` path is
            # sufficient and avoids the chat payload entirely.
            return await self.embed(text)
        self.logger.debug(
            "vllm-subprocess embed_multimodal(): text=%d chars, +image",
            len(text) if text else 0,
        )
        messages = self._build_chat_messages(text=text, image=image)
        return await self._embed_chat_messages(messages)

    @property
    def dimensions(self) -> int:
        return self._dimensions


class VLLMSubprocessEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    """Plugin entry point. Wired through the embed-server's
    ``_init_single_vector_provider`` dispatcher; not auto-selected via
    ``MEMORYLAYER_EMBEDDING_PROVIDER``.
    """

    PROVIDER_NAME = PROVIDER_NAME_VLLM_SUBPROCESS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        # Model name follows the same resolution order as the in-process
        # vLLM provider: VLLM_MODEL override → shared MODEL → built-in default.
        model_name = v.environ(MEMORYLAYER_EMBEDDING_VLLM_MODEL, default=None)
        if not model_name:
            model_name = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL)

        return VLLMSubprocessEmbeddingProvider(
            v=v,
            model_name=model_name,
            dtype=v.environ(MEMORYLAYER_EMBEDDING_VLLM_DTYPE, default=DEFAULT_DTYPE),
            max_model_len=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH,
                default=DEFAULT_MAX_LENGTH,
                type_fn=int,
            ),
            output_dimensions=v.environ(
                MEMORYLAYER_EMBEDDING_DIMENSIONS,
                default=DEFAULT_EMBEDDING_DIMENSIONS,
                type_fn=int,
            ),
            gpu_memory_utilization=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL,
                default=DEFAULT_GPU_MEMORY_UTILIZATION,
                type_fn=float,
            ),
            enforce_eager=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER,
                default=DEFAULT_ENFORCE_EAGER,
                type_fn=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
            ),
            host=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_HOST,
                default=DEFAULT_VLLM_SUBPROCESS_HOST,
            ),
            port=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_PORT,
                default=DEFAULT_VLLM_SUBPROCESS_PORT,
                type_fn=int,
            ),
            startup_timeout_sec=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
                default=DEFAULT_VLLM_SUBPROCESS_STARTUP_TIMEOUT_SEC,
                type_fn=float,
            ),
            cmd=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_SUBPROCESS_CMD,
                default=DEFAULT_VLLM_SUBPROCESS_CMD,
            ),
        )
