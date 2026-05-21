"""Out-of-process vLLM multi-vector (ColPali-family) embedding provider.

Spawns ``vllm serve --runner pooling`` as a child process for a ColPali
checkpoint (default ``ModernVBERT/colmodernvbert-merged``) and proxies
per-token embedding requests over HTTP. vLLM's ``/pooling`` endpoint
returns one vector per token; we package those into the same
``MultiVectorEmbedding`` shape the in-process colpali-engine provider
emits so the rest of the embed-server (``/v1/embeddings/multi``,
``/v1/embeddings/images``, ``/v1/score``) doesn't care which backend
served the request.

Selected on the embed-server with::

    EMBED_SERVER_MULTI_VECTOR_PROVIDER=vllm_subprocess

ColPali checkpoints often ship without a populated ``architectures``
field (``colmodernvbert-merged`` is one example); the runner forwards
``--hf-overrides '{"architectures": [...]}'`` so vLLM routes the model
through the correct arch class (e.g. ``ColModernVBertForRetrieval``).

Subprocess lifecycle (spawn / health-poll / SIGTERM-then-SIGKILL) is
delegated to :class:`memorylayer_embed_server.services._vllm_runner.VLLMSubprocessRunner`.
Hierarchical token pooling is applied client-side post-response, using
the same factor as the in-process colpali path so MaxSim geometry stays
consistent across providers.
"""

from __future__ import annotations

import asyncio
import base64
from logging import Logger
from pathlib import Path

import numpy as np
from memorylayer_server.services.embedding._maxsim import MultiVectorEmbedding
from memorylayer_server.services.embedding.base import (
    EmbeddingProviderPluginBase,
    MultimodalEmbeddingProvider,
)
from scitrera_app_framework import Variables, get_logger

from .._vllm_runner import VLLMSubprocessRunner

# Reuse the existing ColPali config knobs where they overlap so operators
# only learn one surface.
from .colpali import (
    DEFAULT_COLPALI_POOL_FACTOR,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL,
    MEMORYLAYER_EMBEDDING_COLPALI_MODEL,
    MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR,
    MEMORYLAYER_EMBEDDING_REVISION,
)
from .vllm import (
    DEFAULT_DTYPE,
    DEFAULT_ENFORCE_EAGER,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MAX_LENGTH,
    MEMORYLAYER_EMBEDDING_VLLM_DTYPE,
    MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER,
    MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL,
    MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH,
)

# Multi-vec-specific subprocess + override env vars.
MEMORYLAYER_EMBEDDING_VLLM_MV_HOST = "MEMORYLAYER_EMBEDDING_VLLM_MV_HOST"
MEMORYLAYER_EMBEDDING_VLLM_MV_PORT = "MEMORYLAYER_EMBEDDING_VLLM_MV_PORT"
MEMORYLAYER_EMBEDDING_VLLM_MV_STARTUP_TIMEOUT_SEC = "MEMORYLAYER_EMBEDDING_VLLM_MV_STARTUP_TIMEOUT_SEC"
MEMORYLAYER_EMBEDDING_VLLM_MV_CMD = "MEMORYLAYER_EMBEDDING_VLLM_MV_CMD"
MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES = "MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES"
MEMORYLAYER_EMBEDDING_VLLM_MV_DIMENSIONS = "MEMORYLAYER_EMBEDDING_VLLM_MV_DIMENSIONS"
# Per-multi-vec max length override. Unset → let vLLM derive from the model's
# config (ColModernVBert caps at 7999; ColQwen3.5 at ~32k). The single-vec
# VLLM_MAX_LENGTH default (32768) is wrong for ColModernVBert and trips a
# pydantic validation error at vLLM boot.
MEMORYLAYER_EMBEDDING_VLLM_MV_MAX_LENGTH = "MEMORYLAYER_EMBEDDING_VLLM_MV_MAX_LENGTH"

DEFAULT_MV_VLLM_HOST = "127.0.0.1"
DEFAULT_MV_VLLM_PORT = 18001  # one above the single-vec subprocess default
DEFAULT_MV_VLLM_STARTUP_TIMEOUT_SEC = 600.0
DEFAULT_MV_VLLM_CMD = "vllm"
# ColModernVBert-merged ships ``architectures: None`` in its config — point
# vLLM at the right routing class. ColQwen3.5 ships the right value in
# config.json so an empty default is fine when an operator swaps models.
DEFAULT_MV_VLLM_ARCHITECTURES = "ColModernVBertForRetrieval"
# ColPali family default projection dim — 128 across ColPali, ColModernVBert,
# ColQwen2/2.5. ColQwen3.5-4.5B-v3 ships 320; override via env when swapping.
DEFAULT_MV_VLLM_DIMENSIONS = 128

PROVIDER_NAME_VLLM_MULTI_VECTOR = "vllm_multi_vector"


class VLLMMultiVectorProvider(MultimodalEmbeddingProvider):
    """Talks to a child ``vllm serve --runner pooling`` process over HTTP."""

    def __init__(
        self,
        v: Variables = None,
        model_name: str = DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL,
        dtype: str = DEFAULT_DTYPE,
        max_model_len: int | None = None,
        output_dimensions: int = DEFAULT_MV_VLLM_DIMENSIONS,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
        architectures: list[str] | None = None,
        pool_factor: int = DEFAULT_COLPALI_POOL_FACTOR,
        host: str = DEFAULT_MV_VLLM_HOST,
        port: int = DEFAULT_MV_VLLM_PORT,
        startup_timeout_sec: float = DEFAULT_MV_VLLM_STARTUP_TIMEOUT_SEC,
        cmd: str = DEFAULT_MV_VLLM_CMD,
    ):
        super().__init__(v, output_dimensions=output_dimensions)
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.architectures = list(architectures) if architectures else []
        self.pool_factor = max(1, int(pool_factor))
        self.host = host
        self.port = int(port)
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.cmd = cmd
        self._dimensions = output_dimensions

        self.logger = get_logger(v, name=self.__class__.__name__)

        self._runner = VLLMSubprocessRunner(
            role="multi_vector",
            model_name=model_name,
            host=host,
            port=int(port),
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            architectures=self.architectures,
            cmd=cmd,
            startup_timeout_sec=float(startup_timeout_sec),
            logger=self.logger,
        )

        self._http = None  # lazy httpx.AsyncClient
        self._pooler = None  # lazy HierarchicalTokenPooler
        self._start_lock = asyncio.Lock()
        self._ready: bool = False
        self._skip_subprocess: bool = False  # test-harness hook

        self.logger.info(
            "Initialized VLLMMultiVectorProvider: model=%s, dtype=%s, host=%s, port=%d, "
            "gpu_memory_utilization=%.2f, architectures=%s, pool_factor=%d",
            model_name,
            dtype,
            host,
            port,
            gpu_memory_utilization,
            self.architectures or "(auto)",
            self.pool_factor,
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

    async def _ensure_started(self):
        if self._ready:
            return self._http
        async with self._start_lock:
            if self._ready:
                return self._http
            if not self._skip_subprocess:
                await self._runner.start()
            import httpx

            # vLLM's /pooling endpoint is at the server root, NOT under /v1
            # (it's a vLLM extension, not OpenAI-compat). Strip the /v1 suffix
            # that VLLMSubprocessRunner.base_url tacks on so client paths like
            # "/pooling" land at the right URL.
            root_url = self._runner.base_url.removesuffix("/v1")
            self._http = httpx.AsyncClient(base_url=root_url, timeout=300.0)
            self._ready = True
            return self._http

    async def preload(self) -> None:
        await self._ensure_started()

    async def shutdown(self) -> None:
        await self._runner.shutdown()
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            self._http = None
        self._ready = False

    # ------------------------------------------------------------------
    # Pooling helper (mirrors the in-process colpali path)
    # ------------------------------------------------------------------

    def _apply_pooling(self, vectors_list: list[list[list[float]]]) -> list[list[list[float]]]:
        """Apply HierarchicalTokenPooler client-side on the per-token vectors.

        Same factor as the in-process colpali path keeps MaxSim geometry
        consistent across backends. ``vectors_list`` is a list of 2D
        ``(num_tokens, dim)`` python-list matrices, one per input.
        """
        if self.pool_factor <= 1:
            return vectors_list
        if self._pooler is None:
            from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

            self._pooler = HierarchicalTokenPooler()
        import torch

        tensors = [torch.tensor(v, dtype=torch.float32) for v in vectors_list]
        pooled = self._pooler.pool_embeddings(tensors, pool_factor=self.pool_factor)
        return [t.numpy().tolist() for t in pooled]

    # ------------------------------------------------------------------
    # /pooling protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pooling_response(data: dict, expected_count: int) -> list[list[list[float]]]:
        """Extract per-input per-token vectors from vLLM's ``/pooling`` JSON.

        vLLM returns ``{"data": [{"index": i, "data": [[…], [...]], ...}, ...]}``
        — each element's nested ``data`` is the 2D ``(tokens, dim)`` matrix.
        We tolerate either ``embedding`` or ``data`` as the inner field name
        because vLLM has shifted naming across minor versions.
        """
        try:
            items = data["data"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected vllm /pooling response: missing 'data' key in {data!r}") from e
        if len(items) != expected_count:
            raise RuntimeError(
                f"vllm /pooling returned {len(items)} items, expected {expected_count}: {data!r}"
            )
        items_sorted = sorted(items, key=lambda d: d.get("index", 0))
        out: list[list[list[float]]] = []
        for item in items_sorted:
            inner = item.get("data")
            if inner is None:
                inner = item.get("embedding")
            if inner is None:
                raise RuntimeError(f"vllm /pooling item missing 'data'/'embedding' field: {item!r}")
            # Some vLLM versions emit a flat list when the model collapses
            # to one vector; defend against that by wrapping into one row.
            if inner and not isinstance(inner[0], list):
                inner = [inner]
            out.append(inner)
        return out

    async def _pooling_text(self, texts: list[str]) -> list[list[list[float]]]:
        client = await self._ensure_started()
        resp = await client.post(
            "/pooling",
            json={"model": self.model_name, "input": texts},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"vllm /pooling failed with status {resp.status_code}: {resp.text[:500]}"
            )
        return self._parse_pooling_response(resp.json(), expected_count=len(texts))

    @staticmethod
    def _image_to_data_url(image: str | bytes | Path) -> str:
        if isinstance(image, str):
            if image.startswith("data:image"):
                return image
            if image.startswith(("http://", "https://")):
                return image
            p = Path(image)
            if len(image) <= 500 or p.exists():
                try:
                    raw = p.read_bytes()
                    return f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
                except (OSError, ValueError):
                    pass
            return f"data:image/jpeg;base64,{image}"
        if isinstance(image, bytes):
            return f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
        if isinstance(image, Path):
            return f"data:image/jpeg;base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    async def _pooling_image(self, image: str | bytes | Path) -> list[list[float]]:
        """Single image → multi-vector via vLLM's chat-style payload.

        Mirrors the multimodal path used by ``VLLMSubprocessEmbeddingProvider``
        for ``/v1/embeddings``; ``/pooling`` accepts the same ``messages``
        field for vision-language pooling models.
        """
        client = await self._ensure_started()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image)}},
                ],
            }
        ]
        resp = await client.post(
            "/pooling",
            json={"model": self.model_name, "messages": messages},
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"vllm /pooling (image) failed with status {resp.status_code}: {resp.text[:500]}"
            )
        parsed = self._parse_pooling_response(resp.json(), expected_count=1)
        return parsed[0]

    # ------------------------------------------------------------------
    # MultimodalEmbeddingProvider interface
    # ------------------------------------------------------------------

    async def embed_text_multivector(self, text: str) -> MultiVectorEmbedding:
        vectors_list = await self._pooling_text([text])
        vectors_list = self._apply_pooling(vectors_list)
        return MultiVectorEmbedding(vectors=vectors_list[0])

    async def embed_batch_multivector(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[MultiVectorEmbedding]:
        # vLLM handles batching internally; ``batch_size`` becomes a
        # client-side chunk size to bound request payload + memory growth.
        out: list[MultiVectorEmbedding] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + batch_size]
            vectors_chunk = await self._pooling_text(chunk)
            vectors_chunk = self._apply_pooling(vectors_chunk)
            out.extend(MultiVectorEmbedding(vectors=v) for v in vectors_chunk)
        return out

    async def embed_image_multivector(self, image: str | bytes | Path) -> MultiVectorEmbedding:
        vectors = await self._pooling_image(image)
        pooled = self._apply_pooling([vectors])[0]
        return MultiVectorEmbedding(vectors=pooled)

    async def embed_images_batch_multivector(
        self,
        images: list[str | bytes | Path],
        batch_size: int = 4,
    ) -> list[MultiVectorEmbedding]:
        # vLLM's /pooling messages payload is single-input; loop client-side
        # rather than fabricate a batch endpoint. The vLLM engine still
        # batches internally across concurrent requests.
        del batch_size
        return [await self.embed_image_multivector(img) for img in images]

    # --- Single-vector compatibility shims (mean-pool) ---

    async def embed(self, text: str) -> list[float]:
        mv = await self.embed_text_multivector(text)
        return np.mean(mv.vectors, axis=0).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        mvs = await self.embed_batch_multivector(texts)
        return [np.mean(mv.vectors, axis=0).tolist() for mv in mvs]

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        mv = await self.embed_image_multivector(image)
        return np.mean(mv.vectors, axis=0).tolist()

    async def embed_multimodal(
        self,
        text: str | None = None,
        image: str | bytes | Path | None = None,
    ) -> list[float]:
        if image is not None:
            return await self.embed_image(image)
        if text is not None:
            return await self.embed(text)
        raise ValueError("At least one of text or image must be provided")

    @property
    def dimensions(self) -> int:
        return self._dimensions


class VLLMMultiVectorProviderPlugin(EmbeddingProviderPluginBase):
    """Plugin entry point. Dispatched via ``EMBED_SERVER_MULTI_VECTOR_PROVIDER=vllm``
    in ``_setup_dual_embedding_service``; not auto-selected by the framework's
    provider-name machinery.
    """

    PROVIDER_NAME = PROVIDER_NAME_VLLM_MULTI_VECTOR

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        model_name = v.environ(MEMORYLAYER_EMBEDDING_COLPALI_MODEL, default=DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL)
        if model_name == "ModernVBERT/colmodernvbert":
            # The base ColModernVBert repo is a LoRA adapter — vLLM can't
            # load it. Auto-upgrade to the merged checkpoint so operators
            # don't trip over it on first boot.
            model_name = "ModernVBERT/colmodernvbert-merged"
            logger.info(
                "Auto-upgraded multi-vector model to ModernVBERT/colmodernvbert-merged "
                "(base 'colmodernvbert' is a LoRA adapter and not vLLM-loadable)."
            )

        architectures_raw = v.environ(
            MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES,
            default=DEFAULT_MV_VLLM_ARCHITECTURES,
        )
        architectures = [a.strip() for a in str(architectures_raw).split(",") if a.strip()]

        # Unset → vLLM derives max length from the model's config; ColModernVBert
        # caps at 7999 so the single-vec default of 32768 is wrong here.
        max_model_len_raw = v.environ(MEMORYLAYER_EMBEDDING_VLLM_MV_MAX_LENGTH, default=None)
        max_model_len = int(max_model_len_raw) if max_model_len_raw not in (None, "") else None

        return VLLMMultiVectorProvider(
            v=v,
            model_name=model_name,
            dtype=v.environ(MEMORYLAYER_EMBEDDING_VLLM_DTYPE, default=DEFAULT_DTYPE),
            max_model_len=max_model_len,
            output_dimensions=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_MV_DIMENSIONS,
                default=DEFAULT_MV_VLLM_DIMENSIONS,
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
            architectures=architectures,
            pool_factor=v.environ(
                MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR,
                default=DEFAULT_COLPALI_POOL_FACTOR,
                type_fn=int,
            ),
            host=v.environ(MEMORYLAYER_EMBEDDING_VLLM_MV_HOST, default=DEFAULT_MV_VLLM_HOST),
            port=v.environ(MEMORYLAYER_EMBEDDING_VLLM_MV_PORT, default=DEFAULT_MV_VLLM_PORT, type_fn=int),
            startup_timeout_sec=v.environ(
                MEMORYLAYER_EMBEDDING_VLLM_MV_STARTUP_TIMEOUT_SEC,
                default=DEFAULT_MV_VLLM_STARTUP_TIMEOUT_SEC,
                type_fn=float,
            ),
            cmd=v.environ(MEMORYLAYER_EMBEDDING_VLLM_MV_CMD, default=DEFAULT_MV_VLLM_CMD),
        )
