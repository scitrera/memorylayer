import asyncio
import uuid
from logging import Logger
from pathlib import Path
from typing import Optional, Any, Union

from scitrera_app_framework import Variables as Variables, get_logger

from memorylayer_server.config import MEMORYLAYER_EMBEDDING_MODEL, MEMORYLAYER_EMBEDDING_DIMENSIONS
from memorylayer_server.services.embedding.base import MultimodalEmbeddingProvider, EmbeddingProviderPluginBase

# Provider name as string (enum lives in OSS; enterprise uses string directly)
PROVIDER_NAME_VLLM = "vllm"

MEMORYLAYER_EMBEDDING_VLLM_MODEL = 'MEMORYLAYER_EMBEDDING_VLLM_MODEL'
MEMORYLAYER_EMBEDDING_VLLM_DTYPE = 'MEMORYLAYER_EMBEDDING_VLLM_DTYPE'
MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH = 'MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH'
MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL = 'MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL'
MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER = 'MEMORYLAYER_EMBEDDING_VLLM_ENFORCE_EAGER'

DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-VL-Embedding-2B'
DEFAULT_EMBEDDING_DIMENSIONS = 2048
DEFAULT_DTYPE = 'bfloat16'
DEFAULT_MAX_LENGTH = 32768
# vLLM's stock default is 0.92 which is catastrophic on unified-memory
# systems (DGX Spark, Jetson, integrated GPUs) and on shared GPUs where
# ColPali also lives. Default conservative; operators bump up on dedicated
# inference boxes via MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.25
# Default off — production wants torch.compile + CUDA graph capture for
# throughput. Flip to true for runtime images that don't ship nvcc (CUDA
# *-runtime-* base) or to slash cold-start time at the cost of latency.
DEFAULT_ENFORCE_EAGER = False


class VLLMEmbeddingProvider(MultimodalEmbeddingProvider):
    """
    vLLM-based embedding provider for high-performance inference.

    Uses vLLM's pooling runner for efficient embedding generation.
    Supports any model compatible with vLLM's embedding mode.

    Features:
    - High throughput with batching
    - GPU optimized
    - Supports multimodal models
    """

    def __init__(
            self,
            v: Variables = None,
            model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
            dtype: str = "bfloat16",
            max_model_len: int = 32768,
            output_dimensions: int = 2048,
            gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
            enforce_eager: bool = DEFAULT_ENFORCE_EAGER,
    ):
        super().__init__(v, output_dimensions=output_dimensions)
        self.model_name = model_name
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.output_dimensions = output_dimensions
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self._engine = None
        self._engine_lock = asyncio.Lock()
        self._dimensions = output_dimensions
        self.logger.info(
            "Initialized VLLMEmbeddingProvider with model: %s, dtype: %s, "
            "gpu_memory_utilization: %.2f, enforce_eager: %s",
            model_name, dtype, gpu_memory_utilization, enforce_eager,
        )

    async def _get_engine(self):
        """Lazy-load the async vLLM engine (vLLM v1 ``AsyncLLM``).

        Using ``AsyncLLM`` so request streams cooperate with the FastAPI
        event loop instead of blocking it with synchronous ``LLM.embed``.

        vLLM 0.6+ replaced ``task="embed"`` with ``runner="pooling"`` (plus
        optional ``convert="embed"``). We pass the modern kwargs and fall
        back to the legacy ``task`` kwarg for older installs.
        """
        if self._engine is not None:
            return self._engine

        async with self._engine_lock:
            if self._engine is not None:  # second waiter
                return self._engine

            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm.engine.arg_utils import AsyncEngineArgs

            self.logger.info(
                "Loading vLLM AsyncLLM engine: model=%s, gpu_mem_util=%.2f",
                self.model_name, self.gpu_memory_utilization,
            )

            common = dict(
                model=self.model_name,
                dtype=self.dtype,
                max_model_len=self.max_model_len,
                trust_remote_code=True,
                gpu_memory_utilization=self.gpu_memory_utilization,
                enforce_eager=self.enforce_eager,
            )
            try:
                args = AsyncEngineArgs(runner="pooling", convert="embed", **common)
            except TypeError:
                # Legacy vLLM (<0.6) — keep it working for older installs.
                args = AsyncEngineArgs(task="embed", **common)

            # ``from_engine_args`` is sync (spawns engine processes) but
            # offload to a thread so we don't pin the FastAPI loop while
            # the model loads.
            self._engine = await asyncio.to_thread(AsyncLLM.from_engine_args, args)
            self.logger.info("vLLM AsyncLLM engine loaded successfully")
            return self._engine

    async def _encode_one(self, prompt) -> list[float]:
        """Run a single ``AsyncLLM.encode`` stream and return the embedding."""
        from vllm import PoolingParams

        engine = await self._get_engine()
        # task="embed" selects sentence-level pooling. Without it,
        # PoolingParams.task=None yields per-token output (shape
        # [seq_len, hidden_dim]) which would not fit our flat
        # ``EmbeddingData.embedding: list[float]`` response model.
        params = PoolingParams(task="embed")
        final_output = None
        async for output in engine.encode(
            prompt, params, request_id=str(uuid.uuid4()),
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError("vLLM encode returned no output")

        # PoolingRequestOutput shape varies slightly across vLLM versions:
        # output.outputs may be a list or a single PoolingOutput.
        out = final_output.outputs
        if isinstance(out, list):
            out = out[0]
        # The pooled embedding tensor lives at .data on PoolingOutput.
        data = getattr(out, "data", None)
        if data is None:
            data = getattr(out, "embedding", None)
        if data is None:
            raise RuntimeError(f"Unexpected vLLM pooling output shape: {out!r}")
        embedding = data.tolist() if hasattr(data, "tolist") else list(data)
        if len(embedding) > self.output_dimensions:
            embedding = embedding[: self.output_dimensions]
        return embedding

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text using vLLM (async)."""
        self.logger.debug("Generating vLLM embedding for text: %s chars", len(text))
        return await self._encode_one(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch concurrently.

        ``AsyncLLM`` batches internally via continuous batching, so issuing
        the per-text requests in parallel is the recommended pattern.
        """
        self.logger.debug("Generating vLLM embeddings for batch of %s texts", len(texts))
        return await asyncio.gather(*[self._encode_one(t) for t in texts])

    def _build_multimodal_prompt(
            self,
            text: Optional[str],
            image: Optional[Union[str, bytes, Path]],
    ) -> dict:
        """Build the vLLM multimodal prompt dict for an image (+optional text)."""
        from PIL import Image
        import io

        image_bytes = self.load_image_bytes(image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return {
            "prompt": text or "",
            "multi_modal_data": {"image": pil_image},
        }

    async def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Image embedding via vLLM multimodal input (Qwen3-VL-Embedding-2B).

        Reuses the same multimodal prompt-dict shape as the legacy sync
        ``LLM.embed`` ({"prompt": "", "multi_modal_data": {"image": pil}})
        — vLLM accepts it as a ``MultiModalInput`` for ``AsyncLLM.encode``.
        """
        self.logger.debug("Generating vLLM embedding for image")
        prompt = self._build_multimodal_prompt(text=None, image=image)
        return await self._encode_one(prompt)

    async def embed_multimodal(
            self,
            text: Optional[str] = None,
            image: Optional[Union[str, bytes, Path]] = None,
    ) -> list[float]:
        """Combined text + image embedding via vLLM."""
        if image is None and text is None:
            raise ValueError("At least one of text or image must be provided")
        if image is None:
            return await self.embed(text)
        self.logger.debug(
            "Generating vLLM multimodal embedding (text=%s chars, image)",
            len(text) if text else 0,
        )
        prompt = self._build_multimodal_prompt(text=text, image=image)
        return await self._encode_one(prompt)

    @property
    def dimensions(self) -> int:
        return self._dimensions


class VLLMEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = PROVIDER_NAME_VLLM

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        # Provider-specific override (MEMORYLAYER_EMBEDDING_VLLM_MODEL) wins
        # over the shared MEMORYLAYER_EMBEDDING_MODEL — necessary when
        # vLLM runs alongside another embedding provider (e.g. ColPali for
        # multi-vector) that wants a different model name.
        model_name = v.environ(MEMORYLAYER_EMBEDDING_VLLM_MODEL, default=None)
        if not model_name:
            model_name = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL)
        return VLLMEmbeddingProvider(
            v=v,
            model_name=model_name,
            dtype=v.environ(MEMORYLAYER_EMBEDDING_VLLM_DTYPE, default=DEFAULT_DTYPE),
            max_model_len=v.environ(MEMORYLAYER_EMBEDDING_VLLM_MAX_LENGTH, default=DEFAULT_MAX_LENGTH, type_fn=int),
            output_dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
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
        )
