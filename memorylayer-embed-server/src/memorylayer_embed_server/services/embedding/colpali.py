"""In-process ColPali multimodal embedding provider.

Lives in the embed-server (where ``colpali-engine`` and ``torch`` are
actually installed via the ``[colpali]`` / ``[gpu]`` extras). The OSS
main server consumes this transparently over HTTP via
``EmbedServerEmbeddingProvider`` — no torch in the OSS server.

``MultiVectorEmbedding`` and ``maxsim_score`` deliberately re-import
from ``memorylayer_server.services.embedding._maxsim`` so the wire-side
contract used by ``/v1/score`` and the late-interaction scoring helper
have a single canonical definition.
"""
import asyncio
from logging import Logger
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scitrera_app_framework import Variables, get_extension

from memorylayer_server.config import MEMORYLAYER_EMBEDDING_MODEL
from memorylayer_server.services.embedding._maxsim import MultiVectorEmbedding, maxsim_score
from memorylayer_server.services.embedding.base import (
    EmbeddingProviderPluginBase,
    MultimodalEmbeddingProvider,
)

MEMORYLAYER_EMBEDDING_COLPALI_MODEL = 'MEMORYLAYER_EMBEDDING_COLPALI_MODEL'
MEMORYLAYER_EMBEDDING_DEVICE = 'MEMORYLAYER_EMBEDDING_DEVICE'
MEMORYLAYER_EMBEDDING_REVISION = 'MEMORYLAYER_EMBEDDING_REVISION'
# Max concurrent ColPali GPU requests in flight. Excess requests wait
# in an asyncio.Semaphore queue. Tune up on dedicated GPUs; ~4 is a
# reasonable default for ColModernVBert on a 24GB GPU.
MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT = 'MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT'
DEFAULT_COLPALI_MAX_CONCURRENT = 4
# Max seconds a request may wait for a semaphore slot before being
# rejected. 0 (default) means wait forever — relies on client timeouts
# for upstream backpressure. Set to a positive number (e.g. 30) on
# load-shedding deployments.
MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC = 'MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC'
DEFAULT_COLPALI_QUEUE_TIMEOUT_SEC = 0.0

# Default to ModernVBERT - MIT licensed, smaller, better performance
DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL = "ModernVBERT/colmodernvbert"
DEFAULT_EMBEDDING_DEVICE = None
DEFAULT_EMBEDDING_REVISION = 'main'


class ColPaliQueueTimeoutError(Exception):
    """Raised when a ColPali request waits longer than the configured
    queue-timeout for a GPU concurrency slot. The FastAPI route handler
    converts this into a 503 with a ``Retry-After`` hint.
    """

    def __init__(self, wait_seconds: float, max_concurrent: int) -> None:
        self.wait_seconds = wait_seconds
        self.max_concurrent = max_concurrent
        super().__init__(
            f"ColPali GPU queue saturated: waited {wait_seconds:.2f}s for "
            f"a slot (max_concurrent={max_concurrent}); rejecting request."
        )


class ColPaliEmbeddingProvider(MultimodalEmbeddingProvider):
    """
    Multi-model ColPali-family embedding provider.

    Supports multiple vision-language retrieval models with late interaction
    (multi-vector) approach for document retrieval. Particularly effective
    for documents with visual elements like PDFs.

    Supported Models:
    - ModernVBERT/colmodernvbert (default) - MIT licensed, smaller, excellent performance
    TODO: athrael-soju/colqwen3.5-4.5B-v3 - Qwen3.5 based (Apache 2.0 licensed)
    - vidore/colqwen2.5-* or tsystems/colqwen2.5-* - ColQwen2.5 models with bfloat16
    - vidore/colqwen2-* - ColQwen2 models with bfloat16
    - vidore/colpali-* - Original ColPali models

    Features:
    - Multi-vector embeddings (one per image patch/token)
    - Late interaction scoring (MaxSim)
    - Flash Attention 2 support when available
    - Automatic model/processor selection based on model name
    - Self-hostable
    """

    def __init__(
            self,
            v: Variables = None,
            model_name: str = DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL,
            device: Optional[str] = None,
            revision: str = DEFAULT_EMBEDDING_REVISION,
            output_dimensions: int = 128,
            max_concurrent: int = DEFAULT_COLPALI_MAX_CONCURRENT,
            queue_timeout_sec: float = DEFAULT_COLPALI_QUEUE_TIMEOUT_SEC,
    ):
        super().__init__(v, output_dimensions)  # Default dimension per vector
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.max_concurrent = max(1, int(max_concurrent))
        self.queue_timeout_sec = max(0.0, float(queue_timeout_sec))
        self._v = v
        self._model = None
        self._processor = None
        # Constructed lazily so the Semaphore binds to the running event
        # loop rather than whichever loop happens to exist at import time.
        self._gpu_semaphore: Optional[asyncio.Semaphore] = None
        # In-flight gauge — exposed via /health/load on the embed-server
        # so an LB can route to the least-utilised replica.
        self._in_flight: int = 0
        self.logger.info(
            "Initialized ColPaliEmbeddingProvider: model=%s (revision=%s), "
            "max_concurrent=%d, queue_timeout_sec=%s",
            model_name, revision, self.max_concurrent,
            "off" if self.queue_timeout_sec == 0 else f"{self.queue_timeout_sec:.1f}s",
        )

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._gpu_semaphore is None:
            self._gpu_semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._gpu_semaphore

    def _get_metrics(self):
        """Resolve the active metrics service. Returns ``None`` if unavailable."""
        try:
            from memorylayer_server.services._constants import EXT_METRICS_SERVICE
            return get_extension(EXT_METRICS_SERVICE, self._v)
        except Exception:  # noqa: BLE001 - metrics are best-effort
            return None

    def get_load_snapshot(self) -> dict:
        """Return current load — used by ``GET /health/load`` for LB routing."""
        max_c = self.max_concurrent
        in_flight = self._in_flight
        return {
            "in_flight": in_flight,
            "max_concurrent": max_c,
            "utilization": (in_flight / max_c) if max_c > 0 else 0.0,
        }

    def _gpu_slot(self):
        """Async context manager around the GPU semaphore.

        * Waits up to ``queue_timeout_sec`` for a slot (forever when 0).
        * Logs at debug for every wait; at warning when wait > 100ms so
          contention is visible in production logs.
        * Raises :class:`ColPaliQueueTimeoutError` on timeout.
        * Emits Prometheus/OTel-compatible metrics via the embed-server's
          ``MetricsService`` when one is registered:
            - ``embed_server_colpali_gpu_slot_total{result}`` counter
              (result=acquired|timeout)
            - ``embed_server_colpali_gpu_slot_wait_seconds`` histogram
            - ``embed_server_colpali_gpu_in_flight`` gauge
            - ``embed_server_colpali_gpu_utilization`` gauge (0..1)
        """
        provider = self

        class _Slot:
            async def __aenter__(self_inner):  # noqa: N805 - dunder style
                import time as _time

                sem = provider._get_semaphore()
                metrics = provider._get_metrics()
                start = _time.monotonic()
                timeout = provider.queue_timeout_sec
                try:
                    if timeout > 0:
                        try:
                            await asyncio.wait_for(sem.acquire(), timeout=timeout)
                        except asyncio.TimeoutError:
                            waited = _time.monotonic() - start
                            if metrics is not None:
                                metrics.counter(
                                    "embed_server_colpali_gpu_slot_total",
                                    labels={"result": "timeout"},
                                )
                                metrics.histogram(
                                    "embed_server_colpali_gpu_slot_wait_seconds",
                                    waited,
                                    labels={"result": "timeout"},
                                )
                            raise ColPaliQueueTimeoutError(
                                wait_seconds=waited,
                                max_concurrent=provider.max_concurrent,
                            )
                    else:
                        await sem.acquire()
                    waited = _time.monotonic() - start
                except ColPaliQueueTimeoutError:
                    raise
                # Successful acquire
                provider._in_flight += 1
                if metrics is not None:
                    metrics.counter(
                        "embed_server_colpali_gpu_slot_total",
                        labels={"result": "acquired"},
                    )
                    metrics.histogram(
                        "embed_server_colpali_gpu_slot_wait_seconds",
                        waited,
                        labels={"result": "acquired"},
                    )
                    metrics.gauge(
                        "embed_server_colpali_gpu_in_flight",
                        float(provider._in_flight),
                    )
                    metrics.gauge(
                        "embed_server_colpali_gpu_utilization",
                        float(provider._in_flight) / max(provider.max_concurrent, 1),
                    )
                if waited > 0.1:
                    provider.logger.warning(
                        "ColPali GPU slot acquired after %.3fs wait "
                        "(in_flight=%d/%d); consider scaling out.",
                        waited, provider._in_flight, provider.max_concurrent,
                    )
                else:
                    provider.logger.debug(
                        "ColPali GPU slot acquired in %.4fs (in_flight=%d/%d)",
                        waited, provider._in_flight, provider.max_concurrent,
                    )
                return None

            async def __aexit__(self_inner, exc_type, exc, tb):  # noqa: N805
                provider._get_semaphore().release()
                provider._in_flight = max(0, provider._in_flight - 1)
                metrics = provider._get_metrics()
                if metrics is not None:
                    metrics.gauge(
                        "embed_server_colpali_gpu_in_flight",
                        float(provider._in_flight),
                    )
                    metrics.gauge(
                        "embed_server_colpali_gpu_utilization",
                        float(provider._in_flight) / max(provider.max_concurrent, 1),
                    )
                return False

        return _Slot()

    def _get_model(self):
        """Lazy load the appropriate model based on model name."""
        if self._model is None:
            import torch
            from transformers.utils.import_utils import is_flash_attn_2_available

            self.logger.info("Loading model: %s (revision: %s)", self.model_name, self.revision)

            # Determine device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Check Flash Attention 2 availability
            use_flash_attn = is_flash_attn_2_available()
            if use_flash_attn:
                self.logger.info("Flash Attention 2 is available and will be used")

            model_name_lower = self.model_name.lower()

            try:
                if 'qwen2.5' in model_name_lower:
                    self._load_colqwen2_5(torch, use_flash_attn)
                elif 'qwen2' in model_name_lower:
                    self._load_colqwen2(torch, use_flash_attn)
                elif 'modernvbert' in model_name_lower:
                    self._load_colmodernvbert(torch, use_flash_attn)
                elif 'colpali' in model_name_lower:
                    self._load_colpali(torch)
                else:
                    # Default to ModernVBERT loading pattern for unknown models
                    self.logger.warning(
                        "Unknown model pattern '%s', attempting ModernVBERT loading",
                        self.model_name
                    )
                    self._load_colmodernvbert(torch, use_flash_attn)

            except ImportError as e:
                self.logger.warning(
                    "colpali_engine not installed or missing dependencies. "
                    "Install with: pip install colpali-engine. Error: %s", e
                )
                raise

        return self._model, self._processor

    def _load_colqwen2_5(self, torch, use_flash_attn: bool):
        """Load ColQwen2.5 model and processor."""
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self.logger.info("Loading ColQwen2.5 model with bfloat16")
        self._model = ColQwen2_5.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if use_flash_attn else None,
        ).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

    def _load_colqwen2(self, torch, use_flash_attn: bool):
        """Load ColQwen2 model and processor."""
        from colpali_engine.models import ColQwen2, ColQwen2Processor

        self.logger.info("Loading ColQwen2 model with bfloat16")
        self._model = ColQwen2.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if use_flash_attn else None,
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.model_name)

    def _load_colmodernvbert(self, torch, use_flash_attn: bool):
        """Load ColModernVBert model and processor."""
        from colpali_engine.models import ColModernVBert, ColModernVBertProcessor

        self.logger.info("Loading ColModernVBert model with float32")
        # ModernVBERT requires high precision for float32 matmul
        torch.set_float32_matmul_precision('high')

        self._model = ColModernVBert.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=self.device,
            attn_implementation="flash_attention_2" if use_flash_attn else None,
        ).eval()
        self._processor = ColModernVBertProcessor.from_pretrained(self.model_name)

    def _load_colpali(self, torch):
        """Load original ColPali model and processor."""
        from colpali_engine.models import ColPali
        from colpali_engine.utils.colpali_processing_utils import ColPaliProcessor

        self.logger.info("Loading ColPali model with float16 (CUDA) or float32 (CPU)")
        self._model = ColPali.from_pretrained(
            self.model_name,
            revision=self.revision,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._model.eval()
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)

    async def embed(self, text: str) -> list[float]:
        """
        Generate single-vector embedding for text (averaged from multi-vector).

        For compatibility with standard vector databases.
        """
        multi_vec = await self.embed_text_multivector(text)
        # Average all vectors to get single vector
        return np.mean(multi_vec.vectors, axis=0).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate single-vector embeddings for batch (averaged from multi-vector)."""
        multi_vecs = await self.embed_batch_multivector(texts)
        return [np.mean(mv.vectors, axis=0).tolist() for mv in multi_vecs]

    async def embed_batch_multivector(
            self,
            texts: list[str],
            batch_size: int = 8,
    ) -> list[MultiVectorEmbedding]:
        """
        Generate multi-vector embeddings for multiple texts efficiently.

        Uses DataLoader for batched processing to improve throughput.

        Args:
            texts: List of text queries to embed
            batch_size: Number of texts to process per batch

        Returns:
            List of MultiVectorEmbedding, one per input text
        """
        import torch
        from torch.utils.data import DataLoader

        self.logger.debug("Batch embedding %d texts with batch_size=%d", len(texts), batch_size)
        async with self._gpu_slot():
            model, processor = self._get_model()

            dataloader = DataLoader(
                texts,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch_texts: processor.process_queries(batch_texts).to(self.device),
            )

            def process_batch(batch_inputs):
                with torch.no_grad():
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                    embeddings = model(**batch_inputs)
                    return list(torch.unbind(embeddings.to('cpu')))

            all_results = []
            for batch_inputs in dataloader:
                batch_tensors = await asyncio.to_thread(process_batch, batch_inputs)
                for tensor in batch_tensors:
                    vectors = tensor.numpy().tolist()
                    all_results.append(MultiVectorEmbedding(vectors=vectors))
        return all_results

    async def embed_images_batch_multivector(
            self,
            images: list[Union[str, bytes, Path]],
            batch_size: int = 4,
    ) -> list[MultiVectorEmbedding]:
        """
        Generate multi-vector embeddings for multiple images efficiently.

        Uses DataLoader for batched processing to improve throughput.

        Args:
            images: List of image paths, bytes, or URLs to embed
            batch_size: Number of images to process per batch

        Returns:
            List of MultiVectorEmbedding, one per input image
        """
        import torch
        from torch.utils.data import DataLoader
        from PIL import Image
        import io

        self.logger.debug("Batch embedding %d images with batch_size=%d", len(images), batch_size)
        # Image decode happens before we grab the GPU slot — keeps the
        # GPU critical section tight.
        pil_images = []
        for img in images:
            image_bytes = self.load_image_bytes(img)
            pil_images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        async with self._gpu_slot():
            model, processor = self._get_model()

            dataloader = DataLoader(
                pil_images,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch_imgs: processor.process_images(batch_imgs).to(self.device),
            )

            def process_batch(batch_inputs):
                with torch.no_grad():
                    batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
                    embeddings = model(**batch_inputs)
                    return list(torch.unbind(embeddings.to('cpu')))

            all_results = []
            for batch_inputs in dataloader:
                batch_tensors = await asyncio.to_thread(process_batch, batch_inputs)
                for tensor in batch_tensors:
                    vectors = tensor.numpy().tolist()
                    all_results.append(MultiVectorEmbedding(vectors=vectors))
        return all_results

    async def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
        """Generate single-vector embedding for image (averaged)."""
        multi_vec = await self.embed_image_multivector(image)
        return np.mean(multi_vec.vectors, axis=0).tolist()

    async def embed_multimodal(
            self,
            text: Optional[str] = None,
            image: Optional[Union[str, bytes, Path]] = None
    ) -> list[float]:
        """Generate single-vector embedding for multimodal content."""
        if image:
            return await self.embed_image(image)
        elif text:
            return await self.embed(text)
        raise ValueError("At least one of text or image must be provided")

    async def embed_text_multivector(self, text: str) -> MultiVectorEmbedding:
        """Generate multi-vector embedding for text (native ColPali format)."""
        self.logger.debug("Generating multi-vector for text: %s chars", len(text))
        import torch

        async with self._gpu_slot():
            model, processor = self._get_model()
            inputs = processor.process_queries([text]).to(self.device)
            with torch.no_grad():
                embeddings = model(**inputs)
            vectors = embeddings[0].cpu().numpy().tolist()
        return MultiVectorEmbedding(vectors=vectors)

    async def embed_image_multivector(
            self,
            image: Union[str, bytes, Path]
    ) -> MultiVectorEmbedding:
        """Generate multi-vector embedding for image (native ColPali format)."""
        self.logger.debug("Generating multi-vector for image")
        import torch
        from PIL import Image
        import io

        # Image decode (CPU) before grabbing the GPU slot.
        image_bytes = self.load_image_bytes(image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        async with self._gpu_slot():
            return await self._embed_image_multivector_locked(pil_image)

    async def _embed_image_multivector_locked(self, pil_image) -> MultiVectorEmbedding:
        """GPU-bound body for embed_image_multivector; called inside the slot."""
        import torch

        model, processor = self._get_model()
        inputs = processor.process_images([pil_image]).to(self.device)
        with torch.no_grad():
            embeddings = model(**inputs)
        vectors = embeddings[0].cpu().numpy().tolist()
        return MultiVectorEmbedding(vectors=vectors)

    @staticmethod
    def maxsim_score(
            query_vectors: MultiVectorEmbedding,
            doc_vectors: MultiVectorEmbedding,
    ) -> float:
        """Calculate MaxSim score (delegates to canonical numpy helper)."""
        return maxsim_score(query_vectors, doc_vectors)

    @property
    def dimensions(self) -> int:
        return self._dimensions


class ColPaliEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    """ColPali embedding provider plugin supporting multiple model families."""
    PROVIDER_NAME: str = "colpali"

    def initialize(self, v: Variables, logger: Logger) -> MultimodalEmbeddingProvider:
        # Provider-specific override (MEMORYLAYER_EMBEDDING_COLPALI_MODEL)
        # wins over the shared MEMORYLAYER_EMBEDDING_MODEL — necessary
        # when ColPali runs alongside another single-vector provider
        # (vLLM / OpenAI / Google) that wants a different model.
        model = v.environ(MEMORYLAYER_EMBEDDING_COLPALI_MODEL, default=None)
        if not model:
            model = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_MEMORYLAYER_COLPALI_EMBEDDING_MODEL)
        device = v.environ(MEMORYLAYER_EMBEDDING_DEVICE, default=DEFAULT_EMBEDDING_DEVICE)
        revision = v.environ(MEMORYLAYER_EMBEDDING_REVISION, default=DEFAULT_EMBEDDING_REVISION)
        max_concurrent = v.environ(
            MEMORYLAYER_EMBEDDING_COLPALI_MAX_CONCURRENT,
            default=DEFAULT_COLPALI_MAX_CONCURRENT,
            type_fn=int,
        )
        queue_timeout_sec = v.environ(
            MEMORYLAYER_EMBEDDING_COLPALI_QUEUE_TIMEOUT_SEC,
            default=DEFAULT_COLPALI_QUEUE_TIMEOUT_SEC,
            type_fn=float,
        )
        return ColPaliEmbeddingProvider(
            v=v,
            model_name=model,
            device=device,
            revision=revision,
            max_concurrent=max_concurrent,
            queue_timeout_sec=queue_timeout_sec,
        )
