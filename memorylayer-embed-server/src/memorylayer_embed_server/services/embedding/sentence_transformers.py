"""Single-vector embedding provider backed by sentence-transformers.

This is the local/CPU-friendly single-vector backend and the embed server's
default. Unlike the vLLM providers it needs no GPU, no CUDA toolchain, and no
model server: it loads a small model in-process and encodes on the CPU.

The default model, ``sentence-transformers/all-MiniLM-L6-v2``, is ~90 MB and
produces **384-dimensional** vectors. That dimension is a durable property of a
deployment's stored data — see ``MEMORYLAYER_EMBEDDING_DIMENSIONS`` and the
"Embedding dimensions" section of the README before changing the model.

Requires the ``local`` extra::

    pip install "memorylayer-embed-server[local]"
"""

from __future__ import annotations

import asyncio
from logging import Logger

from memorylayer_server.config import MEMORYLAYER_EMBEDDING_DIMENSIONS, MEMORYLAYER_EMBEDDING_MODEL
from memorylayer_server.services.embedding.base import EmbeddingProvider, EmbeddingProviderPluginBase
from scitrera_app_framework import Variables as Variables

PROVIDER_NAME_SENTENCE_TRANSFORMERS = "sentence_transformers"

MEMORYLAYER_EMBEDDING_ST_DEVICE = "MEMORYLAYER_EMBEDDING_ST_DEVICE"
MEMORYLAYER_EMBEDDING_ST_BATCH_SIZE = "MEMORYLAYER_EMBEDDING_ST_BATCH_SIZE"

# all-MiniLM-L6-v2: 6-layer MiniLM, ~90 MB, 384-d, 256-token window. The
# standard "small and good enough" sentence embedder — fast on CPU, which is
# what makes it usable as a default.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSIONS = 384
# None lets sentence-transformers pick (CUDA when present, else CPU). Pin to
# "cpu" to keep the GPU free for the multi-vector/ColPali provider.
DEFAULT_DEVICE = None
DEFAULT_BATCH_SIZE = 32


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """In-process sentence-transformers embeddings (CPU-friendly)."""

    def __init__(
        self,
        v: Variables = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        output_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        device: str | None = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        super().__init__(v, output_dimensions=output_dimensions)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._model_lock = asyncio.Lock()
        self.logger.info(
            "Initialized SentenceTransformersEmbeddingProvider (model=%s, dimensions=%d, device=%s)",
            model_name,
            output_dimensions,
            device or "auto",
        )

    async def _get_model(self):
        """Load the model once, off the event loop.

        First call downloads the weights if they are not already in the HF
        cache, so this can take a while; ``preload()`` exists to move that cost
        to startup instead of the first request.
        """
        if self._model is not None:
            return self._model
        async with self._model_lock:
            if self._model is not None:  # another coroutine won the race
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "The 'sentence_transformers' single-vector provider requires the `local` extra: "
                    'pip install "memorylayer-embed-server[local]". '
                    "Alternatively set MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER to 'openai', 'google', "
                    "'vllm_subprocess', or 'mock'."
                ) from e

            self.logger.info("Loading sentence-transformers model: %s", self.model_name)
            model = await asyncio.to_thread(SentenceTransformer, self.model_name, device=self.device)

            # Trust the model over configuration: a dimension mismatch here means
            # every vector written would be the wrong width, which is exactly the
            # failure that is painful to undo once data exists.
            #
            # sentence-transformers 5.x renamed get_sentence_embedding_dimension()
            # to get_embedding_dimension(); prefer the new name and fall back so
            # both major versions work.
            size_fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            actual = int(size_fn())
            if self._dimensions and actual != self._dimensions:
                self.logger.warning(
                    "Model %s produces %d-dimensional vectors but MEMORYLAYER_EMBEDDING_DIMENSIONS is %d. "
                    "Using the model's %d. Set MEMORYLAYER_EMBEDDING_DIMENSIONS=%d to make this explicit, and note "
                    "that memories already stored at a different dimension will not be searchable alongside these.",
                    self.model_name,
                    actual,
                    self._dimensions,
                    actual,
                    actual,
                )
            self._dimensions = actual
            self._model = model
            self.logger.info(
                "Loaded sentence-transformers model %s (dimensions=%d, device=%s)",
                self.model_name,
                actual,
                getattr(model, "device", self.device or "auto"),
            )
            return self._model

    async def preload(self):
        """Load (and download, if needed) the model at startup."""
        await self._get_model()

    async def embed(self, text: str) -> list[float]:
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._get_model()

        def _encode() -> list[list[float]]:
            # normalize_embeddings=True makes the vectors unit-length so a dot
            # product is cosine similarity, which is what recall() assumes.
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in vectors]

        # encode() is synchronous and CPU-bound; running it inline would stall
        # the event loop (and the liveness probe) for the whole batch.
        return await asyncio.to_thread(_encode)


class SentenceTransformersEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = PROVIDER_NAME_SENTENCE_TRANSFORMERS

    def initialize(self, v: Variables, logger: Logger) -> SentenceTransformersEmbeddingProvider:
        device = v.environ(MEMORYLAYER_EMBEDDING_ST_DEVICE, default=DEFAULT_DEVICE)
        return SentenceTransformersEmbeddingProvider(
            v=v,
            model_name=v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL),
            output_dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
            device=device.strip() or None if isinstance(device, str) else device,
            batch_size=v.environ(MEMORYLAYER_EMBEDDING_ST_BATCH_SIZE, default=DEFAULT_BATCH_SIZE, type_fn=int),
        )
