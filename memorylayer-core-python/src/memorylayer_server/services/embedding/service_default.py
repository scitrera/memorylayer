import hashlib
import time
from logging import Logger
from pathlib import Path
from typing import Any

from scitrera_app_framework import Variables as Variables
from scitrera_app_framework import get_extension, get_logger

from ...utils import cosine_similarity as _cosine_similarity
from .._constants import EXT_METRICS_SERVICE
from ..cache import EXT_CACHE_SERVICE
from .base import (
    EXT_EMBEDDING_PROVIDER,
    EmbeddingInput,
    EmbeddingProvider,
    EmbeddingServicePluginBase,
    EmbeddingType,
    MultimodalEmbeddingProvider,
)


class EmbeddingService:
    """
    Embedding service that wraps providers and adds caching.

    Supports text embeddings with optional multimodal content
    when a multimodal provider is configured.
    """

    def __init__(self, v: Variables = None, provider: EmbeddingProvider = None, cache: Any | None = None):
        self.provider = provider
        self.cache = cache
        self.logger = get_logger(v, name=self.__class__.__name__)
        self._is_multimodal = isinstance(provider, MultimodalEmbeddingProvider)
        try:
            self.metrics = get_extension(EXT_METRICS_SERVICE, v) if v is not None else None
        except Exception:
            self.metrics = None

        self.logger.info(
            "Initialized EmbeddingService with provider: %s, dimensions: %s, multimodal: %s",
            provider.__class__.__name__,
            provider.dimensions,
            self._is_multimodal,
        )

    def _record_model_call(
        self,
        modality: str,
        outcome: str,
        item_count: int,
        elapsed_seconds: float,
    ) -> None:
        """Record provider work separately from generative model calls."""
        if self.metrics is None:
            return
        labels = {"modality": modality, "outcome": outcome}
        try:
            self.metrics.counter("memorylayer_embedding_calls_total", labels=labels)
            self.metrics.counter(
                "memorylayer_embedding_inputs_total",
                item_count,
                labels=labels,
            )
            self.metrics.histogram(
                "memorylayer_embedding_latency_seconds",
                elapsed_seconds,
                labels=labels,
            )
        except Exception:
            self.logger.debug("Embedding metric emission failed", exc_info=True)

    async def _provider_call(self, awaitable, *, modality: str, item_count: int):
        started = time.monotonic()
        try:
            result = await awaitable
        except Exception:
            self._record_model_call(
                modality,
                "failed",
                item_count,
                time.monotonic() - started,
            )
            raise
        self._record_model_call(
            modality,
            "completed",
            item_count,
            time.monotonic() - started,
        )
        return result

    @property
    def is_multimodal(self) -> bool:
        """Whether this service supports multimodal (text + image) embeddings."""
        return self._is_multimodal

    async def embed(self, text: str) -> list[float]:
        """Generate embedding with optional caching."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Check cache first
        if self.cache:
            cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
            cached = await self.cache.get(cache_key)
            if cached:
                self.logger.debug("Cache hit for embedding: %s", cache_key)
                return cached

        # Generate embedding
        embedding = await self._provider_call(
            self.provider.embed(text),
            modality="text",
            item_count=1,
        )

        # Cache result
        if self.cache:
            await self.cache.set(cache_key, embedding, ttl_seconds=3600)  # 1 hour TTL
            self.logger.debug("Cached embedding: %s", cache_key)

        return embedding

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        """
        Generate embedding for an image.

        Requires a multimodal provider.
        """
        if not self._is_multimodal:
            raise ValueError(
                f"Provider {self.provider.__class__.__name__} does not support image embeddings. "
                "Use a multimodal embedding provider to enable image support."
            )

        provider: MultimodalEmbeddingProvider = self.provider
        return await self._provider_call(
            provider.embed_image(image),
            modality="image",
            item_count=1,
        )

    async def embed_multimodal(self, text: str | None = None, image: str | bytes | Path | None = None) -> list[float]:
        """
        Generate embedding for combined text and image.

        Requires a multimodal provider.
        """
        if not self._is_multimodal:
            if image:
                raise ValueError(f"Provider {self.provider.__class__.__name__} does not support image embeddings.")
            return await self.embed(text)

        provider: MultimodalEmbeddingProvider = self.provider
        return await self._provider_call(
            provider.embed_multimodal(text, image),
            modality="multimodal",
            item_count=1,
        )

    async def embed_input(self, input: EmbeddingInput) -> list[float]:
        """Generate embedding for EmbeddingInput (convenience method)."""
        if input.embedding_type == EmbeddingType.TEXT:
            return await self.embed(input.text)
        elif self._is_multimodal:
            provider: MultimodalEmbeddingProvider = self.provider
            return await self._provider_call(
                provider.embed_input(input),
                modality=input.embedding_type.value,
                item_count=1,
            )
        else:
            raise ValueError("Multimodal input requires a multimodal provider")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch (more efficient)."""
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        return await self._provider_call(
            self.provider.embed_batch(valid_texts),
            modality="text_batch",
            item_count=len(valid_texts),
        )

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return _cosine_similarity(a, b)


class EmbeddingServicePlugin(EmbeddingServicePluginBase):
    """Default plugin for embedding service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        cache_service = self.get_extension(EXT_CACHE_SERVICE, v)
        embedding_provider: EmbeddingProvider = self.get_extension(EXT_EMBEDDING_PROVIDER, v)
        return EmbeddingService(v=v, provider=embedding_provider, cache=cache_service)
