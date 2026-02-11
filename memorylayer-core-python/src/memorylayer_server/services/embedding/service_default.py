import hashlib
from logging import Logger

from pathlib import Path
from typing import Any, Optional, Union, Iterable

from scitrera_app_framework import get_logger, Variables as Variables

from .base import (
    EmbeddingProvider, MultimodalEmbeddingProvider,
    EmbeddingInput, EmbeddingType,
    EmbeddingServicePluginBase, EXT_EMBEDDING_PROVIDER,
)
from ..cache import EXT_CACHE_SERVICE
from ...utils import cosine_similarity as _cosine_similarity


class EmbeddingService:
    """
    Embedding service that wraps providers and adds caching.

    Supports text embeddings with optional multimodal content
    when a multimodal provider is configured.
    """

    def __init__(self, v: Variables = None, provider: EmbeddingProvider = None, cache: Optional[Any] = None):
        self.provider = provider
        self.cache = cache
        self.logger = get_logger(v, name=self.__class__.__name__)
        self._is_multimodal = isinstance(provider, MultimodalEmbeddingProvider)

        self.logger.info(
            "Initialized EmbeddingService with provider: %s, dimensions: %s, multimodal: %s",
            provider.__class__.__name__,
            provider.dimensions,
            self._is_multimodal
        )

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
        embedding = await self.provider.embed(text)

        # Cache result
        if self.cache:
            await self.cache.set(cache_key, embedding, ttl_seconds=3600)  # 1 hour TTL
            self.logger.debug("Cached embedding: %s", cache_key)

        return embedding

    async def embed_image(self, image: Union[str, bytes, Path]) -> list[float]:
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
        return await provider.embed_image(image)

    async def embed_multimodal(
            self,
            text: Optional[str] = None,
            image: Optional[Union[str, bytes, Path]] = None
    ) -> list[float]:
        """
        Generate embedding for combined text and image.

        Requires a multimodal provider.
        """
        if not self._is_multimodal:
            if image:
                raise ValueError(
                    f"Provider {self.provider.__class__.__name__} does not support image embeddings."
                )
            return await self.embed(text)

        provider: MultimodalEmbeddingProvider = self.provider
        return await provider.embed_multimodal(text, image)

    async def embed_input(self, input: EmbeddingInput) -> list[float]:
        """Generate embedding for EmbeddingInput (convenience method)."""
        if input.embedding_type == EmbeddingType.TEXT:
            return await self.embed(input.text)
        elif self._is_multimodal:
            provider: MultimodalEmbeddingProvider = self.provider
            return await provider.embed_input(input)
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

        return await self.provider.embed_batch(valid_texts)

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return _cosine_similarity(a, b)


class EmbeddingServicePlugin(EmbeddingServicePluginBase):
    """Default plugin for embedding service."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        cache_service = self.get_extension(EXT_CACHE_SERVICE, v)
        embedding_provider: EmbeddingProvider = self.get_extension(EXT_EMBEDDING_PROVIDER, v)
        return EmbeddingService(
            v=v,
            provider=embedding_provider,
            cache=cache_service
        )
