"""Dual embedding service managing single-vector and multi-vector providers."""

from logging import Logger
from pathlib import Path

from memorylayer_server.services.embedding.base import (
    EmbeddingProvider,
    MultimodalEmbeddingProvider,
)
from scitrera_app_framework import Variables, get_logger

# Extension point names for the dual embedding service
EXT_SINGLE_VECTOR_PROVIDER = "embed-server-single-vector-provider"
EXT_MULTI_VECTOR_PROVIDER = "embed-server-multi-vector-provider"
EXT_DUAL_EMBEDDING_SERVICE = "embed-server-dual-embedding-service"


class DualEmbeddingService:
    """
    Manages two simultaneous embedding providers:
    - Single-vector (vLLM): Standard embeddings for vector DB storage
    - Multi-vector (ColPali): Late interaction embeddings for document retrieval

    Unlike upstream's single-provider pattern, this service explicitly
    wires and exposes both providers for different use cases.
    """

    def __init__(
        self,
        v: Variables = None,
        single_vector_provider: EmbeddingProvider = None,
        multi_vector_provider: MultimodalEmbeddingProvider = None,
    ):
        self.logger: Logger = get_logger(v, name=self.__class__.__name__)
        self._single_vector = single_vector_provider
        self._multi_vector = multi_vector_provider
        self._v = v

        self.logger.info(
            "Initialized DualEmbeddingService: single=%s, multi=%s",
            type(single_vector_provider).__name__ if single_vector_provider else "None",
            type(multi_vector_provider).__name__ if multi_vector_provider else "None",
        )

    @property
    def single_vector(self) -> EmbeddingProvider:
        """Get the single-vector embedding provider."""
        if self._single_vector is None:
            raise RuntimeError("Single-vector embedding provider not configured")
        return self._single_vector

    @property
    def multi_vector(self) -> MultimodalEmbeddingProvider:
        """Get the multi-vector embedding provider."""
        if self._multi_vector is None:
            raise RuntimeError("Multi-vector embedding provider not configured")
        return self._multi_vector

    @property
    def has_single_vector(self) -> bool:
        return self._single_vector is not None

    @property
    def has_multi_vector(self) -> bool:
        return self._multi_vector is not None

    async def preload(self):
        """Preload both embedding providers."""
        if self._single_vector:
            self.logger.info("Preloading single-vector provider")
            await self._single_vector.preload()

        if self._multi_vector:
            self.logger.info("Preloading multi-vector provider")
            await self._multi_vector.preload()

    # --- Single-Vector Operations ---

    async def embed(self, text: str) -> list[float]:
        """Generate single-vector embedding for text."""
        return await self.single_vector.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate single-vector embeddings for batch."""
        return await self.single_vector.embed_batch(texts)

    async def embed_image_single(self, image: str | bytes | Path) -> list[float]:
        """Generate single-vector embedding for image."""
        if isinstance(self._single_vector, MultimodalEmbeddingProvider):
            return await self._single_vector.embed_image(image)
        raise RuntimeError("Single-vector provider does not support image embedding")

    @property
    def single_vector_dimensions(self) -> int:
        return self.single_vector.dimensions

    @property
    def single_vector_model_name(self) -> str:
        """Get the model name from the single-vector provider."""
        return getattr(self._single_vector, "model_name", "unknown")

    # --- Multi-Vector Operations ---

    async def embed_text_multivector(self, text: str):
        """Generate multi-vector embedding for text."""
        provider = self.multi_vector
        if hasattr(provider, "embed_text_multivector"):
            return await provider.embed_text_multivector(text)
        raise RuntimeError("Multi-vector provider does not support text multivector embedding")

    async def embed_batch_multivector(self, texts: list[str], batch_size: int = 8):
        """Generate multi-vector embeddings for batch of texts."""
        provider = self.multi_vector
        if hasattr(provider, "embed_batch_multivector"):
            return await provider.embed_batch_multivector(texts, batch_size=batch_size)
        raise RuntimeError("Multi-vector provider does not support batch multivector embedding")

    async def embed_image_multivector(self, image: str | bytes | Path):
        """Generate multi-vector embedding for image."""
        provider = self.multi_vector
        if hasattr(provider, "embed_image_multivector"):
            return await provider.embed_image_multivector(image)
        raise RuntimeError("Multi-vector provider does not support image multivector embedding")

    async def embed_images_batch_multivector(
        self,
        images: list[str | bytes | Path],
        batch_size: int = 4,
    ):
        """Generate multi-vector embeddings for batch of images."""
        provider = self.multi_vector
        if hasattr(provider, "embed_images_batch_multivector"):
            return await provider.embed_images_batch_multivector(images, batch_size=batch_size)
        raise RuntimeError("Multi-vector provider does not support batch image multivector embedding")

    @property
    def multi_vector_dimensions(self) -> int:
        return self.multi_vector.dimensions

    @property
    def multi_vector_model_name(self) -> str:
        """Get the model name from the multi-vector provider."""
        return getattr(self._multi_vector, "model_name", "unknown")

    # --- MaxSim Scoring ---

    @staticmethod
    def maxsim_score(query_vectors, doc_vectors) -> float:
        """Calculate MaxSim score between query and document multi-vectors."""
        from memorylayer_server.services.embedding._maxsim import maxsim_score

        return maxsim_score(query_vectors, doc_vectors)
