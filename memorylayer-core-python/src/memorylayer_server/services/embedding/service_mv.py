from logging import Logger
from pathlib import Path
from typing import Any, Optional, Union

from scitrera_app_framework import Variables

from memorylayer_server.services.cache import EXT_CACHE_SERVICE
from memorylayer_server.services.embedding import EXT_EMBEDDING_PROVIDER
from memorylayer_server.services.embedding.base import (
    EmbeddingProvider,
    EmbeddingServicePluginBase,
    MultimodalEmbeddingProvider,
)
from memorylayer_server.services.embedding.service_default import EmbeddingService

from ._maxsim import MultiVectorEmbedding, maxsim_score


class EmbeddingServiceMV(EmbeddingService):

    def __init__(self, v: Variables = None, provider: EmbeddingProvider = None, cache: Optional[Any] = None):
        super().__init__(v, provider, cache)
        # Multi-vector capability is duck-typed: any provider that exposes
        # the ColPali-style multivector method set is considered capable.
        # The canonical multivector-capable provider today is
        # EmbedServerEmbeddingProvider (delegates to memorylayer-embed-server).
        self._is_multivector = isinstance(provider, MultimodalEmbeddingProvider) and hasattr(
            provider, "embed_text_multivector"
        )

    @property
    def is_multivector(self) -> bool:
        """Whether this service supports multi-vector embeddings (ColPali-style)."""
        return self._is_multivector

    async def embed_multivector(self, text: str) -> MultiVectorEmbedding:
        """Generate multi-vector embedding (multivector-capable provider only)."""
        if not self._is_multivector:
            raise ValueError(
                f"Provider {self.provider.__class__.__name__} does not support multi-vector embeddings. "
                "Configure MEMORYLAYER_EMBEDDING_PROVIDER=embed_server."
            )
        return await self.provider.embed_text_multivector(text)

    async def embed_image_multivector(
        self,
        image: Union[str, bytes, Path],
    ) -> MultiVectorEmbedding:
        """Generate multi-vector embedding for image (multivector-capable provider only)."""
        if not self._is_multivector:
            raise ValueError("Multi-vector embeddings require a multivector-capable provider")
        return await self.provider.embed_image_multivector(image)

    async def embed_batch_multivector(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[MultiVectorEmbedding]:
        """Generate multi-vector embeddings for multiple texts efficiently."""
        if not self._is_multivector:
            raise ValueError("Multi-vector embeddings require a multivector-capable provider")
        return await self.provider.embed_batch_multivector(texts, batch_size=batch_size)

    async def embed_images_batch_multivector(
        self,
        images: list[Union[str, bytes, Path]],
        batch_size: int = 4,
    ) -> list[MultiVectorEmbedding]:
        """Generate multi-vector embeddings for multiple images efficiently."""
        if not self._is_multivector:
            raise ValueError("Multi-vector embeddings require a multivector-capable provider")
        return await self.provider.embed_images_batch_multivector(images, batch_size=batch_size)

    @staticmethod
    def maxsim_score(
        query_vectors: MultiVectorEmbedding,
        doc_vectors: MultiVectorEmbedding,
    ) -> float:
        """Calculate MaxSim score for multi-vector embeddings (numpy-local)."""
        return maxsim_score(query_vectors, doc_vectors)


class EmbeddingServiceMVPlugin(EmbeddingServicePluginBase):
    """Plugin for EmbeddingServiceMV with a multivector-capable provider."""

    PROVIDER_NAME: str = "mv"

    def initialize(self, v: Variables, logger: Logger) -> EmbeddingService:
        cache_service = self.get_extension(EXT_CACHE_SERVICE, v)
        embedding_provider: EmbeddingProvider = self.get_extension(EXT_EMBEDDING_PROVIDER, v)
        return EmbeddingServiceMV(v=v, provider=embedding_provider, cache=cache_service)
