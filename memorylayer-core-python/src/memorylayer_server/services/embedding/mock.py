import hashlib
import math
import random

from logging import Logger

from scitrera_app_framework import Variables as Variables, get_logger

from ...config import EmbeddingProviderType, MEMORYLAYER_EMBEDDING_DIMENSIONS

from .base import EmbeddingProvider, EmbeddingProviderPluginBase

DEFAULT_EMBEDDING_DIMENSIONS = 384


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Simple mock embedding provider for testing without heavy ML dependencies.

    Generates deterministic embeddings based on content hash using a seeded
    PRNG for full-dimensional coverage. Embeddings use non-negative values
    to ensure cosine similarities are non-negative, matching the behavior
    of real embedding models.

    Not suitable for production - use for testing only.
    """

    def __init__(self, v: Variables = None, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS):
        super().__init__(v, dimensions)
        self.logger.info("Initialized MockEmbeddingProvider with dimensions=%d", dimensions)

    async def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding based on text hash."""

        # Use hash as PRNG seed for full-dimensional unique values
        text_hash = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(text_hash[:8], byteorder="big")
        rng = random.Random(seed)

        # Generate non-negative embedding values (like real embedding models)
        # so cosine similarities are always >= 0
        embedding = [rng.random() for _ in range(self._dimensions)]

        # L2-normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for batch."""
        return [await self.embed(text) for text in texts]


class MockEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = EmbeddingProviderType.MOCK

    def initialize(self, v: Variables, logger: Logger) -> MockEmbeddingProvider:
        return MockEmbeddingProvider(
            v=v,
            dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int)
        )
