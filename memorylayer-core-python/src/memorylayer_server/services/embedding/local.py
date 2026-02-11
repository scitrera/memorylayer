from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables as Variables, get_logger

from .base import EmbeddingProvider, EmbeddingProviderPluginBase
from ...config import MEMORYLAYER_EMBEDDING_MODEL, EmbeddingProviderType

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding using sentence-transformers.

    For self-hosted deployments without external API calls.
    Text-only fallback when multimodal not needed.
    """

    def __init__(self, v: Variables = None, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(v)
        self.model_name = model_name
        self._model = None
        self.logger.info("Initialized LocalEmbeddingProvider with model: %s", model_name)

    def _get_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self.logger.info("Loading sentence-transformers model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        self.logger.debug("Generating local embedding for text: %s chars", len(text))
        model = self._get_model()
        embedding = model.encode(text)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        self.logger.debug("Generating local embeddings for batch of %s texts", len(texts))
        model = self._get_model()
        embeddings = model.encode(texts)
        return [e.tolist() for e in embeddings]

    @property
    def dimensions(self) -> int:
        model = self._get_model()
        return model.get_sentence_embedding_dimension()


class LocalEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = EmbeddingProviderType.LOCAL

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return LocalEmbeddingProvider(
            v=v,
            model_name=v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL),
        )
