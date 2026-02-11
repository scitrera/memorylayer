"""Google GenAI (Gemini) embedding provider."""
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_logger

from ...config import EmbeddingProviderType, MEMORYLAYER_EMBEDDING_MODEL, MEMORYLAYER_EMBEDDING_DIMENSIONS
from .base import EmbeddingProvider, EmbeddingProviderPluginBase

MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY = 'MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY'

DEFAULT_EMBEDDING_MODEL = 'gemini-embedding-001'
DEFAULT_EMBEDDING_DIMENSIONS = 768


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google GenAI embedding provider using Gemini embedding models.

    Uses the google-genai SDK for text embedding generation.
    """

    def __init__(
            self,
            v: Variables = None,
            api_key: Optional[str] = None,
            model: str = DEFAULT_EMBEDDING_MODEL,
            dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ):
        super().__init__(v, output_dimensions=dimensions)
        self._api_key = api_key
        self.model = model
        self._output_dimensionality = dimensions
        self._client = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized GoogleEmbeddingProvider: model=%s, dimensions=%s",
            model, dimensions,
        )

    def _get_client(self):
        """Lazy-load Google GenAI client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. Install with: pip install google-genai"
                )
        return self._client

    def _get_config(self):
        """Build EmbedContentConfig with output dimensionality."""
        from google.genai import types
        return types.EmbedContentConfig(
            output_dimensionality=self._output_dimensionality,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        client = self._get_client()
        self.logger.debug("Generating Google embedding for text: %s chars", len(text))

        response = await client.aio.models.embed_content(
            model=self.model,
            contents=text,
            config=self._get_config(),
        )
        return list(response.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        client = self._get_client()
        self.logger.debug("Generating Google embeddings for batch of %s texts", len(texts))

        response = await client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=self._get_config(),
        )
        return [list(emb.values) for emb in response.embeddings]


class GoogleEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    """Plugin for Google GenAI embedding provider."""
    PROVIDER_NAME = EmbeddingProviderType.GOOGLE

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return GoogleEmbeddingProvider(
            v=v,
            api_key=v.environ(MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY, default=None),
            model=v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL),
            dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
        )
