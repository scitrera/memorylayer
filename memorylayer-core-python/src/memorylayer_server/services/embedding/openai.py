from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables as Variables, get_logger

from ...config import EmbeddingProviderType, MEMORYLAYER_EMBEDDING_MODEL, MEMORYLAYER_EMBEDDING_DIMENSIONS

from .base import EmbeddingProvider, EmbeddingProviderPluginBase

MEMORYLAYER_EMBEDDING_OPENAI_API_KEY = 'MEMORYLAYER_EMBEDDING_OPENAI_API_KEY'
MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL = 'MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL'

DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_OPENAI_API_KEY = 'x'
DEFAULT_OPENAI_BASE_URL = None


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small.

    Also supports OpenAI-compatible APIs (vLLM, Ollama, LocalAI, etc.)
    by specifying base_url.
    """

    def __init__(
            self,
            v: Variables = None,
            api_key: Optional[str] = None,
            model: str = "text-embedding-3-small",
            base_url: Optional[str] = None,
            dimensions: int = 1536,
    ):
        super().__init__(v, output_dimensions=dimensions)
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._base_url = base_url

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        self.logger.debug("Generating OpenAI embedding for text: %s chars", len(text))
        response = await self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        self.logger.debug("Generating OpenAI embeddings for batch of %s texts", len(texts))
        response = await self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]


class OpenAIEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = EmbeddingProviderType.OPENAI

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return OpenAIEmbeddingProvider(
            v=v,
            api_key=v.environ(MEMORYLAYER_EMBEDDING_OPENAI_API_KEY, default=DEFAULT_OPENAI_API_KEY),
            model=v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL),
            base_url=v.environ(MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL, default=DEFAULT_OPENAI_BASE_URL),
            dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
        )
