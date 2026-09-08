"""Google GenAI (Gemini) embedding provider."""

from logging import Logger

from scitrera_app_framework import Variables, get_logger

from ...config import MEMORYLAYER_EMBEDDING_DIMENSIONS, MEMORYLAYER_EMBEDDING_MODEL, EmbeddingProviderType
from .._constants import EXT_API_KEY_STORE
from ..api_key_store import get_api_key_store, resolve_api_key_ref
from .base import EmbeddingProvider, EmbeddingProviderPluginBase

MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY = "MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY"
# Optional override for the API-key-store name to resolve when no literal
# MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY is set. Defaults to the provider's
# own DEFAULT_API_KEY_NAME (``GOOGLE_API_KEY``).
MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY_NAME = "MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY_NAME"
# Provider-specific model override; wins over MEMORYLAYER_EMBEDDING_MODEL.
# Useful when the Google provider runs alongside another (e.g. on the
# embed-server with ColPali for multi-vector — both providers would
# otherwise collide on the shared MEMORYLAYER_EMBEDDING_MODEL key).
MEMORYLAYER_EMBEDDING_GOOGLE_MODEL = "MEMORYLAYER_EMBEDDING_GOOGLE_MODEL"

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_EMBEDDING_DIMENSIONS = 768


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google GenAI embedding provider using Gemini embedding models.

    Uses the google-genai SDK for text embedding generation.

    The API key is resolved through the API key store (env by default;
    Aether-first in enterprise) unless a literal ``api_key`` is supplied, in
    which case it is used statically. The SDK client is built lazily and
    rebuilt only when the resolved key changes (supports rotation).
    """

    # Default key name resolved from the API key store when no literal
    # api_key is given and no override is configured.
    DEFAULT_API_KEY_NAME = "GOOGLE_API_KEY"

    def __init__(
        self,
        v: Variables = None,
        api_key: str | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        api_key_name: str | None = None,
        api_key_store=None,
    ):
        super().__init__(v, output_dimensions=dimensions)
        self._api_key, self._key_ref = resolve_api_key_ref(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            default_name=self.DEFAULT_API_KEY_NAME,
        )
        self.model = model
        self._output_dimensionality = dimensions
        self._client = None
        self._client_key = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized GoogleEmbeddingProvider: model=%s, dimensions=%s",
            model,
            dimensions,
        )

    def _build_client(self, api_key):
        """Construct a Google GenAI client for ``api_key``."""
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        return genai.Client(api_key=api_key)

    def _get_client(self):
        """Lazy-load Google GenAI client from the static key (no store)."""
        if self._client is None:
            self._client = self._build_client(self._api_key)
            self._client_key = self._api_key
        return self._client

    async def _ensure_client(self):
        """Return a client built with the currently-resolved key.

        Rebuilds the cached client only when the store-resolved key changes.
        """
        if self._key_ref is None:
            return self._get_client()
        key = await self._key_ref.resolve()
        if self._client is None or key != self._client_key:
            self._client = self._build_client(key)
            self._client_key = key
        return self._client

    def _get_config(self):
        """Build EmbedContentConfig with output dimensionality."""
        from google.genai import types

        return types.EmbedContentConfig(
            output_dimensionality=self._output_dimensionality,
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        client = await self._ensure_client()
        self.logger.debug("Generating Google embedding for text: %s chars", len(text))

        response = await client.aio.models.embed_content(
            model=self.model,
            contents=text,
            config=self._get_config(),
        )
        return list(response.embeddings[0].values)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        client = await self._ensure_client()
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
        model = v.environ(MEMORYLAYER_EMBEDDING_GOOGLE_MODEL, default=None)
        if not model:
            model = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL)
        return GoogleEmbeddingProvider(
            v=v,
            api_key=v.environ(MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY, default=None),
            api_key_name=v.environ(MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY_NAME, default=None),
            api_key_store=get_api_key_store(v),
            model=model,
            dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
        )

    def get_dependencies(self, v: Variables):
        return (EXT_API_KEY_STORE,)
