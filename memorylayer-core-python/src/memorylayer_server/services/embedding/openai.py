from logging import Logger

from scitrera_app_framework import Variables as Variables, ext_parse_bool

from ...config import MEMORYLAYER_EMBEDDING_DIMENSIONS, MEMORYLAYER_EMBEDDING_MODEL, EmbeddingProviderType
from .._constants import EXT_API_KEY_STORE
from ..api_key_store import get_api_key_store, resolve_api_key_ref
from .base import EmbeddingProvider, EmbeddingProviderPluginBase

MEMORYLAYER_EMBEDDING_OPENAI_API_KEY = "MEMORYLAYER_EMBEDDING_OPENAI_API_KEY"
# Optional override for the API-key-store name to resolve when no literal
# MEMORYLAYER_EMBEDDING_OPENAI_API_KEY is set. Defaults to the provider's
# own DEFAULT_API_KEY_NAME (``OPENAI_API_KEY``).
MEMORYLAYER_EMBEDDING_OPENAI_API_KEY_NAME = "MEMORYLAYER_EMBEDDING_OPENAI_API_KEY_NAME"
MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL = "MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL"
# Provider-specific model override; wins over MEMORYLAYER_EMBEDDING_MODEL.
# Useful when the OpenAI provider runs alongside another (e.g. on the
# embed-server with ColPali for multi-vector — both providers would
# otherwise collide on the shared MEMORYLAYER_EMBEDDING_MODEL key).
MEMORYLAYER_EMBEDDING_OPENAI_MODEL = "MEMORYLAYER_EMBEDDING_OPENAI_MODEL"
# When truthy, send the OpenAI ``dimensions`` request param (= configured
# MEMORYLAYER_EMBEDDING_DIMENSIONS) so a matryoshka-capable model/server (OpenAI
# text-embedding-3*, vLLM Qwen3-VL-Embedding, …) returns a truncated vector.
# Default off: older/base models reject the param. Prod uses this to serve 1920-d
# vectors (matryoshka-truncated from 2048), which also keeps them under pgvector's
# 2000-dim HNSW-index ceiling.
MEMORYLAYER_EMBEDDING_OPENAI_PASS_DIMENSIONS = "MEMORYLAYER_EMBEDDING_OPENAI_PASS_DIMENSIONS"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_OPENAI_BASE_URL = None
# Placeholder used when no key resolves — OpenAI-compatible local servers
# (vLLM, Ollama, LocalAI) ignore the key but the SDK requires a non-empty value.
_KEYLESS_PLACEHOLDER = "x"


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small.

    Also supports OpenAI-compatible APIs (vLLM, Ollama, LocalAI, etc.)
    by specifying base_url.

    The API key is resolved through the API key store (env by default;
    Aether-first in enterprise) unless a literal ``api_key`` is supplied, in
    which case it is used statically. The SDK client is built lazily and
    rebuilt only when the resolved key changes (supports rotation).
    """

    # Default key name resolved from the API key store when no literal
    # api_key is given and no override is configured.
    DEFAULT_API_KEY_NAME = "OPENAI_API_KEY"

    def __init__(
        self,
        v: Variables = None,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        dimensions: int = 1536,
        api_key_name: str | None = None,
        api_key_store=None,
        pass_dimensions: bool = False,
    ):
        super().__init__(v, output_dimensions=dimensions)
        # Whether to send the ``dimensions`` request param (matryoshka truncation).
        self._pass_dimensions = pass_dimensions
        self._static_key, self._key_ref = resolve_api_key_ref(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            default_name=self.DEFAULT_API_KEY_NAME,
        )
        self.model = model
        self._base_url = base_url
        self._client = None
        self._client_key = None

    def _build_client(self, api_key):
        """Construct an OpenAI async client for ``api_key`` (placeholder if None)."""
        import openai

        return openai.AsyncOpenAI(api_key=api_key or _KEYLESS_PLACEHOLDER, base_url=self._base_url)

    async def _ensure_client(self):
        """Return a client built with the currently-resolved key.

        Rebuilds the cached client only when the resolved key changes.
        """
        key = self._static_key if self._key_ref is None else await self._key_ref.resolve()
        if self._client is None or key != self._client_key:
            self._client = self._build_client(key)
            self._client_key = key
        return self._client

    def _create_kwargs(self, input_) -> dict:
        """Build ``embeddings.create`` kwargs, optionally requesting matryoshka dims."""
        kwargs = {"input": input_, "model": self.model}
        if self._pass_dimensions and self._dimensions:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for single text."""
        self.logger.debug("Generating OpenAI embedding for text: %s chars", len(text))
        client = await self._ensure_client()
        response = await client.embeddings.create(**self._create_kwargs(text))
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (more efficient)."""
        self.logger.debug("Generating OpenAI embeddings for batch of %s texts", len(texts))
        client = await self._ensure_client()
        response = await client.embeddings.create(**self._create_kwargs(texts))
        return [item.embedding for item in response.data]


class OpenAIEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = EmbeddingProviderType.OPENAI

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        model = v.environ(MEMORYLAYER_EMBEDDING_OPENAI_MODEL, default=None)
        if not model:
            model = v.environ(MEMORYLAYER_EMBEDDING_MODEL, default=DEFAULT_EMBEDDING_MODEL)
        return OpenAIEmbeddingProvider(
            v=v,
            api_key=v.environ(MEMORYLAYER_EMBEDDING_OPENAI_API_KEY, default=None),
            api_key_name=v.environ(MEMORYLAYER_EMBEDDING_OPENAI_API_KEY_NAME, default=None),
            api_key_store=get_api_key_store(v),
            model=model,
            base_url=v.environ(MEMORYLAYER_EMBEDDING_OPENAI_BASE_URL, default=DEFAULT_OPENAI_BASE_URL),
            dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int),
            pass_dimensions=ext_parse_bool(v.environ(MEMORYLAYER_EMBEDDING_OPENAI_PASS_DIMENSIONS, default="")),
        )

    def get_dependencies(self, v: Variables):
        return (EXT_API_KEY_STORE,)
