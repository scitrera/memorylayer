"""Embedding provider that delegates all model work to ``memorylayer-embed-server``.

Single- and multi-vector text embeddings, image embeddings, and the
``embed_multimodal`` convenience all route through ``EmbedServerClient``
(``services/document/embed_client.py``), which itself supports both
direct HTTP and Aether-mTLS transports. No torch / transformers /
sentence-transformers / colpali_engine import lives in this process.

This provider is the canonical replacement for the retired in-process
providers ``local`` (sentence-transformers), ``colpali``
(colpali-engine), and ``qwen3-vl`` (qwen-vl-utils).
"""
from __future__ import annotations

import base64
from logging import Logger
from pathlib import Path

from scitrera_app_framework import Variables, get_extension

from ...config import (
    DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER,
    EmbeddingProviderType,
    MEMORYLAYER_EMBEDDING_DIMENSIONS,
)
from .._constants import EXT_EMBED_SERVER_CLIENT
from ._maxsim import MultiVectorEmbedding
from .base import MultimodalEmbeddingProvider, EmbeddingProviderPluginBase


class EmbedServerEmbeddingProvider(MultimodalEmbeddingProvider):
    """Embedding provider that proxies all calls to the embed-server.

    Implements both single-vector and multi-vector embedding methods.
    The underlying ``EmbedServerClient`` is resolved lazily (on first
    use) via the ``EXT_EMBED_SERVER_CLIENT`` extension so that whichever
    transport (HTTP or Aether) the operator configured is reused.
    """

    def __init__(
        self,
        v: Variables = None,
        output_dimensions: int | None = None,
    ):
        super().__init__(v, output_dimensions)
        self._v = v
        self._client = None
        self.logger.info(
            "Initialized EmbedServerEmbeddingProvider (dimensions=%s)",
            output_dimensions,
        )

    def _get_client(self):
        if self._client is None:
            client = get_extension(EXT_EMBED_SERVER_CLIENT, self._v)
            if client is None:
                raise RuntimeError(
                    "EmbedServerEmbeddingProvider requires the embed-server "
                    "client extension (EXT_EMBED_SERVER_CLIENT) to be "
                    "initialised. Check MEMORYLAYER_EMBED_SERVER_URL and "
                    "MEMORYLAYER_EMBED_TRANSPORT."
                )
            self._client = client
        return self._client

    async def _ensure_connected(self):
        """Ensure the EmbedServerClient is connected; safe to call repeatedly.

        ``EmbedServerClientPlugin.async_ready`` connects at startup, but tests
        or non-standard wirings may construct the provider without the plugin
        lifecycle. The HTTP transport's ``connect()`` is cheap and idempotent
        in the sense that the second call simply reopens the httpx client if
        none exists.
        """
        client = self._get_client()
        # HTTP transport: ``_client`` is the underlying httpx.AsyncClient and
        # is None until connect() is called. Aether transport never sets it.
        if getattr(client, "_client", None) is None and getattr(client, "_transport", "http") == "http":
            await client.connect()
        return client

    # ------------------------------------------------------------------
    # Single-vector text
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> list[float]:
        client = await self._ensure_connected()
        result = await client.embed_texts([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = await self._ensure_connected()
        return await client.embed_texts(texts)

    # ------------------------------------------------------------------
    # Multi-vector (ColPali-style late interaction)
    # ------------------------------------------------------------------

    async def embed_text_multivector(self, text: str) -> MultiVectorEmbedding:
        client = await self._ensure_connected()
        result = await client.embed_texts_multivector([text], input_type="query")
        return MultiVectorEmbedding(vectors=result[0]["vectors"])

    async def embed_batch_multivector(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[MultiVectorEmbedding]:
        # batch_size is informational here; the server controls real batching.
        del batch_size
        if not texts:
            return []
        client = await self._ensure_connected()
        results = await client.embed_texts_multivector(texts, input_type="document")
        return [MultiVectorEmbedding(vectors=r["vectors"]) for r in results]

    async def embed_image_multivector(
        self,
        image: str | bytes | Path,
    ) -> MultiVectorEmbedding:
        client = await self._ensure_connected()
        b64 = _to_base64(image)
        results = await client.embed_images_multivector([b64])
        return MultiVectorEmbedding(vectors=results[0]["vectors"])

    async def embed_images_batch_multivector(
        self,
        images: list[str | bytes | Path],
        batch_size: int = 4,
    ) -> list[MultiVectorEmbedding]:
        del batch_size
        if not images:
            return []
        client = await self._ensure_connected()
        b64_images = [_to_base64(img) for img in images]
        results = await client.embed_images_multivector(b64_images)
        return [MultiVectorEmbedding(vectors=r["vectors"]) for r in results]

    # ------------------------------------------------------------------
    # MultimodalEmbeddingProvider abstract methods
    # ------------------------------------------------------------------

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        """Single-vector embedding for an image (averaged from multi-vector)."""
        import numpy as np

        multi = await self.embed_image_multivector(image)
        if not multi.vectors:
            return []
        return np.mean(np.array(multi.vectors), axis=0).tolist()

    async def embed_multimodal(
        self,
        text: str | None = None,
        image: str | bytes | Path | None = None,
    ) -> list[float]:
        if image is not None:
            return await self.embed_image(image)
        if text is not None:
            return await self.embed(text)
        raise ValueError("At least one of text or image must be provided")


def _to_base64(image: str | bytes | Path) -> str:
    """Coerce an image input to a base64 string (no data: prefix)."""
    if isinstance(image, str):
        if image.startswith("data:image"):
            _, encoded = image.split(",", 1)
            return encoded
        if image.startswith(("http://", "https://")):
            import urllib.request

            with urllib.request.urlopen(image) as response:  # noqa: S310 - operator-controlled URL
                return base64.b64encode(response.read()).decode("ascii")
        path_obj = Path(image)
        if len(image) <= 500 or path_obj.exists():
            try:
                return base64.b64encode(path_obj.read_bytes()).decode("ascii")
            except (OSError, ValueError):
                pass
        return image  # treat as raw base64
    if isinstance(image, bytes):
        return base64.b64encode(image).decode("ascii")
    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("ascii")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


class EmbedServerEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    """Plugin registration for the embed-server-backed embedding provider."""

    PROVIDER_NAME = EmbeddingProviderType.EMBED_SERVER

    def initialize(self, v: Variables, logger: Logger) -> EmbedServerEmbeddingProvider:
        dimensions = v.environ(
            MEMORYLAYER_EMBEDDING_DIMENSIONS,
            default=DEFAULT_EMBEDDING_DIMENSIONS_EMBED_SERVER,
            type_fn=int,
        )
        return EmbedServerEmbeddingProvider(v=v, output_dimensions=dimensions)
