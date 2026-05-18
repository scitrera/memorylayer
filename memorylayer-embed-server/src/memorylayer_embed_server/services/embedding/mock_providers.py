"""Deterministic mock embedding providers — no torch, no models.

Selected via ``EMBED_SERVER_USE_MOCK_PROVIDERS=true``. The mocks expose
the same surfaces as the real single-vector (vLLM) and multi-vector
(ColPali) providers but produce fully deterministic output from a
SHA-256 seed of the input. Intended for:

* Unit tests of the FastAPI routes (no model downloads).
* The lightweight integration-test embed-server image
  (``Dockerfile.test``) so the chain memorylayer-server →
  memorylayer-embed-server can be exercised in CI without GPUs.
"""

from __future__ import annotations

import base64
import hashlib
import math
from pathlib import Path

import numpy as np
from memorylayer_server.services.embedding._maxsim import MultiVectorEmbedding
from memorylayer_server.services.embedding.base import (
    MultimodalEmbeddingProvider,
)
from scitrera_app_framework import Variables

DEFAULT_MOCK_SINGLE_VECTOR_DIMS = 384
DEFAULT_MOCK_MULTI_VECTOR_DIMS = 128
DEFAULT_MOCK_MULTI_VECTOR_TOKENS = 16


def _seeded_rng(payload: bytes) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return np.random.default_rng(seed)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def _image_to_bytes(image: str | bytes | Path) -> bytes:
    """Coerce supported image inputs to raw bytes (used as a hash key)."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, Path):
        return image.read_bytes()
    if isinstance(image, str):
        if image.startswith("data:image"):
            _, encoded = image.split(",", 1)
            return base64.b64decode(encoded)
        # Long base64 strings (no path/URL prefix) — decode and hash.
        try:
            return base64.b64decode(image, validate=False)
        except Exception:
            return image.encode("utf-8")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


class MockSingleVectorProvider(MultimodalEmbeddingProvider):
    """Deterministic dense embeddings, L2-normalized, hash-seeded.

    Inherits ``MultimodalEmbeddingProvider`` to mirror the real
    ``VLLMEmbeddingProvider`` (Qwen3-VL-Embedding-2B is multimodal) so
    ``/v1/embeddings/images?mode=single`` exercises the same code path.
    """

    PROVIDER_NAME = "mock-single"

    def __init__(self, v: Variables = None, dimensions: int = DEFAULT_MOCK_SINGLE_VECTOR_DIMS):
        super().__init__(v, output_dimensions=dimensions)
        self.model_name = f"mock-single-d{dimensions}"
        self.logger.info("Initialized MockSingleVectorProvider (dims=%d)", dimensions)

    async def preload(self) -> None:
        return

    def _vec_from_payload(self, payload: bytes) -> list[float]:
        rng = _seeded_rng(payload)
        vec = rng.standard_normal(self._dimensions).astype(np.float32)
        return _l2_normalize(vec).tolist()

    async def embed(self, text: str) -> list[float]:
        return self._vec_from_payload(text.encode("utf-8"))

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec_from_payload(t.encode("utf-8")) for t in texts]

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        return self._vec_from_payload(_image_to_bytes(image))

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


class MockMultiVectorProvider(MultimodalEmbeddingProvider):
    """Deterministic ColPali-shaped multi-vector embeddings.

    Each call produces ``num_tokens`` vectors of ``dimensions`` each,
    seeded from a SHA-256 of the input payload.
    """

    PROVIDER_NAME = "mock-multi"

    def __init__(
        self,
        v: Variables = None,
        dimensions: int = DEFAULT_MOCK_MULTI_VECTOR_DIMS,
        num_tokens: int = DEFAULT_MOCK_MULTI_VECTOR_TOKENS,
    ):
        super().__init__(v, output_dimensions=dimensions)
        self.num_tokens = num_tokens
        self.model_name = f"mock-multi-t{num_tokens}-d{dimensions}"
        self.logger.info(
            "Initialized MockMultiVectorProvider (num_tokens=%d, dims=%d)",
            num_tokens,
            dimensions,
        )

    async def preload(self) -> None:
        return

    def _mv_from_payload(self, payload: bytes) -> MultiVectorEmbedding:
        rng = _seeded_rng(payload)
        matrix = rng.standard_normal((self.num_tokens, self._dimensions)).astype(np.float32)
        matrix = _l2_normalize(matrix)
        return MultiVectorEmbedding(vectors=matrix.tolist())

    # Single-vector compatibility: mean-pool the multi-vector.
    async def embed(self, text: str) -> list[float]:
        mv = self._mv_from_payload(text.encode("utf-8"))
        return np.mean(np.array(mv.vectors), axis=0).tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]

    # ColPali-shaped methods
    async def embed_text_multivector(self, text: str) -> MultiVectorEmbedding:
        return self._mv_from_payload(text.encode("utf-8"))

    async def embed_batch_multivector(
        self,
        texts: list[str],
        batch_size: int = 8,
    ) -> list[MultiVectorEmbedding]:
        del batch_size
        return [self._mv_from_payload(t.encode("utf-8")) for t in texts]

    async def embed_image(self, image: str | bytes | Path) -> list[float]:
        mv = self._mv_from_payload(_image_to_bytes(image))
        return np.mean(np.array(mv.vectors), axis=0).tolist()

    async def embed_image_multivector(
        self,
        image: str | bytes | Path,
    ) -> MultiVectorEmbedding:
        return self._mv_from_payload(_image_to_bytes(image))

    async def embed_images_batch_multivector(
        self,
        images: list[str | bytes | Path],
        batch_size: int = 4,
    ) -> list[MultiVectorEmbedding]:
        del batch_size
        return [self._mv_from_payload(_image_to_bytes(img)) for img in images]

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


# Sigmoid used by callers that want bounded similarity scores from
# unbounded mock vectors (kept here for symmetry with the OSS reranker).
def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
