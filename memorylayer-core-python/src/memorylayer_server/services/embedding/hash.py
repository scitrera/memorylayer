"""Deterministic lexical hashing embedding provider.

Unlike :class:`MockEmbeddingProvider` (which hashes the *entire* string into a
random vector and therefore carries no token-level signal), this provider uses
the classic feature-hashing trick: each token is hashed to a dimension and
accumulated, then the vector is L2-normalized. Two texts that share words get a
positive cosine similarity proportional to their lexical overlap.

That makes it fully deterministic, dependency-free, and offline while still
producing *meaningful* relative rankings — which is exactly what the retrieval
eval harness (``memorylayer_server.eval``) needs to score recall quality and
detect ranking regressions. It is NOT a semantic model; do not use it in
production.
"""

import hashlib
import math
import re
from logging import Logger

from scitrera_app_framework import Variables as Variables

from ...config import MEMORYLAYER_EMBEDDING_DIMENSIONS, EmbeddingProviderType
from .base import EmbeddingProvider, EmbeddingProviderPluginBase

DEFAULT_EMBEDDING_DIMENSIONS = 384

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbeddingProvider(EmbeddingProvider):
    """Feature-hashing bag-of-words embeddings (deterministic, offline)."""

    def __init__(self, v: Variables = None, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS):
        super().__init__(v, dimensions)
        self.logger.info("Initialized HashEmbeddingProvider with dimensions=%d", dimensions)
        # This is the default provider so the server works with no external
        # dependencies, which means most operators reach it without choosing it.
        # Say plainly what they got and how to upgrade.
        self.logger.warning(
            "Using the 'hash' embedding provider: lexical bag-of-words matching, NOT semantic search. "
            "Good for local development, tests, and the retrieval-eval harness; not for production recall quality. "
            "For real embeddings set MEMORYLAYER_EMBEDDING_PROVIDER to one of: "
            "'openai' (+MEMORYLAYER_EMBEDDING_OPENAI_API_KEY), "
            "'google' (+MEMORYLAYER_EMBEDDING_GOOGLE_API_KEY), or "
            "'embed_server' (+MEMORYLAYER_EMBED_SERVER_URL, self-hosted GPU)."
        )

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic, non-negative, lexically-grounded embedding."""
        dims = self._dimensions
        vec = [0.0] * dims

        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            # Empty/punctuation-only text: hash the raw string into one slot so
            # the result is still deterministic and L2-normalizable.
            digest = hashlib.sha1(text.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "big") % dims] = 1.0
        else:
            for token in tokens:
                digest = hashlib.sha1(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "big") % dims
                # Non-negative accumulation keeps cosine similarity >= 0, matching
                # the behaviour real embedding models exhibit in this codebase.
                vec[idx] += 1.0

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class HashEmbeddingProviderPlugin(EmbeddingProviderPluginBase):
    PROVIDER_NAME = EmbeddingProviderType.HASH

    def initialize(self, v: Variables, logger: Logger) -> HashEmbeddingProvider:
        return HashEmbeddingProvider(
            v=v, dimensions=v.environ(MEMORYLAYER_EMBEDDING_DIMENSIONS, default=DEFAULT_EMBEDDING_DIMENSIONS, type_fn=int)
        )
