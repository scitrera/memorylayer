"""Reranker provider that delegates scoring to ``memorylayer-embed-server``.

Uses the embed-server's multi-vector embedding endpoints plus the
``/v1/score`` MaxSim endpoint to score query-document pairs without
running any model in-process.

Replaces the retired in-process providers ``local`` (sentence-transformers
CrossEncoder) and ``qwen3-vl`` (qwen-vl-utils reranker).
"""
from __future__ import annotations

from logging import Logger

from scitrera_app_framework import Variables, get_extension

from ....config import RerankerProviderType
from ..._constants import EXT_EMBED_SERVER_CLIENT
from ..base import RerankerProvider, RerankerProviderPluginBase


class EmbedServerRerankerProvider(RerankerProvider):
    """Reranker that scores documents via embed-server MaxSim.

    Each ``rerank`` call:
      1. Embeds the query as a multi-vector via ``/v1/embeddings/multi``
         (input_type=``query``).
      2. Embeds the documents as multi-vectors via the same endpoint
         (input_type=``document``).
      3. Sends both to ``/v1/score`` for MaxSim scoring.
      4. Sigmoid-normalises the raw scores into the 0..1 range expected
         by the rest of the reranker abstraction.
    """

    def __init__(self, v: Variables = None):
        super().__init__(v)
        self._v = v
        self._client = None
        self.logger.info("Initialized EmbedServerRerankerProvider")

    def _get_client(self):
        if self._client is None:
            client = get_extension(EXT_EMBED_SERVER_CLIENT, self._v)
            if client is None:
                raise RuntimeError(
                    "EmbedServerRerankerProvider requires the embed-server "
                    "client extension (EXT_EMBED_SERVER_CLIENT) to be "
                    "initialised. Check MEMORYLAYER_EMBED_SERVER_URL and "
                    "MEMORYLAYER_EMBED_TRANSPORT."
                )
            self._client = client
        return self._client

    async def _ensure_connected(self):
        """Ensure the EmbedServerClient is connected; safe to call repeatedly."""
        client = self._get_client()
        if getattr(client, "_client", None) is None and getattr(client, "_transport", "http") == "http":
            await client.connect()
        return client

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        if not documents:
            return []

        effective_query = f"{instruction} {query}" if instruction else query

        client = await self._ensure_connected()
        query_mv = await client.embed_texts_multivector(
            [effective_query], input_type="query"
        )
        doc_mv = await client.embed_texts_multivector(
            documents, input_type="document"
        )

        scores_payload = await client.score_maxsim(
            query_vectors=query_mv[0]["vectors"],
            document_vectors=[d["vectors"] for d in doc_mv],
        )

        # /v1/score returns a list of {index, score} dicts; restore input order
        # and squash unbounded MaxSim scores to a 0..1 range via sigmoid.
        ordered = sorted(scores_payload, key=lambda s: s["index"])
        return [_sigmoid(float(item["score"])) for item in ordered]


def _sigmoid(x: float) -> float:
    import math

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EmbedServerRerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin registration for the embed-server-backed reranker provider."""

    PROVIDER_NAME = RerankerProviderType.EMBED_SERVER

    def initialize(self, v: Variables, logger: Logger) -> EmbedServerRerankerProvider:
        return EmbedServerRerankerProvider(v=v)
