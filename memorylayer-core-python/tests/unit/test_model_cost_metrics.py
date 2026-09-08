"""Embedding and dedicated-reranker metrics stay separate from generation."""

from unittest.mock import AsyncMock

import pytest

from memorylayer_server.services.embedding.service_default import EmbeddingService
from memorylayer_server.services.reranker.base import RerankerProvider, RerankerService


class SpyMetrics:
    def __init__(self):
        self.counters = []
        self.histograms = []

    def counter(self, name, value=1, labels=None):
        self.counters.append((name, value, labels))

    def histogram(self, name, value, labels=None):
        self.histograms.append((name, value, labels))


@pytest.mark.asyncio
async def test_embedding_provider_calls_have_their_own_metrics():
    provider = AsyncMock()
    provider.dimensions = 3
    provider.embed.return_value = [1.0, 0.0, 0.0]
    service = EmbeddingService(provider=provider)
    service.metrics = SpyMetrics()

    assert await service.embed("text") == [1.0, 0.0, 0.0]
    names = [name for name, _value, _labels in service.metrics.counters]
    assert "memorylayer_embedding_calls_total" in names
    assert not any("generation" in name for name in names)


class _DedicatedReranker(RerankerProvider):
    async def rerank(self, query, documents, instruction=None):
        return [1.0 - index / 10 for index, _document in enumerate(documents)]


class _GenerativeReranker(_DedicatedReranker):
    generative = True


@pytest.mark.asyncio
async def test_only_dedicated_rerankers_use_non_generative_metrics():
    metrics = SpyMetrics()
    dedicated = RerankerService(_DedicatedReranker())
    dedicated.metrics = metrics
    await dedicated.rerank("query", ["a", "b"])
    assert any(name == "memorylayer_reranker_calls_total" for name, _value, _labels in metrics.counters)

    generative_metrics = SpyMetrics()
    generative = RerankerService(_GenerativeReranker())
    generative.metrics = generative_metrics
    await generative.rerank("query", ["a", "b"])
    assert generative_metrics.counters == []
