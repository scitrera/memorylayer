"""Unit tests for EmbedServerRerankerProvider.

Covers the rerank pipeline: query is multivector-embedded with
input_type=``query``, documents with input_type=``document``, and the
score endpoint is invoked with both. Raw MaxSim scores are sigmoid-normalised
before being returned.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.reranker.embed_server import (
    EmbedServerRerankerProvider,
)


def _provider_with_client(client) -> EmbedServerRerankerProvider:
    p = EmbedServerRerankerProvider(v=None)
    p._client = client
    return p


@pytest.fixture
def stub_client() -> MagicMock:
    client = MagicMock(name="EmbedServerClient")
    client.embed_texts_multivector = AsyncMock()
    client.score_maxsim = AsyncMock()
    return client


async def test_rerank_empty_documents(stub_client):
    provider = _provider_with_client(stub_client)
    out = await provider.rerank("query", [])
    assert out == []
    stub_client.embed_texts_multivector.assert_not_called()
    stub_client.score_maxsim.assert_not_called()


async def test_rerank_routes_query_and_documents(stub_client):
    # First call: query embedding (input_type=query)
    # Second call: document embeddings (input_type=document)
    stub_client.embed_texts_multivector.side_effect = [
        [{"vectors": [[1.0, 0.0]], "num_vectors": 1}],
        [
            {"vectors": [[0.5, 0.5]], "num_vectors": 1},
            {"vectors": [[0.0, 1.0]], "num_vectors": 1},
        ],
    ]
    # Score endpoint returns out-of-order results to assert ordering
    stub_client.score_maxsim.return_value = [
        {"index": 1, "score": -1.0},
        {"index": 0, "score": 2.0},
    ]

    provider = _provider_with_client(stub_client)
    scores = await provider.rerank("question", ["doc-a", "doc-b"])

    # Two scores, in input order, sigmoid-normalised into (0, 1)
    assert len(scores) == 2
    assert all(0.0 < s < 1.0 for s in scores)
    # Index 0 had higher raw MaxSim → larger sigmoid value
    assert scores[0] > scores[1]

    # Verify the two embed calls had the right input_types
    embed_calls = stub_client.embed_texts_multivector.await_args_list
    assert embed_calls[0].kwargs == {"input_type": "query"}
    assert embed_calls[0].args == (["question"],)
    assert embed_calls[1].kwargs == {"input_type": "document"}
    assert embed_calls[1].args == (["doc-a", "doc-b"],)

    # Verify score_maxsim was called with the unwrapped vectors
    score_kwargs = stub_client.score_maxsim.await_args.kwargs
    assert score_kwargs["query_vectors"] == [[1.0, 0.0]]
    assert score_kwargs["document_vectors"] == [[[0.5, 0.5]], [[0.0, 1.0]]]


async def test_rerank_applies_instruction_prefix(stub_client):
    stub_client.embed_texts_multivector.side_effect = [
        [{"vectors": [[1.0, 0.0]], "num_vectors": 1}],
        [{"vectors": [[1.0, 0.0]], "num_vectors": 1}],
    ]
    stub_client.score_maxsim.return_value = [{"index": 0, "score": 1.0}]

    provider = _provider_with_client(stub_client)
    await provider.rerank("query", ["doc"], instruction="Find all medical docs:")

    # First call (query embedding) should have the instruction prepended
    first_call = stub_client.embed_texts_multivector.await_args_list[0]
    assert first_call.args == (["Find all medical docs: query"],)


async def test_missing_embed_server_client_raises(monkeypatch):
    import memorylayer_server.services.reranker.embed_server.provider as mod

    monkeypatch.setattr(mod, "get_extension", lambda *_a, **_k: None)
    provider = EmbedServerRerankerProvider(v=None)

    with pytest.raises(RuntimeError, match="EXT_EMBED_SERVER_CLIENT"):
        await provider.rerank("q", ["d"])
