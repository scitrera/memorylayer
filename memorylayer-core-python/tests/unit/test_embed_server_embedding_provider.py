"""Unit tests for EmbedServerEmbeddingProvider.

The provider sits between the embedding-service abstraction and
EmbedServerClient. These tests stub the client surface and assert
that each public provider method routes to the right client method
with the expected payload shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.embedding.embed_server import (
    EmbedServerEmbeddingProvider,
)


def _provider_with_client(client) -> EmbedServerEmbeddingProvider:
    """Build a provider with a stubbed EmbedServerClient already wired."""
    p = EmbedServerEmbeddingProvider(v=None, output_dimensions=4)
    p._client = client
    return p


@pytest.fixture
def stub_client() -> MagicMock:
    client = MagicMock(name="EmbedServerClient")
    client.embed_texts = AsyncMock()
    client.embed_texts_multivector = AsyncMock()
    client.embed_images_multivector = AsyncMock()
    return client


# ---------------------------------------------------------------------------
# Single-vector text
# ---------------------------------------------------------------------------


async def test_embed_single_routes_to_embed_texts(stub_client):
    stub_client.embed_texts.return_value = [[0.1, 0.2, 0.3, 0.4]]
    provider = _provider_with_client(stub_client)

    result = await provider.embed("hello")

    assert result == [0.1, 0.2, 0.3, 0.4]
    stub_client.embed_texts.assert_awaited_once_with(["hello"])


async def test_embed_batch_routes_to_embed_texts(stub_client):
    stub_client.embed_texts.return_value = [[0.1] * 4, [0.2] * 4]
    provider = _provider_with_client(stub_client)

    result = await provider.embed_batch(["a", "b"])

    assert len(result) == 2
    stub_client.embed_texts.assert_awaited_once_with(["a", "b"])


async def test_embed_batch_empty_short_circuits(stub_client):
    provider = _provider_with_client(stub_client)
    assert await provider.embed_batch([]) == []
    stub_client.embed_texts.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-vector text
# ---------------------------------------------------------------------------


async def test_embed_text_multivector_routes_with_query_input_type(stub_client):
    stub_client.embed_texts_multivector.return_value = [{"vectors": [[0.1, 0.1], [0.2, 0.2]], "num_vectors": 2}]
    provider = _provider_with_client(stub_client)

    mv = await provider.embed_text_multivector("query text")

    assert mv.vectors == [[0.1, 0.1], [0.2, 0.2]]
    stub_client.embed_texts_multivector.assert_awaited_once_with(["query text"], input_type="query")


async def test_embed_batch_multivector_routes_with_document_input_type(stub_client):
    stub_client.embed_texts_multivector.return_value = [
        {"vectors": [[0.1] * 2], "num_vectors": 1},
        {"vectors": [[0.2] * 2], "num_vectors": 1},
    ]
    provider = _provider_with_client(stub_client)

    out = await provider.embed_batch_multivector(["doc1", "doc2"])

    assert len(out) == 2
    stub_client.embed_texts_multivector.assert_awaited_once_with(["doc1", "doc2"], input_type="document")


# ---------------------------------------------------------------------------
# Multi-vector image
# ---------------------------------------------------------------------------


async def test_embed_image_multivector_sends_base64(stub_client):
    stub_client.embed_images_multivector.return_value = [{"vectors": [[0.1, 0.2, 0.3, 0.4]], "num_vectors": 1}]
    provider = _provider_with_client(stub_client)

    raw_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"
    mv = await provider.embed_image_multivector(raw_bytes)

    assert len(mv.vectors) == 1
    # Verify the call args contain a single base64 string
    args, _ = stub_client.embed_images_multivector.call_args
    sent = args[0]
    assert len(sent) == 1
    import base64

    assert base64.b64decode(sent[0]) == raw_bytes


# ---------------------------------------------------------------------------
# Single-vector image (averaged from multivector)
# ---------------------------------------------------------------------------


async def test_embed_image_returns_average(stub_client):
    stub_client.embed_images_multivector.return_value = [{"vectors": [[1.0, 0.0], [0.0, 1.0]], "num_vectors": 2}]
    provider = _provider_with_client(stub_client)

    avg = await provider.embed_image(b"png-bytes")

    assert avg == pytest.approx([0.5, 0.5])


# ---------------------------------------------------------------------------
# Missing extension behaviour
# ---------------------------------------------------------------------------


async def test_missing_embed_server_client_raises(monkeypatch):
    import memorylayer_server.services.embedding.embed_server as mod

    monkeypatch.setattr(mod, "get_extension", lambda *_a, **_k: None)
    provider = EmbedServerEmbeddingProvider(v=None, output_dimensions=4)

    with pytest.raises(RuntimeError, match="EXT_EMBED_SERVER_CLIENT"):
        await provider.embed("hi")
