"""Unit tests for EmbedServerClient transport switch (Phase 3).

Covers the dual-transport contract added in Phase 3 of the Aether
convergence: ``EmbedServerClient`` selects between direct HTTP (legacy
behaviour) and Aether ``proxy_http_async`` based on the ``transport``
constructor arg, with otherwise-identical public method shapes.

What is tested here:
- HTTP transport: requires ``connect()``, calls ``httpx.AsyncClient.post``;
  request_id and body round-trip through the standard FastAPI shape.
- Aether transport: requires no ``connect()``; calls
  ``scitrera_aether_client.proxy.proxy_http_async`` with the configured
  ``aether_target`` topic and serialised JSON body.
- Custom target topic via ``aether_target`` is honoured.
- Construct-time validation: invalid transport string raises ``ValueError``;
  ``transport='aether'`` without an ``aether_connection`` raises ``ValueError``.

Out of scope here:
- The full embed server inference path — that lives in
  ``oss/memorylayer-embed-server/tests/``.
- Real Aether gateway round-trip — covered by the in-process integration
  tests in ``tests/integration/test_aether_terminator_chain.py`` (which
  prove the terminator side; this client is the initiator complement).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.document.embed_client import (
    EmbedServerClient,
    TRANSPORT_AETHER,
    TRANSPORT_HTTP,
)


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


async def test_embed_client_http_transport_calls_httpx(monkeypatch):
    """HTTP transport: connect() opens httpx.AsyncClient; embed_texts hits POST /v1/embeddings."""
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }
    fake_response.raise_for_status = MagicMock()
    fake_client.post = AsyncMock(return_value=fake_response)
    fake_client.aclose = AsyncMock()

    # Patch httpx.AsyncClient so connect() returns our mock.
    import memorylayer_server.services.document.embed_client as ec_mod

    fake_async_client_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr(ec_mod.httpx, "AsyncClient", fake_async_client_cls)

    logger = MagicMock()
    client = EmbedServerClient(
        base_url="http://localhost:61051",
        timeout=30.0,
        logger=logger,
        transport=TRANSPORT_HTTP,
    )
    await client.connect()
    out = await client.embed_texts(["hello", "world"])

    assert out == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.post.assert_awaited_once_with("/v1/embeddings", json={"input": ["hello", "world"]})


# ---------------------------------------------------------------------------
# Aether transport
# ---------------------------------------------------------------------------


async def test_embed_client_aether_transport_calls_proxy_http_async(monkeypatch):
    """Aether transport: embed_texts issues proxy_http_async against the default target."""
    # Fake proxy_http_async response
    fake_response = SimpleNamespace(
        status_code=200,
        body=json.dumps(
            {"data": [{"index": 0, "embedding": [0.5]}]}
        ).encode("utf-8"),
    )
    fake_proxy = AsyncMock(return_value=fake_response)

    # Patch the *late* import target (proxy_http_async is imported inside
    # _request_json on the aether path).  Replace at the source module so
    # the import sees the mock.
    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    fake_aether_conn = SimpleNamespace(client=MagicMock(name="AsyncServiceClient"))
    logger = MagicMock()

    client = EmbedServerClient(
        base_url="http://unused",
        timeout=30.0,
        logger=logger,
        transport=TRANSPORT_AETHER,
        aether_connection=fake_aether_conn,
    )
    await client.connect()  # no-op for aether
    out = await client.embed_texts(["hi"])

    assert out == [[0.5]]
    assert fake_proxy.await_count == 1
    call = fake_proxy.await_args
    # Positional: (client,)
    assert call.args[0] is fake_aether_conn.client
    # Keyword args carry target_topic + path + body
    assert call.kwargs["target_topic"] == "sv::memorylayer-embed::default"
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["path"] == "/v1/embeddings"
    body_decoded = json.loads(call.kwargs["body"])
    assert body_decoded == {"input": ["hi"]}
    assert call.kwargs["headers"]["content-type"] == "application/json"


async def test_embed_client_aether_transport_honours_custom_target(monkeypatch):
    """``aether_target`` constructor arg overrides the default target topic."""
    fake_response = SimpleNamespace(
        status_code=200,
        body=json.dumps({"data": [{"index": 0, "embedding": []}]}).encode("utf-8"),
    )
    fake_proxy = AsyncMock(return_value=fake_response)

    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    fake_aether_conn = SimpleNamespace(client=MagicMock())
    client = EmbedServerClient(
        base_url="http://unused",
        timeout=30.0,
        logger=MagicMock(),
        transport=TRANSPORT_AETHER,
        aether_connection=fake_aether_conn,
        aether_target="sv::memorylayer-embed::pod-7",
    )
    await client.embed_texts([])

    assert fake_proxy.await_args.kwargs["target_topic"] == "sv::memorylayer-embed::pod-7"


# ---------------------------------------------------------------------------
# Construct-time validation
# ---------------------------------------------------------------------------


def test_embed_client_rejects_invalid_transport():
    """Unknown transport identifiers raise at construction."""
    with pytest.raises(ValueError, match="Unsupported embed transport"):
        EmbedServerClient(
            base_url="http://localhost:61051",
            logger=MagicMock(),
            transport="grpc-direct",
        )


def test_embed_client_aether_transport_requires_connection():
    """``transport='aether'`` without an aether_connection raises."""
    with pytest.raises(ValueError, match="aether_connection"):
        EmbedServerClient(
            base_url="http://unused",
            logger=MagicMock(),
            transport=TRANSPORT_AETHER,
            aether_connection=None,
        )
