"""Unit tests for the Aether transport mode (Phase 5).

Covers ``MemoryLayerClient(transport='aether', aether_client=...)``: requests
route through ``scitrera_aether_client.proxy.proxy_http_async`` against the
configured target topic, ``acting_for()`` propagates as a structured
``AuthorizationContext`` proto field on the proxy envelope, the SDK does NOT
own the underlying aether connection, and constructor-time validation
catches the missing-aether_client case.

Out of scope:
- Bespoke httpx paths (file uploads, NDJSON streaming) — those raise
  ``NotImplementedError`` on Aether transport by design.  Documented in the
  client.py docstring on ``_ensure_client``.
- Sync client (``SyncMemoryLayerClient``) — async-first per Phase 5 scope.
"""

from __future__ import annotations

import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer._transport import AetherTransportResponse
from memorylayer.client import MemoryLayerClient


def _fake_proxy_response(
    status_code: int = 200,
    body: bytes = b'{"ok": true}',
    headers: dict[str, str] | None = None,
) -> SimpleNamespace:
    """Build a minimal stand-in for ``ProxyHttpResponse``.

    Only the attrs the transport actually reads matter: ``status_code``,
    ``headers`` (a Mapping), ``body`` (bytes), and a ``HasField('error')``
    that returns False.
    """
    return SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        body=body,
        HasField=lambda f: False,
    )


# ---------------------------------------------------------------------------
# Construct-time validation
# ---------------------------------------------------------------------------


def test_aether_transport_requires_aether_client():
    """``transport='aether'`` without ``aether_client`` raises at construction."""
    with pytest.raises(ValueError, match="aether_client"):
        MemoryLayerClient(transport="aether", aether_client=None)


def test_unknown_transport_string_raises_on_aenter():
    """Unknown transport strings raise when entering the context manager."""
    client = MemoryLayerClient(transport="grpc-direct")  # type: ignore[arg-type]

    async def _enter():
        async with client:
            pass

    import asyncio

    with pytest.raises(ValueError, match="Unknown transport"):
        asyncio.get_event_loop().run_until_complete(_enter())


# ---------------------------------------------------------------------------
# Aether transport request routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aether_transport_routes_through_proxy_http_async(monkeypatch):
    """Standard request goes via proxy_http_async with the configured target."""
    fake_proxy = AsyncMock(
        return_value=_fake_proxy_response(
            status_code=200,
            body=_json.dumps({"id": "mem_1", "content": "hi"}).encode("utf-8"),
        )
    )

    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    fake_aether = MagicMock(name="AsyncServiceClient")
    async with MemoryLayerClient(
        transport="aether",
        aether_client=fake_aether,
        api_key="test-key",
    ) as client:
        # Drive any path that goes through _request; remember() is convenient.
        # Drive the transport directly — we're testing the transport layer,
        # not the SDK methods' response parsing.
        await client._transport.request("POST", "/memories", json={"content": "hi"})

    assert fake_proxy.await_count == 1
    call = fake_proxy.await_args
    # Positional first arg is the aether client
    assert call.args[0] is fake_aether
    # Default target topic
    assert call.kwargs["target_topic"] == "sv::memorylayer::default"
    # Path is mounted at /v1/* to match the terminator allow_paths config
    assert call.kwargs["path"].startswith("/v1/")
    # Body round-trips as JSON bytes
    body_decoded = _json.loads(call.kwargs["body"])
    assert body_decoded["content"] == "hi"
    # Authorization header carried as ordinary HTTP header
    assert call.kwargs["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_aether_transport_obo_via_authorization_field(monkeypatch):
    """``acting_for()`` populates the structured AuthorizationContext kwargs.

    The transport intercepts X-Aether-* headers and translates them into the
    proxy_http_async ``grant_id`` / ``authority_mode`` / ``subject_type`` /
    ``subject_id`` kwargs (proto AuthorizationContext field on the envelope).
    The terminator's _mint_auth_headers reads the structured field, not the
    headers, so this is the canonical OBO surface.
    """
    fake_proxy = AsyncMock(return_value=_fake_proxy_response(body=_json.dumps({"results": []}).encode("utf-8")))

    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    fake_aether = MagicMock(name="AsyncAgentClient")
    async with MemoryLayerClient(transport="aether", aether_client=fake_aether, api_key="k") as client:
        async with client.acting_for("g_abc", subject=("user", "alice")) as proxy:
            await proxy.recall("preferences")

    call = fake_proxy.await_args
    # OBO routes via the structured kwargs, not the X-Aether-* headers.
    assert call.kwargs["grant_id"] == "g_abc"
    assert call.kwargs["authority_mode"] == "on_behalf_of"
    assert call.kwargs["subject_type"] == "user"
    assert call.kwargs["subject_id"] == "alice"
    # The X-Aether-* headers are NOT forwarded as ordinary headers — strict
    # terminator would strip them anyway, but the SDK should not even send
    # them since the structured field is the authoritative surface.
    forwarded_hdrs = call.kwargs["headers"]
    for k in forwarded_hdrs:
        assert not k.lower().startswith("x-aether-"), f"X-Aether-* header leaked through structured-OBO path: {k!r}"


@pytest.mark.asyncio
async def test_aether_transport_custom_target_honoured(monkeypatch):
    """``aether_target`` constructor arg overrides the default topic."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response(body=b'{"ok": true}'))

    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    async with MemoryLayerClient(
        transport="aether",
        aether_client=MagicMock(),
        aether_target="sv::memorylayer::pod-7",
    ) as client:
        # Drive the transport directly — we're testing the transport layer,
        # not the SDK methods' response parsing.
        await client._transport.request("POST", "/memories", json={"content": "hi"})

    assert fake_proxy.await_args.kwargs["target_topic"] == "sv::memorylayer::pod-7"


# ---------------------------------------------------------------------------
# Lifecycle: SDK does NOT own the aether connection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aether_transport_aclose_does_not_close_shared_client(monkeypatch):
    """Exiting the SDK context does not call ``aclose`` on the aether client."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response())

    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    fake_aether = MagicMock(name="AsyncServiceClient")
    fake_aether.aclose = AsyncMock()

    async with MemoryLayerClient(transport="aether", aether_client=fake_aether) as client:
        # Drive the transport directly — we're testing the transport layer,
        # not the SDK methods' response parsing.
        await client._transport.request("POST", "/memories", json={"content": "hi"})

    fake_aether.aclose.assert_not_called()


# ---------------------------------------------------------------------------
# Direct-mode: AetherTransportResponse json/raise_for_status sanity
# ---------------------------------------------------------------------------


def test_aether_transport_response_json_roundtrip():
    """``AetherTransportResponse.json()`` decodes a UTF-8 body."""
    resp = AetherTransportResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body=b'{"a": 1}',
    )
    assert resp.json() == {"a": 1}
    # raise_for_status is a no-op on 2xx
    resp.raise_for_status()


def test_aether_transport_response_empty_body_returns_dict():
    """Empty 204-style body decodes to an empty dict."""
    resp = AetherTransportResponse(status_code=204, headers={}, body=b"")
    assert resp.json() == {}


# ---------------------------------------------------------------------------
# Phase 5.1 — content path (NDJSON, binary, multipart)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aether_transport_content_bytes_round_trip(monkeypatch):
    """``content=<bytes>`` is forwarded as-is, no JSON encoding applied."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response(body=b'{"imported": 0}'))
    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    ndjson = b'{"a":1}\n{"b":2}\n'
    async with MemoryLayerClient(transport="aether", aether_client=MagicMock()) as client:
        await client._transport.request(
            "POST",
            "/workspaces/ws_1/import",
            content=ndjson,
            headers={"Content-Type": "application/x-ndjson"},
        )

    call = fake_proxy.await_args
    assert call.kwargs["body"] == ndjson
    # SDK-supplied content-type wins; no application/json default applied.
    assert call.kwargs["headers"].get("Content-Type") == "application/x-ndjson"


@pytest.mark.asyncio
async def test_aether_transport_content_string_is_utf8_encoded(monkeypatch):
    """``content=<str>`` is utf-8 encoded before shipping."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response())
    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    ndjson_str = '{"a":"\u00e9"}\n'  # UTF-8 multibyte char
    async with MemoryLayerClient(transport="aether", aether_client=MagicMock()) as client:
        await client._transport.request("POST", "/x", content=ndjson_str, headers={"Content-Type": "text/plain"})

    assert fake_proxy.await_args.kwargs["body"] == ndjson_str.encode("utf-8")


@pytest.mark.asyncio
async def test_aether_transport_rejects_both_json_and_content(monkeypatch):
    """Passing both ``json`` and ``content`` is a programming error."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response())
    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    async with MemoryLayerClient(transport="aether", aether_client=MagicMock()) as client:
        with pytest.raises(ValueError, match="json or content, not both"):
            await client._transport.request("POST", "/x", json={"a": 1}, content=b"hi")


def test_encode_multipart_produces_well_formed_body():
    """``MemoryLayerClient._encode_multipart`` returns bytes parseable as multipart.

    We don't try to fully parse the multipart body — that's httpx's job and
    httpx already covers it.  We just confirm the boundary is in both the
    content-type header and the body, the file marker is present, and the
    form fields are encoded.
    """
    body, content_type = MemoryLayerClient._encode_multipart(
        files={"file": ("test.csv", b"a,b,c\n1,2,3\n")},
        data={"name": "demo", "importance": "0.5"},
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1]
    assert boundary.encode() in body
    # Form fields appear as name="..." Content-Disposition lines.
    assert b'name="file"' in body
    assert b"test.csv" in body
    assert b'name="name"' in body
    assert b"demo" in body
    assert b'name="importance"' in body
    assert b"0.5" in body
    # File body bytes are present verbatim.
    assert b"a,b,c\n1,2,3\n" in body


@pytest.mark.asyncio
async def test_aether_transport_carries_multipart_unchanged(monkeypatch):
    """Multipart body produced by ``_encode_multipart`` reaches proxy_http_async unchanged."""
    fake_proxy = AsyncMock(return_value=_fake_proxy_response(body=_json.dumps({"document": {}, "job": {}}).encode("utf-8")))
    import scitrera_aether_client.proxy as proxy_mod

    monkeypatch.setattr(proxy_mod, "proxy_http_async", fake_proxy)

    body, content_type = MemoryLayerClient._encode_multipart(
        files={"file": ("doc.txt", b"hello world")},
        data={"target_context_id": "ctx_1", "importance": "0.5"},
    )

    async with MemoryLayerClient(transport="aether", aether_client=MagicMock()) as client:
        await client._transport.request(
            "POST",
            "/documents",
            content=body,
            headers={"content-type": content_type},
        )

    call = fake_proxy.await_args
    assert call.kwargs["body"] == body
    assert call.kwargs["headers"]["content-type"] == content_type
