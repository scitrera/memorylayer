"""Unit tests for ``memorylayer_server.services.aether_service.asgi_bridge``.

Builds a tiny FastAPI app (defined inline, no conftest dependency) and
drives ``asgi_dispatch`` with synthetic ``MintedRequest`` objects.  The
tests confirm that the ASGI bridge correctly converts a ``MintedRequest``
into an ASGI ``http`` scope, invokes the app, and packages the response
back into a ``ProxyHttpResponse``.

What is tested here:
- ``test_asgi_dispatch_basic_get`` — GET /healthz → 200, body present.
- ``test_asgi_dispatch_post_json_body`` — POST body round-trips through the
  bridge (handler reads the raw body, echoes it; bytes match end-to-end).
- ``test_asgi_dispatch_query_string_preserved`` — query string is placed in
  ``scope["query_string"]`` so FastAPI's Query params parse correctly.
- ``test_asgi_dispatch_response_headers_round_trip`` — handler sets
  ``Content-Type: application/json``; the header appears in the
  ``ProxyHttpResponse.headers`` map.
- ``test_asgi_dispatch_x_auth_headers_visible_to_handler`` — the
  load-bearing OBO test: ``MintedRequest.headers`` contains ``X-Auth-*``
  entries minted by the terminator; the ASGI bridge must surface them in
  the Starlette request so the FastAPI handler (or auth middleware) can
  read them.  This confirms the MintedRequest → ASGI scope → Starlette
  Request header chain works end-to-end.
- ``test_asgi_dispatch_status_code_propagates`` — handler raises
  ``HTTPException(status_code=403)``; response carries 403.

Out of scope here:
- The full terminator→ASGI→FastAPI stack (covered in
  ``tests/integration/test_aether_terminator_chain.py``).
- Streaming responses — the asgi_bridge module documents this as a known
  limitation (Phase 2a explicitly excluded indefinite streaming).
- Live Aether gateway / dev compose infrastructure.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query, Request, Response
from scitrera_aether_client.proxy_terminator import MintedRequest

from memorylayer_server.services.aether_service.asgi_bridge import asgi_dispatch

# ---------------------------------------------------------------------------
# Mini FastAPI app shared across tests
# ---------------------------------------------------------------------------

_app = FastAPI()


@_app.get("/healthz")
async def healthz():
    return Response(content=b"ok", media_type="text/plain", status_code=200)


@_app.post("/api/echo")
async def echo(request: Request):
    body = await request.body()
    return Response(content=body, media_type="application/octet-stream", status_code=200)


@_app.get("/api/search")
async def search(a: str = Query(default=""), b: str = Query(default="")):
    payload = json.dumps({"a": a, "b": b}).encode()
    return Response(content=payload, media_type="application/json", status_code=200)


@_app.get("/api/typed")
async def typed_response():
    return Response(
        content=b'{"ok": true}',
        media_type="application/json",
        status_code=200,
    )


@_app.get("/api/auth-echo")
async def auth_echo(request: Request):
    subject = request.headers.get("x-auth-subject-id", "missing")
    mode = request.headers.get("x-auth-authority-mode", "missing")
    payload = json.dumps({"subject_id": subject, "mode": mode}).encode()
    return Response(content=payload, media_type="application/json", status_code=200)


@_app.get("/api/forbidden")
async def forbidden():
    raise HTTPException(status_code=403, detail="no access")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _minted(
    method: str = "GET",
    path: str = "/healthz",
    query: str = "",
    headers: dict | None = None,
    body: bytes = b"",
    request_id: str = "test-req-1",
) -> MintedRequest:
    return MintedRequest(
        method=method,
        path=path,
        query=query,
        headers=headers or {},
        body=body,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_asgi_dispatch_basic_get():
    """GET /healthz returns 200 with non-empty body."""
    req = _minted(method="GET", path="/healthz")
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 200
    assert resp.body == b"ok"
    assert resp.request_id == "test-req-1"


async def test_asgi_dispatch_post_json_body():
    """POST /api/echo: handler reads the request body; response body matches."""
    payload = b'{"message": "hello from aether"}'
    req = _minted(
        method="POST",
        path="/api/echo",
        headers={"content-type": "application/octet-stream"},
        body=payload,
        request_id="test-req-echo",
    )
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 200
    assert resp.body == payload


async def test_asgi_dispatch_query_string_preserved():
    """Query string ``a=1&b=2`` is parsed by FastAPI's Query mechanism."""
    req = _minted(
        method="GET",
        path="/api/search",
        query="a=hello&b=world",
        request_id="test-req-qs",
    )
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["a"] == "hello"
    assert data["b"] == "world"


async def test_asgi_dispatch_response_headers_round_trip():
    """Handler sets ``Content-Type: application/json``; header present in response."""
    req = _minted(method="GET", path="/api/typed", request_id="test-req-hdrs")
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 200
    # Headers map is lowercase-normalised by ASGI / Starlette.
    ct = resp.headers.get("content-type", "")
    assert "application/json" in ct


async def test_asgi_dispatch_x_auth_headers_visible_to_handler():
    """X-Auth-* headers in MintedRequest reach the FastAPI handler.

    This is the load-bearing test for the OBO header chain:
    MintedRequest.headers → asgi_bridge scope → Starlette Request.headers.
    The terminator mints X-Auth-* headers from the AuthorizationContext
    before calling the handler; if they don't survive the scope-building
    step, OBO-gated endpoints would silently see no authority context.
    """
    req = _minted(
        method="GET",
        path="/api/auth-echo",
        headers={
            "X-Auth-Subject-ID": "user-alice",
            "X-Auth-Authority-Mode": "on_behalf_of",
        },
        request_id="test-req-auth",
    )
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["subject_id"] == "user-alice"
    assert data["mode"] == "on_behalf_of"


async def test_asgi_dispatch_status_code_propagates():
    """HTTPException(status_code=403) from a handler becomes response status 403."""
    req = _minted(method="GET", path="/api/forbidden", request_id="test-req-403")
    resp = await asgi_dispatch(_app, req)

    assert resp.status_code == 403
