"""Bridge from Aether ProxyHttpTerminator MintedRequest -> ASGI scope -> FastAPI.

This module is the per-request glue between the Aether
:class:`~scitrera_aether_client.proxy_terminator.ProxyHttpTerminator` and
MemoryLayer's FastAPI application. The terminator hands us a fully-minted
:class:`~scitrera_aether_client.proxy_terminator.MintedRequest`
(``X-Auth-*`` headers already stamped from the validated
``AuthorizationContext``); we drive the ASGI app once and accumulate the
response into a :class:`~scitrera_aether_client.proto.aether_pb2.ProxyHttpResponse`.

Notes / decisions
-----------------
* **Header encoding**: ASGI 3.0 mandates latin-1 encoding for headers in the
  scope (bytes-in, bytes-out per the spec). MintedRequest carries headers
  as ``Dict[str, str]``; non-latin-1 characters in either direction will
  raise ``UnicodeEncodeError``. That is a wire-protocol issue: HTTP headers
  are not transport-safe outside of latin-1 and a caller producing such a
  request has bigger problems.
* **Streaming responses**: TODO. Phase 2a explicitly excluded streaming on
  the terminator. We accumulate the full response body and return it
  inline. ``ProxyHttpTerminator._send_response`` chunks bodies > 256 KiB
  before sending them upstream, so this still scales for large bounded
  responses; what is NOT supported is open-ended streaming
  (``StreamingResponse`` over SSE etc). When the terminator is upgraded to
  support ``stream_response_indefinitely`` we will need to switch to a
  send-callback that forwards each ``http.response.body`` chunk as a
  separate ``ProxyHttpBodyChunk``.
* **Header collisions**: ASGI permits duplicate header names as separate
  ``(name, value)`` tuples; ``MintedRequest.headers`` is a ``Dict``, so
  duplicates are last-write-wins on the way IN. Going OUT we preserve
  whatever the ASGI app sent. This is a small fidelity loss for inbound
  ``Set-Cookie`` / ``Cookie`` echoing scenarios but matches the Go
  terminator's envelope shape, which also models headers as ``map<str,str>``.
* **app_workspace**: surfaced via ``scope["state"]["app_workspace"]`` so
  middleware that wants to read the originating Aether workspace doesn't
  have to re-parse a header. Cheap and side-effect-free.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from scitrera_aether_client.proto import aether_pb2
from scitrera_aether_client.proxy_terminator import MintedRequest


async def asgi_dispatch(
    app: Callable[..., Awaitable[None]],
    req: MintedRequest,
) -> aether_pb2.ProxyHttpResponse:
    """Drive an ASGI HTTP-protocol app from a ``MintedRequest``, return response.

    Builds a synthetic ``http`` scope, runs the app once, and accumulates
    the response body into a single :class:`ProxyHttpResponse`. Suitable
    for FastAPI / Starlette / any ASGI 3.0 app.

    Args:
        app: An ASGI 3.0 callable
            (``app(scope, receive, send)``).
        req: The fully assembled, header-minted request from the
            ProxyHttpTerminator.

    Returns:
        A :class:`ProxyHttpResponse` carrying the status, response headers,
        and accumulated body. ``request_id`` is stamped from ``req`` so
        the upstream gateway correlates the reply with the originating
        ``ProxyHttpRequest``.
    """
    received = False
    response_status = 500
    response_headers: list[tuple[bytes, bytes]] = []
    response_chunks: list[bytes] = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": req.body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal response_status, response_headers
        msg_type = message["type"]
        if msg_type == "http.response.start":
            response_status = message["status"]
            response_headers = list(message.get("headers", []))
        elif msg_type == "http.response.body":
            chunk = message.get("body", b"")
            if chunk:
                response_chunks.append(chunk)
            # ``more_body`` is intentionally ignored: we accumulate every
            # chunk until the ASGI coroutine returns. Streaming responses
            # need a different code path (see module docstring TODO).

    query = req.query
    query_bytes = query.encode("latin-1") if isinstance(query, str) else query

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": req.method,
        "scheme": "http",
        "path": req.path,
        "raw_path": req.path.encode("utf-8"),
        "query_string": query_bytes,
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in req.headers.items()
        ],
        "client": ("aether", 0),
        "server": ("memorylayer", 0),
        "root_path": "",
        # Surface the originating Aether workspace for downstream middleware
        # that wants it without re-parsing headers. Cheap, side-effect free.
        "state": {"app_workspace": req.app_workspace},
    }

    await app(scope, receive, send)

    return aether_pb2.ProxyHttpResponse(
        request_id=req.request_id,
        status_code=response_status,
        headers={k.decode("latin-1"): v.decode("latin-1") for k, v in response_headers},
        body=b"".join(response_chunks),
    )


__all__ = ["asgi_dispatch"]
