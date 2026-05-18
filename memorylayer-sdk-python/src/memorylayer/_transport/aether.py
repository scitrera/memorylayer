"""Aether transport — routes SDK requests through ``proxy_http_async``.

When ``MemoryLayerClient(transport='aether', aether_client=..., aether_target=...)``
is constructed, every request goes via the Aether SDK's ``proxy_http_async``
initiator against the MemoryLayer service principal (default
``sv::memorylayer::default``).  The on-prem Aether terminator running inside
MemoryLayer (Phase 2c) handles the inbound envelope, mints
``X-Auth-*`` headers from the validated authority, and forwards to FastAPI.

OBO surface
-----------
The SDK's ``acting_for(...)`` context manager produces ``X-Aether-*`` headers.
This transport intercepts those headers, translates them into the canonical
``AuthorizationContext`` proto field on ``ProxyHttpRequest``, and lets
``proxy_http_async`` carry them as a structured envelope.  The terminator's
``_mint_auth_headers()`` reads ``req.authorization`` directly — that is the
canonical OBO surface, and is preferable to header-based propagation
(strict-mode terminators strip inbound ``X-Auth-*`` / ``X-Aether-*`` headers
before minting anyway, so the proto field is the only reliable channel).

Connection ownership
--------------------
``AetherTransport`` does NOT own the underlying ``aether_client``.  ``aclose()``
is a no-op.  Callers are expected to construct + dispose of the connection
themselves (cowork, embed-server's pattern, etc.).
"""

from __future__ import annotations

import json as _json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode

# Headers that ``acting_for()`` injects on the SDK side.  We intercept these
# at the transport boundary and convert them into a structured proto field.
_OBO_HEADER_GRANT_ID = "X-Aether-Grant-ID"
_OBO_HEADER_MODE = "X-Aether-Authority-Mode"
_OBO_HEADER_SUBJECT_TYPE = "X-Aether-Subject-Type"
_OBO_HEADER_SUBJECT_ID = "X-Aether-Subject-ID"

# API key / session headers that travel as ordinary HTTP headers in the
# outbound ProxyHttpRequest.  The terminator forwards them unchanged.
_DEFAULT_OBO_HEADER_NAMES = (
    _OBO_HEADER_GRANT_ID,
    _OBO_HEADER_MODE,
    _OBO_HEADER_SUBJECT_TYPE,
    _OBO_HEADER_SUBJECT_ID,
)


class AetherTransportResponse:
    """``TransportResponse``-compatible wrapper around ``ProxyHttpResponse``."""

    def __init__(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body

    @property
    def content(self) -> bytes:
        return self._body

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", errors="replace") if self._body else ""

    def json(self) -> Any:
        if not self._body:
            return {}
        return _json.loads(self._body.decode("utf-8"))

    def raise_for_status(self) -> None:
        # Mirror ``httpx.Response.raise_for_status`` semantics so the SDK's
        # request layer can stay transport-agnostic.  The MemoryLayer error
        # mapping in client._request runs on status_code first, so this is
        # only a fallback for any path that calls raise_for_status directly.
        if 400 <= self.status_code < 600:
            import httpx

            raise httpx.HTTPStatusError(
                f"{self.status_code} response from aether transport",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )


class AetherTransport:
    """Routes SDK requests via ``scitrera_aether_client.proxy.proxy_http_async``."""

    def __init__(
        self,
        aether_client: Any,
        target: str = "sv::memorylayer::default",
        *,
        api_key: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
        base_path: str = "/v1",
    ) -> None:
        if aether_client is None:
            raise ValueError(
                "AetherTransport requires aether_client (an AsyncAgentClient or AsyncServiceClient with a live gateway connection)"
            )
        self._aether_client = aether_client
        self._target = target
        self._timeout = timeout
        # The terminator allow_paths config in MemoryLayer covers /v1/* + /healthz +
        # /metrics; we mount the SDK on /v1 to mirror what HttpTransport does.
        self._base_path = base_path.rstrip("/")
        self._default_headers: dict[str, str] = {}
        if api_key:
            self._default_headers["Authorization"] = f"Bearer {api_key}"
        if session_id:
            self._default_headers["X-Session-ID"] = session_id
        self._session_id: str | None = session_id

    def set_session(self, session_id: str) -> None:
        self._session_id = session_id
        self._default_headers["X-Session-ID"] = session_id

    def clear_session(self) -> None:
        self._session_id = None
        self._default_headers.pop("X-Session-ID", None)

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> AetherTransportResponse:
        # Late import: scitrera_aether_client is an optional runtime dep
        # selected by the [aether] extra. OSS deployments using HTTP-only
        # transport never hit this code.
        from scitrera_aether_client.proxy import proxy_http_async

        # Build the full path with query string. The SDK's HttpTransport
        # uses base_url=".../v1" so paths like "/memories" become absolute
        # "/v1/memories"; we replicate that here so the terminator's
        # allow_paths (["/v1/*", ...]) match.
        full_path = path
        if not full_path.startswith("/"):
            full_path = "/" + full_path
        full_path = f"{self._base_path}{full_path}"

        if params:
            stringified = {str(k): str(v) for k, v in params.items() if v is not None}
            if stringified:
                full_path = f"{full_path}?{urlencode(stringified)}"

        merged_headers: dict[str, str] = dict(self._default_headers)
        if headers:
            merged_headers.update(headers)

        # Extract OBO headers and convert to a proxy_http_async kwarg surface.
        grant_id = merged_headers.pop(_OBO_HEADER_GRANT_ID, None)
        authority_mode = merged_headers.pop(_OBO_HEADER_MODE, None)
        subject_type = merged_headers.pop(_OBO_HEADER_SUBJECT_TYPE, None)
        subject_id = merged_headers.pop(_OBO_HEADER_SUBJECT_ID, None)
        # Lower-cased variants belt-and-braces — middleware sometimes
        # normalises header keys.
        for hdr in _DEFAULT_OBO_HEADER_NAMES:
            merged_headers.pop(hdr.lower(), None)

        body = b""
        if json is not None and content is not None:
            raise ValueError("AetherTransport.request: pass either json or content, not both")
        if json is not None:
            body = _json.dumps(json).encode("utf-8")
            merged_headers.setdefault("content-type", "application/json")
        elif content is not None:
            body = content if isinstance(content, bytes) else content.encode("utf-8")
            # Caller is responsible for setting any required content-type via headers.

        kwargs: dict[str, Any] = dict(
            target_topic=self._target,
            method=method.upper(),
            path=full_path,
            headers=merged_headers,
            body=body,
            timeout=self._timeout,
        )
        if grant_id:
            kwargs["grant_id"] = grant_id
            kwargs["authority_mode"] = authority_mode or "on_behalf_of"
            if subject_type:
                kwargs["subject_type"] = subject_type
            if subject_id:
                kwargs["subject_id"] = subject_id

        response = await proxy_http_async(self._aether_client, **kwargs)

        # ``proxy_http_async`` returns a ``ProxyHttpResponse`` proto with
        # ``status_code``, ``headers`` (ScalarMap), ``body`` (bytes), and
        # ``error`` (ProxyError).  We project to our compatibility shape.
        if response.HasField("error"):
            from ..exceptions import MemoryLayerError

            raise MemoryLayerError(
                f"Aether proxy error: {response.error.message or 'unknown'}",
                status_code=response.status_code or 502,
            )

        return AetherTransportResponse(
            status_code=response.status_code,
            headers={k: v for k, v in response.headers.items()},
            body=bytes(response.body),
        )

    async def aclose(self) -> None:
        # We do NOT own the aether_client; closing is the caller's job.
        return None
