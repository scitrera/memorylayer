"""Async HTTP client for the MemoryLayer Embed Server.

Wraps the embed server REST API for transcription, single-vector embeddings,
multi-vector embeddings, and MaxSim scoring.

Two transports are supported, selected by ``MEMORYLAYER_EMBED_TRANSPORT``:

* ``http`` (default) — direct ``httpx.AsyncClient`` calls against
  ``MEMORYLAYER_EMBED_SERVER_URL``. Suitable for OSS deployments running the
  embed server as a peer container reachable via plain HTTP.
* ``aether`` — issues ``proxy_http_async`` calls through the existing
  ``AetherServiceConnection`` client against
  ``sv::memorylayer-embed::{specifier}`` (default ``default``). Suitable for
  cross-DC GPU placement where the embed server sits behind a Go
  proxy-sidecar terminator and is only reachable over Aether mTLS.

This client is authority-context-agnostic: requests are made under the
host MemoryLayer service's direct authority. OBO scoping for embedding
access is a follow-up; today every embedding call is service→service.
"""

from __future__ import annotations

import json as _json
from logging import Logger
from typing import Any

import httpx
from scitrera_app_framework import Variables, get_extension

from ...config import (
    DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
    DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
    DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT,
    DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
    DEFAULT_MEMORYLAYER_EMBED_TRANSPORT,
    MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
    MEMORYLAYER_EMBED_AETHER_TARGET,
    MEMORYLAYER_EMBED_SERVER_TIMEOUT,
    MEMORYLAYER_EMBED_SERVER_URL,
    MEMORYLAYER_EMBED_TRANSPORT,
)
from .._constants import EXT_AETHER_SERVICE_CONNECTION
from . import EmbedServerClientPluginBase

# Transport identifiers
TRANSPORT_HTTP = "http"
TRANSPORT_AETHER = "aether"


class EmbedServerClient:
    """Async client for the MemoryLayer Embed Server.

    Provides methods for transcription, embedding generation (single-vector
    and multi-vector), and MaxSim scoring. The wire transport is selected at
    construction time:

    * ``transport='http'`` → direct httpx call to ``base_url``
    * ``transport='aether'`` → ``proxy_http_async`` against ``aether_target``
      using the supplied ``aether_connection`` (an
      ``AetherServiceConnection`` whose ``client`` attribute is an
      ``AsyncServiceClient``).

    Either way the public method shapes are identical so callers do not
    branch on transport.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 300.0,
        logger: Logger = None,
        *,
        transport: str = TRANSPORT_HTTP,
        aether_connection: Any | None = None,
        aether_target: str = DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
        aether_stream_idle_timeout_ms: int = DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
    ):
        """Initialize the embed server client.

        Args:
            base_url: Base URL of the embed server (HTTP transport only).
            timeout: Per-request timeout in seconds. For streaming RPCs over
                Aether this becomes the time-to-first-byte deadline.
            logger: Logger instance.
            transport: ``'http'`` or ``'aether'``.
            aether_connection: AetherServiceConnection providing ``.client``;
                required for ``transport='aether'``.
            aether_target: Target topic for Aether transport.
            aether_stream_idle_timeout_ms: Idle timeout for streaming RPCs
                over Aether, passed to ``proxy_http_async``.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self.logger = logger
        self._transport = transport
        self._aether_connection = aether_connection
        self._aether_target = aether_target
        self._aether_stream_idle_timeout_ms = int(aether_stream_idle_timeout_ms)
        if transport not in (TRANSPORT_HTTP, TRANSPORT_AETHER):
            raise ValueError(f"Unsupported embed transport: {transport!r} (must be {TRANSPORT_HTTP!r} or {TRANSPORT_AETHER!r})")
        if transport == TRANSPORT_AETHER and aether_connection is None:
            raise ValueError("transport='aether' requires aether_connection (an AetherServiceConnection with a live .client)")

    async def connect(self) -> None:
        """Initialize the underlying transport.

        For HTTP transport this opens an ``httpx.AsyncClient``; for Aether
        transport this is a no-op (the shared service connection is owned
        by ``AetherServiceConnection`` and already connected).
        """
        if self._transport == TRANSPORT_HTTP:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
            self.logger.info("Connected to embed server (http) at %s", self._base_url)
        else:
            self.logger.info(
                "Embed server client using aether transport, target=%s",
                self._aether_target,
            )

    async def close(self) -> None:
        """Close the underlying transport (HTTP only).

        Aether transport leaves the shared service connection alone.
        """
        if self._transport == TRANSPORT_HTTP and self._client is not None:
            await self._client.aclose()
            self._client = None
            self.logger.info("Disconnected from embed server")

    # ------------------------------------------------------------------
    # Internal: shared request shape across transports.
    # ------------------------------------------------------------------

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        """Issue a JSON request via the configured transport and return the
        decoded JSON body. Raises an HTTP-status-style error on non-2xx.
        """
        if self._transport == TRANSPORT_HTTP:
            assert self._client is not None, "EmbedServerClient.connect() not called"
            if method.upper() == "POST":
                resp = await self._client.post(path, json=payload)
            elif method.upper() == "GET":
                resp = await self._client.get(path)
            else:  # pragma: no cover - defensive; only POST/GET used today
                resp = await self._client.request(method, path, json=payload)
            resp.raise_for_status()
            return resp.json()

        # ── Aether transport ────────────────────────────────────────────
        # Late import: ``scitrera_aether_client`` is an optional runtime dep
        # for OSS deployments that never use aether transport.
        from scitrera_aether_client.proxy import proxy_http_async

        body = b""
        headers: dict[str, str] = {}
        if payload is not None:
            body = _json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        client = self._aether_connection.client
        response = await proxy_http_async(
            client,
            target_topic=self._aether_target,
            method=method.upper(),
            path=path,
            headers=headers,
            body=body,
            timeout=self._timeout,
        )

        status = response.status_code
        if status < 200 or status >= 300:
            raise EmbedServerHTTPError(
                status_code=status,
                body=response.body,
                target=self._aether_target,
                path=path,
            )
        if not response.body:
            return {}
        return _json.loads(response.body)

    async def _request_stream(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ):
        """Issue a streaming request and return an ``AsyncIterator[bytes]``.

        The caller iterates raw SSE-formatted chunks. Both transports
        yield bytes verbatim from the upstream:

        * HTTP: ``httpx.AsyncClient.stream()`` → ``aiter_bytes()``.
        * Aether: ``proxy_http_async(stream_response=True)`` →
          ``StreamingProxyResponse.aiter()``.
        """
        if self._transport == TRANSPORT_HTTP:
            assert self._client is not None, "EmbedServerClient.connect() not called"

            async def _http_iter():
                async with self._client.stream(method.upper(), path, json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk

            return _http_iter()

        # ── Aether transport ────────────────────────────────────────────
        from scitrera_aether_client.proxy import proxy_http_async

        body = b""
        headers: dict[str, str] = {}
        if payload is not None:
            body = _json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        client = self._aether_connection.client
        streaming_resp = await proxy_http_async(
            client,
            target_topic=self._aether_target,
            method=method.upper(),
            path=path,
            headers=headers,
            body=body,
            timeout=self._timeout,
            stream_response=True,
            stream_idle_timeout_ms=self._aether_stream_idle_timeout_ms,
        )

        # ``StreamingProxyResponse`` returns header info on the wrapper +
        # ``aiter()`` for chunks. The wrapper raises mid-stream errors
        # from the iterator, so caller-side error handling stays consistent.
        status = getattr(streaming_resp, "status_code", None)
        if status is not None and (status < 200 or status >= 300):
            raise EmbedServerHTTPError(
                status_code=status,
                body=b"",
                target=self._aether_target,
                path=path,
            )

        async def _aether_iter():
            async for chunk in streaming_resp.aiter():
                if chunk:
                    yield chunk

        return _aether_iter()

    # ------------------------------------------------------------------
    # Public API (transport-agnostic)
    # ------------------------------------------------------------------

    async def chat_completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ):
        """Forward an OpenAI-shape chat completion to the embed-server.

        Returns a ``dict`` (the upstream JSON response) when ``stream=False``,
        or an ``AsyncIterator[bytes]`` yielding raw SSE chunks when
        ``stream=True``.

        The payload is forwarded as-is, so tools, response_format, multimodal
        content blocks, reasoning fields, and any other OpenAI-compatible
        extension pass through to whatever LLM the embed-server is hosting.
        """
        if stream:
            request_payload = {**payload, "stream": True}
            return await self._request_stream("POST", "/v1/chat/completions", request_payload)
        return await self._request_json("POST", "/v1/chat/completions", payload)

    async def completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ):
        """Legacy text completions. Same contract as :meth:`chat_completions`."""
        if stream:
            request_payload = {**payload, "stream": True}
            return await self._request_stream("POST", "/v1/completions", request_payload)
        return await self._request_json("POST", "/v1/completions", payload)

    async def transcribe_pages(
        self,
        images_b64: list[str],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Transcribe page images to markdown text."""
        payload: dict = {"images": images_b64}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.logger.debug("Transcribing %d page images", len(images_b64))
        return await self._request_json("POST", "/v1/transcribe", payload)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Get single-vector embeddings for texts."""
        payload = {"input": texts}

        self.logger.debug("Embedding %d texts (single-vector)", len(texts))
        data = await self._request_json("POST", "/v1/embeddings", payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    async def embed_texts_multivector(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[dict]:
        """Get multi-vector embeddings for texts via ColPali."""
        payload = {"input": texts, "input_type": input_type}

        self.logger.debug("Embedding %d texts (multi-vector, type=%s)", len(texts), input_type)
        data = await self._request_json("POST", "/v1/embeddings/multi", payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [{"vectors": item["vectors"], "num_vectors": item["num_vectors"]} for item in sorted_data]

    async def embed_images_multivector(self, images_b64: list[str]) -> list[dict]:
        """Get multi-vector embeddings from images via ColPali."""
        payload = {"images": images_b64, "mode": "multi"}

        self.logger.debug("Embedding %d images (multi-vector)", len(images_b64))
        data = await self._request_json("POST", "/v1/embeddings/images", payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [{"vectors": item["vectors"], "num_vectors": item["num_vectors"]} for item in sorted_data]

    async def score_maxsim(
        self,
        query_vectors: list[list[float]],
        document_vectors: list[list[list[float]]],
    ) -> list[dict]:
        """Score query against documents via MaxSim."""
        payload = {
            "query_vectors": query_vectors,
            "document_vectors": document_vectors,
        }

        self.logger.debug(
            "Scoring MaxSim: query (%d vectors) vs %d documents",
            len(query_vectors),
            len(document_vectors),
        )
        data = await self._request_json("POST", "/v1/score", payload)
        return data["scores"]


class EmbedServerHTTPError(Exception):
    """Raised when the embed server returns a non-2xx status (any transport).

    Provides the same surface for callers regardless of whether the underlying
    failure originated from httpx (HTTP transport) or proxy_http_async
    (Aether transport).
    """

    def __init__(self, *, status_code: int, body: bytes, target: str, path: str) -> None:
        self.status_code = status_code
        self.body = body
        self.target = target
        self.path = path
        try:
            detail = body.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            detail = "<undecodable body>"
        super().__init__(f"embed server returned {status_code} for {target}{path}: {detail!r}")


class EmbedServerClientPlugin(EmbedServerClientPluginBase):
    """Plugin for the default embed server client.

    Reads ``MEMORYLAYER_EMBED_TRANSPORT`` to decide which transport to wire up.
    For ``aether`` transport it pulls the shared ``AetherServiceConnection``
    via ``EXT_AETHER_SERVICE_CONNECTION`` (the Phase-1 service connection).

    ``async_ready`` calls ``client.connect()`` so the underlying
    ``httpx.AsyncClient`` is ready before any provider tries to use it.
    """

    PROVIDER_NAME = "default"

    async def async_ready(self, v: Variables, logger: Logger, value: object | None) -> None:
        if value is None:
            return
        try:
            await value.connect()
        except Exception as e:  # noqa: BLE001 — connect failures shouldn't crash boot
            logger.warning("EmbedServerClient.connect() failed at startup: %s", e)

    def initialize(self, v: Variables, logger: Logger) -> EmbedServerClient:
        base_url = v.environ(
            MEMORYLAYER_EMBED_SERVER_URL,
            default=DEFAULT_MEMORYLAYER_EMBED_SERVER_URL,
        )
        timeout = float(
            v.environ(
                MEMORYLAYER_EMBED_SERVER_TIMEOUT,
                default=str(DEFAULT_MEMORYLAYER_EMBED_SERVER_TIMEOUT),
            )
        )
        transport = v.environ(
            MEMORYLAYER_EMBED_TRANSPORT,
            default=DEFAULT_MEMORYLAYER_EMBED_TRANSPORT,
        ).lower()

        aether_connection = None
        aether_target = DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET
        if transport == TRANSPORT_AETHER:
            aether_connection = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
            if aether_connection is None:
                raise RuntimeError(
                    "MEMORYLAYER_EMBED_TRANSPORT=aether requires the "
                    "AetherServiceConnection extension (EXT_AETHER_SERVICE_CONNECTION) "
                    "to be initialised first; no connection found."
                )
            aether_target = v.environ(
                MEMORYLAYER_EMBED_AETHER_TARGET,
                default=DEFAULT_MEMORYLAYER_EMBED_AETHER_TARGET,
            )

        aether_stream_idle_ms = int(
            v.environ(
                MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS,
                default=str(DEFAULT_MEMORYLAYER_EMBED_AETHER_STREAM_IDLE_TIMEOUT_MS),
            )
        )

        logger.info(
            "Initializing embed server client: transport=%s, url=%s, aether_target=%s, timeout=%.0fs",
            transport,
            base_url if transport == TRANSPORT_HTTP else "<aether>",
            aether_target if transport == TRANSPORT_AETHER else "<n/a>",
            timeout,
        )
        return EmbedServerClient(
            base_url=base_url,
            timeout=timeout,
            logger=logger,
            transport=transport,
            aether_connection=aether_connection,
            aether_target=aether_target,
            aether_stream_idle_timeout_ms=aether_stream_idle_ms,
        )
