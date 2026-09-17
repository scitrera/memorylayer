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

import asyncio
import json as _json
import math
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
#: Transport faults worth another attempt: "the connection broke", not "the
#: server took too long". ``NetworkError`` is httpx's family for a dead socket
#: (ConnectError/ReadError/WriteError/CloseError) and excludes the timeout
#: family, so a genuinely slow embed still fails fast instead of being retried
#: into a multiple of its own budget.
_RETRYABLE_TRANSPORT = (
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
)
_TRANSPORT_RETRY_ATTEMPTS = 3
_TRANSPORT_RETRY_BACKOFF_SEC = 0.5

#: Idle lifetime for pooled connections, set BELOW the shortest keep-alive on
#: the path (uvicorn defaults to 5s) so this client always retires a connection
#: before the server can close it underneath us.
KEEPALIVE_EXPIRY_SEC = 4.0

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
        image_batch_size: int = 0,
        image_concurrency: int = 1,
        text_concurrency: int = 1,
        transcription_concurrency: int | None = None,
        transcription_providers: list[str] | None = None,
        text_batch_size: int = 0,
        text_batch_bytes: int = 0,
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
        if image_batch_size < 0:
            raise ValueError("image_batch_size must be nonnegative")
        self._image_batch_size = image_batch_size
        if not 1 <= image_concurrency <= 32:
            raise ValueError("image_concurrency must be between 1 and 32")
        self._image_concurrency = image_concurrency
        transcription_concurrency = image_concurrency if transcription_concurrency is None else transcription_concurrency
        for name, value in (("text_concurrency", text_concurrency), ("transcription_concurrency", transcription_concurrency)):
            if type(value) is not int or not 1 <= value <= 32:
                raise ValueError(f"{name} must be between 1 and 32")
        if transcription_providers is not None and (
                not isinstance(transcription_providers, list) or not transcription_providers or
                any(not isinstance(p, str) or not p.strip() for p in transcription_providers) or
                len(set(transcription_providers)) != len(transcription_providers)):
            raise ValueError("transcription_providers must be a nonempty list of unique provider names")
        self._transcription_providers = list(transcription_providers) if transcription_providers else None
        self._text_concurrency = text_concurrency
        self._transcription_concurrency = transcription_concurrency
        # A client is process-wide: concurrent documents share these role limits.
        self._embedding_slots = asyncio.Semaphore(max(image_concurrency, text_concurrency))
        self._transcription_slots = asyncio.Semaphore(transcription_concurrency)
        if text_batch_size < 0 or text_batch_bytes < 0:
            raise ValueError("text batch limits must be nonnegative")
        self._text_batch_size = text_batch_size
        self._text_batch_bytes = text_batch_bytes
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
            if self._client is not None:
                return  # idempotent: already connected (shared client reuse)
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                # Retire pooled connections before the far end does. Embedding
                # traffic is bursty -- a batch per document phase, then a gap --
                # so sockets sit idle long enough for the server (uvicorn
                # defaults to a 5s keep-alive) to close them. Reusing one that
                # is already gone fails before any response byte arrives.
                limits=httpx.Limits(keepalive_expiry=KEEPALIVE_EXPIRY_SEC),
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
    # Public request seam: shared request shape across transports.
    # ------------------------------------------------------------------

    async def _send_with_transport_retry(self, send, method: str, path: str):
        """Retry pre-send connection failures; never replay ambiguously accepted POSTs.

        A read error or closed keepalive connection does not prove the server
        failed to accept the request. GET diagnostics may be retried; billed
        inference POSTs only retry ConnectError/ConnectTimeout.
        """
        last: Exception | None = None
        for attempt_number in range(1, _TRANSPORT_RETRY_ATTEMPTS + 1):
            try:
                return await send()
            except _RETRYABLE_TRANSPORT as exc:
                if method.upper() != "GET" and not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                    raise
                last = exc
                if attempt_number == _TRANSPORT_RETRY_ATTEMPTS:
                    break
                delay = _TRANSPORT_RETRY_BACKOFF_SEC * (2 ** (attempt_number - 1))
                self.logger.info(
                    "embed server %s %s: connection dropped (%s: %s); retry %d/%d in %.1fs",
                    method.upper(), path, type(exc).__name__, exc or "no detail",
                    attempt_number, _TRANSPORT_RETRY_ATTEMPTS - 1, delay,
                )
                await asyncio.sleep(delay)
        raise last

    async def request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ) -> dict:
        """Public seam for calling arbitrary embed-server endpoints (e.g.
        enterprise/custom endpoints not hard-coded on this client).

        Issue a JSON request via the configured transport and return the
        decoded JSON body. Raises an HTTP-status-style error on non-2xx.
        """
        if self._transport == TRANSPORT_HTTP:
            assert self._client is not None, "EmbedServerClient.connect() not called"

            async def _send() -> httpx.Response:
                if method.upper() == "POST":
                    return await self._client.post(path, json=payload)
                if method.upper() == "GET":
                    return await self._client.get(path)
                # pragma: no cover - defensive; only POST/GET used today
                return await self._client.request(method, path, json=payload)

            resp = await self._send_with_transport_retry(_send, method, path)
            if resp.is_error:
                # Surface the concrete reason (e.g. "Embedding input length ...
                # exceeds max_model_len") in ML logs; ``raise_for_status`` only
                # reports the status code, which previously hid the cause.
                self.logger.error(
                    "embed server %s %s -> %s: %s",
                    method.upper(), path, resp.status_code, resp.text[:500],
                )
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

    async def request_stream(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
    ):
        """Public seam for streaming arbitrary embed-server endpoints (e.g.
        enterprise/custom endpoints not hard-coded on this client).

        Issue a streaming request and return an ``AsyncIterator[bytes]``.

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
            return await self.request_stream("POST", "/v1/chat/completions", request_payload)
        return await self.request_json("POST", "/v1/chat/completions", payload)

    async def completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ):
        """Legacy text completions. Same contract as :meth:`chat_completions`."""
        if stream:
            request_payload = {**payload, "stream": True}
            return await self.request_stream("POST", "/v1/completions", request_payload)
        return await self.request_json("POST", "/v1/completions", payload)

    async def _request_images(self, path: str, payload: dict) -> dict:
        """Split bounded serving requests while preserving batch-relative indexes."""
        images = payload["images"]
        limit = self._image_batch_size
        transcription = path == "/v1/transcribe"
        concurrency = self._transcription_concurrency if transcription else self._image_concurrency
        slots = self._transcription_slots if transcription else self._embedding_slots
        if not limit or len(images) <= limit:
            async with slots:
                return await self._request_image_batch(path, payload)
        key, index_key = ("results", "page_index") if path == "/v1/transcribe" else ("data", "index")
        merged: dict = {key: []}
        stats: dict = {}
        offsets = iter(range(0, len(images), limit))
        parts = {}

        async def worker():
            for offset in offsets:
                async with slots:
                    parts[offset] = await self._request_image_batch(path, {**payload, "images": images[offset:offset + limit]})

        if concurrency == 1:
            await worker()
        else:
            async with asyncio.TaskGroup() as group:
                for _ in range(min(concurrency, math.ceil(len(images) / limit))):
                    group.create_task(worker())
        for offset, part in sorted(parts.items()):
            seen = set()
            for entry in part[key]:
                index = entry[index_key]
                if type(index) is not int or not 0 <= index < min(limit, len(images) - offset) or index in seen:
                    raise ValueError("Invalid image index from embed server")
                seen.add(index)
                merged[key].append({**entry, index_key: index + offset})
            for name, value in part.get("stats", {}).items():
                if isinstance(value, (float, int)):
                    stats[name] = stats.get(name, 0) + value
        if stats:
            merged["stats"] = stats
        return merged

    async def _request_image_batch(self, path: str, payload: dict) -> dict:
        if path != "/v1/transcribe" or not self._transcription_providers:
            return await self.request_json("POST", path, payload)
        # Explicit provider selection disables the remote default cascade. Only
        # confirmed failed pages advance; transport/protocol failures propagate.
        images = payload["images"]
        pending = list(range(len(images)))
        results = {}
        totals = {"total_tokens_in": 0, "total_tokens_out": 0, "total_latency_ms": 0}
        for provider in self._transcription_providers:
            response = await self.request_json("POST", path,
                {**payload, "images": [images[i] for i in pending], "provider": provider})
            entries = response["results"]
            indexes = [entry.get("page_index") for entry in entries]
            if (any(type(i) is not int for i in indexes) or sorted(indexes) != list(range(len(pending)))
                    or any(type(entry.get("success")) is not bool for entry in entries)):
                raise ValueError("Invalid transcription page indexes or success flags")
            remaining = []
            for entry in entries:
                index = pending[entry["page_index"]]
                attempts = results.get(index, {}).get("attempts", []) + entry.get("attempts", [])
                results[index] = {**entry, "page_index": index, "attempts": attempts}
                if not entry["success"]:
                    remaining.append(index)
            for name in totals:
                totals[name] += response.get("stats", {}).get(name, 0)
            pending = sorted(remaining)
            if not pending:
                break
        ordered = [results[i] for i in range(len(images))]
        successful = sum(entry["success"] for entry in ordered)
        return {"results": ordered, "stats": {**totals, "total_pages": len(images),
                "successful_pages": successful, "failed_pages": len(images) - successful}}

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
        return await self._request_images("/v1/transcribe", payload)

    async def _request_texts(self, path: str, payload: dict) -> dict:
        texts = payload["input"]
        if not self._text_batch_size and not self._text_batch_bytes:
            async with self._embedding_slots:
                return await self.request_json("POST", path, payload)
        batches: list[list[str]] = []
        batch: list[str] = []
        total_bytes = 0
        for text in texts:
            size = len(text.encode("utf-8"))
            if self._text_batch_bytes and size > self._text_batch_bytes:
                raise ValueError("Text exceeds embed-server byte budget; split the text before embedding")
            if batch and ((self._text_batch_size and len(batch) >= self._text_batch_size)
                          or (self._text_batch_bytes and total_bytes + size > self._text_batch_bytes)):
                batches.append(batch)
                batch, total_bytes = [], 0
            batch.append(text)
            total_bytes += size
        if batch:
            batches.append(batch)
        parts = {}
        pending = iter(enumerate(batches))

        async def worker():
            for number, batch in pending:
                async with self._embedding_slots:
                    parts[number] = await self.request_json("POST", path, {**payload, "input": batch})

        if self._text_concurrency == 1:
            await worker()
        else:
            async with asyncio.TaskGroup() as group:
                for _ in range(min(self._text_concurrency, len(batches))):
                    group.create_task(worker())
        merged: dict = {"data": []}
        offset = 0
        for number, batch in enumerate(batches):
            part = parts[number]
            seen = set()
            for entry in part["data"]:
                index = entry["index"]
                if type(index) is not int or not 0 <= index < len(batch) or index in seen:
                    raise ValueError("Invalid text index from embed server")
                seen.add(index)
                merged["data"].append({**entry, "index": index + offset})
            if len(seen) != len(batch):
                raise ValueError("Missing text index from embed server")
            offset += len(batch)
        return merged

    async def embed_texts(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        """Get single-vector embeddings for texts.

        ``dimensions`` (optional) forwards the OpenAI ``dimensions`` field so a
        Matryoshka-capable server truncates each vector to the requested size
        (mirrors the enterprise tokenary client). Omitted by default so servers
        that do not support the field are unaffected.

        With ``text_batch_bytes`` configured, oversized inputs are split without
        dropping characters. Chunk vectors are weighted by UTF-8 byte length,
        summed and L2-normalized into one page vector. Short inputs retain their
        original server vector. This is an approximate page representation; the
        source transcript is never changed. Keep the byte budget stable for an
        existing index, or explicitly re-embed when changing this policy.
        """
        # Page transcripts can exceed the request budget. Embed every UTF-8-safe
        # chunk and pool back to one vector per original input; never truncate
        # the source or silently shift vectors onto the wrong page.
        chunks, groups = [], []
        for text in texts:
            pieces = _split_embedding_text(text, self._text_batch_bytes)
            groups.append((len(chunks), len(pieces)))
            chunks.extend(pieces)
        if not chunks:
            return []
        payload: dict = {"input": chunks}
        if dimensions is not None:
            payload["dimensions"] = dimensions

        self.logger.debug("Embedding %d texts in %d chunks (dimensions=%s)", len(texts), len(chunks), dimensions)
        data = await self._request_texts("/v1/embeddings", payload)
        entries = data["data"]
        indices = [entry["index"] for entry in entries]
        if (any(type(i) is not int for i in indices)
                or sorted(indices) != list(range(len(chunks)))):
            raise ValueError("Embed server must return exactly one vector per text chunk")
        vectors = [item["embedding"] for item in sorted(entries, key=lambda x: x["index"])]
        result = []
        for start, count in groups:
            if count == 1:
                result.append(vectors[start])  # Preserve existing short-input vectors.
                continue
            group = vectors[start:start + count]
            width = len(group[0])
            if not width or any(len(vector) != width for vector in group):
                raise ValueError("Inconsistent embedding dimensions across text chunks")
            weights = [len(chunk.encode("utf-8")) for chunk in chunks[start:start + count]]
            pooled = [math.fsum(vector[i] * weight for vector, weight in zip(group, weights))
                      for i in range(width)]
            norm = math.hypot(*pooled)
            if not math.isfinite(norm) or norm == 0:
                raise ValueError("Invalid pooled text embedding")
            result.append([value / norm for value in pooled])
        return result

    async def embed_texts_multivector(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[dict]:
        """Get multi-vector embeddings for texts via ColPali."""
        payload = {"input": texts, "input_type": input_type}

        self.logger.debug("Embedding %d texts (multi-vector, type=%s)", len(texts), input_type)
        data = await self._request_texts("/v1/embeddings/multi", payload)
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [{"vectors": item["vectors"], "num_vectors": item["num_vectors"]} for item in sorted_data]

    async def embed_images(self, images_b64: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        """Get image vectors from the server's single-vector vision model."""
        payload: dict = {"images": images_b64, "mode": "single"}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        data = await self._request_images("/v1/embeddings/images", payload)
        return [item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"])]

    async def embed_images_multivector(self, images_b64: list[str]) -> list[dict]:
        """Get multi-vector embeddings from images via ColPali."""
        payload = {"images": images_b64, "mode": "multi"}

        self.logger.debug("Embedding %d images (multi-vector)", len(images_b64))
        data = await self._request_images("/v1/embeddings/images", payload)
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
        data = await self.request_json("POST", "/v1/score", payload)
        return data["scores"]


def _split_embedding_text(text: str, byte_limit: int) -> list[str]:
    """Losslessly split at UTF-8 boundaries, preferring nearby whitespace.

    Long inputs use byte-weighted, L2-normalized pooling in ``embed_texts``.
    The byte limit still applies to the entire outgoing batch in _request_texts.
    """
    encoded = text.encode("utf-8")
    if not byte_limit or len(encoded) <= byte_limit:
        return [text]
    chunks = []
    start = 0
    while start < len(encoded):
        chunk = encoded[start:start + byte_limit].decode("utf-8", errors="ignore")
        if not chunk:
            raise ValueError("Text byte budget cannot hold one UTF-8 character")
        if start + len(chunk.encode("utf-8")) < len(encoded):
            boundary = next((i + 1 for i in range(len(chunk) - 1, len(chunk) // 2, -1)
                             if chunk[i].isspace()), None)
            if boundary:
                chunk = chunk[:boundary]
        chunks.append(chunk)
        start += len(chunk.encode("utf-8"))
    return chunks


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
            image_batch_size=int(v.environ("MEMORYLAYER_EMBED_IMAGE_BATCH_SIZE", default=0)),
            image_concurrency=int(v.environ("MEMORYLAYER_EMBED_IMAGE_CONCURRENCY", default=1)),
            text_concurrency=int(v.environ("MEMORYLAYER_EMBED_TEXT_CONCURRENCY", default=1)),
            transcription_concurrency=int(v.environ("MEMORYLAYER_EMBED_TRANSCRIPTION_CONCURRENCY",
                default=v.environ("MEMORYLAYER_EMBED_IMAGE_CONCURRENCY", default=1))),
            transcription_providers=(str(v.environ("MEMORYLAYER_EMBED_TRANSCRIPTION_PROVIDERS", default="")).split(",")
                if v.environ("MEMORYLAYER_EMBED_TRANSCRIPTION_PROVIDERS", default="") else None),
            text_batch_size=int(v.environ("MEMORYLAYER_EMBED_TEXT_BATCH_SIZE", default=0)),
            text_batch_bytes=int(v.environ("MEMORYLAYER_EMBED_TEXT_BATCH_BYTES", default=0)),
            base_url=base_url,
            timeout=timeout,
            logger=logger,
            transport=transport,
            aether_connection=aether_connection,
            aether_target=aether_target,
            aether_stream_idle_timeout_ms=aether_stream_idle_ms,
        )
