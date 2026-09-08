"""HTTP transport — wraps ``httpx.AsyncClient`` with the SDK's expected headers."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Methods safe to retry automatically. POST is intentionally excluded — it is
# not idempotent in general, so a transient 5xx after the server may have
# already applied the write would risk duplicate side effects.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE", "PATCH"})

# HTTP status codes considered transient and worth retrying.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class HttpTransport:
    """Default SDK transport: one ``httpx.AsyncClient`` per SDK instance.

    Adds bounded retry-with-backoff for idempotent requests on transient
    failures (connection/timeout errors and retryable 5xx/429 responses),
    honoring the ``Retry-After`` header when present.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        session_id: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ) -> None:
        """
        Args:
            base_url: API base URL (already suffixed with the version prefix).
            api_key: Optional bearer token.
            session_id: Optional X-Session-ID header value.
            timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts for idempotent requests on
                transient failures (default: 3). 0 disables retries.
            backoff_factor: Base seconds for exponential backoff between
                retries (delay = backoff_factor * 2**attempt). ``Retry-After``
                on a response takes precedence when present.
        """
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if session_id:
            headers["X-Session-ID"] = session_id
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )
        self._max_retries = max(0, max_retries)
        self._backoff_factor = backoff_factor

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        """Underlying ``httpx.AsyncClient``.

        Exposed so the small set of bespoke SDK paths that need raw httpx
        (file uploads, NDJSON streaming) can keep working unchanged.  Aether
        transport intentionally does NOT expose an equivalent — those bespoke
        paths raise ``NotImplementedError`` when running on Aether transport.
        """
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        kwargs: dict[str, Any] = {"params": params, "headers": headers}
        if json is not None:
            kwargs["json"] = json
        if content is not None:
            kwargs["content"] = content

        # Only idempotent methods are retried, to avoid duplicate writes.
        retryable = method.upper() in _IDEMPOTENT_METHODS and self._max_retries > 0
        if not retryable:
            return await self._client.request(method, path, **kwargs)

        last_exc: Exception | None = None
        # attempt 0 is the initial try; up to max_retries additional attempts.
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(method, path, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.WriteError) as exc:
                last_exc = exc
                if attempt >= self._max_retries:
                    raise
                await self._sleep_before_retry(attempt, None)
                logger.debug("Retrying %s %s after transport error (attempt %d): %s", method, path, attempt + 1, exc)
                continue

            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_retries:
                await self._sleep_before_retry(attempt, response)
                logger.debug(
                    "Retrying %s %s after HTTP %d (attempt %d)", method, path, response.status_code, attempt + 1
                )
                continue

            return response

        # Unreachable in practice (loop either returns or raises), but keeps
        # type-checkers happy and re-raises the last transport error if any.
        if last_exc is not None:
            raise last_exc
        return response

    async def _sleep_before_retry(self, attempt: int, response: httpx.Response | None) -> None:
        """Sleep before the next retry, honoring Retry-After when present."""
        delay = self._backoff_factor * (2**attempt)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    # Retry-After may be an HTTP-date; fall back to backoff.
                    logger.debug("Non-numeric Retry-After header %r; using backoff", retry_after)
        await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self._client.aclose()

    def set_session(self, session_id: str) -> None:
        """Update the ``X-Session-ID`` header on the underlying httpx client."""
        self._client.headers["X-Session-ID"] = session_id

    def clear_session(self) -> None:
        if "X-Session-ID" in self._client.headers:
            del self._client.headers["X-Session-ID"]
