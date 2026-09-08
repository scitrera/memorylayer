"""Transport-fault retry on the embed client.

A pooled keep-alive connection closed by the far end fails on the NEXT request,
before any response byte arrives -- so the request never reached the server.
That is why the embed server logged clean 200s while document_embed raised
httpx.ReadError and failed the whole phase.

The distinction that matters: retry a broken CONNECTION, never a slow server
(retrying a timeout multiplies a budget that was already exhausted) and never a
rejected request (deterministic; retrying triples load and still fails).
"""

from __future__ import annotations

import httpx
import pytest

from memorylayer_server.services.document.embed_client import (
    _RETRYABLE_TRANSPORT,
    _TRANSPORT_RETRY_ATTEMPTS,
    EmbedServerClient,
)


def _client() -> EmbedServerClient:
    c = EmbedServerClient.__new__(EmbedServerClient)
    c.logger = __import__("logging").getLogger("test-embed-retry")
    return c


class _Sender:
    """Raises `exc` for the first `fail_times` calls, then returns a response."""

    def __init__(self, exc: Exception, fail_times: int):
        self._exc = exc
        self._fail_times = fail_times
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return httpx.Response(200, json={"ok": True})


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [
    httpx.ReadError("connection lost"),
    httpx.ConnectError("refused"),
    httpx.WriteError("broken pipe"),
    httpx.RemoteProtocolError("server disconnected"),
    httpx.ConnectTimeout("connect timed out"),
])
async def test_transport_faults_are_retried_and_recover(exc):
    send = _Sender(exc, fail_times=1)

    resp = await _client()._send_with_transport_retry(send, "POST", "/embed")

    assert resp.status_code == 200
    assert send.calls == 2  # failed once, succeeded on a fresh connection


@pytest.mark.asyncio
async def test_a_read_timeout_is_not_retried():
    """A slow embed must fail fast. Retrying spends a multiple of a budget that
    was already exhausted -- the reasoning that kept timeouts out of the
    transcription retry set too."""
    send = _Sender(httpx.ReadTimeout("too slow"), fail_times=99)

    with pytest.raises(httpx.ReadTimeout):
        await _client()._send_with_transport_retry(send, "POST", "/embed")

    assert send.calls == 1


@pytest.mark.asyncio
async def test_timeouts_are_excluded_from_the_retryable_family():
    # NetworkError covers dead sockets; TimeoutException is a sibling branch.
    # If httpx ever reparented them this would silently start retrying slow
    # requests, so assert the relationship rather than trusting it.
    assert not issubclass(httpx.ReadTimeout, _RETRYABLE_TRANSPORT)
    assert issubclass(httpx.ReadError, _RETRYABLE_TRANSPORT)


@pytest.mark.asyncio
async def test_persistent_failure_still_raises_after_the_attempt_budget():
    send = _Sender(httpx.ReadError("gone"), fail_times=99)

    with pytest.raises(httpx.ReadError):
        await _client()._send_with_transport_retry(send, "POST", "/embed")

    assert send.calls == _TRANSPORT_RETRY_ATTEMPTS


@pytest.mark.asyncio
async def test_a_healthy_request_is_sent_exactly_once():
    send = _Sender(httpx.ReadError("unused"), fail_times=0)

    await _client()._send_with_transport_retry(send, "GET", "/health")

    assert send.calls == 1
