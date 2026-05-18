"""In-process integration tests for the full ProxyHttpTerminator → ASGI bridge → FastAPI chain.

This is the most valuable test in Phase 4: it stitches together the three
independently-tested pieces — the Aether SDK's ``ProxyHttpTerminator``
(Phase 2a), the ``asgi_bridge`` (Phase 2b), and a real FastAPI app — and
exercises the full dispatch path from a ``ProxyHttpRequest`` envelope all
the way through to a FastAPI response, without touching the network or
spinning up Docker infrastructure.

The injection mechanism mirrors ``sdk/python-client/tests/test_proxy_terminator.py``'s
``_AsyncClientStub`` / ``_get_terminator_dispatcher`` pattern: we build a
minimal client stub that owns an ``asyncio.Queue`` (the terminator's
outbound channel), then push ``ProxyHttpRequest`` envelopes via
``dispatcher.handle_request()``, and drain the queue to assert on the
``ProxyHttpResponse``.

What is tested here (verification matrix from the Phase 4 plan):

| Test                                              | AuthorizationContext          | Resolver      | Expected                              |
|---------------------------------------------------|-------------------------------|---------------|---------------------------------------|
| test_direct_mode_passes_through                   | mode=direct / no authz        | None          | 200; mode header is ``direct``        |
| test_obo_mode_no_resolver_rejected_default_policy | mode=obo, grant=g1, sub=usr-a | None          | ProxyError ACL_DENIED; handler not called |
| test_obo_mode_with_resolver_passes_extended_headers | mode=obo, grant=g1, sub=usr-a | stub (success) | 200; subject-id, max-access-level, workspace-scope headers present |
| test_strict_mode_strips_inbound_xauth_spoofs      | mode=obo                      | stub (success) | 200; FastAPI sees subject-id from envelope, NOT inbound spoof |
| test_path_filtering_denies_unlisted_path          | any                           | any           | ProxyError ACL_DENIED for /admin/secret |

OBO resolution is exercised through a small ``_StubResolver`` that conforms
to ``AuthorityResolverProtocol`` and returns a fixed ``ResolvedAuthorityInfo``
without hitting the real Aether SDK RPC — no live infrastructure required.

Out of scope (deferred, documented at module bottom):
- ``backend-future/tests/integration/test_cowork_memorylayer_obo.py`` — the
  cowork-level end-to-end test depends on Phase 5 (SDK Aether transport
  mode), which is not yet implemented.
- Live-gateway smoke check — requires a real Aether gateway instance (dev
  compose). See the manual smoke checklist at the end of this file.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request, Response
from scitrera_aether_client.authority import ResolvedAuthorityInfo
from scitrera_aether_client.proto import aether_pb2
from scitrera_aether_client.proxy_terminator import (
    MintedRequest,
    ProxyHttpTerminator,
    _get_terminator_dispatcher,
)

from memorylayer_server.services.aether_service.asgi_bridge import asgi_dispatch

# The ``integration`` marker is applied at the directory level via
# ``tests/integration/conftest.py``'s ``pytest_collection_modifyitems``
# hook, so this module is automatically deselected by ``-m "not integration"``.

# ---------------------------------------------------------------------------
# Minimal FastAPI app used as the backend across all tests
# ---------------------------------------------------------------------------

_app = FastAPI()


@_app.get("/healthz")
async def healthz(request: Request):
    """Echoes authority headers so tests can assert on what the handler saw."""
    mode = request.headers.get("x-auth-authority-mode", "missing")
    subject = request.headers.get("x-auth-subject-id", "none")
    max_level = request.headers.get("x-auth-max-access-level", "none")
    ws_scope = request.headers.get("x-auth-workspace-scope", "none")
    payload = json.dumps(
        {
            "mode": mode,
            "subject_id": subject,
            "max_access_level": max_level,
            "workspace_scope": ws_scope,
        }
    ).encode()
    return Response(content=payload, media_type="application/json", status_code=200)


# ---------------------------------------------------------------------------
# Fake async client stub (mirrors test_proxy_terminator._AsyncClientStub)
# ---------------------------------------------------------------------------


class _AsyncClientStub:
    """Minimal async client surface the terminator dispatcher needs.

    Owns an ``asyncio.Queue`` that the terminator writes outbound
    ``UpstreamMessage`` frames into.  Tests drain this queue to inspect
    the ``ProxyHttpResponse``.
    """

    def __init__(self) -> None:
        self._request_queue: asyncio.Queue = asyncio.Queue()
        self._init_msg = None  # No identity needed for these tests.

    @property
    def identity(self):
        """Return None — no service identity injected for these tests."""
        from scitrera_aether_client.client_async import BaseAsyncAetherClient

        return BaseAsyncAetherClient.identity.fget(self)

    async def drain_upstream(self) -> list[aether_pb2.UpstreamMessage]:
        out: list[aether_pb2.UpstreamMessage] = []
        while True:
            try:
                out.append(self._request_queue.get_nowait())
            except asyncio.QueueEmpty:
                return out


# ---------------------------------------------------------------------------
# Stub authority resolver
# ---------------------------------------------------------------------------


class _StubResolver:
    """In-process ``AuthorityResolverProtocol`` implementation.

    Returns a fixed ``ResolvedAuthorityInfo`` (or ``None``) without any
    network call, so tests exercise the full terminator header-minting path
    without a live Aether gateway.
    """

    def __init__(self, info: ResolvedAuthorityInfo | None = None) -> None:
        self._info = info
        self.calls: list[dict] = []

    async def resolve(
        self,
        grant_id: str,
        subject_type: str,
        subject_id: str,
        *,
        actor_type: str | None = None,
        actor_id: str | None = None,
    ) -> ResolvedAuthorityInfo | None:
        self.calls.append(
            {
                "grant_id": grant_id,
                "subject_type": subject_type,
                "subject_id": subject_id,
            }
        )
        return self._info


def _good_resolver(
    subject_id: str = "user-a",
    max_access_level: int = 20,
    workspace_scope: tuple[str, ...] = ("ws-1", "ws-2"),
) -> _StubResolver:
    info = ResolvedAuthorityInfo(
        grant_id="g1",
        subject_type="user",
        subject_id=subject_id,
        root_subject_type="user",
        root_subject_id=subject_id,
        audience_type="service",
        audience_id="sv::memorylayer::main",
        max_access_level=max_access_level,
        workspace_scope=workspace_scope,
        expires_at=0,
        revoked=False,
    )
    return _StubResolver(info=info)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_request(
    request_id: str = "req-1",
    method: str = "GET",
    path: str = "/healthz",
    headers: dict | None = None,
    body: bytes = b"",
    authorization: aether_pb2.AuthorizationContext | None = None,
) -> aether_pb2.ProxyHttpRequest:
    req = aether_pb2.ProxyHttpRequest(
        request_id=request_id,
        target_topic="sv::memorylayer::main",
        method=method,
        path=path,
        body=body,
    )
    if headers:
        for k, v in headers.items():
            req.headers[k] = v
    if authorization is not None:
        req.authorization.CopyFrom(authorization)
    return req


def _obo_authz(
    grant_id: str = "g1",
    subject_id: str = "user-a",
) -> aether_pb2.AuthorizationContext:
    return aether_pb2.AuthorizationContext(
        authority_mode="on_behalf_of",
        subject=aether_pb2.PrincipalRef(principal_type="user", principal_id=subject_id),
        grant_id=grant_id,
    )


async def _make_terminator(
    client: _AsyncClientStub,
    resolver: _StubResolver | None = None,
    allow_paths: list[str] | None = None,
    obo_policy: str = "require_resolver",
) -> ProxyHttpTerminator:
    """Construct and start a ProxyHttpTerminator backed by ``_app`` via asgi_bridge."""

    async def _handler(req: MintedRequest):
        return await asgi_dispatch(_app, req)

    term = ProxyHttpTerminator(
        client=client,
        handler=_handler,
        allow_paths=allow_paths or ["/healthz", "/v1/*"],
        header_mode="strict",
        resolver=resolver,
        obo_policy=obo_policy,
    )
    await term.start()
    return term


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_direct_mode_passes_through():
    """Direct-mode request (no AuthorizationContext) reaches FastAPI and returns 200.

    The handler receives ``X-Auth-Authority-Mode: direct`` from the
    terminator's header minting; no subject headers are set.
    """
    client = _AsyncClientStub()
    term = await _make_terminator(client)

    dispatcher = _get_terminator_dispatcher(client)
    req = _build_request("req-direct", method="GET", path="/healthz")
    await dispatcher.handle_request(req)

    msgs = await client.drain_upstream()
    assert len(msgs) == 1
    resp = msgs[0].proxy_http_response
    assert not resp.HasField("error"), f"unexpected error: {resp.error}"
    assert resp.status_code == 200

    data = json.loads(resp.body)
    assert data["mode"] == "direct"
    assert data["subject_id"] == "none"

    await term.stop()


async def test_obo_mode_no_resolver_rejected_default_policy():
    """OBO request with no resolver is rejected (``require_resolver`` default).

    The handler must NOT be invoked — the terminator returns an ACL_DENIED
    ProxyError before dispatch.  This is the critical security gate: without
    a resolver the terminator cannot validate or mint full authority headers,
    so it refuses rather than forwarding a partially-minted request.
    """
    client = _AsyncClientStub()
    # No resolver passed — require_resolver is the default.
    term = await _make_terminator(client, resolver=None, obo_policy="require_resolver")

    dispatcher = _get_terminator_dispatcher(client)
    req = _build_request(
        "req-obo-no-resolver",
        path="/healthz",
        authorization=_obo_authz(grant_id="g1", subject_id="user-a"),
    )
    await dispatcher.handle_request(req)

    msgs = await client.drain_upstream()
    assert len(msgs) == 1
    resp = msgs[0].proxy_http_response
    assert resp.HasField("error")
    assert resp.error.kind == aether_pb2.ProxyError.ACL_DENIED

    await term.stop()


async def test_obo_mode_with_resolver_passes_extended_headers():
    """Happy OBO path: resolver returns grant info; FastAPI sees all minted headers.

    Verifies the full chain from envelope AuthorizationContext through the
    resolver overlay to the FastAPI handler: subject-id, max-access-level,
    and workspace-scope must all appear in the headers the handler reads.
    """
    client = _AsyncClientStub()
    resolver = _good_resolver(subject_id="user-a", max_access_level=20, workspace_scope=("ws-1", "ws-2"))
    term = await _make_terminator(client, resolver=resolver)

    dispatcher = _get_terminator_dispatcher(client)
    req = _build_request(
        "req-obo-ok",
        path="/healthz",
        authorization=_obo_authz(grant_id="g1", subject_id="user-a"),
    )
    await dispatcher.handle_request(req)

    msgs = await client.drain_upstream()
    assert len(msgs) == 1
    resp = msgs[0].proxy_http_response
    assert not resp.HasField("error"), f"unexpected error: {resp.error}"
    assert resp.status_code == 200

    data = json.loads(resp.body)
    assert data["mode"] == "on_behalf_of"
    assert data["subject_id"] == "user-a"
    assert data["max_access_level"] == "20"
    assert data["workspace_scope"] == "ws-1,ws-2"

    await term.stop()


async def test_strict_mode_strips_inbound_xauth_spoofs():
    """Inbound ``X-Auth-*`` spoofs are stripped; handler sees envelope-derived values only.

    A malicious caller who injects ``X-Auth-Subject-ID: attacker`` in the
    inbound headers must not be able to impersonate a different subject.
    Strict mode strips ALL inbound X-Auth-*/X-Aether-* headers and replaces
    them with values minted from the validated ``AuthorizationContext``.
    """
    client = _AsyncClientStub()
    resolver = _good_resolver(subject_id="real-user", max_access_level=10)
    term = await _make_terminator(client, resolver=resolver)

    dispatcher = _get_terminator_dispatcher(client)
    req = _build_request(
        "req-spoof",
        path="/healthz",
        headers={
            # Attacker tries to claim a different subject identity.
            "X-Auth-Subject-ID": "attacker",
            "X-Auth-Authority-Mode": "direct",  # tries to downgrade mode
            "X-Aether-Grant-ID": "spoofed-grant",
        },
        authorization=_obo_authz(grant_id="g1", subject_id="real-user"),
    )
    await dispatcher.handle_request(req)

    msgs = await client.drain_upstream()
    assert len(msgs) == 1
    resp = msgs[0].proxy_http_response
    assert not resp.HasField("error"), f"unexpected error: {resp.error}"
    assert resp.status_code == 200

    data = json.loads(resp.body)
    # Spoofed subject rejected; envelope-minted subject-id wins.
    assert data["subject_id"] == "real-user"
    # Mode is OBO (from envelope), not the spoofed "direct".
    assert data["mode"] == "on_behalf_of"

    await term.stop()


async def test_path_filtering_denies_unlisted_path():
    """Requests to paths outside ``allow_paths`` are denied with ACL_DENIED.

    The terminator is configured with ``["/healthz", "/v1/*"]``.  A request
    to ``/admin/secret`` must be rejected before the handler runs.
    """
    client = _AsyncClientStub()
    term = await _make_terminator(client, allow_paths=["/healthz", "/v1/*"], obo_policy="allow_partial")

    dispatcher = _get_terminator_dispatcher(client)
    req = _build_request("req-denied", method="GET", path="/admin/secret")
    await dispatcher.handle_request(req)

    msgs = await client.drain_upstream()
    assert len(msgs) == 1
    resp = msgs[0].proxy_http_response
    assert resp.HasField("error")
    assert resp.error.kind == aether_pb2.ProxyError.ACL_DENIED

    await term.stop()


# ---------------------------------------------------------------------------
# Deferred / out-of-scope notes
# ---------------------------------------------------------------------------
#
# 1. backend-future/tests/integration/test_cowork_memorylayer_obo.py
#    DEFERRED — requires Phase 5 (MemoryLayer SDK Aether transport mode).
#    The cowork ``MemoryLayerBackend`` calls the SDK; the SDK needs an
#    Aether transport branch to route those calls through the terminator
#    rather than plain HTTP.  Phase 5 is not yet implemented.
#
# 2. Live-gateway smoke check (from the plan's "Verification (end-to-end)" section).
#    These tests run entirely in-process.  The manual verification sequence
#    after a real dev compose stack is up:
#
#    cd /home/drew/scitrera-app-monorepo2/backend-future
#    ./dev-platform.sh   # bring up Aether gateway with sv::memorylayer::* certs + ACL
#    ./dev.sh            # bring up MemoryLayer + deps
#
#    # Confirm MemoryLayer registers as a service, not an agent:
#    docker logs ml-server | grep -E "AetherServiceConnection ready|ProxyHttpTerminator registered"
#    # Expected: single line per — sv.memorylayer.<specifier>
#
#    # Drive a direct-mode proxy call from a Python REPL:
#    # from scitrera_aether_client import AsyncServiceClient
#    # from scitrera_aether_client.proxy import proxy_http_async
#    # client = AsyncServiceClient(...)
#    # await client.connect("localhost:50051")
#    # resp = await proxy_http_async(client, "sv::memorylayer::main", "GET", "/healthz")
#    # assert resp.status_code == 200
#
#    # Drive an OBO call (requires a real grant in the dev Aether instance):
#    # authz = acting_for(grant_id=<grant>, subject=PrincipalRef(...))
#    # resp = await proxy_http_async(client, target, "GET", "/v1/memories", authorization=authz)
#    # assert resp.status_code == 200
#
#    # Confirm identity introspection shows NO ag::*::memorylayer::* registration.
#    # (No CLI tool exists yet; inspect Aether gateway logs for "RegisterPrincipal"
#    #  events and confirm principal_type == SERVICE not AGENT.)
