"""SECURITY: owner-scoped user chat threads (_user_chat sentinel).

Regression tests for the cross-user chat leak: user-owned chat threads all live
in the shared ``_user_chat`` sentinel workspace and the harness sends a shared
client thread_id ("_default") for every user. Isolation is enforced by SCOPING
on the OBO human subject — thread identity is (workspace_id, user_id, id) with
``id`` stored VERBATIM and an internal surrogate ``row_id`` as the physical PK.
Without the owner scope, every user's ("_user_chat", "_default") is ONE shared
row and User A/B chats converge.

These tests exercise the API-layer chokepoint (``api/v1/chat.py``) directly by
calling the route functions with:
- a real chat service + audit service (from the isolated test framework), and
- crafted ``RequestContext`` objects (OBO subject = the human user) returned by
  a mock auth service.

The stored surrogate ``row_id`` is INTERNAL and must NEVER appear in a response,
so tests re-read a user's thread via the owner-scoped
``get_thread(ws, "_default", user_id=<human>)`` rather than any mangled id.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from scitrera_app_framework import get_extension

from memorylayer_server.api.v1 import chat as chat_api
from memorylayer_server.models.auth import (
    AuthorityContext,
    PrincipalRef,
    RequestContext,
)
from memorylayer_server.models.chat import USER_CHAT_HOME_WORKSPACE
from memorylayer_server.services.audit import EXT_AUDIT_SERVICE
from memorylayer_server.services.chat import EXT_CHAT_SERVICE


# ---------------------------------------------------------------------------
# API chokepoint fixtures
# ---------------------------------------------------------------------------


def _ctx_for_user(user_email: str, workspace_id: str, tenant_id: str = "t_scoping") -> RequestContext:
    """Build an OBO RequestContext whose human subject is ``user_email`` and whose
    *actor* is the sahara sandbox (must NOT be used for scoping)."""
    return RequestContext(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=user_email,  # proxy echoes subject here under OBO
        actor=PrincipalRef(type="agent", id="ag::_sandbox::sahara::sbx-123"),
        authority=AuthorityContext(
            mode="on_behalf_of",
            subject=PrincipalRef(type="user", id=user_email),
            grant_id="g_test",
        ),
    )


def _ctx_no_user(workspace_id: str, tenant_id: str = "t_scoping") -> RequestContext:
    """A context with no resolvable OBO human user (e.g. a bare service actor)."""
    return RequestContext(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        user_id=None,
        actor=PrincipalRef(type="service", id="sv::sandbox-provider"),
        authority=AuthorityContext(mode="direct"),
    )


class _AuthStub:
    """Mock AuthenticationService whose build_context returns a preset context."""

    def __init__(self, ctx: RequestContext):
        self._ctx = ctx

    async def build_context(self, request, body=None):
        return self._ctx


def _authz_stub() -> MagicMock:
    authz = MagicMock()
    authz.require_authorization = AsyncMock(return_value=None)
    return authz


def _http_request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    req.query_params = {}
    return req


@pytest_asyncio.fixture(autouse=True)
async def _ensure_workspaces(v):
    """Create the workspaces used by these tests.

    In production the auth service's ``resolve_workspace`` auto-creates the
    workspace before any chat write; these tests call the route functions
    directly (bypassing auth), so we ensure the FK target rows exist here.
    """
    from datetime import UTC, datetime

    from memorylayer_server.models.workspace import Workspace
    from memorylayer_server.services.storage import EXT_STORAGE_BACKEND

    storage = get_extension(EXT_STORAGE_BACKEND, v)
    for ws in (USER_CHAT_HOME_WORKSPACE, "shared_project_ws"):
        if not await storage.get_workspace(ws):
            await storage.create_workspace(
                Workspace(
                    id=ws,
                    tenant_id="t_scoping",
                    name=ws,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )


@pytest_asyncio.fixture
async def chat_service(v):
    return get_extension(EXT_CHAT_SERVICE, v)


@pytest_asyncio.fixture
async def audit_service(v):
    return get_extension(EXT_AUDIT_SERVICE, v)


@pytest.fixture
def logger():
    import logging

    return logging.getLogger("test_chat_user_scoping")


# ---------------------------------------------------------------------------
# (a) two different OBO users appending the SAME client thread_id "_default"
#     get DIFFERENT stored threads; each read returns only their own messages.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_users_same_client_thread_are_isolated(chat_service, audit_service, logger):
    from memorylayer_server.api.v1.schemas import MessageCreateRequest, MessagesAppendRequest

    alice_ctx = _ctx_for_user("alice@example.com", USER_CHAT_HOME_WORKSPACE)
    bob_ctx = _ctx_for_user("bob@example.com", USER_CHAT_HOME_WORKSPACE)

    async def _append(ctx, text):
        return await chat_api.append_messages(
            http_request=_http_request(),
            thread_id="_default",
            request=MessagesAppendRequest(
                messages=[MessageCreateRequest(role="user", content=text)]
            ),
            workspace_id=USER_CHAT_HOME_WORKSPACE,
            auth_service=_AuthStub(ctx),
            authz_service=_authz_stub(),
            chat_service=chat_service,
            audit_service=audit_service,
            logger=logger,
        )

    resp_a = await _append(alice_ctx, "alice-secret")
    resp_b = await _append(bob_ctx, "bob-secret")

    # The harness contract is preserved: both see "_default" echoed back.
    assert resp_a.thread_id == "_default"
    assert resp_b.thread_id == "_default"

    # Distinct users writing the SAME client id "_default" get SEPARATE stored
    # threads — each resolved by the owner-scoped (workspace, user_id, id) lookup.
    alice_thread = await chat_service.get_thread(
        USER_CHAT_HOME_WORKSPACE, "_default", user_id="alice@example.com"
    )
    bob_thread = await chat_service.get_thread(
        USER_CHAT_HOME_WORKSPACE, "_default", user_id="bob@example.com"
    )
    assert alice_thread is not None and bob_thread is not None
    # Both present the SAME verbatim client id, but they are distinct rows.
    assert alice_thread.id == bob_thread.id == "_default"
    # The stored surrogate row_id never leaks into the client-facing id.
    assert "::u::" not in alice_thread.id
    # Ownership stamped to the OBO human, not the sahara actor.
    assert alice_thread.user_id == "alice@example.com"
    assert bob_thread.user_id == "bob@example.com"

    # Each user reads ONLY their own messages via the API chokepoint.
    async def _get_messages(ctx):
        return await chat_api.get_messages(
            http_request=_http_request(),
            thread_id="_default",
            workspace_id=USER_CHAT_HOME_WORKSPACE,
            limit=100,
            offset=0,
            after_index=None,
            order="asc",
            auth_service=_AuthStub(ctx),
            authz_service=_authz_stub(),
            chat_service=chat_service,
            audit_service=audit_service,
            logger=logger,
        )

    alice_msgs = await _get_messages(alice_ctx)
    bob_msgs = await _get_messages(bob_ctx)

    def _texts(resp):
        return [m.content for m in resp.messages]

    assert _texts(alice_msgs) == ["alice-secret"]
    assert _texts(bob_msgs) == ["bob-secret"]
    # Response ids round-trip back to the client id.
    assert alice_msgs.thread_id == "_default"
    assert all(m.thread_id == "_default" for m in alice_msgs.messages)


# ---------------------------------------------------------------------------
# (b) a workspace-homed thread: id stored verbatim (no materialization) AND
#     owner-keyed per-user, same as the sentinel. Workspace-homed is NOT a
#     collaboration feature — see _scope_user_id in api/v1/chat.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_homed_thread_verbatim_id_and_owner_keyed(chat_service, audit_service, logger):
    from memorylayer_server.api.v1.schemas import MessageCreateRequest, MessagesAppendRequest

    ws = "shared_project_ws"
    ctx = _ctx_for_user("alice@example.com", ws)

    resp = await chat_api.append_messages(
        http_request=_http_request(),
        thread_id="shared-thread",
        request=MessagesAppendRequest(
            messages=[MessageCreateRequest(role="user", content="hello team")]
        ),
        workspace_id=ws,
        auth_service=_AuthStub(ctx),
        authz_service=_authz_stub(),
        chat_service=chat_service,
        audit_service=audit_service,
        logger=logger,
    )
    assert resp.thread_id == "shared-thread"
    # The stored id is the raw client id — NOT materialized. Owner scoping is a
    # COLUMN, so the lookup supplies the owner key rather than expecting an
    # unscoped row: workspace-homed threads are keyed (workspace_id, user_id, id)
    # exactly like sentinel threads. What differs between the homes is only the
    # id materialization — which is absent in both, and is what this test guards.
    stored = await chat_service.get_thread(ws, "shared-thread", user_id="alice@example.com")
    assert stored is not None
    assert stored.id == "shared-thread"
    # No per-user suffix was appended.
    assert "::u::" not in stored.id
    # Workspace-homed threads are per-user, NOT shared across the workspace's
    # members: the append stamps the OBO human as owner and labels the row for
    # its home, so the auto-title task (and every other thread-id-keyed route)
    # resolves the same single row the create path wrote.
    assert stored.user_id == "alice@example.com"
    assert stored.ownership == "workspace"
    # ...and therefore another member of the same workspace does not see it.
    assert await chat_service.get_thread(ws, "shared-thread", user_id="bob@example.com") is None


# ---------------------------------------------------------------------------
# (c) _user_chat access with no OBO user is DENIED (fail-closed, 401).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sentinel_without_obo_user_denied(chat_service, audit_service, logger):
    from fastapi import HTTPException

    from memorylayer_server.api.v1.schemas import MessageCreateRequest, MessagesAppendRequest

    ctx = _ctx_no_user(USER_CHAT_HOME_WORKSPACE)

    with pytest.raises(HTTPException) as exc:
        await chat_api.append_messages(
            http_request=_http_request(),
            thread_id="_default",
            request=MessagesAppendRequest(
                messages=[MessageCreateRequest(role="user", content="who am i")]
            ),
            workspace_id=USER_CHAT_HOME_WORKSPACE,
            auth_service=_AuthStub(ctx),
            authz_service=_authz_stub(),
            chat_service=chat_service,
            audit_service=audit_service,
            logger=logger,
        )
    assert exc.value.status_code == 401

    # And reads are denied too — never fall back to the shared thread.
    with pytest.raises(HTTPException) as exc2:
        await chat_api.get_messages(
            http_request=_http_request(),
            thread_id="_default",
            workspace_id=USER_CHAT_HOME_WORKSPACE,
            auth_service=_AuthStub(ctx),
            authz_service=_authz_stub(),
            chat_service=chat_service,
            audit_service=audit_service,
            logger=logger,
        )
    assert exc2.value.status_code == 401


# ---------------------------------------------------------------------------
# (d) round-trip: write "_default" then read "_default" as the same user
#     returns their messages.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_trip_same_user(chat_service, audit_service, logger):
    from memorylayer_server.api.v1.schemas import MessageCreateRequest, MessagesAppendRequest

    ctx = _ctx_for_user("carol@example.com", USER_CHAT_HOME_WORKSPACE)

    await chat_api.append_messages(
        http_request=_http_request(),
        thread_id="_default",
        request=MessagesAppendRequest(
            messages=[
                MessageCreateRequest(role="user", content="msg-1"),
                MessageCreateRequest(role="assistant", content="msg-2"),
            ]
        ),
        workspace_id=USER_CHAT_HOME_WORKSPACE,
        auth_service=_AuthStub(ctx),
        authz_service=_authz_stub(),
        chat_service=chat_service,
        audit_service=audit_service,
        logger=logger,
    )

    # get_thread_full round-trips the client id on both thread + messages.
    full = await chat_api.get_thread_full(
        http_request=_http_request(),
        thread_id="_default",
        workspace_id=USER_CHAT_HOME_WORKSPACE,
        limit=100,
        offset=0,
        order="asc",
        auth_service=_AuthStub(ctx),
        authz_service=_authz_stub(),
        chat_service=chat_service,
        audit_service=audit_service,
        logger=logger,
    )
    assert full.thread.id == "_default"
    assert [m.content for m in full.messages] == ["msg-1", "msg-2"]
    assert all(m.thread_id == "_default" for m in full.messages)

    # list_user_threads (the sidebar route) is force-scoped to the OBO human and
    # presents the client id back. A spoofed path user_id must be ignored.
    listing = await chat_api.list_user_threads(
        http_request=_http_request(),
        user_id="attacker@example.com",  # spoof attempt — must be ignored
        ownership="user",
        scope_filter=None,
        limit=50,
        offset=0,
        include_hidden=False,
        parent_thread=None,
        auth_service=_AuthStub(ctx),
        authz_service=_authz_stub(),
        chat_service=chat_service,
        audit_service=audit_service,
        logger=logger,
    )
    ids = {t.id for t in listing.threads}
    assert "_default" in ids
    # The materialized storage id must never leak to the client.
    assert not any("::u::" in t.id for t in listing.threads)


@pytest.mark.asyncio
async def test_list_user_threads_spoofed_user_isolated(chat_service, audit_service, logger):
    """A caller cannot enumerate another user's threads by supplying their id in
    the /user/{user_id} path — the route force-scopes to the OBO subject."""
    from memorylayer_server.api.v1.schemas import MessageCreateRequest, MessagesAppendRequest

    dave_ctx = _ctx_for_user("dave@example.com", USER_CHAT_HOME_WORKSPACE)
    eve_ctx = _ctx_for_user("eve@example.com", USER_CHAT_HOME_WORKSPACE)

    for ctx, txt in ((dave_ctx, "dave-only"), (eve_ctx, "eve-only")):
        await chat_api.append_messages(
            http_request=_http_request(),
            thread_id="_default",
            request=MessagesAppendRequest(
                messages=[MessageCreateRequest(role="user", content=txt)]
            ),
            workspace_id=USER_CHAT_HOME_WORKSPACE,
            auth_service=_AuthStub(ctx),
            authz_service=_authz_stub(),
            chat_service=chat_service,
            audit_service=audit_service,
            logger=logger,
        )

    # Eve tries to read Dave's threads by passing his id in the path — denied by
    # force-scoping to her own OBO subject.
    listing = await chat_api.list_user_threads(
        http_request=_http_request(),
        user_id="dave@example.com",
        ownership="user",
        scope_filter=None,
        limit=50,
        offset=0,
        include_hidden=False,
        parent_thread=None,
        auth_service=_AuthStub(eve_ctx),
        authz_service=_authz_stub(),
        chat_service=chat_service,
        audit_service=audit_service,
        logger=logger,
    )
    # Eve only ever sees her own thread; her stored thread carries her ownership.
    eve_thread = await chat_service.get_thread(
        USER_CHAT_HOME_WORKSPACE, "_default", user_id="eve@example.com"
    )
    assert eve_thread is not None
    assert eve_thread.user_id == "eve@example.com"
    # Listing is scoped to Eve; Dave's message content is never returned here.
    for t in listing.threads:
        assert t.user_id in (None, "eve@example.com")
