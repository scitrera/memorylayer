"""Auth must not create workspaces as a side effect of resolving them.

Authentication resolves a workspace on EVERY request, and it used to call
``ensure_workspace(auto_create=True)`` unconditionally while doing so. That made
a plain read create rows.

It reached production through the admin console: the cross-workspace list
endpoints (``GET /v1/admin/skills``, ``/memories``, ``/jobs``, ``/documents``,
...) take ``workspace_id`` as a *filter*, and the console's "Filter workspace"
box is a debounced text input. Typing ``jgl-field`` into it left eight
workspaces behind — ``j``, ``jg``, ``jgl``, ``jgl-``, ``jgl-f``, ``jgl-fi``,
``jgl-fie``, ``jgl-fiel`` — one per debounce window, 3.8s apart end to end.

Two independent rules keep it from recurring, and both are pinned here:

1. Safe HTTP methods (GET/HEAD/OPTIONS) never create. This is what stops the
   admin-filter case, and it is the HTTP contract besides.
2. Only well-formed ids are creatable. This is what stops the other observed
   shapes — ``/workspace`` and ``/sahara`` (a path passed where an id was
   expected) and ``workspace:_global`` (a caller that prefixed the id) — which
   arrived on write paths that rule 1 does not cover.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from pydantic import BaseModel

from memorylayer_server.services.authentication.aether import AetherAuthenticationService
from memorylayer_server.services.authentication.default import OpenAuthenticationService
from memorylayer_server.services.workspace.default import is_creatable_workspace_id


class _Body(BaseModel):
    workspace_id: str | None = None


def _make_request(method: str, query: dict | None = None, headers: dict | None = None) -> MagicMock:
    req = MagicMock()
    req.method = method
    req.query_params = query or {}
    req.headers = headers or {}
    return req


def _make_service(**kwargs) -> tuple[OpenAuthenticationService, MagicMock]:
    session_service = MagicMock()
    session_service.get = AsyncMock(side_effect=Exception("not found"))
    workspace_service = MagicMock()
    workspace_service.ensure_workspace = AsyncMock()
    svc = OpenAuthenticationService(
        session_service=session_service,
        workspace_service=workspace_service,
        **kwargs,
    )
    return svc, workspace_service


def _make_aether_service(**kwargs) -> tuple[AetherAuthenticationService, MagicMock]:
    session_service = MagicMock()
    session_service.get = AsyncMock(side_effect=Exception("not found"))
    workspace_service = MagicMock()
    workspace_service.ensure_workspace = AsyncMock()
    svc = AetherAuthenticationService(
        session_service=session_service,
        workspace_service=workspace_service,
        allow_default_tenant_fallback=True,
        **kwargs,
    )
    return svc, workspace_service


# ---------------------------------------------------------------------------
# Rule 1: safe methods do not create
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "get"])
@pytest.mark.asyncio
async def test_safe_methods_never_create_a_workspace(method):
    """The admin-filter case: a listing must not mint what it filters on."""
    svc, workspace_service = _make_service()
    request = _make_request(method, query={"workspace_id": "jgl-f"})

    ctx = await svc.build_context(request, None)

    # Resolution still yields the id — the handler goes on to filter by it and
    # correctly finds nothing.
    assert ctx.workspace_id == "jgl-f"
    workspace_service.ensure_workspace.assert_not_awaited()


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.asyncio
async def test_unsafe_methods_still_create(method):
    """The OSS "just works" path: a write into a fresh workspace still lands.

    MemoryLayer's MCP server derives a workspace from the git repo name and
    expects the first remember() to succeed, and ``memories.workspace_id`` is a
    foreign key — so removing creation from the write path would turn that into
    an opaque 500.
    """
    svc, workspace_service = _make_service()
    request = _make_request(method, query={"workspace_id": "my-repo"})

    ctx = await svc.build_context(request, None)

    assert ctx.workspace_id == "my-repo"
    assert workspace_service.ensure_workspace.await_args.kwargs["workspace_id"] == "my-repo"


@pytest.mark.asyncio
async def test_safe_method_does_not_implicitly_create_a_session_either():
    """A session row carries a workspace foreign key, so it follows the same rule."""
    svc, _ = _make_service()
    svc.ensure_session = AsyncMock()
    request = _make_request(
        "GET",
        query={"workspace_id": "ws-read"},
        headers={"X-Session-ID": "sess-unknown"},
    )

    await svc.build_context(request, None)

    svc.ensure_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsafe_method_still_implicitly_creates_a_session():
    svc, _ = _make_service()
    svc.ensure_session = AsyncMock(return_value=None)
    request = _make_request(
        "POST",
        query={"workspace_id": "ws-write"},
        headers={"X-Session-ID": "sess-unknown"},
    )

    await svc.build_context(request, None)

    svc.ensure_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_aether_path_applies_the_same_rule():
    """The leak was observed on the Aether authenticator, so pin it there too."""
    svc, workspace_service = _make_aether_service()

    await svc.build_context(_make_request("GET", query={"workspace_id": "jgl-fie"}), None)
    workspace_service.ensure_workspace.assert_not_awaited()

    await svc.build_context(_make_request("POST", query={"workspace_id": "jgl-field"}), None)
    assert workspace_service.ensure_workspace.await_args.kwargs["workspace_id"] == "jgl-field"


# ---------------------------------------------------------------------------
# Rule 2: the kill switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_implicit_create_rejects_a_write_naming_an_unknown_workspace():
    """Enterprise default: a workspace comes from an explicit create, never from
    a request that merely names one.

    404 rather than silence, because the handler's very next act is an insert
    against a workspace foreign key — letting it through would surface as an
    opaque 500 naming nothing about the workspace.
    """
    svc, workspace_service = _make_service(implicit_workspace_create=False)
    workspace_service.ensure_workspace = AsyncMock(return_value=None)  # does not exist

    with pytest.raises(HTTPException) as exc:
        await svc.build_context(_make_request("POST", query={"workspace_id": "nope"}), None)

    assert exc.value.status_code == 404
    assert "nope" in str(exc.value.detail)
    # Checked existence only -- never asked for creation.
    assert workspace_service.ensure_workspace.await_args.kwargs["auto_create"] is False


@pytest.mark.asyncio
async def test_disabled_implicit_create_allows_a_write_to_an_existing_workspace():
    svc, workspace_service = _make_service(implicit_workspace_create=False)
    workspace_service.ensure_workspace = AsyncMock(return_value=MagicMock(id="real-ws"))

    ctx = await svc.build_context(_make_request("POST", query={"workspace_id": "real-ws"}), None)

    assert ctx.workspace_id == "real-ws"


@pytest.mark.asyncio
async def test_disabled_implicit_create_still_reads_quietly():
    """The admin filter box must show an empty table, not an error banner."""
    svc, workspace_service = _make_service(implicit_workspace_create=False)
    workspace_service.ensure_workspace = AsyncMock(return_value=None)

    ctx = await svc.build_context(_make_request("GET", query={"workspace_id": "jgl-f"}), None)

    assert ctx.workspace_id == "jgl-f"
    workspace_service.ensure_workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_disabled_implicit_create_does_not_deadlock_bootstrap():
    """``POST /v1/workspaces`` resolves ``_default`` on its way to creating the
    first workspace, and names no workspace of its own. 404-ing that fallback
    would make it impossible to ever create one.
    """
    svc, workspace_service = _make_service(implicit_workspace_create=False)
    workspace_service.ensure_workspace = AsyncMock(return_value=None)

    ctx = await svc.build_context(_make_request("POST"), None)

    assert ctx.workspace_id == "_default"
    workspace_service.ensure_workspace.assert_not_awaited()


# ---------------------------------------------------------------------------
# Rule 3: malformed ids are not creatable
# ---------------------------------------------------------------------------


async def _ensure_like_the_real_service(workspace_id: str, tenant_id: str, auto_create: bool):
    """Stand-in for WorkspaceService.ensure_workspace's create-time validation."""
    if auto_create and not is_creatable_workspace_id(workspace_id):
        raise ValueError(f"cannot auto-create workspace with malformed id: {workspace_id!r}")
    return MagicMock(id=workspace_id)


@pytest.mark.parametrize(
    "workspace_id",
    [
        "/workspace",  # observed: a path passed where an id was expected
        "/sahara",  # observed: same, from a sahara command
        "workspace:_global",  # observed: a caller that prefixed the id
        "ws with spaces",
        "-leading-dash",
    ],
)
@pytest.mark.asyncio
async def test_malformed_ids_are_rejected_rather_than_created(workspace_id):
    svc, workspace_service = _make_service()
    workspace_service.ensure_workspace = AsyncMock(side_effect=_ensure_like_the_real_service)

    with pytest.raises(HTTPException) as exc:
        await svc.build_context(_make_request("POST", query={"workspace_id": workspace_id}), None)

    assert exc.value.status_code == 400
    assert workspace_id in str(exc.value.detail), "the rejected value must be named for the broken caller"


@pytest.mark.asyncio
async def test_a_request_naming_no_workspace_still_resolves_the_default():
    """``_default`` is well-formed, so the fallback path is untouched."""
    svc, workspace_service = _make_service()
    workspace_service.ensure_workspace = AsyncMock(side_effect=_ensure_like_the_real_service)

    ctx = await svc.build_context(_make_request("POST"), None)

    assert ctx.workspace_id == "_default"
