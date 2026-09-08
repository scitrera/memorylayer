"""Auto-creation refuses ids that nobody chose.

``ensure_workspace`` turns a caller's typo into a permanent row, so the rule
guards CREATION only: an existing workspace resolves however it is spelled, and
tightening the charset never strands data already in the table.

The rejected shapes here are the ones production actually accumulated, not
hypotheticals — see the module docstring of test_auth_workspace_autocreate.py.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.workspace.default import (
    WorkspaceService,
    is_creatable_workspace_id,
)


def _make_service(existing=None) -> tuple[WorkspaceService, MagicMock]:
    storage = MagicMock()
    storage.get_workspace = AsyncMock(return_value=existing)
    storage.create_workspace = AsyncMock(side_effect=lambda ws: ws)
    return WorkspaceService(storage=storage), storage


@pytest.mark.parametrize(
    "workspace_id",
    ["_default", "_global", "_global_user", "jgl-field", "my-repo", "ws_a1B2", "a.b-c_d", "9lives"],
)
def test_ordinary_ids_are_creatable(workspace_id):
    assert is_creatable_workspace_id(workspace_id)


@pytest.mark.parametrize(
    "workspace_id",
    [
        "/workspace",
        "/sahara",
        "workspace:_global",
        "ws with spaces",
        "-leading-dash",
        ".leading-dot",
        "ws/nested",
        "ws\nnewline",
        "",
        "x" * 129,
    ],
)
def test_malformed_ids_are_not_creatable(workspace_id):
    assert not is_creatable_workspace_id(workspace_id)


@pytest.mark.asyncio
async def test_ensure_workspace_raises_rather_than_creating_junk():
    svc, storage = _make_service(existing=None)

    with pytest.raises(ValueError, match="malformed id"):
        await svc.ensure_workspace("workspace:_global", tenant_id="t1", auto_create=True)

    storage.create_workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_existing_malformed_workspace_still_resolves():
    """Validation gates creation, not access — old rows must stay reachable."""
    existing = MagicMock(id="/sahara")
    svc, storage = _make_service(existing=existing)

    got = await svc.ensure_workspace("/sahara", tenant_id="t1", auto_create=True)

    assert got is existing
    storage.create_workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_lookup_only_never_validates():
    svc, storage = _make_service(existing=None)

    assert await svc.ensure_workspace("/sahara", tenant_id="t1", auto_create=False) is None
    storage.create_workspace.assert_not_awaited()


@pytest.mark.asyncio
async def test_well_formed_id_still_auto_creates():
    svc, storage = _make_service(existing=None)

    created = await svc.ensure_workspace("my-repo", tenant_id="t1", auto_create=True)

    assert created.id == "my-repo"
    assert created.tenant_id == "t1"
    storage.create_workspace.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_workspace_propagates_backend_refusal():
    """An existing row is not reported deleted when its backend refuses."""
    existing = MagicMock(id="protected")
    svc, storage = _make_service(existing=existing)
    storage.delete_workspace = AsyncMock(return_value=False)

    assert await svc.delete_workspace("protected") is False

    storage.delete_workspace.assert_awaited_once_with("protected")
