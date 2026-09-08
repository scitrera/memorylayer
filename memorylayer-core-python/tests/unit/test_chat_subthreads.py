"""Tests for sub-threads (parent_thread): top-level-default listing, listing a
parent's children, the shared-context invariant, and child cascade on delete."""

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from scitrera_app_framework.api import Variables

from memorylayer_server.models.chat import ChatThread, CreateThreadInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.chat.default import DefaultChatService
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend

WS = "ws-sub"


@pytest_asyncio.fixture
async def storage(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "sub.db"))
    await backend.connect()
    await backend.create_workspace(
        Workspace(id=WS, tenant_id="_default", name="Sub WS",
                  created_at=datetime.now(UTC), updated_at=datetime.now(UTC))
    )
    yield backend
    await backend.disconnect()


class _FakeTaskService:
    async def schedule_task(self, task_type, payload):
        return "task-id"


@pytest_asyncio.fixture
def service(storage):
    return DefaultChatService(storage=storage, task_service=_FakeTaskService(), v=Variables())


# Owner used by the storage-level helpers below. Threads are now owner-scoped
# (workspace_id, user_id, id), so reads must pass the same user_id.
U = "u1"


async def _mk(storage, tid, *, parent_thread=None, ownership="user", user_id=U):
    now = datetime.now(UTC)
    return await storage.create_thread(ChatThread(
        id=tid, workspace_id=WS, tenant_id="_default", user_id=user_id,
        ownership=ownership, parent_thread=parent_thread, created_at=now, updated_at=now,
    ))


# ---------------------------------------------------------------------------
# Storage: listing semantics + cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parent_thread_persists(storage):
    await _mk(storage, "p")
    await _mk(storage, "c", parent_thread="p")
    assert (await storage.get_thread(WS, "c", user_id=U)).parent_thread == "p"
    assert (await storage.get_thread(WS, "p", user_id=U)).parent_thread is None


@pytest.mark.asyncio
async def test_list_defaults_to_top_level_only(storage):
    await _mk(storage, "p1")
    await _mk(storage, "p2")
    await _mk(storage, "c1", parent_thread="p1")
    await _mk(storage, "c2", parent_thread="p1")

    top = {t.id for t in await storage.list_threads(WS)}
    assert top == {"p1", "p2"}  # children excluded by default

    children = {t.id for t in await storage.list_threads(WS, parent_thread="p1")}
    assert children == {"c1", "c2"}

    # user-scoped listing has the same default + child behavior
    top_user = {t.id for t in await storage.list_user_threads("_default", "u1")}
    assert top_user == {"p1", "p2"}
    child_user = {t.id for t in await storage.list_user_threads("_default", "u1", parent_thread="p1")}
    assert child_user == {"c1", "c2"}


@pytest.mark.asyncio
async def test_delete_parent_cascades_children_recursively(storage):
    await _mk(storage, "p")
    await _mk(storage, "c", parent_thread="p")
    await _mk(storage, "g", parent_thread="c")  # grandchild
    await _mk(storage, "other")

    assert await storage.delete_thread(WS, "p", user_id=U) is True
    assert await storage.get_thread(WS, "p", user_id=U) is None
    assert await storage.get_thread(WS, "c", user_id=U) is None
    assert await storage.get_thread(WS, "g", user_id=U) is None  # recursive
    assert await storage.get_thread(WS, "other", user_id=U) is not None


# ---------------------------------------------------------------------------
# Service: shared-context invariant
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_inherits_parent_context(service):
    # Threads are owner-scoped (workspace_id, user_id, id); a sub-thread is
    # created under the SAME owner as its parent (in production the API passes
    # the same OBO user_key for both), so the parent resolves. The child still
    # requests conflicting ownership/scope which are overridden to the parent's.
    parent = await service.create_thread(
        WS, "_default",
        CreateThreadInput(title="p", ownership="workspace", scope="office", user_id="owner"),
    )
    child = await service.create_thread(
        WS, "_default",
        CreateThreadInput(title="c", parent_thread=parent.id,
                          ownership="user", scope="web", user_id="owner"),
    )
    assert child.parent_thread == parent.id
    assert child.ownership == "workspace"  # inherited
    assert child.scope == "office"
    assert child.user_id == "owner"
    assert child.tenant_id == parent.tenant_id


@pytest.mark.asyncio
async def test_child_with_missing_parent_raises(service):
    with pytest.raises(ValueError, match="parent thread"):
        await service.create_thread(
            WS, "_default", CreateThreadInput(title="orphan", parent_thread="nonexistent"),
        )
