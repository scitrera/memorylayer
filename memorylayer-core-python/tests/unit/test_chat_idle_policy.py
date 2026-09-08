"""Tests for per-thread idle policy (idle_action), archiving (hidden_at), and
the background chat-thread cleanup sweeps."""

import logging
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import pytest_asyncio
from scitrera_app_framework.api import Variables

from memorylayer_server.models.chat import ChatThread, MessageInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.tasks.chat_thread_cleanup_handler import (
    periodic_chat_thread_cleanup_task,
)

WS = "ws-idle"


@pytest_asyncio.fixture
async def storage(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "idle.db"))
    await backend.connect()
    await backend.create_workspace(
        Workspace(id=WS, tenant_id="_default", name="Idle WS",
                  created_at=datetime.now(UTC), updated_at=datetime.now(UTC))
    )
    yield backend
    await backend.disconnect()


async def _mk(storage, tid, *, idle_action=None, hidden_at=None, age_days=0,
              message_count=0, last_decomposed_index=0, expires_at=None,
              user_id=None) -> ChatThread:
    # Threads are owner-scoped (workspace_id, user_id, id). These storage-layer
    # tests use workspace-owned threads (user_id=None) so reads/updates via
    # ``get_thread(WS, id)`` (user_id=None) resolve; the cleanup sweeps carry
    # ``thread.user_id`` through automatically.
    ts = datetime.now(UTC) - timedelta(days=age_days)
    thread = ChatThread(
        id=tid, workspace_id=WS, tenant_id="_default", user_id=user_id,
        idle_action=idle_action, hidden_at=hidden_at,
        message_count=message_count, last_decomposed_index=last_decomposed_index,
        expires_at=expires_at, created_at=ts, updated_at=ts,
    )
    return await storage.create_thread(thread)


class _FakeTaskService:
    def __init__(self):
        self.scheduled = []

    async def schedule_task(self, task_type, payload):
        self.scheduled.append((task_type, payload))
        return "task-id"


# ---------------------------------------------------------------------------
# Storage layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_action_persists(storage):
    await _mk(storage, "t1", idle_action="hide")
    got = await storage.get_thread(WS, "t1")
    assert got.idle_action == "hide"
    assert got.hidden_at is None
    assert got.is_hidden is False


@pytest.mark.asyncio
async def test_hide_excluded_from_list_until_include_hidden(storage):
    # "visible" is owned by u1 so the user-scoped listing below returns it.
    await _mk(storage, "visible", user_id="u1")
    await _mk(storage, "archived", user_id="u1")
    await storage.hide_thread(WS, "archived", user_id="u1")

    ids = {t.id for t in await storage.list_threads(WS)}
    assert ids == {"visible"}
    ids_all = {t.id for t in await storage.list_threads(WS, include_hidden=True)}
    assert ids_all == {"visible", "archived"}

    # user-scoped listing honors the same filter
    user_ids = {t.id for t in await storage.list_user_threads("_default", "u1")}
    assert user_ids == {"visible"}


@pytest.mark.asyncio
async def test_hide_does_not_bump_updated_at_unhide_clears(storage):
    t = await _mk(storage, "t1", age_days=3)
    before = (await storage.get_thread(WS, "t1")).updated_at
    hidden = await storage.hide_thread(WS, "t1")
    assert hidden.hidden_at is not None
    # hiding is not activity -> updated_at unchanged
    assert (await storage.get_thread(WS, "t1")).updated_at == before
    restored = await storage.unhide_thread(WS, "t1")
    assert restored.hidden_at is None


@pytest.mark.asyncio
async def test_append_revives_hidden_thread(storage):
    await _mk(storage, "t1")
    await storage.hide_thread(WS, "t1")
    assert (await storage.get_thread(WS, "t1")).is_hidden
    await storage.append_messages(WS, "t1", [MessageInput(role="user", content="hi")])
    got = await storage.get_thread(WS, "t1")
    assert got.is_hidden is False  # revived


@pytest.mark.asyncio
async def test_list_idle_threads_filters(storage):
    await _mk(storage, "del_old", idle_action="delete", age_days=10)
    await _mk(storage, "del_new", idle_action="delete", age_days=1)
    await _mk(storage, "hide_old", idle_action="hide", age_days=10)
    await _mk(storage, "none_old", idle_action=None, age_days=10)
    cutoff = datetime.now(UTC) - timedelta(days=7)

    dels = {t.id for t in await storage.list_idle_threads(cutoff, idle_action="delete")}
    assert dels == {"del_old"}  # del_new too recent; none/hide excluded by action
    hides = {t.id for t in await storage.list_idle_threads(cutoff, idle_action="hide", only_hidden=False)}
    assert hides == {"hide_old"}


@pytest.mark.asyncio
async def test_list_hidden_threads_by_age(storage):
    await _mk(storage, "old", hidden_at=datetime.now(UTC) - timedelta(days=40))
    await _mk(storage, "new", hidden_at=datetime.now(UTC) - timedelta(days=2))
    cutoff = datetime.now(UTC) - timedelta(days=30)
    got = {t.id for t in await storage.list_hidden_threads(cutoff)}
    assert got == {"old"}


# ---------------------------------------------------------------------------
# Cleanup task sweeps
# ---------------------------------------------------------------------------


async def _run_cleanup(storage, task_service=None, env=None):
    with patch.dict(os.environ, env or {}, clear=False):
        v = Variables()
        await periodic_chat_thread_cleanup_task(
            storage=storage, task_service=task_service, v=v,
            logger=logging.getLogger("test-cleanup"),
        )


@pytest.mark.asyncio
async def test_delete_action_idle_deleted_when_fully_decomposed(storage):
    await _mk(storage, "doomed", idle_action="delete", age_days=10,
              message_count=3, last_decomposed_index=3)  # unprocessed=0
    ts = _FakeTaskService()
    await _run_cleanup(storage, ts)
    assert await storage.get_thread(WS, "doomed") is None
    assert ts.scheduled == []  # nothing to decompose


@pytest.mark.asyncio
async def test_delete_action_deferred_when_undecomposed(storage):
    await _mk(storage, "pending", idle_action="delete", age_days=10,
              message_count=5, last_decomposed_index=2)  # unprocessed=3
    ts = _FakeTaskService()
    await _run_cleanup(storage, ts)
    # deferred: NOT deleted, decomposition scheduled
    assert await storage.get_thread(WS, "pending") is not None
    assert ts.scheduled and ts.scheduled[0][0] == "chat_decomposition"
    assert ts.scheduled[0][1]["thread_id"] == "pending"


@pytest.mark.asyncio
async def test_hide_action_archives_idle_thread(storage):
    await _mk(storage, "arch", idle_action="hide", age_days=10,
              message_count=2, last_decomposed_index=0)  # unprocessed=2
    ts = _FakeTaskService()
    await _run_cleanup(storage, ts)
    got = await storage.get_thread(WS, "arch")
    assert got is not None and got.is_hidden  # archived, not deleted
    # undecomposed tail also scheduled for decomposition
    assert any(t == "chat_decomposition" for t, _ in ts.scheduled)


@pytest.mark.asyncio
async def test_idle_none_and_recent_untouched(storage):
    await _mk(storage, "keep_none", idle_action=None, age_days=30)
    await _mk(storage, "keep_recent", idle_action="delete", age_days=1)
    await _run_cleanup(storage, _FakeTaskService())
    assert await storage.get_thread(WS, "keep_none") is not None
    assert await storage.get_thread(WS, "keep_recent") is not None


@pytest.mark.asyncio
async def test_grace_purge_disabled_by_default(storage):
    await _mk(storage, "hidden_old", idle_action="hide",
              hidden_at=datetime.now(UTC) - timedelta(days=60), age_days=60)
    # default HIDDEN_DELETE_DAYS=0 -> no purge
    await _run_cleanup(storage, _FakeTaskService())
    assert await storage.get_thread(WS, "hidden_old") is not None


@pytest.mark.asyncio
async def test_grace_purge_deletes_when_enabled(storage):
    await _mk(storage, "hidden_old", idle_action="hide",
              hidden_at=datetime.now(UTC) - timedelta(days=60), age_days=60,
              message_count=1, last_decomposed_index=1)  # fully decomposed
    await _mk(storage, "hidden_new", idle_action="hide",
              hidden_at=datetime.now(UTC) - timedelta(days=5), age_days=5)
    await _run_cleanup(storage, _FakeTaskService(),
                       env={"MEMORYLAYER_CHAT_THREAD_HIDDEN_DELETE_DAYS": "30"})
    assert await storage.get_thread(WS, "hidden_old") is None  # past 30d grace
    assert await storage.get_thread(WS, "hidden_new") is not None  # within grace


@pytest.mark.asyncio
async def test_absolute_expires_at_still_deleted(storage):
    await _mk(storage, "expired", expires_at=datetime.now(UTC) - timedelta(hours=1))
    await _run_cleanup(storage, _FakeTaskService())
    assert await storage.get_thread(WS, "expired") is None
