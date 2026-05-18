"""
Integration tests for chat history storage in the SQLite backend.

Tests the SQLite storage backend directly with chat thread and message operations.
Each test class gets its own isolated storage backend with an in-memory (temp file) database.
"""

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from memorylayer_server.models.chat import (
    ChatMessageContent,
    ChatThread,
    MessageInput,
)
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def storage(tmp_path):
    """
    Create an isolated SQLiteStorageBackend for each test.

    Uses a temp file (not :memory:) so aiosqlite works correctly with
    WAL mode and FK constraints.  The backend is connected and seeded with
    a workspace before the test runs, and disconnected afterwards.
    """
    db_path = str(tmp_path / "test_chat.db")
    backend = SQLiteStorageBackend(db_path)
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest_asyncio.fixture
async def workspace_id(storage) -> str:
    """Create a test workspace and return its ID."""
    ws_id = "test-workspace-chat"
    workspace = Workspace(
        id=ws_id,
        tenant_id="_default",
        name="Chat Test Workspace",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    await storage.create_workspace(workspace)
    return ws_id


def _make_thread(workspace_id: str, thread_id: str = "thread-1", **kwargs) -> ChatThread:
    """Helper: build a ChatThread with sensible defaults."""
    now = datetime.now(UTC)
    return ChatThread(
        id=thread_id,
        workspace_id=workspace_id,
        tenant_id="_default",
        created_at=now,
        updated_at=now,
        **kwargs,
    )


def _make_msg_input(role: str = "user", content: str = "hello") -> MessageInput:
    """Helper: build a simple MessageInput."""
    return MessageInput(role=role, content=content)


# ---------------------------------------------------------------------------
# TestChatThreadStorage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestChatThreadStorage:
    """Tests for chat thread CRUD in SQLite."""

    async def test_create_thread(self, storage, workspace_id):
        """Create a ChatThread, store it, verify all fields round-trip."""
        thread = _make_thread(
            workspace_id,
            thread_id="t-create",
            user_id="user-drew",
            context_id="ctx-1",
            observer_id="claude",
            subject_id="drew",
            title="Test Thread",
            metadata={"source": "test"},
        )
        result = await storage.create_thread(thread)

        assert result.id == "t-create"
        assert result.workspace_id == workspace_id
        assert result.user_id == "user-drew"
        assert result.context_id == "ctx-1"
        assert result.observer_id == "claude"
        assert result.subject_id == "drew"
        assert result.title == "Test Thread"
        assert result.metadata == {"source": "test"}
        assert result.message_count == 0
        assert result.last_decomposed_index == 0
        assert result.expires_at is None

    async def test_get_thread(self, storage, workspace_id):
        """Create then retrieve a thread by workspace_id + thread_id."""
        thread = _make_thread(workspace_id, thread_id="t-get", title="Get Me")
        await storage.create_thread(thread)

        fetched = await storage.get_thread(workspace_id, "t-get")

        assert fetched is not None
        assert fetched.id == "t-get"
        assert fetched.workspace_id == workspace_id
        assert fetched.title == "Get Me"

    async def test_get_thread_not_found(self, storage, workspace_id):
        """get_thread returns None for a nonexistent thread_id."""
        result = await storage.get_thread(workspace_id, "does-not-exist")
        assert result is None

    async def test_list_threads(self, storage, workspace_id):
        """Create multiple threads and verify list_threads returns all of them."""
        for i in range(3):
            await storage.create_thread(_make_thread(workspace_id, thread_id=f"t-list-{i}", title=f"Thread {i}"))

        threads = await storage.list_threads(workspace_id)

        thread_ids = {t.id for t in threads}
        assert "t-list-0" in thread_ids
        assert "t-list-1" in thread_ids
        assert "t-list-2" in thread_ids

    async def test_list_threads_by_user(self, storage, workspace_id):
        """list_threads with user_id filters to that user's threads only."""
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-user-a", user_id="alice"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-user-b", user_id="bob"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-user-a2", user_id="alice"))

        alice_threads = await storage.list_threads(workspace_id, user_id="alice")
        thread_ids = {t.id for t in alice_threads}

        assert "t-user-a" in thread_ids
        assert "t-user-a2" in thread_ids
        assert "t-user-b" not in thread_ids

    async def test_list_threads_excludes_expired(self, storage, workspace_id):
        """list_threads does not return threads whose expires_at is in the past."""
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

        await storage.create_thread(_make_thread(workspace_id, thread_id="t-expired", expires_at=past))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-active", expires_at=future))
        await storage.create_thread(
            _make_thread(workspace_id, thread_id="t-permanent")  # no expiry
        )

        threads = await storage.list_threads(workspace_id)
        thread_ids = {t.id for t in threads}

        assert "t-expired" not in thread_ids
        assert "t-active" in thread_ids
        assert "t-permanent" in thread_ids

    async def test_update_thread(self, storage, workspace_id):
        """update_thread modifies the specified fields and returns the updated thread."""
        thread = _make_thread(workspace_id, thread_id="t-update", title="Original")
        await storage.create_thread(thread)

        updated = await storage.update_thread(
            workspace_id,
            "t-update",
            title="Updated Title",
            last_decomposed_index=5,
        )

        assert updated is not None
        assert updated.title == "Updated Title"
        assert updated.last_decomposed_index == 5
        # Unchanged fields stay intact
        assert updated.workspace_id == workspace_id

    async def test_delete_thread(self, storage, workspace_id):
        """delete_thread removes the thread; get_thread returns None afterwards."""
        thread = _make_thread(workspace_id, thread_id="t-delete")
        await storage.create_thread(thread)

        deleted = await storage.delete_thread(workspace_id, "t-delete")

        assert deleted is True
        assert await storage.get_thread(workspace_id, "t-delete") is None

    # ------------------------------------------------------------------
    # scope field tests
    # ------------------------------------------------------------------

    async def test_create_thread_scope_roundtrip(self, storage, workspace_id):
        """scope field persists through create → get round-trip."""
        web_thread = _make_thread(workspace_id, thread_id="t-scope-web", scope="web")
        office_thread = _make_thread(workspace_id, thread_id="t-scope-office", scope="office")
        null_thread = _make_thread(workspace_id, thread_id="t-scope-null")  # scope=None

        await storage.create_thread(web_thread)
        await storage.create_thread(office_thread)
        await storage.create_thread(null_thread)

        fetched_web = await storage.get_thread(workspace_id, "t-scope-web")
        fetched_office = await storage.get_thread(workspace_id, "t-scope-office")
        fetched_null = await storage.get_thread(workspace_id, "t-scope-null")

        assert fetched_web is not None and fetched_web.scope == "web"
        assert fetched_office is not None and fetched_office.scope == "office"
        assert fetched_null is not None and fetched_null.scope is None

    async def test_list_threads_scope_filter_web(self, storage, workspace_id):
        """scope_filter='web' returns web threads AND null-scope threads (NULL ≡ web)."""
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sf-web", scope="web"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sf-office", scope="office"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sf-null"))  # NULL ≡ web

        threads = await storage.list_threads(workspace_id, scope_filter="web")
        thread_ids = {t.id for t in threads}

        assert "t-sf-web" in thread_ids
        assert "t-sf-null" in thread_ids
        assert "t-sf-office" not in thread_ids

    async def test_list_threads_scope_filter_office(self, storage, workspace_id):
        """scope_filter='office' returns only office threads."""
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sfo-web", scope="web"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sfo-office", scope="office"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-sfo-null"))

        threads = await storage.list_threads(workspace_id, scope_filter="office")
        thread_ids = {t.id for t in threads}

        assert "t-sfo-office" in thread_ids
        assert "t-sfo-web" not in thread_ids
        assert "t-sfo-null" not in thread_ids

    async def test_list_threads_scope_filter_none_returns_all(self, storage, workspace_id):
        """scope_filter=None (default) returns all threads regardless of scope."""
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-all-web", scope="web"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-all-office", scope="office"))
        await storage.create_thread(_make_thread(workspace_id, thread_id="t-all-null"))

        threads = await storage.list_threads(workspace_id)
        thread_ids = {t.id for t in threads}

        assert "t-all-web" in thread_ids
        assert "t-all-office" in thread_ids
        assert "t-all-null" in thread_ids


# ---------------------------------------------------------------------------
# TestChatMessageStorage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestChatMessageStorage:
    """Tests for chat message operations in SQLite."""

    async def test_append_messages(self, storage, workspace_id):
        """Appending messages assigns IDs and sequential message_index values."""
        thread = _make_thread(workspace_id, thread_id="t-append")
        await storage.create_thread(thread)

        msgs = [
            _make_msg_input("user", "Hello"),
            _make_msg_input("assistant", "Hi there"),
        ]
        created = await storage.append_messages(workspace_id, "t-append", msgs)

        assert len(created) == 2
        assert created[0].id  # non-empty ID was assigned
        assert created[1].id
        assert created[0].message_index == 0
        assert created[1].message_index == 1
        assert created[0].role == "user"
        assert created[1].role == "assistant"
        assert created[0].thread_id == "t-append"

    async def test_append_increments_count(self, storage, workspace_id):
        """thread.message_count is updated after each append."""
        thread = _make_thread(workspace_id, thread_id="t-count")
        await storage.create_thread(thread)

        await storage.append_messages(
            workspace_id,
            "t-count",
            [
                _make_msg_input("user", "msg 1"),
                _make_msg_input("assistant", "msg 2"),
            ],
        )
        t = await storage.get_thread(workspace_id, "t-count")
        assert t.message_count == 2

        await storage.append_messages(
            workspace_id,
            "t-count",
            [
                _make_msg_input("user", "msg 3"),
            ],
        )
        t = await storage.get_thread(workspace_id, "t-count")
        assert t.message_count == 3

    async def test_get_messages_ordered(self, storage, workspace_id):
        """Messages are returned in ascending message_index order by default."""
        thread = _make_thread(workspace_id, thread_id="t-order")
        await storage.create_thread(thread)

        await storage.append_messages(
            workspace_id,
            "t-order",
            [
                _make_msg_input("user", "first"),
                _make_msg_input("assistant", "second"),
                _make_msg_input("user", "third"),
            ],
        )

        messages = await storage.get_messages(workspace_id, "t-order")

        assert len(messages) == 3
        assert messages[0].message_index == 0
        assert messages[1].message_index == 1
        assert messages[2].message_index == 2
        assert messages[0].content == "first"
        assert messages[2].content == "third"

    async def test_get_messages_with_limit_offset(self, storage, workspace_id):
        """Pagination via limit and offset works correctly."""
        thread = _make_thread(workspace_id, thread_id="t-page")
        await storage.create_thread(thread)

        inputs = [_make_msg_input("user", f"msg {i}") for i in range(5)]
        await storage.append_messages(workspace_id, "t-page", inputs)

        page1 = await storage.get_messages(workspace_id, "t-page", limit=2, offset=0)
        page2 = await storage.get_messages(workspace_id, "t-page", limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].message_index == 0
        assert page1[1].message_index == 1
        assert page2[0].message_index == 2
        assert page2[1].message_index == 3

    async def test_get_messages_after_index(self, storage, workspace_id):
        """after_index filter returns only messages with message_index > after_index."""
        thread = _make_thread(workspace_id, thread_id="t-after")
        await storage.create_thread(thread)

        inputs = [_make_msg_input("user", f"msg {i}") for i in range(5)]
        await storage.append_messages(workspace_id, "t-after", inputs)

        # after_index=2 should return messages at index 3 and 4
        messages = await storage.get_messages(workspace_id, "t-after", after_index=2)

        assert len(messages) == 2
        assert messages[0].message_index == 3
        assert messages[1].message_index == 4

    async def test_structured_content_roundtrip(self, storage, workspace_id):
        """Structured content (list[ChatMessageContent]) serializes and deserializes correctly."""
        thread = _make_thread(workspace_id, thread_id="t-structured")
        await storage.create_thread(thread)

        blocks = [
            ChatMessageContent(type="text", text="Here is the result:"),
            ChatMessageContent(type="tool_result", data={"output": "42", "tool": "calc"}),
        ]
        structured_msg = MessageInput(role="assistant", content=blocks)
        await storage.append_messages(workspace_id, "t-structured", [structured_msg])

        messages = await storage.get_messages(workspace_id, "t-structured")

        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[0].text == "Here is the result:"
        assert msg.content[1].type == "tool_result"
        assert msg.content[1].data["output"] == "42"

    async def test_delete_thread_cascades_messages(self, storage, workspace_id):
        """Deleting a thread removes its messages from the messages table."""
        thread = _make_thread(workspace_id, thread_id="t-cascade")
        await storage.create_thread(thread)

        await storage.append_messages(
            workspace_id,
            "t-cascade",
            [
                _make_msg_input("user", "will be gone"),
                _make_msg_input("assistant", "also gone"),
            ],
        )

        # Confirm messages exist before deletion
        before = await storage.get_messages(workspace_id, "t-cascade")
        assert len(before) == 2

        await storage.delete_thread(workspace_id, "t-cascade")

        # Thread gone
        assert await storage.get_thread(workspace_id, "t-cascade") is None
        # Messages gone (querying should return empty list, not raise)
        after = await storage.get_messages(workspace_id, "t-cascade")
        assert len(after) == 0

    async def test_delete_message_removes_row(self, storage, workspace_id):
        """delete_message returns True and removes the targeted row."""
        thread = _make_thread(workspace_id, thread_id="t-del-msg")
        await storage.create_thread(thread)

        created = await storage.append_messages(
            workspace_id,
            "t-del-msg",
            [
                _make_msg_input("user", "first"),
                _make_msg_input("assistant", "second"),
            ],
        )
        target_id = created[0].id

        result = await storage.delete_message(workspace_id, "t-del-msg", target_id)

        assert result is True
        remaining = await storage.get_messages(workspace_id, "t-del-msg")
        assert len(remaining) == 1
        assert remaining[0].id == created[1].id

    async def test_delete_message_not_found_returns_false(self, storage, workspace_id):
        """delete_message returns False without raising when message does not exist."""
        thread = _make_thread(workspace_id, thread_id="t-del-missing")
        await storage.create_thread(thread)

        result = await storage.delete_message(workspace_id, "t-del-missing", "nonexistent-id")

        assert result is False

    async def test_delete_message_wrong_workspace_returns_false(self, storage, workspace_id):
        """delete_message respects workspace scoping — wrong workspace returns False."""
        thread = _make_thread(workspace_id, thread_id="t-del-ws")
        await storage.create_thread(thread)

        created = await storage.append_messages(
            workspace_id,
            "t-del-ws",
            [_make_msg_input("user", "hello")],
        )
        msg_id = created[0].id

        result = await storage.delete_message("other-workspace", "t-del-ws", msg_id)

        assert result is False
        # Original message still present in the correct workspace
        remaining = await storage.get_messages(workspace_id, "t-del-ws")
        assert len(remaining) == 1

    async def test_client_supplied_id_is_used(self, storage, workspace_id):
        """When a client supplies an id in MessageInput, the stored row uses that id."""
        thread = _make_thread(workspace_id, thread_id="t-client-id")
        await storage.create_thread(thread)

        client_id = "msg_test_123"
        created = await storage.append_messages(
            workspace_id,
            "t-client-id",
            [MessageInput(role="user", content="hello", id=client_id)],
        )

        assert len(created) == 1
        assert created[0].id == client_id

        # delete_message should succeed using that exact id
        deleted = await storage.delete_message(workspace_id, "t-client-id", client_id)
        assert deleted is True

        # Row is gone
        remaining = await storage.get_messages(workspace_id, "t-client-id")
        assert len(remaining) == 0
