"""Unit tests for chat history models and service."""

from datetime import UTC, datetime, timedelta

import pytest

from memorylayer_server.models.chat import (
    AppendMessagesInput,
    ChatMessage,
    ChatMessageContent,
    ChatThread,
    ChatThreadWithMessages,
    CreateThreadInput,
    DecompositionResult,
    MessageInput,
)


class TestChatMessageContent:
    """Tests for ChatMessageContent structured blocks."""

    def test_text_content(self):
        block = ChatMessageContent(type="text", text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"
        assert block.data is None

    def test_tool_call_content(self):
        block = ChatMessageContent(
            type="tool_call",
            data={"name": "search", "arguments": {"query": "test"}},
        )
        assert block.type == "tool_call"
        assert block.data["name"] == "search"

    def test_image_content(self):
        block = ChatMessageContent(
            type="image",
            data={"url": "https://example.com/img.png", "alt": "screenshot"},
        )
        assert block.type == "image"


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_simple_string_content(self):
        msg = ChatMessage(
            id="msg-1",
            thread_id="thread-1",
            message_index=0,
            role="user",
            content="Hello!",
        )
        assert msg.content == "Hello!"
        assert msg.role == "user"
        assert msg.message_index == 0

    def test_structured_content(self):
        blocks = [
            ChatMessageContent(type="text", text="Here's an image:"),
            ChatMessageContent(type="image", data={"url": "https://example.com/img.png"}),
        ]
        msg = ChatMessage(
            id="msg-2",
            thread_id="thread-1",
            message_index=1,
            role="assistant",
            content=blocks,
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"

    def test_role_validation_strips_and_lowercases(self):
        msg = ChatMessage(
            id="msg-3",
            thread_id="thread-1",
            message_index=0,
            role="  USER  ",
            content="hi",
        )
        assert msg.role == "user"

    def test_role_validation_rejects_empty(self):
        with pytest.raises(ValueError):
            ChatMessage(
                id="msg-4",
                thread_id="thread-1",
                message_index=0,
                role="   ",
                content="hi",
            )


class TestChatThread:
    """Tests for ChatThread model."""

    def test_basic_creation(self):
        thread = ChatThread(
            id="thread-1",
            workspace_id="ws-1",
            tenant_id="_default",
        )
        assert thread.id == "thread-1"
        assert thread.workspace_id == "ws-1"
        assert thread.message_count == 0
        assert thread.last_decomposed_index == 0
        assert thread.expires_at is None
        assert not thread.is_expired

    def test_with_entity_attribution(self):
        thread = ChatThread(
            id="thread-2",
            workspace_id="ws-1",
            observer_id="claude",
            subject_id="user-drew",
        )
        assert thread.observer_id == "claude"
        assert thread.subject_id == "user-drew"

    def test_is_expired_false_when_no_expiry(self):
        thread = ChatThread(id="t1", workspace_id="ws")
        assert not thread.is_expired

    def test_is_expired_true_when_past(self):
        thread = ChatThread(
            id="t1",
            workspace_id="ws",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        assert thread.is_expired

    def test_is_expired_false_when_future(self):
        thread = ChatThread(
            id="t1",
            workspace_id="ws",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
        )
        assert not thread.is_expired

    def test_unprocessed_count(self):
        thread = ChatThread(
            id="t1",
            workspace_id="ws",
            message_count=15,
            last_decomposed_index=10,
        )
        assert thread.unprocessed_count == 5

    def test_unprocessed_count_zero_when_caught_up(self):
        thread = ChatThread(
            id="t1",
            workspace_id="ws",
            message_count=10,
            last_decomposed_index=10,
        )
        assert thread.unprocessed_count == 0

    def test_default_context_id(self):
        thread = ChatThread(id="t1", workspace_id="ws")
        assert thread.context_id == "_default"


class TestChatThreadWithMessages:
    """Tests for the combined thread+messages model."""

    def test_basic(self):
        thread = ChatThread(id="t1", workspace_id="ws", message_count=2)
        messages = [
            ChatMessage(id="m1", thread_id="t1", message_index=0, role="user", content="hi"),
            ChatMessage(id="m2", thread_id="t1", message_index=1, role="assistant", content="hello"),
        ]
        combined = ChatThreadWithMessages(
            thread=thread,
            messages=messages,
            total_messages=2,
        )
        assert combined.thread.id == "t1"
        assert len(combined.messages) == 2
        assert combined.total_messages == 2


class TestCreateThreadInput:
    """Tests for thread creation input."""

    def test_defaults(self):
        inp = CreateThreadInput()
        assert inp.thread_id is None
        assert inp.context_id == "_default"
        assert inp.user_id is None
        assert inp.observer_id is None
        assert inp.expires_at is None

    def test_with_all_fields(self):
        inp = CreateThreadInput(
            thread_id="my-thread",
            user_id="drew",
            context_id="project-x",
            observer_id="claude",
            subject_id="drew",
            title="Planning Session",
            metadata={"source": "mcp"},
            expires_at=datetime(2026, 12, 31, tzinfo=UTC),
        )
        assert inp.thread_id == "my-thread"
        assert inp.title == "Planning Session"
        assert inp.expires_at.year == 2026


class TestAppendMessagesInput:
    """Tests for message append input."""

    def test_requires_at_least_one_message(self):
        with pytest.raises(ValueError):
            AppendMessagesInput(messages=[])

    def test_accepts_messages(self):
        inp = AppendMessagesInput(
            messages=[
                MessageInput(role="user", content="hello"),
                MessageInput(role="assistant", content="hi there"),
            ]
        )
        assert len(inp.messages) == 2


class TestMessageInput:
    """Tests for individual message input."""

    def test_simple_text(self):
        msg = MessageInput(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_structured_content(self):
        msg = MessageInput(
            role="assistant",
            content=[
                ChatMessageContent(type="text", text="Here's the result"),
                ChatMessageContent(type="tool_result", data={"output": "42"}),
            ],
        )
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2


class TestDecompositionResult:
    """Tests for decomposition result."""

    def test_basic(self):
        result = DecompositionResult(
            thread_id="t1",
            workspace_id="ws",
            messages_processed=10,
            memories_created=3,
            from_index=5,
            to_index=15,
        )
        assert result.messages_processed == 10
        assert result.memories_created == 3


class TestChatMessageSerialization:
    """Tests for content serialization behavior."""

    def test_string_content_serializes(self):
        msg = ChatMessage(
            id="m1",
            thread_id="t1",
            message_index=0,
            role="user",
            content="hello",
        )
        data = msg.model_dump(mode="json")
        assert data["content"] == "hello"

    def test_structured_content_serializes(self):
        msg = ChatMessage(
            id="m1",
            thread_id="t1",
            message_index=0,
            role="assistant",
            content=[ChatMessageContent(type="text", text="hi")],
        )
        data = msg.model_dump(mode="json")
        assert isinstance(data["content"], list)
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"] == "hi"
