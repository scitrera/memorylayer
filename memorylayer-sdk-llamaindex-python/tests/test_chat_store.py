"""Unit tests for MemoryLayerChatStore and utility functions."""

import pytest
import respx
from httpx import Response
from llama_index.core.llms import ChatMessage, MessageRole

from memorylayer_llamaindex import (
    CHAT_KEY_TAG_PREFIX,
    MemoryLayerChatStore,
    chat_message_to_memory_payload,
    get_chat_key,
    get_message_index,
    memory_to_chat_message,
    message_role_to_string,
    string_to_message_role,
)


# ========== Test Fixtures ==========


@pytest.fixture
def base_url() -> str:
    """Test base URL."""
    return "http://test.memorylayer.ai"


@pytest.fixture
def chat_store(base_url: str) -> MemoryLayerChatStore:
    """Create test chat store."""
    return MemoryLayerChatStore(
        base_url=base_url,
        api_key="test_api_key",
        workspace_id="ws_test",
    )


@pytest.fixture
def sample_chat_message() -> ChatMessage:
    """Create a sample chat message."""
    return ChatMessage.from_str("Hello, how can I help you?", role=MessageRole.USER)


@pytest.fixture
def sample_memory() -> dict:
    """Create a sample memory response."""
    return {
        "id": "mem_123",
        "workspace_id": "ws_test",
        "content": "user: Hello, how can I help you?",
        "type": "episodic",
        "importance": 0.5,
        "tags": ["llamaindex_chat_key:user_123"],
        "metadata": {
            "chat_key": "user_123",
            "message_index": 0,
            "role": "user",
            "additional_kwargs": {},
            "blocks": [{"block_type": "text", "text": "Hello, how can I help you?"}],
        },
        "access_count": 0,
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
    }


def create_memory_response(
    memory_id: str = "mem_new",
    content: str = "test content",
    memory_type: str = "episodic",
    importance: float = 0.5,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Create a complete memory response for mocking."""
    return {
        "id": memory_id,
        "workspace_id": "ws_test",
        "content": content,
        "type": memory_type,
        "importance": importance,
        "tags": tags or [],
        "metadata": metadata or {},
        "access_count": 0,
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
    }


def create_chat_memory(
    memory_id: str,
    content: str,
    message_index: int,
    role: str,
    chat_key: str,
    additional_metadata: dict | None = None,
) -> dict:
    """Create a complete chat memory for mocking."""
    metadata = {
        "message_index": message_index,
        "role": role,
        "chat_key": chat_key,
        "additional_kwargs": {},
        "blocks": [{"block_type": "text", "text": content}],
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    return create_memory_response(
        memory_id=memory_id,
        content=content,
        metadata=metadata,
        tags=[f"llamaindex_chat_key:{chat_key}"],
    )


# ========== Utility Function Tests ==========


class TestMessageRoleToString:
    """Tests for message_role_to_string function."""

    def test_user_role(self) -> None:
        """Test converting USER role to string."""
        result = message_role_to_string(MessageRole.USER)
        assert result == "user"

    def test_assistant_role(self) -> None:
        """Test converting ASSISTANT role to string."""
        result = message_role_to_string(MessageRole.ASSISTANT)
        assert result == "assistant"

    def test_system_role(self) -> None:
        """Test converting SYSTEM role to string."""
        result = message_role_to_string(MessageRole.SYSTEM)
        assert result == "system"

    def test_tool_role(self) -> None:
        """Test converting TOOL role to string."""
        result = message_role_to_string(MessageRole.TOOL)
        assert result == "tool"

    def test_chatbot_role(self) -> None:
        """Test converting CHATBOT role to string."""
        result = message_role_to_string(MessageRole.CHATBOT)
        assert result == "chatbot"

    def test_model_role(self) -> None:
        """Test converting MODEL role to string."""
        result = message_role_to_string(MessageRole.MODEL)
        assert result == "model"

    def test_function_role(self) -> None:
        """Test converting FUNCTION role to string."""
        result = message_role_to_string(MessageRole.FUNCTION)
        assert result == "function"


class TestStringToMessageRole:
    """Tests for string_to_message_role function."""

    def test_user_string(self) -> None:
        """Test converting 'user' string to role."""
        result = string_to_message_role("user")
        assert result == MessageRole.USER

    def test_human_alias(self) -> None:
        """Test converting 'human' alias to USER role."""
        result = string_to_message_role("human")
        assert result == MessageRole.USER

    def test_assistant_string(self) -> None:
        """Test converting 'assistant' string to role."""
        result = string_to_message_role("assistant")
        assert result == MessageRole.ASSISTANT

    def test_ai_alias(self) -> None:
        """Test converting 'ai' alias to ASSISTANT role."""
        result = string_to_message_role("ai")
        assert result == MessageRole.ASSISTANT

    def test_system_string(self) -> None:
        """Test converting 'system' string to role."""
        result = string_to_message_role("system")
        assert result == MessageRole.SYSTEM

    def test_tool_string(self) -> None:
        """Test converting 'tool' string to role."""
        result = string_to_message_role("tool")
        assert result == MessageRole.TOOL

    def test_case_insensitive(self) -> None:
        """Test case insensitive conversion."""
        assert string_to_message_role("USER") == MessageRole.USER
        assert string_to_message_role("Assistant") == MessageRole.ASSISTANT
        assert string_to_message_role("SYSTEM") == MessageRole.SYSTEM

    def test_whitespace_handling(self) -> None:
        """Test handling of whitespace."""
        assert string_to_message_role("  user  ") == MessageRole.USER
        assert string_to_message_role("\tassistant\n") == MessageRole.ASSISTANT

    def test_unknown_role_defaults_to_user(self) -> None:
        """Test that unknown roles default to USER."""
        result = string_to_message_role("unknown_role")
        assert result == MessageRole.USER


class TestChatMessageToMemoryPayload:
    """Tests for chat_message_to_memory_payload function."""

    def test_basic_conversion(self) -> None:
        """Test basic message to payload conversion."""
        message = ChatMessage.from_str("Hello!", role=MessageRole.USER)
        payload = chat_message_to_memory_payload(message, key="user_123", index=0)

        assert payload["type"] == "episodic"
        assert payload["importance"] == 0.5
        assert f"{CHAT_KEY_TAG_PREFIX}user_123" in payload["tags"]
        assert payload["metadata"]["chat_key"] == "user_123"
        assert payload["metadata"]["message_index"] == 0
        assert payload["metadata"]["role"] == "user"

    def test_assistant_message(self) -> None:
        """Test conversion of assistant message."""
        message = ChatMessage.from_str("I can help you!", role=MessageRole.ASSISTANT)
        payload = chat_message_to_memory_payload(message, key="sess_456", index=1)

        assert payload["metadata"]["role"] == "assistant"
        assert payload["metadata"]["message_index"] == 1
        assert payload["metadata"]["chat_key"] == "sess_456"

    def test_custom_importance(self) -> None:
        """Test custom importance setting."""
        message = ChatMessage.from_str("Important!", role=MessageRole.SYSTEM)
        payload = chat_message_to_memory_payload(
            message, key="key", index=0, importance=0.9
        )

        assert payload["importance"] == 0.9

    def test_custom_memory_type(self) -> None:
        """Test custom memory type setting."""
        message = ChatMessage.from_str("Fact", role=MessageRole.USER)
        payload = chat_message_to_memory_payload(
            message, key="key", index=0, memory_type="semantic"
        )

        assert payload["type"] == "semantic"

    def test_additional_tags(self) -> None:
        """Test adding additional tags."""
        message = ChatMessage.from_str("Tagged", role=MessageRole.USER)
        payload = chat_message_to_memory_payload(
            message, key="key", index=0, additional_tags=["important", "urgent"]
        )

        assert "important" in payload["tags"]
        assert "urgent" in payload["tags"]
        assert f"{CHAT_KEY_TAG_PREFIX}key" in payload["tags"]

    def test_additional_metadata(self) -> None:
        """Test adding additional metadata."""
        message = ChatMessage.from_str("Meta", role=MessageRole.USER)
        payload = chat_message_to_memory_payload(
            message, key="key", index=0, additional_metadata={"custom_field": "value"}
        )

        assert payload["metadata"]["custom_field"] == "value"

    def test_custom_tag_prefix(self) -> None:
        """Test custom tag prefix."""
        message = ChatMessage.from_str("Custom", role=MessageRole.USER)
        payload = chat_message_to_memory_payload(
            message, key="key", index=0, tag_prefix="custom_prefix:"
        )

        assert "custom_prefix:key" in payload["tags"]


class TestMemoryToChatMessage:
    """Tests for memory_to_chat_message function."""

    def test_basic_conversion(self, sample_memory: dict) -> None:
        """Test basic memory to message conversion."""
        message = memory_to_chat_message(sample_memory)

        assert message.role == MessageRole.USER

    def test_assistant_memory(self) -> None:
        """Test conversion of assistant memory."""
        memory = {
            "id": "mem_456",
            "content": "assistant: Sure, I can help!",
            "metadata": {
                "role": "assistant",
                "message_index": 1,
                "blocks": [{"block_type": "text", "text": "Sure, I can help!"}],
            },
        }

        message = memory_to_chat_message(memory)
        assert message.role == MessageRole.ASSISTANT

    def test_fallback_to_content(self) -> None:
        """Test fallback when no blocks in metadata."""
        memory = {
            "id": "mem_789",
            "content": "Hello there!",
            "metadata": {"role": "user"},
        }

        message = memory_to_chat_message(memory)
        assert message.role == MessageRole.USER

    def test_empty_metadata(self) -> None:
        """Test handling of empty metadata."""
        memory = {"id": "mem_000", "content": "No metadata"}

        message = memory_to_chat_message(memory)
        assert message.role == MessageRole.USER  # Default


class TestGetMessageIndex:
    """Tests for get_message_index function."""

    def test_extracts_index(self, sample_memory: dict) -> None:
        """Test extracting message index from memory."""
        index = get_message_index(sample_memory)
        assert index == 0

    def test_returns_default_when_missing(self) -> None:
        """Test default return when index is missing."""
        memory = {"id": "mem_123", "content": "test"}
        index = get_message_index(memory)
        assert index == 0

    def test_handles_empty_metadata(self) -> None:
        """Test handling of empty metadata."""
        memory = {"id": "mem_123", "content": "test", "metadata": {}}
        index = get_message_index(memory)
        assert index == 0


class TestGetChatKey:
    """Tests for get_chat_key function."""

    def test_extracts_key(self, sample_memory: dict) -> None:
        """Test extracting chat key from memory."""
        key = get_chat_key(sample_memory)
        assert key == "user_123"

    def test_returns_none_when_missing(self) -> None:
        """Test None return when key is missing."""
        memory = {"id": "mem_123", "content": "test"}
        key = get_chat_key(memory)
        assert key is None

    def test_handles_empty_metadata(self) -> None:
        """Test handling of empty metadata."""
        memory = {"id": "mem_123", "content": "test", "metadata": {}}
        key = get_chat_key(memory)
        assert key is None


# ========== MemoryLayerChatStore Class Tests ==========


class TestMemoryLayerChatStoreInit:
    """Tests for MemoryLayerChatStore initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization values."""
        store = MemoryLayerChatStore()

        assert store.base_url == "http://localhost:61001"
        assert store.api_key is None
        assert store.workspace_id is None
        assert store.timeout == 30.0

    def test_custom_initialization(self, base_url: str) -> None:
        """Test custom initialization values."""
        store = MemoryLayerChatStore(
            base_url=base_url,
            api_key="my_key",
            workspace_id="ws_custom",
            timeout=60.0,
        )

        assert store.base_url == base_url
        assert store.api_key == "my_key"
        assert store.workspace_id == "ws_custom"
        assert store.timeout == 60.0

    def test_class_name(self) -> None:
        """Test class_name method."""
        assert MemoryLayerChatStore.class_name() == "MemoryLayerChatStore"


# ========== Sync Method Tests ==========


class TestSyncSetMessages:
    """Tests for sync set_messages method."""

    @respx.mock
    def test_set_messages_empty_list(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test setting empty message list."""
        # Mock recall (for delete check) - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        chat_store.set_messages("user_123", [])

    @respx.mock
    def test_set_messages_single_message(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test setting a single message."""
        # Mock recall (for delete) - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Mock memory creation
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        message = ChatMessage.from_str("Hello!", role=MessageRole.USER)
        chat_store.set_messages("user_123", [message])

    @respx.mock
    def test_set_messages_multiple_messages(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test setting multiple messages."""
        # Mock recall - returns empty (no existing messages)
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Mock memory creation
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        messages = [
            ChatMessage.from_str("Hi!", role=MessageRole.USER),
            ChatMessage.from_str("Hello! How can I help?", role=MessageRole.ASSISTANT),
            ChatMessage.from_str("Tell me a joke", role=MessageRole.USER),
        ]
        chat_store.set_messages("user_123", messages)


class TestSyncGetMessages:
    """Tests for sync get_messages method."""

    @respx.mock
    def test_get_messages_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test getting messages when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        messages = chat_store.get_messages("user_123")

        # Verify
        assert messages == []

    @respx.mock
    def test_get_messages_returns_sorted(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test that messages are returned sorted by index."""
        # Mock recall - returns messages in reverse order
        mock_memories = [
            create_chat_memory("mem_2", "Message 2", 2, "user", "user_123"),
            create_chat_memory("mem_0", "Message 0", 0, "user", "user_123"),
            create_chat_memory("mem_1", "Message 1", 1, "assistant", "user_123"),
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 3}
            )
        )

        # Test
        messages = chat_store.get_messages("user_123")

        # Verify - should be sorted by index
        assert len(messages) == 3
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[2].role == MessageRole.USER


class TestSyncAddMessage:
    """Tests for sync add_message method."""

    @respx.mock
    def test_add_message_to_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test adding message when no messages exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Mock memory creation
        create_route = respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        message = ChatMessage.from_str("Hello!", role=MessageRole.USER)
        chat_store.add_message("user_123", message)

        # Verify - check that memory was created
        assert create_route.called

    @respx.mock
    def test_add_message_to_existing(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test adding message when messages already exist."""
        # Mock recall - returns one existing message
        mock_memories = [
            create_chat_memory("mem_0", "First message", 0, "user", "user_123"),
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 1}
            )
        )

        # Mock memory creation
        create_route = respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        message = ChatMessage.from_str("Second message", role=MessageRole.ASSISTANT)
        chat_store.add_message("user_123", message)

        # Verify
        assert create_route.called


class TestSyncDeleteMessages:
    """Tests for sync delete_messages method."""

    @respx.mock
    def test_delete_messages_none_exist(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting when no messages exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = chat_store.delete_messages("user_123")

        # Verify
        assert result is None

    @respx.mock
    def test_delete_messages_returns_deleted(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting messages returns the deleted messages."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "Message to delete",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            }
        ]

        # Mock recall - returns messages
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 1}
            )
        )

        # Mock delete
        respx.delete(f"{base_url}/v1/memories/mem_0").mock(
            return_value=Response(204)
        )

        # Test
        result = chat_store.delete_messages("user_123")

        # Verify
        assert result is not None
        assert len(result) == 1


class TestSyncDeleteMessage:
    """Tests for sync delete_message method."""

    @respx.mock
    def test_delete_message_not_found(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting message that doesn't exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = chat_store.delete_message("user_123", 0)

        # Verify
        assert result is None

    @respx.mock
    def test_delete_message_by_index(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting a specific message by index."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "Keep this",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            },
            {
                "id": "mem_1",
                "content": "Delete this",
                "metadata": {"message_index": 1, "role": "assistant", "chat_key": "user_123"},
            },
        ]

        # Mock recall
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 2}
            )
        )

        # Mock delete
        delete_route = respx.delete(f"{base_url}/v1/memories/mem_1").mock(
            return_value=Response(204)
        )

        # Test
        result = chat_store.delete_message("user_123", 1)

        # Verify
        assert result is not None
        assert result.role == MessageRole.ASSISTANT
        assert delete_route.called


class TestSyncDeleteLastMessage:
    """Tests for sync delete_last_message method."""

    @respx.mock
    def test_delete_last_message_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting last message when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = chat_store.delete_last_message("user_123")

        # Verify
        assert result is None

    @respx.mock
    def test_delete_last_message_success(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test deleting the last message."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "First",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            },
            {
                "id": "mem_1",
                "content": "Last",
                "metadata": {"message_index": 1, "role": "assistant", "chat_key": "user_123"},
            },
        ]

        # Mock recall (called twice - once for get_messages, once for delete_message)
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 2}
            )
        )

        # Mock delete
        respx.delete(f"{base_url}/v1/memories/mem_1").mock(
            return_value=Response(204)
        )

        # Test
        result = chat_store.delete_last_message("user_123")

        # Verify
        assert result is not None
        assert result.role == MessageRole.ASSISTANT


class TestSyncGetKeys:
    """Tests for sync get_keys method."""

    @respx.mock
    def test_get_keys_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test getting keys when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        keys = chat_store.get_keys()

        # Verify
        assert keys == []

    @respx.mock
    def test_get_keys_returns_unique(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test getting unique keys."""
        mock_memories = [
            create_chat_memory("mem_0", "content", 0, "user", "user_1"),
            create_chat_memory("mem_1", "content", 0, "user", "user_2"),
            create_chat_memory("mem_2", "content", 0, "user", "user_1"),  # Duplicate
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 3}
            )
        )

        # Test
        keys = chat_store.get_keys()

        # Verify - should be unique
        assert len(keys) == 2
        assert set(keys) == {"user_1", "user_2"}


# ========== Async Method Tests ==========


class TestAsyncSetMessages:
    """Tests for async aset_messages method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_aset_messages_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async setting empty message list."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        await chat_store.aset_messages("user_123", [])

    @pytest.mark.asyncio
    @respx.mock
    async def test_aset_messages_with_messages(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async setting messages."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Mock memory creation
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        messages = [
            ChatMessage.from_str("Hello", role=MessageRole.USER),
            ChatMessage.from_str("Hi there!", role=MessageRole.ASSISTANT),
        ]
        await chat_store.aset_messages("user_123", messages)


class TestAsyncGetMessages:
    """Tests for async aget_messages method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_aget_messages_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async getting messages when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        messages = await chat_store.aget_messages("user_123")

        # Verify
        assert messages == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_aget_messages_returns_sorted(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async messages are returned sorted by index."""
        mock_memories = [
            {
                "id": "mem_1",
                "content": "Second",
                "metadata": {"message_index": 1, "role": "assistant", "chat_key": "user_123"},
            },
            {
                "id": "mem_0",
                "content": "First",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            },
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 2}
            )
        )

        # Test
        messages = await chat_store.aget_messages("user_123")

        # Verify
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT


class TestAsyncAddMessage:
    """Tests for async async_add_message method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_add_message(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async adding a message."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Mock memory creation
        create_route = respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(200, json=create_memory_response())
        )

        # Test
        message = ChatMessage.from_str("Hello!", role=MessageRole.USER)
        await chat_store.async_add_message("user_123", message)

        # Verify
        assert create_route.called


class TestAsyncDeleteMessages:
    """Tests for async adelete_messages method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_messages_none_exist(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting when no messages exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = await chat_store.adelete_messages("user_123")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_messages_success(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting messages."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "To delete",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            }
        ]

        # Mock recall
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 1}
            )
        )

        # Mock delete
        respx.delete(f"{base_url}/v1/memories/mem_0").mock(
            return_value=Response(204)
        )

        # Test
        result = await chat_store.adelete_messages("user_123")

        # Verify
        assert result is not None
        assert len(result) == 1


class TestAsyncDeleteMessage:
    """Tests for async adelete_message method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_message_not_found(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting message that doesn't exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = await chat_store.adelete_message("user_123", 0)

        # Verify
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_message_success(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting a specific message."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "Delete me",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            }
        ]

        # Mock recall
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 1}
            )
        )

        # Mock delete
        respx.delete(f"{base_url}/v1/memories/mem_0").mock(
            return_value=Response(204)
        )

        # Test
        result = await chat_store.adelete_message("user_123", 0)

        # Verify
        assert result is not None
        assert result.role == MessageRole.USER


class TestAsyncDeleteLastMessage:
    """Tests for async adelete_last_message method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_last_message_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting last message when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        result = await chat_store.adelete_last_message("user_123")

        # Verify
        assert result is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_adelete_last_message_success(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async deleting the last message."""
        mock_memories = [
            {
                "id": "mem_0",
                "content": "First",
                "metadata": {"message_index": 0, "role": "user", "chat_key": "user_123"},
            },
            {
                "id": "mem_1",
                "content": "Last",
                "metadata": {"message_index": 1, "role": "assistant", "chat_key": "user_123"},
            },
        ]

        # Mock recall
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 2}
            )
        )

        # Mock delete
        respx.delete(f"{base_url}/v1/memories/mem_1").mock(
            return_value=Response(204)
        )

        # Test
        result = await chat_store.adelete_last_message("user_123")

        # Verify
        assert result is not None
        assert result.role == MessageRole.ASSISTANT


class TestAsyncGetKeys:
    """Tests for async aget_keys method."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_aget_keys_empty(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async getting keys when none exist."""
        # Mock recall - returns empty
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Test
        keys = await chat_store.aget_keys()

        # Verify
        assert keys == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_aget_keys_returns_unique(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async getting unique keys."""
        mock_memories = [
            create_chat_memory("mem_0", "content", 0, "user", "key_a"),
            create_chat_memory("mem_1", "content", 0, "user", "key_b"),
            create_chat_memory("mem_2", "content", 0, "user", "key_a"),  # Duplicate
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(
                200, json={"memories": mock_memories, "total_count": 3}
            )
        )

        # Test
        keys = await chat_store.aget_keys()

        # Verify
        assert len(keys) == 2
        assert set(keys) == {"key_a", "key_b"}


# ========== Error Handling Tests ==========


class TestErrorHandling:
    """Tests for error handling."""

    @respx.mock
    def test_api_error_handling(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test handling of API errors."""
        # Mock error response
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(500, json={"detail": "Internal server error"})
        )

        # Test
        with pytest.raises(RuntimeError) as exc_info:
            chat_store.get_messages("user_123")

        assert "500" in str(exc_info.value)
        assert "Internal server error" in str(exc_info.value)

    @respx.mock
    def test_unauthorized_error(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test handling of unauthorized errors."""
        # Mock 401 response
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(401, json={"detail": "Invalid API key"})
        )

        # Test
        with pytest.raises(RuntimeError) as exc_info:
            chat_store.get_messages("user_123")

        assert "401" in str(exc_info.value)
        assert "Invalid API key" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_async_api_error_handling(
        self, chat_store: MemoryLayerChatStore, base_url: str
    ) -> None:
        """Test async handling of API errors."""
        # Mock error response
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(400, json={"detail": "Bad request"})
        )

        # Test
        with pytest.raises(RuntimeError) as exc_info:
            await chat_store.aget_messages("user_123")

        assert "400" in str(exc_info.value)
        assert "Bad request" in str(exc_info.value)
