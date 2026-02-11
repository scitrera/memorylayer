"""Unit tests for MemoryLayerChatMessageHistory."""

import pytest
import respx
from httpx import Response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memorylayer_langchain import MemoryLayerChatMessageHistory


@pytest.fixture
def chat_history(base_url: str, api_key: str, workspace_id: str, session_id: str) -> MemoryLayerChatMessageHistory:
    """Create test chat message history."""
    return MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url=base_url,
        api_key=api_key,
        workspace_id=workspace_id,
    )


@respx.mock
def test_add_message_human(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test adding a human message."""
    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock the POST response for storing memory
    mock_response = {
        "id": "mem_123",
        "workspace_id": "ws_test",
        "content": "Hello, how are you?",
        "type": "episodic",
        "importance": 0.5,
        "tags": ["session:sess_test_123", "chat_message", "role:human"],
        "metadata": {"session_id": "sess_test_123", "role": "human", "message_index": 0},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }
    respx.post(f"{base_url}/v1/memories").mock(return_value=Response(200, json=mock_response))

    # Test
    chat_history.add_message(HumanMessage(content="Hello, how are you?"))

    # Verify the request was made
    assert len(respx.calls) == 2
    # First call is to get current messages
    assert respx.calls[0].request.url.path == "/v1/memories/recall"
    # Second call is to create the memory
    assert respx.calls[1].request.url.path == "/v1/memories"


@respx.mock
def test_add_message_ai(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test adding an AI message."""
    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock the POST response for storing memory
    mock_response = {
        "id": "mem_124",
        "workspace_id": "ws_test",
        "content": "I'm doing well, thank you!",
        "type": "episodic",
        "importance": 0.5,
        "tags": ["session:sess_test_123", "chat_message", "role:ai"],
        "metadata": {"session_id": "sess_test_123", "role": "ai", "message_index": 0},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }
    respx.post(f"{base_url}/v1/memories").mock(return_value=Response(200, json=mock_response))

    # Test
    chat_history.add_message(AIMessage(content="I'm doing well, thank you!"))

    # Verify the request was made with correct role tag
    assert len(respx.calls) == 2
    request_body = respx.calls[1].request.content
    assert b"role:ai" in request_body


@respx.mock
def test_add_messages_multiple(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test adding multiple messages at once."""
    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock POST responses for storing memories
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_125",
            "workspace_id": "ws_test",
            "content": "test",
            "type": "episodic",
            "importance": 0.5,
            "tags": [],
            "metadata": {},
            "access_count": 0,
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:00:00Z",
        })
    )

    # Test
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
        HumanMessage(content="How are you?"),
    ]
    chat_history.add_messages(messages)

    # Verify: 1 recall + 3 memory creates
    assert len(respx.calls) == 4


@respx.mock
def test_messages_property_empty(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test retrieving messages when history is empty."""
    # Mock empty response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Test
    messages = chat_history.messages

    # Verify
    assert messages == []


@respx.mock
def test_messages_property_with_messages(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test retrieving messages from history."""
    # Mock response with messages
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "workspace_id": "ws_test",
                "content": "Hello",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message", "role:human"],
                "metadata": {
                    "session_id": "sess_test_123",
                    "role": "human",
                    "message_index": 0,
                },
                "access_count": 0,
                "created_at": "2026-01-26T10:00:00Z",
                "updated_at": "2026-01-26T10:00:00Z",
            },
            {
                "id": "mem_124",
                "workspace_id": "ws_test",
                "content": "Hi there!",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message", "role:ai"],
                "metadata": {
                    "session_id": "sess_test_123",
                    "role": "ai",
                    "message_index": 1,
                },
                "access_count": 0,
                "created_at": "2026-01-26T10:00:01Z",
                "updated_at": "2026-01-26T10:00:01Z",
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    messages = chat_history.messages

    # Verify
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there!"


@respx.mock
def test_messages_sorted_by_index(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test that messages are sorted by message_index."""
    # Mock response with messages out of order
    mock_response = {
        "memories": [
            {
                "id": "mem_124",
                "workspace_id": "ws_test",
                "content": "Second message",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message", "role:ai"],
                "metadata": {
                    "session_id": "sess_test_123",
                    "role": "ai",
                    "message_index": 1,
                },
                "access_count": 0,
                "created_at": "2026-01-26T10:00:01Z",
                "updated_at": "2026-01-26T10:00:01Z",
            },
            {
                "id": "mem_123",
                "workspace_id": "ws_test",
                "content": "First message",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message", "role:human"],
                "metadata": {
                    "session_id": "sess_test_123",
                    "role": "human",
                    "message_index": 0,
                },
                "access_count": 0,
                "created_at": "2026-01-26T10:00:00Z",
                "updated_at": "2026-01-26T10:00:00Z",
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    messages = chat_history.messages

    # Verify order is correct (sorted by message_index)
    assert len(messages) == 2
    assert messages[0].content == "First message"
    assert messages[1].content == "Second message"


@respx.mock
def test_clear(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test clearing message history."""
    # Mock recall response
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "workspace_id": "ws_test",
                "content": "Hello",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message"],
                "metadata": {"session_id": "sess_test_123"},
                "access_count": 0,
                "created_at": "2026-01-26T10:00:00Z",
                "updated_at": "2026-01-26T10:00:00Z",
            },
            {
                "id": "mem_124",
                "workspace_id": "ws_test",
                "content": "Hi",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message"],
                "metadata": {"session_id": "sess_test_123"},
                "access_count": 0,
                "created_at": "2026-01-26T10:00:01Z",
                "updated_at": "2026-01-26T10:00:01Z",
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Mock delete responses
    respx.delete(f"{base_url}/v1/memories/mem_123").mock(return_value=Response(204))
    respx.delete(f"{base_url}/v1/memories/mem_124").mock(return_value=Response(204))

    # Test
    chat_history.clear()

    # Verify: 1 recall + 2 deletes
    assert len(respx.calls) == 3


@respx.mock
def test_clear_empty_history(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test clearing when history is already empty."""
    # Mock empty recall response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Test - should not raise
    chat_history.clear()

    # Verify only recall was called
    assert len(respx.calls) == 1


@respx.mock
def test_system_message(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test adding and retrieving system messages."""
    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock the POST response for storing memory
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_126",
            "workspace_id": "ws_test",
            "content": "You are a helpful assistant.",
            "type": "episodic",
            "importance": 0.5,
            "tags": ["session:sess_test_123", "chat_message", "role:system"],
            "metadata": {"session_id": "sess_test_123", "role": "system", "message_index": 0},
            "access_count": 0,
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:00:00Z",
        })
    )

    # Test
    chat_history.add_message(SystemMessage(content="You are a helpful assistant."))

    # Verify the request was made with correct role tag
    request_body = respx.calls[1].request.content
    assert b"role:system" in request_body


@respx.mock
def test_message_with_full_reconstruction(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test that messages with message_data are fully reconstructed."""
    # Mock response with message_data for full reconstruction
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "workspace_id": "ws_test",
                "content": "Hello",
                "type": "episodic",
                "importance": 0.5,
                "tags": ["session:sess_test_123", "chat_message", "role:human"],
                "metadata": {
                    "session_id": "sess_test_123",
                    "role": "human",
                    "message_index": 0,
                    "message_data": {
                        "type": "human",
                        "data": {
                            "content": "Hello",
                            "additional_kwargs": {},
                            "response_metadata": {},
                        },
                    },
                },
                "access_count": 0,
                "created_at": "2026-01-26T10:00:00Z",
                "updated_at": "2026-01-26T10:00:00Z",
            },
        ],
        "total_count": 1,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    messages = chat_history.messages

    # Verify
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"


@respx.mock
def test_custom_memory_tags(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test that custom memory_tags are included."""
    chat_history = MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url=base_url,
        api_key=api_key,
        workspace_id=workspace_id,
        memory_tags=["user:user_123", "project:project_abc"],
    )

    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock the POST response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_127",
            "workspace_id": "ws_test",
            "content": "Test",
            "type": "episodic",
            "importance": 0.5,
            "tags": [],
            "metadata": {},
            "access_count": 0,
            "created_at": "2026-01-26T10:00:00Z",
            "updated_at": "2026-01-26T10:00:00Z",
        })
    )

    # Test
    chat_history.add_message(HumanMessage(content="Test"))

    # Verify custom tags are in the request
    request_body = respx.calls[1].request.content
    assert b"user:user_123" in request_body
    assert b"project:project_abc" in request_body


@respx.mock
def test_http_error_on_add_message(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test HTTP error handling when adding messages."""
    import httpx

    # Mock the empty messages response for getting current count
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Mock error response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test
    with pytest.raises(httpx.HTTPStatusError):
        chat_history.add_message(HumanMessage(content="Test"))


@respx.mock
def test_http_error_on_clear(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test HTTP error handling when clearing."""
    import httpx

    # Mock error response on recall
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test
    with pytest.raises(httpx.HTTPStatusError):
        chat_history.clear()


@respx.mock
def test_http_error_on_messages_returns_empty(chat_history: MemoryLayerChatMessageHistory, base_url: str) -> None:
    """Test that messages property returns empty list on HTTP error."""
    # Mock error response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test - should not raise but return empty list
    messages = chat_history.messages

    # Verify
    assert messages == []


def test_headers_with_api_key_and_workspace() -> None:
    """Test that headers are set correctly with API key and workspace."""
    history = MemoryLayerChatMessageHistory(
        session_id="test",
        base_url="http://localhost:61001",
        api_key="my_api_key",
        workspace_id="ws_123",
    )

    assert history._headers["Authorization"] == "Bearer my_api_key"
    assert history._headers["X-Workspace-ID"] == "ws_123"


def test_headers_without_api_key() -> None:
    """Test that headers are set correctly without API key."""
    history = MemoryLayerChatMessageHistory(
        session_id="test",
        base_url="http://localhost:61001",
    )

    assert "Authorization" not in history._headers
    assert "X-Workspace-ID" not in history._headers


def test_base_url_trailing_slash_stripped() -> None:
    """Test that trailing slash is stripped from base_url."""
    history = MemoryLayerChatMessageHistory(
        session_id="test",
        base_url="http://localhost:61001/",
    )

    assert history.base_url == "http://localhost:61001"


def test_message_to_role_conversion(chat_history: MemoryLayerChatMessageHistory) -> None:
    """Test message to role conversion."""
    assert chat_history._message_to_role(HumanMessage(content="test")) == "human"
    assert chat_history._message_to_role(AIMessage(content="test")) == "ai"
    assert chat_history._message_to_role(SystemMessage(content="test")) == "system"


def test_role_to_message_class_conversion(chat_history: MemoryLayerChatMessageHistory) -> None:
    """Test role to message class conversion."""
    assert chat_history._role_to_message_class("human") == HumanMessage
    assert chat_history._role_to_message_class("ai") == AIMessage
    assert chat_history._role_to_message_class("system") == SystemMessage
    # Unknown role defaults to HumanMessage
    assert chat_history._role_to_message_class("unknown") == HumanMessage
