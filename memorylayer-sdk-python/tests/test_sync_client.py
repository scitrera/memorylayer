"""Unit tests for synchronous MemoryLayer client."""

import pytest
import respx
from httpx import Response

from memorylayer import (
    AuthenticationError,
    Memory,
    MemoryType,
    NotFoundError,
    RecallResult,
    ReflectResult,
    RelationshipType,
    SyncMemoryLayerClient,
    sync_client,
)


@pytest.fixture
def base_url() -> str:
    """Test base URL."""
    return "http://test.memorylayer.ai"


@pytest.fixture
def client(base_url: str) -> SyncMemoryLayerClient:
    """Create test client."""
    return SyncMemoryLayerClient(
        base_url=base_url,
        api_key="test_api_key",
        workspace_id="ws_test",
    )


@respx.mock
def test_remember(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test storing a memory."""
    # Mock response
    mock_response = {
        "id": "mem_123",
        "workspace_id": "ws_test",
        "content": "User prefers Python",
        "type": "semantic",
        "importance": 0.8,
        "tags": ["preferences"],
        "metadata": {},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }

    respx.post(f"{base_url}/v1/memories").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        memory = client.remember(
            content="User prefers Python",
            type=MemoryType.SEMANTIC,
            importance=0.8,
            tags=["preferences"],
        )

    # Verify
    assert isinstance(memory, Memory)
    assert memory.id == "mem_123"
    assert memory.content == "User prefers Python"
    assert memory.type == MemoryType.SEMANTIC
    assert memory.importance == 0.8


@respx.mock
def test_recall(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test recalling memories."""
    # Mock response
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "workspace_id": "ws_test",
                "content": "User prefers Python",
                "type": "semantic",
                "importance": 0.8,
                "tags": ["preferences"],
                "metadata": {},
                "access_count": 1,
                "created_at": "2026-01-26T10:00:00Z",
                "updated_at": "2026-01-26T10:00:00Z",
            }
        ],
        "total_count": 1,
        "query_tokens": 5,
        "search_latency_ms": 45,
    }

    respx.post(f"{base_url}/v1/memories/recall").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        result = client.recall(query="coding preferences", limit=5)

    # Verify
    assert isinstance(result, RecallResult)
    assert len(result.memories) == 1
    assert result.total_count == 1
    assert result.search_latency_ms == 45


@respx.mock
def test_reflect(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test reflecting on memories."""
    # Mock response
    mock_response = {
        "reflection": "User prefers Python and FastAPI for backend development.",
        "source_memories": ["mem_123", "mem_456"],
        "confidence": 0.9,
        "tokens_processed": 150,
    }

    respx.post(f"{base_url}/v1/memories/reflect").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        result = client.reflect(query="summarize user preferences")

    # Verify
    assert isinstance(result, ReflectResult)
    assert "Python" in result.reflection
    assert len(result.source_memories) == 2
    assert result.confidence == 0.9


@respx.mock
def test_get_memory(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test getting a specific memory."""
    # Mock response
    mock_response = {
        "id": "mem_123",
        "workspace_id": "ws_test",
        "content": "User prefers Python",
        "type": "semantic",
        "importance": 0.8,
        "tags": ["preferences"],
        "metadata": {},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }

    respx.get(f"{base_url}/v1/memories/mem_123").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        memory = client.get_memory("mem_123")

    # Verify
    assert memory.id == "mem_123"
    assert memory.content == "User prefers Python"


@respx.mock
def test_forget(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test forgetting a memory."""
    respx.delete(f"{base_url}/v1/memories/mem_123").mock(return_value=Response(204))

    # Test
    with client:
        result = client.forget("mem_123")

    # Verify
    assert result is True


@respx.mock
def test_associate(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test creating an association."""
    # Mock response
    mock_response = {
        "id": "assoc_123",
        "workspace_id": "ws_test",
        "source_id": "mem_123",
        "target_id": "mem_456",
        "relationship": "solves",
        "strength": 0.9,
        "metadata": {},
        "created_at": "2026-01-26T10:00:00Z",
    }

    respx.post(f"{base_url}/v1/memories/mem_123/associate").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        association = client.associate(
            source_id="mem_123",
            target_id="mem_456",
            relationship=RelationshipType.SOLVES,
            strength=0.9,
        )

    # Verify
    assert association.id == "assoc_123"
    assert association.source_id == "mem_123"
    assert association.target_id == "mem_456"
    assert association.relationship == "solves"


@respx.mock
def test_authentication_error(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test authentication error handling."""
    respx.post(f"{base_url}/v1/memories").mock(return_value=Response(401, json={"detail": "Invalid API key"}))

    with pytest.raises(AuthenticationError) as exc_info:
        with client:
            client.remember(content="test")

    assert "Invalid API key" in str(exc_info.value)


@respx.mock
def test_not_found_error(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test not found error handling."""
    respx.get(f"{base_url}/v1/memories/mem_999").mock(return_value=Response(404, json={"detail": "Memory not found"}))

    with pytest.raises(NotFoundError) as exc_info:
        with client:
            client.get_memory("mem_999")

    assert "Memory not found" in str(exc_info.value)


@respx.mock
def test_create_session(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test creating a session."""
    # Mock response
    mock_response = {
        "id": "sess_123",
        "workspace_id": "ws_test",
        "metadata": {},
        "expires_at": "2026-01-26T11:00:00Z",
        "created_at": "2026-01-26T10:00:00Z",
    }

    respx.post(f"{base_url}/v1/sessions").mock(return_value=Response(200, json=mock_response))

    # Test
    with client:
        session = client.create_session(ttl_seconds=3600)

    # Verify
    assert session.id == "sess_123"
    assert session.workspace_id == "ws_test"


@respx.mock
def test_session_context(client: SyncMemoryLayerClient, base_url: str) -> None:
    """Test session context operations."""
    # Mock set context - return empty success response
    respx.post(f"{base_url}/v1/sessions/sess_123/context").mock(return_value=Response(200, json={}))

    # Mock get context
    mock_context = {
        "context": {
            "current_file": {"path": "auth.py", "line": 42},
        }
    }
    respx.get(f"{base_url}/v1/sessions/sess_123/context").mock(return_value=Response(200, json=mock_context))

    # Test
    with client:
        # Set context
        client.set_context("sess_123", "current_file", {"path": "auth.py", "line": 42})

        # Get context
        context = client.get_context("sess_123", ["current_file"])

    # Verify
    assert "current_file" in context
    assert context["current_file"]["path"] == "auth.py"


@respx.mock
def test_sync_client_helper(base_url: str) -> None:
    """Test sync_client() context manager helper."""
    # Mock response
    mock_response = {
        "id": "mem_456",
        "workspace_id": "ws_test",
        "content": "Test memory",
        "type": "working",
        "importance": 0.5,
        "tags": [],
        "metadata": {},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }

    respx.post(f"{base_url}/v1/memories").mock(return_value=Response(200, json=mock_response))

    # Test
    with sync_client(
        base_url=base_url,
        api_key="test_api_key",
        workspace_id="ws_test",
    ) as client:
        memory = client.remember(content="Test memory")

    # Verify
    assert isinstance(memory, Memory)
    assert memory.id == "mem_456"
    assert memory.content == "Test memory"


def test_manual_lifecycle(client: SyncMemoryLayerClient) -> None:
    """Test manual connect() and close() lifecycle."""
    # Verify client not initialized
    assert client._client is None

    # Connect
    client.connect()
    assert client._client is not None

    # Close
    client.close()
    # Note: httpx.Client.close() doesn't set _client to None, just closes connections


def test_ensure_client_raises_error() -> None:
    """Test that using client without initialization raises error."""
    client = SyncMemoryLayerClient(base_url="http://test.memorylayer.ai")

    with pytest.raises(RuntimeError, match="Client not initialized"):
        client._ensure_client()
