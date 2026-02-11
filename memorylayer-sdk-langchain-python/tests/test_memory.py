"""Unit tests for legacy BaseMemory implementations."""

import pytest
import respx
from httpx import Response
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from memorylayer_langchain import MemoryLayerMemory, MemoryLayerConversationSummaryMemory


@pytest.fixture
def memory(base_url: str, api_key: str, workspace_id: str, session_id: str) -> MemoryLayerMemory:
    """Create test MemoryLayerMemory instance."""
    with respx.mock:
        # Mock the initial message count request
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        return MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )


@pytest.fixture
def summary_memory(base_url: str, api_key: str, workspace_id: str, session_id: str) -> MemoryLayerConversationSummaryMemory:
    """Create test MemoryLayerConversationSummaryMemory instance."""
    with respx.mock:
        # Mock the initial message count request
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        return MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )


# ============================================================================
# MemoryLayerMemory Tests
# ============================================================================


def test_memory_variables(memory: MemoryLayerMemory) -> None:
    """Test that memory_variables returns the correct list."""
    assert memory.memory_variables == ["history"]


def test_memory_variables_custom_key(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test that memory_variables returns custom key when set."""
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            memory_key="chat_history",
        )
        assert memory.memory_variables == ["chat_history"]


@respx.mock
def test_load_memory_variables_empty(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test loading memory variables when history is empty."""
    # Mock empty response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify
    assert result == {"history": ""}


@respx.mock
def test_load_memory_variables_with_messages(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test loading memory variables with existing messages."""
    # Mock response with messages
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "content": "Hello",
                "metadata": {"role": "human", "message_index": 0},
            },
            {
                "id": "mem_124",
                "content": "Hi there!",
                "metadata": {"role": "ai", "message_index": 1},
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify - should be formatted string
    assert "Human: Hello" in result["history"]
    assert "AI: Hi there!" in result["history"]


@respx.mock
def test_load_memory_variables_return_messages(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test loading memory variables with return_messages=True."""
    # Create memory with return_messages=True
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            return_messages=True,
        )

    # Mock response with messages
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "content": "Hello",
                "metadata": {"role": "human", "message_index": 0},
            },
            {
                "id": "mem_124",
                "content": "Hi there!",
                "metadata": {"role": "ai", "message_index": 1},
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify - should be list of messages
    messages = result["history"]
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "Hello"
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Hi there!"


@respx.mock
def test_load_memory_variables_custom_prefixes(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test loading memory variables with custom prefixes."""
    # Create memory with custom prefixes
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            human_prefix="User",
            ai_prefix="Assistant",
        )

    # Mock response with messages
    mock_response = {
        "memories": [
            {
                "id": "mem_123",
                "content": "Hello",
                "metadata": {"role": "human", "message_index": 0},
            },
            {
                "id": "mem_124",
                "content": "Hi there!",
                "metadata": {"role": "ai", "message_index": 1},
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify - should use custom prefixes
    assert "User: Hello" in result["history"]
    assert "Assistant: Hi there!" in result["history"]


@respx.mock
def test_save_context(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test saving context (input and output)."""
    # Mock the POST response for storing memories
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
    memory.save_context(
        inputs={"input": "Hello"},
        outputs={"output": "Hi there!"}
    )

    # Verify - 2 memory creates (human + AI)
    assert len(respx.calls) == 2

    # Verify human message was stored
    first_request = respx.calls[0].request.content
    assert b"Hello" in first_request
    assert b"role:human" in first_request

    # Verify AI message was stored
    second_request = respx.calls[1].request.content
    assert b"Hi there!" in second_request
    assert b"role:ai" in second_request


@respx.mock
def test_save_context_with_input_output_keys(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test saving context with custom input/output keys."""
    # Create memory with custom keys
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            input_key="question",
            output_key="answer",
        )

    # Mock the POST response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_126",
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

    # Test with multiple keys in input/output
    memory.save_context(
        inputs={"question": "What is 2+2?", "other": "ignored"},
        outputs={"answer": "4", "confidence": "high"}
    )

    # Verify the correct keys were used
    first_request = respx.calls[0].request.content
    assert b"What is 2+2?" in first_request
    assert b"ignored" not in first_request

    second_request = respx.calls[1].request.content
    assert b'"4"' in second_request or b": 4" in second_request


@respx.mock
def test_save_context_non_string_values(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test saving context with non-string values."""
    # Mock the POST response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_127",
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

    # Test with non-string values (should be converted to strings)
    memory.save_context(
        inputs={"input": 42},
        outputs={"output": ["a", "list"]}
    )

    # Verify - should not raise and should convert to strings
    assert len(respx.calls) == 2


@respx.mock
def test_clear(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test clearing memory."""
    # Mock recall response
    mock_response = {
        "memories": [
            {"id": "mem_123", "content": "Hello", "metadata": {}},
            {"id": "mem_124", "content": "Hi", "metadata": {}},
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
    memory.clear()

    # Verify: 1 recall + 2 deletes
    assert len(respx.calls) == 3


@respx.mock
def test_clear_empty_history(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test clearing when history is already empty."""
    # Mock empty recall response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )

    # Test - should not raise
    memory.clear()

    # Verify only recall was called
    assert len(respx.calls) == 1


def test_headers_with_api_key_and_workspace(memory: MemoryLayerMemory) -> None:
    """Test that headers are set correctly with API key and workspace."""
    assert memory._headers["Authorization"] == "Bearer test_api_key"
    assert memory._headers["X-Workspace-ID"] == "ws_test"


def test_headers_without_api_key(base_url: str, session_id: str) -> None:
    """Test that headers are set correctly without API key."""
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
        )

    assert "Authorization" not in memory._headers
    assert "X-Workspace-ID" not in memory._headers


@respx.mock
def test_custom_memory_tags(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test that custom memory_tags are included."""
    # Create memory with custom tags
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            memory_tags=["user:user_123", "project:project_abc"],
        )

    # Mock the POST response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_128",
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
    memory.save_context(
        inputs={"input": "Test"},
        outputs={"output": "Response"}
    )

    # Verify custom tags are in the requests
    request_body = respx.calls[0].request.content
    assert b"user:user_123" in request_body
    assert b"project:project_abc" in request_body


@respx.mock
def test_http_error_on_save_context(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test HTTP error handling when saving context."""
    import httpx

    # Mock error response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test
    with pytest.raises(httpx.HTTPStatusError):
        memory.save_context(
            inputs={"input": "Test"},
            outputs={"output": "Response"}
        )


@respx.mock
def test_http_error_on_clear_recall(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test that clear handles HTTP error during recall gracefully."""
    # Mock error response on recall - _get_memories catches this and returns []
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test - should not raise since _get_memories catches the error
    memory.clear()

    # Verify only recall was called (no deletes since empty list returned)
    assert len(respx.calls) == 1


@respx.mock
def test_http_error_on_clear_delete(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test HTTP error handling when deleting during clear."""
    import httpx

    # Mock successful recall response
    mock_response = {
        "memories": [
            {"id": "mem_123", "content": "Hello", "metadata": {}},
        ],
        "total_count": 1,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Mock error response on delete - this should log warning but not raise
    respx.delete(f"{base_url}/v1/memories/mem_123").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test - the delete failure logs a warning but doesn't raise
    # (based on the implementation which uses logger.warning for delete failures)
    memory.clear()

    # Verify: 1 recall + 1 delete attempt
    assert len(respx.calls) == 2


@respx.mock
def test_load_memory_variables_on_http_error(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test that load_memory_variables returns empty on HTTP error."""
    # Mock error response
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test - should not raise but return empty
    result = memory.load_memory_variables({})

    # Verify
    assert result == {"history": ""}


@respx.mock
def test_messages_sorted_by_index(memory: MemoryLayerMemory, base_url: str) -> None:
    """Test that messages are sorted by message_index."""
    # Mock response with messages out of order
    mock_response = {
        "memories": [
            {
                "id": "mem_124",
                "content": "Second message",
                "metadata": {"role": "ai", "message_index": 1},
            },
            {
                "id": "mem_123",
                "content": "First message",
                "metadata": {"role": "human", "message_index": 0},
            },
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify order is correct (sorted by message_index)
    assert result["history"].index("First message") < result["history"].index("Second message")


# ============================================================================
# MemoryLayerConversationSummaryMemory Tests
# ============================================================================


def test_summary_memory_variables(summary_memory: MemoryLayerConversationSummaryMemory) -> None:
    """Test that memory_variables returns the correct list."""
    assert summary_memory.memory_variables == ["history"]


@respx.mock
def test_summary_load_memory_variables_empty(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test loading summary memory variables when history is empty."""
    # Mock reflect response with empty summary
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "", "reflection": "", "confidence": 0.9})
    )

    # Test
    result = summary_memory.load_memory_variables({})

    # Verify
    assert result == {"history": ""}


@respx.mock
def test_summary_load_memory_variables_with_summary(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test loading summary memory variables with existing summary."""
    # Mock reflect response
    summary_text = "The user greeted the assistant and discussed the weather."
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": summary_text, "reflection": "", "confidence": 0.9})
    )

    # Test
    result = summary_memory.load_memory_variables({})

    # Verify
    assert result["history"] == summary_text


@respx.mock
def test_summary_load_memory_variables_return_messages(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test loading summary memory variables with return_messages=True."""
    # Create memory with return_messages=True
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            return_messages=True,
        )

    # Mock reflect response
    summary_text = "The user greeted the assistant."
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": summary_text, "confidence": 0.9})
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify - should be list with SystemMessage
    messages = result["history"]
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == summary_text


@respx.mock
def test_summary_load_memory_variables_return_messages_empty(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test loading summary memory variables with return_messages=True when empty."""
    # Create memory with return_messages=True
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            return_messages=True,
        )

    # Mock reflect response with empty summary
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "", "confidence": 0.9})
    )

    # Test
    result = memory.load_memory_variables({})

    # Verify - should be empty list
    assert result["history"] == []


@respx.mock
def test_summary_save_context(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test saving context with summary memory."""
    # Mock the POST response for storing memories
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_130",
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
    summary_memory.save_context(
        inputs={"input": "Hello"},
        outputs={"output": "Hi there!"}
    )

    # Verify - 2 memory creates (human + AI)
    assert len(respx.calls) == 2


@respx.mock
def test_summary_clear(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test clearing summary memory."""
    # Mock recall response
    mock_response = {
        "memories": [
            {"id": "mem_131", "content": "Hello", "metadata": {}},
            {"id": "mem_132", "content": "Hi", "metadata": {}},
        ],
        "total_count": 2,
    }
    respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json=mock_response)
    )

    # Mock delete responses
    respx.delete(f"{base_url}/v1/memories/mem_131").mock(return_value=Response(204))
    respx.delete(f"{base_url}/v1/memories/mem_132").mock(return_value=Response(204))

    # Test
    summary_memory.clear()

    # Verify: 1 recall + 2 deletes
    assert len(respx.calls) == 3


@respx.mock
def test_summary_custom_prompt(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test using custom summary prompt."""
    custom_prompt = "Summarize the conversation for session {session_id} focusing on technical details."

    # Create memory with custom prompt
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            summary_prompt=custom_prompt,
        )

    # Mock reflect response
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "Technical summary", "confidence": 0.9})
    )

    # Test
    memory.load_memory_variables({})

    # Verify the request used the custom prompt
    request_body = respx.calls[0].request.content
    assert b"technical details" in request_body


@respx.mock
def test_summary_max_tokens(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test max_tokens parameter is sent to reflect endpoint."""
    # Create memory with custom max_tokens
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            max_tokens=1000,
        )

    # Mock reflect response
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "Summary", "confidence": 0.9})
    )

    # Test
    memory.load_memory_variables({})

    # Verify the request included max_tokens
    request_body = respx.calls[0].request.content
    assert b"1000" in request_body


@respx.mock
def test_summary_include_sources(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test include_sources parameter is sent to reflect endpoint."""
    # Create memory with include_sources=True
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            include_sources=True,
        )

    # Mock reflect response
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "Summary", "confidence": 0.9})
    )

    # Test
    memory.load_memory_variables({})

    # Verify the request included include_sources
    request_body = respx.calls[0].request.content
    assert b"include_sources" in request_body
    assert b"true" in request_body


@respx.mock
def test_summary_http_error_returns_empty(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test that summary returns empty string on HTTP error."""
    # Mock error response
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(500, json={"detail": "Internal server error"})
    )

    # Test - should not raise but return empty
    result = summary_memory.load_memory_variables({})

    # Verify
    assert result == {"history": ""}


def test_summary_headers_with_api_key_and_workspace(summary_memory: MemoryLayerConversationSummaryMemory) -> None:
    """Test that summary memory headers are set correctly."""
    assert summary_memory._headers["Authorization"] == "Bearer test_api_key"
    assert summary_memory._headers["X-Workspace-ID"] == "ws_test"


@respx.mock
def test_summary_uses_reflection_fallback(summary_memory: MemoryLayerConversationSummaryMemory, base_url: str) -> None:
    """Test that summary falls back to 'reflection' key if 'synthesis' is missing."""
    # Mock reflect response with only 'reflection' key
    respx.post(f"{base_url}/v1/memories/reflect").mock(
        return_value=Response(200, json={"reflection": "Fallback summary", "confidence": 0.9})
    )

    # Test
    result = summary_memory.load_memory_variables({})

    # Verify
    assert result["history"] == "Fallback summary"


@respx.mock
def test_summary_custom_memory_tags(base_url: str, api_key: str, workspace_id: str, session_id: str) -> None:
    """Test that custom memory_tags are included in summary memory."""
    # Create memory with custom tags
    with respx.mock:
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        memory = MemoryLayerConversationSummaryMemory(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            memory_tags=["user:user_456", "topic:weather"],
        )

    # Mock the POST response
    respx.post(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={
            "id": "mem_133",
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
    memory.save_context(
        inputs={"input": "Test"},
        outputs={"output": "Response"}
    )

    # Verify custom tags are in the requests
    request_body = respx.calls[0].request.content
    assert b"user:user_456" in request_body
    assert b"topic:weather" in request_body
