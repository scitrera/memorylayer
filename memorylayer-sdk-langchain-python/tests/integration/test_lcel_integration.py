"""Integration tests for LCEL chains with mocked LLM.

These tests verify that MemoryLayerChatMessageHistory works correctly
with LangChain's RunnableWithMessageHistory for LCEL-based chains.
"""

from typing import Any

import pytest
import respx
from httpx import Response
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from memorylayer_langchain import MemoryLayerChatMessageHistory


def extract_messages(prompt_value: Any) -> list[BaseMessage]:
    """Extract messages from a ChatPromptValue or list of messages."""
    if isinstance(prompt_value, ChatPromptValue):
        return prompt_value.to_messages()
    if isinstance(prompt_value, list):
        return prompt_value
    # Handle other prompt value types that have to_messages()
    if hasattr(prompt_value, "to_messages"):
        return prompt_value.to_messages()
    return list(prompt_value)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def base_url() -> str:
    """Test base URL."""
    return "http://test.memorylayer.ai"


@pytest.fixture
def api_key() -> str:
    """Test API key."""
    return "test_api_key"


@pytest.fixture
def workspace_id() -> str:
    """Test workspace ID."""
    return "ws_test"


def create_mock_memory_response(memory_id: str, content: str, role: str, index: int) -> dict:
    """Create a mock memory response."""
    return {
        "id": memory_id,
        "workspace_id": "ws_test",
        "content": content,
        "type": "episodic",
        "importance": 0.5,
        "tags": [f"session:sess_test", "chat_message", f"role:{role}"],
        "metadata": {
            "session_id": "sess_test",
            "role": role,
            "message_index": index,
        },
        "access_count": 0,
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
    }


# ============================================================================
# Mock LLM for Testing
# ============================================================================


class MockChatModel:
    """A mock chat model for testing LCEL chains.

    This mimics a LangChain chat model that returns predefined responses.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize with a list of responses to return in order."""
        self.responses = responses or ["I'm doing great, thank you for asking!"]
        self.call_count = 0
        self.call_history: list[list[BaseMessage]] = []

    def invoke(self, prompt_value: Any, **kwargs) -> AIMessage:
        """Process messages and return an AI response."""
        messages = extract_messages(prompt_value)
        self.call_history.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)


def mock_llm_function(responses: list[str]) -> RunnableLambda:
    """Create a mock LLM as a RunnableLambda for use in LCEL chains."""
    call_count = [0]  # Use list to allow mutation in closure

    def _invoke(prompt_value: Any) -> AIMessage:
        response = responses[call_count[0] % len(responses)]
        call_count[0] += 1
        return AIMessage(content=response)

    return RunnableLambda(_invoke)


# ============================================================================
# Integration Tests: MemoryLayerChatMessageHistory with LCEL
# ============================================================================


class TestLCELIntegrationWithMockedLLM:
    """Integration tests for LCEL chains with MemoryLayerChatMessageHistory."""

    @respx.mock
    def test_basic_chain_with_message_history(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test basic LCEL chain with RunnableWithMessageHistory."""
        # Setup mocks for MemoryLayer API
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create a simple prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        # Create mock LLM
        mock_llm = mock_llm_function(["Hello! How can I help you today?"])

        # Build the chain
        chain = prompt | mock_llm

        # Create history factory
        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        # Wrap with message history
        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Invoke the chain
        response = chain_with_history.invoke(
            {"input": "Hi there!"},
            config={"configurable": {"session_id": "sess_test"}},
        )

        # Verify response
        assert isinstance(response, AIMessage)
        assert response.content == "Hello! How can I help you today?"

    @respx.mock
    def test_chain_loads_existing_history(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that chain correctly loads existing conversation history."""
        # Mock existing conversation history
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello!", "human", 0),
            create_mock_memory_response("mem_2", "Hi there! How can I help?", "ai", 1),
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": existing_memories, "total_count": 2})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_3", "test", "human", 2),
            )
        )

        # Track what messages the LLM receives
        received_messages: list[list[BaseMessage]] = []

        def capture_invoke(prompt_value: Any) -> AIMessage:
            messages = extract_messages(prompt_value)
            received_messages.append(messages)
            return AIMessage(content="I remember our previous conversation!")

        # Create chain with capturing LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | RunnableLambda(capture_invoke)

        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Invoke chain
        response = chain_with_history.invoke(
            {"input": "What did we talk about?"},
            config={"configurable": {"session_id": "sess_test"}},
        )

        assert response.content == "I remember our previous conversation!"

        # Verify LLM received the history
        assert len(received_messages) == 1
        messages = received_messages[0]

        # Should have: system + history (2 messages) + new human message = 4
        assert len(messages) == 4
        # Check history was included
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello!"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].content == "Hi there! How can I help?"

    @respx.mock
    def test_multi_turn_conversation(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test multi-turn conversation with history persistence."""
        # Track stored memories for simulating persistence
        stored_memories: list[dict] = []
        memory_counter = [0]

        def create_memory_side_effect(request):
            """Side effect to track stored memories."""
            import json

            body = json.loads(request.content)
            mem_id = f"mem_{memory_counter[0]}"
            memory_counter[0] += 1

            memory = {
                "id": mem_id,
                "workspace_id": "ws_test",
                "content": body["content"],
                "type": "episodic",
                "importance": 0.5,
                "tags": body.get("tags", []),
                "metadata": body.get("metadata", {}),
                "access_count": 0,
                "created_at": "2026-01-27T10:00:00Z",
                "updated_at": "2026-01-27T10:00:00Z",
            }
            stored_memories.append(memory)
            return Response(200, json=memory)

        def recall_side_effect(request):
            """Return currently stored memories."""
            return Response(
                200,
                json={"memories": stored_memories.copy(), "total_count": len(stored_memories)},
            )

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(side_effect=create_memory_side_effect)

        # Create chain
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        responses = [
            "Hello! I'm your assistant.",
            "Your name is Alice, nice to meet you!",
            "Your name is Alice, as you told me earlier.",
        ]
        response_idx = [0]

        def mock_invoke(prompt_value: Any) -> AIMessage:
            resp = responses[response_idx[0] % len(responses)]
            response_idx[0] += 1
            return AIMessage(content=resp)

        chain = prompt | RunnableLambda(mock_invoke)

        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": "sess_test"}}

        # Turn 1
        response1 = chain_with_history.invoke({"input": "Hi!"}, config=config)
        assert response1.content == "Hello! I'm your assistant."

        # Turn 2
        response2 = chain_with_history.invoke({"input": "My name is Alice"}, config=config)
        assert response2.content == "Your name is Alice, nice to meet you!"

        # Turn 3
        response3 = chain_with_history.invoke({"input": "What's my name?"}, config=config)
        assert response3.content == "Your name is Alice, as you told me earlier."

        # Verify memories were stored (2 messages per turn = 6 total)
        # Note: The history adds messages after invoke, so we'd have 6 total
        assert len(stored_memories) >= 2  # At least first turn stored

    @respx.mock
    def test_multiple_sessions_isolated(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that different sessions maintain isolated histories."""
        # Track memories by session
        memories_by_session: dict[str, list[dict]] = {}

        def recall_side_effect(request):
            """Return memories for the specific session based on tags."""
            import json

            body = json.loads(request.content)
            tags = body.get("tags", [])

            # Find session tag
            session_id = None
            for tag in tags:
                if tag.startswith("session:"):
                    session_id = tag.split(":", 1)[1]
                    break

            memories = memories_by_session.get(session_id, [])
            return Response(200, json={"memories": memories, "total_count": len(memories)})

        def create_side_effect(request):
            """Store memory for the correct session."""
            import json

            body = json.loads(request.content)
            metadata = body.get("metadata", {})
            session_id = metadata.get("session_id", "unknown")

            if session_id not in memories_by_session:
                memories_by_session[session_id] = []

            memory = {
                "id": f"mem_{len(memories_by_session[session_id])}",
                "workspace_id": "ws_test",
                "content": body["content"],
                "type": "episodic",
                "importance": 0.5,
                "tags": body.get("tags", []),
                "metadata": metadata,
                "access_count": 0,
                "created_at": "2026-01-27T10:00:00Z",
                "updated_at": "2026-01-27T10:00:00Z",
            }
            memories_by_session[session_id].append(memory)
            return Response(200, json=memory)

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(side_effect=create_side_effect)

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        def echo_invoke(prompt_value: Any) -> AIMessage:
            # Count history messages (excluding the current input)
            messages = extract_messages(prompt_value)
            history_count = len(messages) - 1  # -1 for current human message
            return AIMessage(content=f"Received {history_count} history messages")

        chain = prompt | RunnableLambda(echo_invoke)

        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Session 1 - first message
        response1 = chain_with_history.invoke(
            {"input": "Hello from session 1"},
            config={"configurable": {"session_id": "sess_1"}},
        )
        assert "0 history" in response1.content

        # Session 2 - first message (should have empty history)
        response2 = chain_with_history.invoke(
            {"input": "Hello from session 2"},
            config={"configurable": {"session_id": "sess_2"}},
        )
        assert "0 history" in response2.content

        # Verify sessions are isolated
        assert "sess_1" in memories_by_session
        assert "sess_2" in memories_by_session

    @respx.mock
    def test_chain_with_system_prompt(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test LCEL chain with system prompt and message history."""
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Track received messages
        received_messages: list[list[BaseMessage]] = []

        def capture_invoke(prompt_value: Any) -> AIMessage:
            messages = extract_messages(prompt_value)
            received_messages.append(messages)
            return AIMessage(content="I am a Python expert!")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a Python programming expert. Always respond helpfully."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | RunnableLambda(capture_invoke)

        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        response = chain_with_history.invoke(
            {"input": "What's the best way to learn Python?"},
            config={"configurable": {"session_id": "sess_test"}},
        )

        assert response.content == "I am a Python expert!"

        # Verify system message was included
        assert len(received_messages) == 1
        messages = received_messages[0]
        assert len(messages) >= 2  # At least system + human
        # First message should be system message
        assert messages[0].content == "You are a Python programming expert. Always respond helpfully."

    @respx.mock
    def test_chain_handles_api_errors_gracefully(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that chain handles MemoryLayer API errors gracefully for reads."""
        # Simulate API returning empty on error (graceful degradation)
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(500, json={"detail": "Internal server error"})
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        def simple_invoke(prompt_value: Any) -> AIMessage:
            # Should still work with empty history on error
            return AIMessage(content="I can still respond!")

        chain = prompt | RunnableLambda(simple_invoke)

        def get_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
            return MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url=base_url,
                api_key=api_key,
                workspace_id=workspace_id,
            )

        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # Should not raise - history returns empty list on error
        response = chain_with_history.invoke(
            {"input": "Hello"},
            config={"configurable": {"session_id": "sess_test"}},
        )
        assert response.content == "I can still respond!"


class TestChatMessageHistoryDirect:
    """Direct integration tests for MemoryLayerChatMessageHistory."""

    @respx.mock
    def test_history_add_and_retrieve_messages(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test adding and retrieving messages through the history interface."""
        stored_messages: list[dict] = []
        msg_counter = [0]

        def create_side_effect(request):
            import json

            body = json.loads(request.content)
            memory = create_mock_memory_response(
                f"mem_{msg_counter[0]}",
                body["content"],
                body["metadata"]["role"],
                msg_counter[0],
            )
            stored_messages.append(memory)
            msg_counter[0] += 1
            return Response(200, json=memory)

        def recall_side_effect(request):
            return Response(
                200,
                json={"memories": stored_messages.copy(), "total_count": len(stored_messages)},
            )

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(side_effect=create_side_effect)

        history = MemoryLayerChatMessageHistory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Add messages
        history.add_user_message("Hello!")
        history.add_ai_message("Hi there!")
        history.add_user_message("How are you?")

        # Retrieve and verify
        messages = history.messages
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello!"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "Hi there!"
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "How are you?"

    @respx.mock
    def test_history_clear(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test clearing message history."""
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello", "human", 0),
            create_mock_memory_response("mem_2", "Hi!", "ai", 1),
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": existing_memories, "total_count": 2})
        )
        respx.delete(f"{base_url}/v1/memories/mem_1").mock(return_value=Response(204))
        respx.delete(f"{base_url}/v1/memories/mem_2").mock(return_value=Response(204))

        history = MemoryLayerChatMessageHistory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Clear should delete all memories
        history.clear()

        # Verify delete calls were made
        delete_calls = [
            call for call in respx.calls
            if call.request.method == "DELETE"
        ]
        assert len(delete_calls) == 2

    @respx.mock
    def test_history_with_custom_tags(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that custom tags are included in stored memories."""
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        request_bodies: list[dict] = []

        def capture_create(request):
            import json

            body = json.loads(request.content)
            request_bodies.append(body)
            return Response(
                200,
                json=create_mock_memory_response("mem_1", body["content"], "human", 0),
            )

        respx.post(f"{base_url}/v1/memories").mock(side_effect=capture_create)

        history = MemoryLayerChatMessageHistory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            memory_tags=["project:test", "user:alice"],
        )

        history.add_user_message("Test message")

        # Verify custom tags were included
        assert len(request_bodies) == 1
        tags = request_bodies[0].get("tags", [])
        assert "project:test" in tags
        assert "user:alice" in tags
        assert "session:sess_test" in tags
        assert "chat_message" in tags
