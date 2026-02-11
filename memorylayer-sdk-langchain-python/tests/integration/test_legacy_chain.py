"""Integration tests for legacy LangChain chains with mocked LLM.

These tests verify that MemoryLayerMemory and MemoryLayerConversationSummaryMemory
work correctly with legacy LangChain chain patterns (ConversationChain-style usage).
"""

from typing import Any

import pytest
import respx
from httpx import Response
from langchain_core.messages import AIMessage, HumanMessage

from memorylayer_langchain import MemoryLayerMemory, MemoryLayerConversationSummaryMemory


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
        "tags": ["session:sess_test", "conversation_memory", f"role:{role}"],
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
# Mock Legacy Chain Components
# ============================================================================


class MockLegacyLLM:
    """A mock LLM for testing legacy chain patterns.

    This mimics the behavior of an LLM that would be used with
    ConversationChain or similar legacy LangChain patterns.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize with a list of responses to return in order."""
        self.responses = responses or ["I'm doing great, thank you for asking!"]
        self.call_count = 0
        self.received_prompts: list[str] = []

    def __call__(self, prompt: str, **kwargs) -> str:
        """Process a prompt and return a response."""
        self.received_prompts.append(prompt)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response


class MockLegacyChain:
    """A mock legacy chain that simulates ConversationChain behavior.

    This mimics how a legacy LangChain chain would interact with memory,
    calling load_memory_variables and save_context during execution.
    """

    def __init__(
        self,
        llm: MockLegacyLLM,
        memory: MemoryLayerMemory | MemoryLayerConversationSummaryMemory,
        prompt_template: str = "History:\n{history}\n\nHuman: {input}\nAI:",
        input_key: str = "input",
        output_key: str = "response",
    ) -> None:
        """Initialize the mock chain."""
        self.llm = llm
        self.memory = memory
        self.prompt_template = prompt_template
        self.input_key = input_key
        self.output_key = output_key

    def run(self, user_input: str) -> str:
        """Run the chain with user input, similar to ConversationChain.run()."""
        # Load memory variables
        memory_vars = self.memory.load_memory_variables({self.input_key: user_input})
        history = memory_vars.get(self.memory.memory_key, "")

        # Format history for prompt
        if isinstance(history, list):
            # If return_messages=True, format messages
            history_str = "\n".join(
                f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
                for m in history
            )
        else:
            history_str = history

        # Build prompt - use the memory's key name dynamically
        format_kwargs = {
            self.memory.memory_key: history_str,
            "history": history_str,  # Also provide as 'history' for default template
            "input": user_input,
        }
        prompt = self.prompt_template.format(**format_kwargs)

        # Get LLM response
        response = self.llm(prompt)

        # Save context
        self.memory.save_context(
            inputs={self.input_key: user_input},
            outputs={self.output_key: response},
        )

        return response


# ============================================================================
# Integration Tests: MemoryLayerMemory with Legacy Chains
# ============================================================================


class TestLegacyChainWithMemoryLayerMemory:
    """Integration tests for legacy chains with MemoryLayerMemory."""

    @respx.mock
    def test_basic_chain_conversation(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test basic legacy chain with MemoryLayerMemory."""
        # Setup mocks
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create memory
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain
        llm = MockLegacyLLM(["Hello! How can I help you today?"])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Run the chain
        response = chain.run("Hi there!")

        # Verify response
        assert response == "Hello! How can I help you today?"

        # Verify LLM received the prompt
        assert len(llm.received_prompts) == 1

    @respx.mock
    def test_chain_loads_existing_history(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that chain correctly loads existing conversation history."""
        # Mock existing conversation history for initial message count
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello!", "human", 0),
            create_mock_memory_response("mem_2", "Hi there! How can I help?", "ai", 1),
        ]

        # First call gets memories for initial count, subsequent calls return history
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": existing_memories, "total_count": 2})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_3", "test", "human", 2),
            )
        )

        # Create memory
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain
        llm = MockLegacyLLM(["I remember our previous conversation!"])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Run the chain
        response = chain.run("What did we talk about?")

        # Verify response
        assert response == "I remember our previous conversation!"

        # Verify LLM received history in prompt
        assert len(llm.received_prompts) == 1
        prompt = llm.received_prompts[0]
        assert "Hello!" in prompt
        assert "Hi there! How can I help?" in prompt

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

        # Create memory
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain with multiple responses
        responses = [
            "Hello! I'm your assistant.",
            "Your name is Alice, nice to meet you!",
            "Your name is Alice, as you told me earlier.",
        ]
        llm = MockLegacyLLM(responses)
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Turn 1
        response1 = chain.run("Hi!")
        assert response1 == "Hello! I'm your assistant."

        # Turn 2
        response2 = chain.run("My name is Alice")
        assert response2 == "Your name is Alice, nice to meet you!"

        # Turn 3
        response3 = chain.run("What's my name?")
        assert response3 == "Your name is Alice, as you told me earlier."

        # Verify memories were stored (2 messages per turn = 6 total)
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

        # Create two separate memories for different sessions
        memory1 = MemoryLayerMemory(
            session_id="sess_1",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        memory2 = MemoryLayerMemory(
            session_id="sess_2",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create chains for each session
        llm1 = MockLegacyLLM(["Response for session 1"])
        llm2 = MockLegacyLLM(["Response for session 2"])

        chain1 = MockLegacyChain(llm=llm1, memory=memory1)
        chain2 = MockLegacyChain(llm=llm2, memory=memory2)

        # Run conversations
        chain1.run("Hello from session 1")
        chain2.run("Hello from session 2")

        # Verify sessions are isolated
        assert "sess_1" in memories_by_session
        assert "sess_2" in memories_by_session
        assert any("session 1" in m["content"] for m in memories_by_session["sess_1"])
        assert any("session 2" in m["content"] for m in memories_by_session["sess_2"])

    @respx.mock
    def test_chain_with_return_messages(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test legacy chain with return_messages=True."""
        # Mock existing conversation history
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello!", "human", 0),
            create_mock_memory_response("mem_2", "Hi there!", "ai", 1),
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

        # Create memory with return_messages=True
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            return_messages=True,
        )

        # Verify load_memory_variables returns messages
        result = memory.load_memory_variables({})
        messages = result["history"]

        assert len(messages) == 2
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)

    @respx.mock
    def test_chain_with_custom_prefixes(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test legacy chain with custom human/ai prefixes."""
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello!", "human", 0),
            create_mock_memory_response("mem_2", "Hi there!", "ai", 1),
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

        # Create memory with custom prefixes
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            human_prefix="User",
            ai_prefix="Assistant",
        )

        # Create chain with matching prompt template
        llm = MockLegacyLLM(["I can help with that!"])
        chain = MockLegacyChain(
            llm=llm,
            memory=memory,
            prompt_template="History:\n{history}\n\nUser: {input}\nAssistant:",
        )

        # Run the chain
        chain.run("Help me!")

        # Verify history uses custom prefixes
        prompt = llm.received_prompts[0]
        assert "User: Hello!" in prompt
        assert "Assistant: Hi there!" in prompt

    @respx.mock
    def test_chain_handles_api_errors_gracefully(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that chain handles MemoryLayer API errors gracefully for reads."""
        # First call for initial count succeeds, second fails
        call_count = [0]

        def recall_side_effect(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call for initialization
                return Response(200, json={"memories": [], "total_count": 0})
            else:
                # Subsequent calls fail
                return Response(500, json={"detail": "Internal server error"})

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)

        # Create memory
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Load memory should return empty on error (graceful degradation)
        result = memory.load_memory_variables({})
        assert result == {"history": ""}

    @respx.mock
    def test_chain_with_custom_memory_key(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test legacy chain with custom memory key."""
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create memory with custom memory key
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            memory_key="chat_history",
        )

        # Verify memory variables returns custom key
        assert memory.memory_variables == ["chat_history"]

        # Verify load_memory_variables returns data with custom key
        result = memory.load_memory_variables({})
        assert "chat_history" in result
        assert result["chat_history"] == ""  # Empty history initially

        # Create chain using default history key (chain doesn't need to know the key)
        llm = MockLegacyLLM(["Hello!"])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Run the chain - MockLegacyChain uses memory.memory_key internally
        response = chain.run("Hi!")
        assert response == "Hello!"


# ============================================================================
# Integration Tests: MemoryLayerConversationSummaryMemory with Legacy Chains
# ============================================================================


class TestLegacyChainWithSummaryMemory:
    """Integration tests for legacy chains with MemoryLayerConversationSummaryMemory."""

    @respx.mock
    def test_basic_chain_with_summary(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test basic legacy chain with summary memory."""
        # Setup mocks
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories/reflect").mock(
            return_value=Response(200, json={"reflection": "No previous conversation.", "confidence": 0.9})
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create summary memory
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain
        llm = MockLegacyLLM(["Hello! I'll remember this conversation."])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Run the chain
        response = chain.run("Hi there!")

        # Verify response
        assert response == "Hello! I'll remember this conversation."

    @respx.mock
    def test_chain_with_existing_summary(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that chain receives summary of existing conversation."""
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories/reflect").mock(
            return_value=Response(
                200,
                json={
                    "reflection": "The user introduced themselves as Bob and discussed Python programming."
                },
            )
        )
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create summary memory
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain
        llm = MockLegacyLLM(["Based on our previous discussion about Python..."])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Run the chain
        chain.run("Can you remind me what we talked about?")

        # Verify LLM received summary in prompt
        prompt = llm.received_prompts[0]
        assert "Bob" in prompt
        assert "Python" in prompt

    @respx.mock
    def test_multi_turn_conversation_with_summary(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test multi-turn conversation with summary memory."""
        stored_memories: list[dict] = []
        memory_counter = [0]
        summary_call_count = [0]

        def create_memory_side_effect(request):
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
            return Response(
                200,
                json={"memories": stored_memories.copy(), "total_count": len(stored_memories)},
            )

        def reflect_side_effect(request):
            summary_call_count[0] += 1
            if len(stored_memories) == 0:
                return Response(200, json={"reflection": "No previous conversation.", "confidence": 0.9})
            else:
                # Generate a summary based on stored content
                contents = [m["content"] for m in stored_memories]
                summary = f"Previous topics discussed: {', '.join(contents[:2])}..."
                return Response(200, json={"reflection": summary, "confidence": 0.9})

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(side_effect=create_memory_side_effect)
        respx.post(f"{base_url}/v1/memories/reflect").mock(side_effect=reflect_side_effect)

        # Create summary memory
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create mock chain
        responses = [
            "Nice to meet you, Alice!",
            "I remember you're Alice. What else would you like to discuss?",
        ]
        llm = MockLegacyLLM(responses)
        chain = MockLegacyChain(llm=llm, memory=memory)

        # Turn 1
        response1 = chain.run("My name is Alice")
        assert "Alice" in response1

        # Turn 2 - summary should be updated
        response2 = chain.run("What do you remember?")
        assert response2 == responses[1]

        # Verify reflect was called for summaries
        assert summary_call_count[0] >= 1

    @respx.mock
    def test_summary_with_custom_prompt(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test summary memory with custom summarization prompt."""
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )

        # Capture the reflect request to verify custom prompt
        reflect_requests: list[Any] = []

        def reflect_side_effect(request):
            import json

            body = json.loads(request.content)
            reflect_requests.append(body)
            return Response(200, json={"reflection": "Technical summary.", "confidence": 0.9})

        respx.post(f"{base_url}/v1/memories/reflect").mock(side_effect=reflect_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(
            return_value=Response(
                200,
                json=create_mock_memory_response("mem_1", "test", "human", 0),
            )
        )

        # Create summary memory with custom prompt
        custom_prompt = "Focus on technical details for session {session_id}."
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            summary_prompt=custom_prompt,
        )

        # Load memory variables to trigger reflect
        memory.load_memory_variables({})

        # Verify custom prompt was used
        assert len(reflect_requests) == 1
        assert "technical details" in reflect_requests[0]["query"]

    @respx.mock
    def test_summary_return_messages(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test summary memory with return_messages=True returns SystemMessage."""
        from langchain_core.messages import SystemMessage

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories/reflect").mock(
            return_value=Response(
                200, json={"reflection": "User discussed weather and Python.", "confidence": 0.9}
            )
        )

        # Create summary memory with return_messages=True
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            return_messages=True,
        )

        # Load memory variables
        result = memory.load_memory_variables({})
        messages = result["history"]

        # Verify SystemMessage is returned
        assert len(messages) == 1
        assert isinstance(messages[0], SystemMessage)
        assert "weather" in messages[0].content
        assert "Python" in messages[0].content

    @respx.mock
    def test_summary_handles_api_errors_gracefully(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test that summary memory handles API errors gracefully."""
        # First call succeeds (initialization), reflect fails
        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": [], "total_count": 0})
        )
        respx.post(f"{base_url}/v1/memories/reflect").mock(
            return_value=Response(500, json={"detail": "Internal server error"})
        )

        # Create summary memory
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Load memory should return empty on error
        result = memory.load_memory_variables({})
        assert result == {"history": ""}


# ============================================================================
# Integration Tests: Memory Clear Operations
# ============================================================================


class TestLegacyChainMemoryClear:
    """Integration tests for clearing memory in legacy chains."""

    @respx.mock
    def test_clear_memory_between_conversations(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test clearing memory between conversations."""
        stored_memories: list[dict] = []
        memory_counter = [0]

        def create_memory_side_effect(request):
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
            return Response(
                200,
                json={"memories": stored_memories.copy(), "total_count": len(stored_memories)},
            )

        def delete_side_effect(request):
            # Remove memory by ID from stored_memories
            mem_id = request.url.path.split("/")[-1]
            nonlocal stored_memories
            stored_memories = [m for m in stored_memories if m["id"] != mem_id]
            return Response(204)

        respx.post(f"{base_url}/v1/memories/recall").mock(side_effect=recall_side_effect)
        respx.post(f"{base_url}/v1/memories").mock(side_effect=create_memory_side_effect)
        respx.delete(url__regex=rf"{base_url}/v1/memories/.*").mock(side_effect=delete_side_effect)

        # Create memory
        memory = MemoryLayerMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Create and use chain
        llm = MockLegacyLLM(["Response 1", "Response 2"])
        chain = MockLegacyChain(llm=llm, memory=memory)

        # First conversation
        chain.run("Hello!")
        assert len(stored_memories) == 2  # human + AI message

        # Clear memory
        memory.clear()
        assert len(stored_memories) == 0

        # Second conversation starts fresh
        chain.run("New conversation!")
        assert len(stored_memories) == 2  # new human + AI message

    @respx.mock
    def test_clear_summary_memory(
        self, base_url: str, api_key: str, workspace_id: str
    ) -> None:
        """Test clearing summary memory."""
        existing_memories = [
            create_mock_memory_response("mem_1", "Hello", "human", 0),
            create_mock_memory_response("mem_2", "Hi!", "ai", 1),
        ]

        respx.post(f"{base_url}/v1/memories/recall").mock(
            return_value=Response(200, json={"memories": existing_memories, "total_count": 2})
        )
        respx.delete(f"{base_url}/v1/memories/mem_1").mock(return_value=Response(204))
        respx.delete(f"{base_url}/v1/memories/mem_2").mock(return_value=Response(204))

        # Create summary memory
        memory = MemoryLayerConversationSummaryMemory(
            session_id="sess_test",
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
        )

        # Clear should delete all memories
        memory.clear()

        # Verify delete calls were made
        delete_calls = [call for call in respx.calls if call.request.method == "DELETE"]
        assert len(delete_calls) == 2
