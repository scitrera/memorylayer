"""Agent and Chat Engine integration example with MemoryLayer.ai persistent memory.

This example demonstrates how to use MemoryLayerChatStore with LlamaIndex's
agents and chat engines for persistent conversation memory that survives
application restarts.

Prerequisites:
    1. Install the package: pip install memorylayer-llamaindex
    2. Have a MemoryLayer server running (default: http://localhost:61001)
    3. For agent examples, install an LLM: pip install llama-index-llms-openai
       or use a local LLM like Ollama: pip install llama-index-llms-ollama

Usage:
    python agent_with_memory.py

Note: Examples gracefully handle missing LLM dependencies, demonstrating
the memory integration patterns even without a configured LLM.
"""

import asyncio

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from memorylayer_llamaindex import MemoryLayerChatStore


def get_memorylayer_chat_store() -> MemoryLayerChatStore:
    """Create a configured MemoryLayerChatStore instance.

    Returns:
        MemoryLayerChatStore configured for local development
    """
    return MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",  # Replace with your API key
        workspace_id="ws_agents",  # Replace with your workspace ID
    )


def simple_chat_engine_example():
    """Example of SimpleChatEngine with persistent memory.

    This example shows how to integrate MemoryLayerChatStore with
    LlamaIndex's SimpleChatEngine for persistent chat history.
    """
    # Import SimpleChatEngine
    try:
        from llama_index.core.chat_engine import SimpleChatEngine
    except ImportError:
        print("SimpleChatEngine not available. Skipping example.")
        return

    chat_store = get_memorylayer_chat_store()
    session_key = "simple_chat_engine_demo"

    # Clean up any previous session data
    try:
        chat_store.delete_messages(session_key)
    except Exception as e:
        print(f"Warning: failed to delete previous session '{session_key}': {e}")

    # Create memory buffer with persistent storage
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=4000,
    )

    # Note: SimpleChatEngine requires an LLM
    # For production, configure with your LLM of choice:
    #
    # from llama_index.llms.openai import OpenAI
    # llm = OpenAI(model="gpt-4o-mini")
    #
    # from llama_index.llms.ollama import Ollama
    # llm = Ollama(model="llama2")
    #
    # chat_engine = SimpleChatEngine.from_defaults(
    #     memory=memory,
    #     llm=llm,
    #     system_prompt="You are a helpful assistant with persistent memory.",
    # )

    # Demonstrate memory operations without requiring LLM
    print("Simulating chat engine conversation with persistent memory...")

    # First "turn" - user asks a question
    memory.put(ChatMessage(role=MessageRole.USER, content="My name is Sarah and I work as a data scientist."))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Nice to meet you, Sarah! It's great to hear you work as a data scientist. "
            "That's a fascinating field. How can I help you today?",
        )
    )

    # Second "turn" - follow-up
    memory.put(ChatMessage(role=MessageRole.USER, content="What programming languages should I learn?"))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="As a data scientist, I'd recommend Python and R for data analysis, "
            "SQL for databases, and consider Julia for high-performance computing.",
        )
    )

    print(f"Stored {len(memory.get_all())} messages in persistent memory")

    # Verify persistence - simulate restart
    memory2 = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=4000,
    )

    history = memory2.get_all()
    print(f"\nAfter 'restart', retrieved {len(history)} persisted messages:")
    for msg in history:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        content = str(msg.content)[:60] + "..." if len(str(msg.content)) > 60 else str(msg.content)
        print(f"  [{role}]: {content}")

    # Clean up
    chat_store.delete_messages(session_key)
    print("\nCleaned up demo messages.")


def function_agent_example():
    """Example of FunctionAgent with persistent memory.

    This example demonstrates integrating MemoryLayerChatStore with
    LlamaIndex's FunctionAgent for tool-augmented conversations.
    """
    # Try to import agent components
    try:
        from llama_index.core.agent.workflow import FunctionAgent
        from llama_index.core.tools import FunctionTool
    except ImportError:
        print("FunctionAgent not available. Showing memory setup pattern instead.")
        _show_agent_memory_pattern()
        return

    chat_store = get_memorylayer_chat_store()
    session_key = "function_agent_demo"

    # Clean up any previous session
    try:
        chat_store.delete_messages(session_key)
    except Exception:
        # Best-effort cleanup: if deleting prior messages fails (e.g., no existing
        # session or transient backend issue), continue with a fresh session.
        pass

    # Create persistent memory for the agent
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=8000,
    )

    # Define sample tools
    def get_current_time() -> str:
        """Get the current time."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        import ast
        import operator

        try:
            # Use AST-based safe evaluation for mathematical expressions
            # This prevents arbitrary code execution while supporting basic math operations
            def safe_eval(node):
                """Safely evaluate an AST node."""
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.USub: operator.neg,
                }

                if isinstance(node, ast.Constant):
                    return node.value
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](safe_eval(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")

            tree = ast.parse(expression, mode="eval")
            result = safe_eval(tree.body)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    tools = [
        FunctionTool.from_defaults(fn=get_current_time),
        FunctionTool.from_defaults(fn=calculate),
    ]

    # Note: FunctionAgent requires an LLM
    # Production usage:
    #
    # from llama_index.llms.openai import OpenAI
    # llm = OpenAI(model="gpt-4o-mini")
    # agent = FunctionAgent(tools=tools, llm=llm)
    #
    # from llama_index.core.workflow import Context
    # ctx = Context(agent)
    #
    # # Run with persistent memory
    # resp = await agent.run(
    #     "What time is it?",
    #     ctx=ctx,
    #     memory=memory,
    # )

    print("FunctionAgent memory setup demonstrated.")
    print(f"Tools available: {[t.metadata.name for t in tools]}")
    print(f"Memory configured with chat_store_key: {session_key}")

    # Simulate agent conversation for demo
    memory.put(ChatMessage(role=MessageRole.USER, content="What's the current time?"))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Using get_current_time tool... The current time is 2024-01-15 14:30:45.",
        )
    )

    memory.put(ChatMessage(role=MessageRole.USER, content="Calculate 15 * 24 + 100"))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Using calculate tool... 15 * 24 + 100 = 460",
        )
    )

    print(f"\nStored {len(memory.get_all())} messages with tool interactions")

    # Clean up
    chat_store.delete_messages(session_key)


def _show_agent_memory_pattern():
    """Show the pattern for using memory with agents."""
    print("\nAgent Memory Integration Pattern:")
    print("-" * 40)
    print("""
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from memorylayer_llamaindex import MemoryLayerChatStore

# Create persistent chat store
chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_agents",
)

# Create memory with persistent backend
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_session_123",
    token_limit=8000,
)

# Create agent
llm = OpenAI(model="gpt-4o-mini")
agent = FunctionAgent(tools=my_tools, llm=llm)

# Run agent with persistent memory
ctx = Context(agent)
response = await agent.run("Hello!", ctx=ctx, memory=memory)

# Memory is automatically persisted after each interaction!
""")


def multi_turn_persistence_example():
    """Demonstrate persistent multi-turn conversations across sessions.

    This example shows how memory persists across simulated application
    restarts, enabling continuous conversations.
    """
    chat_store = get_memorylayer_chat_store()
    session_key = "multi_turn_demo"

    # Clean up any previous session
    try:
        chat_store.delete_messages(session_key)
    except Exception:
        # Best-effort cleanup: ignore failures so the example continues even if
        # the server is unavailable or no previous session exists.
        pass

    print("=== Session 1: Initial Conversation ===")
    memory1 = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=4000,
    )

    # First session - establish context
    messages_session1 = [
        (MessageRole.USER, "Hi! I'm working on a machine learning project."),
        (MessageRole.ASSISTANT, "That's exciting! What kind of ML project are you building?"),
        (MessageRole.USER, "It's a recommendation system for an e-commerce platform."),
        (
            MessageRole.ASSISTANT,
            "Great choice! Recommendation systems are very impactful. "
            "Are you using collaborative filtering, content-based, or a hybrid approach?",
        ),
        (MessageRole.USER, "I'm thinking hybrid, but starting with collaborative filtering."),
    ]

    for role, content in messages_session1:
        memory1.put(ChatMessage(role=role, content=content))

    print(f"Session 1: Stored {len(memory1.get_all())} messages")
    print("Session 1 ended. (simulating app restart...)\n")

    # Second session - continue conversation with full context
    print("=== Session 2: Continued Conversation ===")
    memory2 = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=4000,
    )

    # Verify history is retrieved
    history = memory2.get_all()
    print(f"Session 2: Retrieved {len(history)} messages from previous session")

    # Continue the conversation
    messages_session2 = [
        (
            MessageRole.ASSISTANT,
            "Welcome back! Last time we discussed your hybrid recommendation "
            "system starting with collaborative filtering. How's it going?",
        ),
        (MessageRole.USER, "I've implemented the basic matrix factorization. Need help with evaluation metrics."),
        (
            MessageRole.ASSISTANT,
            "For recommendation systems, consider: Precision@K, Recall@K, NDCG, "
            "MAP, and for implicit feedback, you might want to look at Hit Rate.",
        ),
    ]

    for role, content in messages_session2:
        memory2.put(ChatMessage(role=role, content=content))

    final_history = memory2.get_all()
    print(f"Session 2: Now have {len(final_history)} total messages")
    print("\nFull conversation history:")
    for i, msg in enumerate(final_history, 1):
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        content = str(msg.content)[:50] + "..." if len(str(msg.content)) > 50 else str(msg.content)
        print(f"  {i}. [{role}]: {content}")

    # Clean up
    chat_store.delete_messages(session_key)
    print("\nCleaned up demo messages.")


async def async_agent_example():
    """Async example of agent with persistent memory.

    Demonstrates async operations for high-performance applications.
    """
    chat_store = get_memorylayer_chat_store()
    session_key = "async_agent_demo"

    # Clean up any previous session
    try:
        await chat_store.adelete_messages(session_key)
    except Exception:
        # Best-effort cleanup: ignore failures (e.g., session not found or backend unavailable)
        pass

    # Create memory for async agent usage
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
        token_limit=4000,
    )

    print("Async Agent Memory Pattern:")
    print("-" * 40)

    # Note: In production, you would use:
    #
    # from llama_index.core.agent.workflow import FunctionAgent
    # from llama_index.core.workflow import Context
    # from llama_index.llms.openai import OpenAI
    #
    # agent = FunctionAgent(tools=[], llm=OpenAI(model="gpt-4o-mini"))
    # ctx = Context(agent)
    #
    # # Run multiple turns asynchronously
    # response1 = await agent.run("Hello!", ctx=ctx, memory=memory)
    # response2 = await agent.run("Follow up question", ctx=ctx, memory=memory)
    #
    # # Memory is automatically persisted via MemoryLayerChatStore

    # Simulate async agent interactions
    messages = [
        (MessageRole.USER, "Async operation 1: What's the weather?"),
        (MessageRole.ASSISTANT, "I don't have real-time weather data, but I can help you find weather APIs."),
        (MessageRole.USER, "Async operation 2: What did I just ask about?"),
        (MessageRole.ASSISTANT, "You asked about the weather. Would you like API recommendations?"),
    ]

    for role, content in messages:
        memory.put(ChatMessage(role=role, content=content))

    print(f"Stored {len(memory.get_all())} messages via async-compatible memory")

    # Demonstrate async retrieval
    retrieved = await chat_store.aget_messages(session_key)
    print(f"Async retrieved {len(retrieved)} messages")

    # Clean up
    await chat_store.adelete_messages(session_key)
    print("Cleaned up async demo.")


def context_retrieval_example():
    """Example showing how to use memory for context in new sessions.

    This pattern is useful for building agents that reference past
    conversations when generating responses.
    """
    chat_store = get_memorylayer_chat_store()
    user_id = "user_context_demo"

    # Clean up any previous session
    try:
        chat_store.delete_messages(user_id)
    except Exception:
        # Best-effort cleanup: ignore errors so the example continues even if
        # the MemoryLayer server is unavailable or the workspace/user is missing.
        pass

    # Store some user preferences and facts over multiple sessions
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=user_id,
        token_limit=8000,
    )

    # Historical conversations establishing user context
    historical_messages = [
        (MessageRole.USER, "I prefer Python over JavaScript."),
        (MessageRole.ASSISTANT, "Noted! Python is great for data science and backend work."),
        (MessageRole.USER, "I'm allergic to shellfish."),
        (MessageRole.ASSISTANT, "I'll remember that - no shellfish recommendations for you."),
        (MessageRole.USER, "My timezone is PST."),
        (MessageRole.ASSISTANT, "Got it, I'll adjust any time-related responses for Pacific Time."),
    ]

    for role, content in historical_messages:
        memory.put(ChatMessage(role=role, content=content))

    print("User context established from previous conversations:")
    print("-" * 40)

    # Simulate new session that needs user context
    new_memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=user_id,
        token_limit=8000,
    )

    # Retrieve context for the new agent/session
    context_messages = new_memory.get_all()
    print(f"Retrieved {len(context_messages)} context messages for new session")

    # In production, you would pass this context to your agent:
    #
    # system_prompt = "Use the following context about the user:\n"
    # for msg in context_messages:
    #     system_prompt += f"{msg.role}: {msg.content}\n"
    #
    # agent = FunctionAgent(
    #     tools=[],
    #     llm=OpenAI(model="gpt-4o-mini"),
    #     system_prompt=system_prompt,
    # )

    # Show the context that would be available
    print("\nContext available for new agent session:")
    for msg in context_messages:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        print(f"  [{role}]: {msg.content}")

    # Clean up
    chat_store.delete_messages(user_id)
    print("\nCleaned up demo data.")


def main():
    """Run all agent integration examples."""
    print("=" * 70)
    print("MemoryLayer LlamaIndex Agent Integration Examples")
    print("=" * 70)

    print("\n1. SimpleChatEngine with Persistent Memory")
    print("-" * 50)
    try:
        simple_chat_engine_example()
    except Exception as e:
        print(f"   Example failed (server may not be running): {e}")

    print("\n2. FunctionAgent with Persistent Memory")
    print("-" * 50)
    try:
        function_agent_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n3. Multi-Turn Persistence Across Sessions")
    print("-" * 50)
    try:
        multi_turn_persistence_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n4. Async Agent Example")
    print("-" * 50)
    try:
        asyncio.run(async_agent_example())
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n5. Context Retrieval for New Sessions")
    print("-" * 50)
    try:
        context_retrieval_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nNote: For full agent functionality, install an LLM provider:")
    print("  pip install llama-index-llms-openai    # For OpenAI")
    print("  pip install llama-index-llms-ollama    # For local Ollama")
    print("\nThen uncomment the agent creation code in each example.")


if __name__ == "__main__":
    main()
