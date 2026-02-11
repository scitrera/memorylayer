"""Legacy LangChain chain examples with MemoryLayer.

This example demonstrates how to use MemoryLayerMemory with LangChain's
legacy chain types such as ConversationChain.

Note: Legacy chains are still widely used but LCEL is recommended for
new projects. See lcel_example.py for the modern approach.
"""

from memorylayer_langchain import MemoryLayerMemory

# ============================================================================
# Configuration
# ============================================================================

MEMORYLAYER_BASE_URL = "http://localhost:61001"
MEMORYLAYER_API_KEY = "your-api-key"
MEMORYLAYER_WORKSPACE_ID = "ws_123"


# ============================================================================
# Basic Memory Example
# ============================================================================


def basic_memory_example():
    """Basic example of using MemoryLayerMemory directly."""
    # Create a memory instance for a specific session
    memory = MemoryLayerMemory(
        session_id="user_123_conversation_1",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Verify memory variables
    print(f"Memory variables: {memory.memory_variables}")

    # Save a conversation turn
    memory.save_context(
        inputs={"input": "Hello! I'm learning about machine learning."},
        outputs={"output": "That's great! Machine learning is a fascinating field."},
    )

    # Load memory variables (returns conversation history)
    history = memory.load_memory_variables({})
    print(f"Loaded history:\n{history['history']}")


# ============================================================================
# ConversationChain Example
# ============================================================================


def conversation_chain_example():
    """Example using MemoryLayerMemory with ConversationChain.

    This is the classic pattern for building conversational agents
    with LangChain's legacy chain architecture.

    Note: This requires langchain-classic and an LLM to be configured.
    """
    # Create memory for this conversation
    memory = MemoryLayerMemory(
        session_id="customer_support_session_1",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Example of how you would use with ConversationChain:
    #
    # from langchain.chains import ConversationChain
    # from langchain_openai import ChatOpenAI
    #
    # llm = ChatOpenAI(model="gpt-4")
    # chain = ConversationChain(
    #     llm=llm,
    #     memory=memory,
    #     verbose=True,
    # )
    #
    # # Have a conversation
    # response1 = chain.run("Hi, I need help with my order #12345")
    # response2 = chain.run("It hasn't arrived yet")
    # response3 = chain.run("I ordered it last week")
    #
    # # Memory persists across chain invocations and even application restarts!

    print("ConversationChain example setup complete")
    print(f"Session ID: {memory.session_id}")


# ============================================================================
# Message Format Options
# ============================================================================


def message_format_example():
    """Example showing different memory output formats."""
    # String format (default)
    string_memory = MemoryLayerMemory(
        session_id="format_example_string",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        return_messages=False,  # Default
        human_prefix="User",
        ai_prefix="Assistant",
    )

    string_memory.save_context(
        inputs={"input": "What's the weather like?"},
        outputs={"output": "I don't have access to weather data."},
    )

    string_history = string_memory.load_memory_variables({})
    print("String format:")
    print(string_history["history"])
    print()

    # Message format (returns LangChain message objects)
    message_memory = MemoryLayerMemory(
        session_id="format_example_messages",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        return_messages=True,
    )

    message_memory.save_context(
        inputs={"input": "What's the weather like?"},
        outputs={"output": "I don't have access to weather data."},
    )

    message_history = message_memory.load_memory_variables({})
    print("Message format:")
    for msg in message_history["history"]:
        print(f"  {msg.__class__.__name__}: {msg.content}")


# ============================================================================
# Custom Keys Example
# ============================================================================


def custom_keys_example():
    """Example using custom input/output keys.

    Some chains use different key names for inputs and outputs.
    MemoryLayerMemory supports custom keys to match your chain.
    """
    memory = MemoryLayerMemory(
        session_id="custom_keys_example",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        input_key="question",
        output_key="answer",
        memory_key="chat_history",  # Custom memory variable name
    )

    # Save context with custom keys
    memory.save_context(
        inputs={"question": "What is Python?"},
        outputs={"answer": "Python is a high-level programming language."},
    )

    # Load with custom memory key
    history = memory.load_memory_variables({})
    print(f"Memory key: {memory.memory_key}")
    print(f"History: {history['chat_history']}")


# ============================================================================
# Persistence Example
# ============================================================================


def persistence_example():
    """Example demonstrating memory persistence across sessions.

    MemoryLayer stores memories persistently, so conversation history
    survives application restarts, crashes, and deployments.
    """
    session_id = "persistent_conversation_123"

    # First "session" - create memory and save some context
    memory1 = MemoryLayerMemory(
        session_id=session_id,
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    memory1.save_context(
        inputs={"input": "My name is Alice"},
        outputs={"output": "Nice to meet you, Alice!"},
    )

    print("First session saved context")
    del memory1  # Simulate application shutdown

    # Second "session" - create new memory instance for same session
    memory2 = MemoryLayerMemory(
        session_id=session_id,
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Load previously stored history
    history = memory2.load_memory_variables({})
    print(f"Second session loaded history:\n{history['history']}")


# ============================================================================
# Clear Memory Example
# ============================================================================


def clear_memory_example():
    """Example showing how to clear conversation history."""
    memory = MemoryLayerMemory(
        session_id="clear_example",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Add some messages
    memory.save_context(
        inputs={"input": "Remember this message"},
        outputs={"output": "I'll remember it!"},
    )

    history_before = memory.load_memory_variables({})
    print(f"Before clear: {len(history_before['history'])} characters")

    # Clear all history for this session
    memory.clear()

    history_after = memory.load_memory_variables({})
    print(f"After clear: {len(history_after['history'])} characters")


# ============================================================================
# Custom Tags Example
# ============================================================================


def custom_tags_example():
    """Example using custom memory tags for categorization."""
    memory = MemoryLayerMemory(
        session_id="tagged_session",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        memory_tags=["department:engineering", "project:alpha"],
    )

    memory.save_context(
        inputs={"input": "Deploy the new feature to staging"},
        outputs={"output": "I'll help coordinate the staging deployment."},
    )

    print(f"Memory saved with tags: {memory.memory_tags}")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Legacy Chain Examples with MemoryLayer")
    print("=" * 60)

    print("\n1. Basic Memory Example")
    print("-" * 40)
    basic_memory_example()

    print("\n2. ConversationChain Example")
    print("-" * 40)
    conversation_chain_example()

    print("\n3. Message Format Options")
    print("-" * 40)
    message_format_example()

    print("\n4. Custom Keys Example")
    print("-" * 40)
    custom_keys_example()

    print("\n5. Persistence Example")
    print("-" * 40)
    persistence_example()

    print("\n6. Clear Memory Example")
    print("-" * 40)
    clear_memory_example()

    print("\n7. Custom Tags Example")
    print("-" * 40)
    custom_tags_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
