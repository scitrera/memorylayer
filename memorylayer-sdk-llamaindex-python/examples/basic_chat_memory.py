"""Basic ChatMemoryBuffer usage example with MemoryLayer.ai LlamaIndex integration.

This example demonstrates how to use MemoryLayerChatStore with LlamaIndex's
ChatMemoryBuffer for persistent chat history storage.

Prerequisites:
    1. Install the package: pip install memorylayer-llamaindex
    2. Have a MemoryLayer server running (default: http://localhost:61001)
    3. Have valid API credentials if authentication is enabled

Usage:
    python basic_chat_memory.py
"""

import asyncio

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer

from memorylayer_llamaindex import MemoryLayerChatStore


def basic_chat_memory_example():
    """Basic example of using ChatMemoryBuffer with MemoryLayerChatStore.

    This example shows how to:
    1. Create a MemoryLayerChatStore instance
    2. Use it with ChatMemoryBuffer
    3. Store and retrieve chat messages
    """
    # Create the chat store connected to MemoryLayer
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",  # Replace with your API key
        workspace_id="ws_demo",   # Replace with your workspace ID
    )

    # Create ChatMemoryBuffer with the chat store
    # The chat_store_key identifies this conversation (e.g., user ID, session ID)
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key="user_alice_session_1",
        token_limit=3000,  # Optional: limit context window size
    )

    # Simulate a conversation
    # Add user message
    user_message = ChatMessage(role=MessageRole.USER, content="Hello! I'm learning Python.")
    memory.put(user_message)
    print(f"Stored user message: {user_message.content}")

    # Add assistant response
    assistant_message = ChatMessage(
        role=MessageRole.ASSISTANT,
        content="Great! Python is a wonderful language. What would you like to learn about?",
    )
    memory.put(assistant_message)
    print(f"Stored assistant message: {assistant_message.content}")

    # Retrieve conversation history
    history = memory.get()
    print(f"\nRetrieved {len(history)} messages from memory:")
    for msg in history:
        print(f"  [{msg.role.value}]: {msg.content}")


def multi_session_example():
    """Example showing multiple user sessions with different chat keys.

    Each user/session gets its own isolated conversation history.
    """
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo",
    )

    # Create separate memories for different users
    alice_memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key="user_alice",
    )

    bob_memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key="user_bob",
    )

    # Alice's conversation
    alice_memory.put(ChatMessage(role=MessageRole.USER, content="What's the weather like?"))
    alice_memory.put(
        ChatMessage(role=MessageRole.ASSISTANT, content="I don't have real-time weather data.")
    )

    # Bob's conversation (completely separate)
    bob_memory.put(ChatMessage(role=MessageRole.USER, content="Help me with Python code."))
    bob_memory.put(
        ChatMessage(role=MessageRole.ASSISTANT, content="I'd be happy to help with Python!")
    )

    # Verify isolation
    print("Alice's history:")
    for msg in alice_memory.get():
        print(f"  [{msg.role.value}]: {msg.content}")

    print("\nBob's history:")
    for msg in bob_memory.get():
        print(f"  [{msg.role.value}]: {msg.content}")


def persistence_example():
    """Example demonstrating persistence across application restarts.

    Messages stored in MemoryLayer persist beyond the application lifecycle.
    """
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo",
    )

    session_key = "persistent_session"

    # First "session" - store some messages
    print("First session - storing messages...")
    memory1 = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
    )
    memory1.put(ChatMessage(role=MessageRole.USER, content="Remember this: my favorite color is blue."))
    memory1.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Got it! I'll remember that your favorite color is blue.",
        )
    )

    # Simulate application restart by creating a new memory buffer
    # (In reality, this would be in a separate script run)
    print("\nSecond session - retrieving persisted messages...")
    memory2 = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=session_key,
    )

    # Retrieve the persisted history
    history = memory2.get()
    print(f"Retrieved {len(history)} messages from previous session:")
    for msg in history:
        print(f"  [{msg.role.value}]: {msg.content}")

    # Clean up for demo purposes
    chat_store.delete_messages(session_key)
    print("\nCleaned up demo messages.")


def direct_store_operations_example():
    """Example showing direct ChatStore operations (without ChatMemoryBuffer).

    Useful when you need more control over message storage.
    """
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo",
    )

    key = "direct_ops_demo"

    # Set messages directly (replaces any existing messages)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=MessageRole.USER, content="Hello!"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Hi there! How can I help?"),
    ]
    chat_store.set_messages(key, messages)
    print(f"Set {len(messages)} messages for key '{key}'")

    # Add a single message
    chat_store.add_message(
        key, ChatMessage(role=MessageRole.USER, content="What time is it?")
    )
    print("Added one more message")

    # Get all messages
    retrieved = chat_store.get_messages(key)
    print(f"\nRetrieved {len(retrieved)} messages:")
    for msg in retrieved:
        print(f"  [{msg.role.value}]: {msg.content}")

    # Delete the last message
    deleted = chat_store.delete_last_message(key)
    if deleted:
        print(f"\nDeleted last message: [{deleted.role.value}]: {deleted.content}")

    # Get all keys in the store
    keys = chat_store.get_keys()
    print(f"\nAll chat keys in store: {keys}")

    # Clean up
    chat_store.delete_messages(key)


async def async_operations_example():
    """Example showing async operations with MemoryLayerChatStore.

    All operations have async equivalents for use in async applications.
    """
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo",
    )

    key = "async_demo"

    # Set messages asynchronously
    messages = [
        ChatMessage(role=MessageRole.USER, content="Async hello!"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Async hi!"),
    ]
    await chat_store.aset_messages(key, messages)
    print("Set messages asynchronously")

    # Add message asynchronously
    await chat_store.async_add_message(
        key, ChatMessage(role=MessageRole.USER, content="Async follow-up")
    )

    # Get messages asynchronously
    retrieved = await chat_store.aget_messages(key)
    print(f"Retrieved {len(retrieved)} messages asynchronously:")
    for msg in retrieved:
        print(f"  [{msg.role.value}]: {msg.content}")

    # Clean up
    await chat_store.adelete_messages(key)
    print("Cleaned up async demo")


def system_message_example():
    """Example showing how to include system messages for context.

    System messages can set the behavior/persona of the assistant.
    """
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo",
    )

    key = "system_msg_demo"

    # Initialize conversation with system message
    chat_store.set_messages(
        key,
        [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="You are a friendly Python tutor. Keep explanations simple and use examples.",
            )
        ],
    )

    # Create memory buffer
    memory = ChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=key,
    )

    # Add conversation
    memory.put(ChatMessage(role=MessageRole.USER, content="What is a list comprehension?"))
    memory.put(
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="A list comprehension is a concise way to create lists. "
            "For example: `[x*2 for x in range(5)]` creates `[0, 2, 4, 6, 8]`.",
        )
    )

    # Get full history including system message
    history = memory.get()
    print("Full conversation with system message:")
    for msg in history:
        print(f"  [{msg.role.value}]: {msg.content[:50]}...")

    # Clean up
    chat_store.delete_messages(key)


def main():
    """Run all examples."""
    print("=" * 60)
    print("MemoryLayer LlamaIndex ChatMemoryBuffer Examples")
    print("=" * 60)

    print("\n1. Basic Chat Memory Example")
    print("-" * 40)
    try:
        basic_chat_memory_example()
    except Exception as e:
        print(f"   Example failed (server may not be running): {e}")

    print("\n2. Multi-Session Example")
    print("-" * 40)
    try:
        multi_session_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n3. Persistence Example")
    print("-" * 40)
    try:
        persistence_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n4. Direct Store Operations Example")
    print("-" * 40)
    try:
        direct_store_operations_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n5. Async Operations Example")
    print("-" * 40)
    try:
        asyncio.run(async_operations_example())
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n6. System Message Example")
    print("-" * 40)
    try:
        system_message_example()
    except Exception as e:
        print(f"   Example failed: {e}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
