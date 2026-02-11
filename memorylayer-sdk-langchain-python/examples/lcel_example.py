"""LCEL (LangChain Expression Language) usage examples with MemoryLayer.

This example demonstrates how to use MemoryLayerChatMessageHistory with
LangChain's modern LCEL chains via RunnableWithMessageHistory.

LCEL is the recommended approach for new LangChain applications as it
provides better composability, streaming support, and async capabilities.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from memorylayer_langchain import MemoryLayerChatMessageHistory

# ============================================================================
# Configuration
# ============================================================================

MEMORYLAYER_BASE_URL = "http://localhost:61001"
MEMORYLAYER_API_KEY = "your-api-key"
MEMORYLAYER_WORKSPACE_ID = "ws_123"


# ============================================================================
# Basic Chat History Example
# ============================================================================


def basic_chat_history_example():
    """Basic example of using MemoryLayerChatMessageHistory directly."""
    # Create a chat history instance for a specific session
    history = MemoryLayerChatMessageHistory(
        session_id="user_123_conversation_1",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Add messages to the history
    history.add_user_message("Hello! I'm interested in learning Python.")
    history.add_ai_message(
        "Great choice! Python is an excellent language for beginners. "
        "What aspects of Python are you most interested in learning?"
    )

    # Retrieve all messages
    messages = history.messages
    print(f"Total messages: {len(messages)}")
    for msg in messages:
        role = "User" if msg.type == "human" else "AI"
        print(f"  {role}: {msg.content[:50]}...")

    # Clear history when done (optional)
    # history.clear()


# ============================================================================
# LCEL with RunnableWithMessageHistory Example
# ============================================================================


def create_session_history(session_id: str) -> MemoryLayerChatMessageHistory:
    """Factory function to create a history instance for a session.

    This function is passed to RunnableWithMessageHistory to create
    or retrieve the history for each session.
    """
    return MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )


def lcel_chain_example():
    """Example of using MemoryLayerChatMessageHistory with LCEL chains.

    This example shows how to integrate MemoryLayer with LangChain's
    RunnableWithMessageHistory for automatic conversation persistence.

    Note: This requires a LangChain LLM (e.g., ChatOpenAI) to be configured.
    """
    # Define the prompt template with message history
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the conversation history "
                "to provide contextual responses.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # In a real application, you would use an actual LLM:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4")
    # chain = prompt | llm

    # For this example, we'll just show the prompt setup
    print("LCEL prompt template created with history placeholder")
    print(f"Input variables: {prompt.input_variables}")

    # Example of how you would wrap the chain with message history:
    #
    # chain_with_history = RunnableWithMessageHistory(
    #     runnable=chain,
    #     get_session_history=create_session_history,
    #     input_messages_key="input",
    #     history_messages_key="history",
    # )
    #
    # # Invoke with session config
    # response = chain_with_history.invoke(
    #     {"input": "What's Python?"},
    #     config={"configurable": {"session_id": "user_123_session_1"}},
    # )


# ============================================================================
# Multi-Session Management Example
# ============================================================================


def multi_session_example():
    """Example managing multiple conversation sessions.

    MemoryLayer supports multiple isolated sessions within the same
    workspace, enabling you to track separate conversations per user
    or per context.
    """
    # Session for user 1
    user1_history = MemoryLayerChatMessageHistory(
        session_id="user_1_main",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Session for user 2
    user2_history = MemoryLayerChatMessageHistory(
        session_id="user_2_main",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Each session is isolated
    user1_history.add_user_message("I prefer TypeScript for web development")
    user2_history.add_user_message("I'm learning Rust for systems programming")

    print(f"User 1 messages: {len(user1_history.messages)}")
    print(f"User 2 messages: {len(user2_history.messages)}")


# ============================================================================
# Custom Tags Example
# ============================================================================


def custom_tags_example():
    """Example using custom memory tags for enhanced filtering.

    Tags allow you to categorize and filter memories across sessions,
    enabling cross-session querying and organization.
    """
    history = MemoryLayerChatMessageHistory(
        session_id="support_ticket_456",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        # Custom tags for this conversation
        memory_tags=["customer:enterprise", "topic:billing", "priority:high"],
    )

    history.add_user_message("I have a question about my enterprise billing")
    history.add_ai_message(
        "I'd be happy to help with your enterprise billing question. "
        "What specific aspect would you like to discuss?"
    )

    print(f"Messages stored with custom tags: {len(history.messages)}")


# ============================================================================
# Streaming Example with LCEL
# ============================================================================


def streaming_example():
    """Example showing how LCEL chains support streaming with MemoryLayer.

    One advantage of LCEL is native streaming support. MemoryLayer
    automatically stores the complete response after streaming finishes.

    Note: This requires a streaming-capable LLM to be configured.
    """
    # Example of streaming chain setup:
    #
    # chain_with_history = RunnableWithMessageHistory(
    #     runnable=chain,
    #     get_session_history=create_session_history,
    #     input_messages_key="input",
    #     history_messages_key="history",
    # )
    #
    # # Stream the response
    # for chunk in chain_with_history.stream(
    #     {"input": "Tell me a story about AI"},
    #     config={"configurable": {"session_id": "story_session"}},
    # ):
    #     print(chunk.content, end="", flush=True)

    print("Streaming is supported with LCEL chains and MemoryLayer")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("LCEL Examples with MemoryLayer")
    print("=" * 60)

    print("\n1. Basic Chat History Example")
    print("-" * 40)
    basic_chat_history_example()

    print("\n2. LCEL Chain Example")
    print("-" * 40)
    lcel_chain_example()

    print("\n3. Multi-Session Example")
    print("-" * 40)
    multi_session_example()

    print("\n4. Custom Tags Example")
    print("-" * 40)
    custom_tags_example()

    print("\n5. Streaming Example")
    print("-" * 40)
    streaming_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
