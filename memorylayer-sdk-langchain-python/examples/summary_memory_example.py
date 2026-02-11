"""Conversation Summary Memory examples with MemoryLayer.

This example demonstrates how to use MemoryLayerConversationSummaryMemory
to provide AI-generated summaries of conversation history instead of
the full conversation transcript.

This is useful for:
- Long conversations that would exceed context windows
- Reducing token usage in API calls
- Extracting key insights from lengthy discussions
"""

from memorylayer_langchain import MemoryLayerConversationSummaryMemory

# ============================================================================
# Configuration
# ============================================================================

MEMORYLAYER_BASE_URL = "http://localhost:61001"
MEMORYLAYER_API_KEY = "your-api-key"
MEMORYLAYER_WORKSPACE_ID = "ws_123"


# ============================================================================
# Basic Summary Memory Example
# ============================================================================


def basic_summary_example():
    """Basic example of using MemoryLayerConversationSummaryMemory."""
    # Create a summary memory instance
    memory = MemoryLayerConversationSummaryMemory(
        session_id="summary_example_1",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    print(f"Memory variables: {memory.memory_variables}")

    # Simulate a multi-turn conversation
    conversations = [
        ("What programming languages do you recommend?",
         "For beginners, I recommend Python for its readability."),
        ("Why Python specifically?",
         "Python has simple syntax and a large ecosystem of libraries."),
        ("What about web development?",
         "For web dev, JavaScript/TypeScript are essential."),
    ]

    for user_input, ai_response in conversations:
        memory.save_context(
            inputs={"input": user_input},
            outputs={"output": ai_response},
        )

    # Load summary instead of full history
    summary = memory.load_memory_variables({})
    print(f"Conversation summary:\n{summary['history']}")


# ============================================================================
# Custom Summary Prompt Example
# ============================================================================


def custom_prompt_example():
    """Example using a custom summarization prompt."""
    # Custom prompt template for domain-specific summaries
    custom_prompt = (
        "Summarize the technical discussion for session {session_id}. "
        "Focus on: 1) Technologies mentioned, 2) Decisions made, "
        "3) Action items or next steps. Be concise."
    )

    memory = MemoryLayerConversationSummaryMemory(
        session_id="custom_prompt_example",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        summary_prompt=custom_prompt,
        max_tokens=300,  # Shorter summaries
    )

    # Simulate a technical discussion
    memory.save_context(
        inputs={"input": "Should we use PostgreSQL or MySQL for the new project?"},
        outputs={
            "output": "PostgreSQL is better for complex queries and JSON support. "
            "Let's go with PostgreSQL."
        },
    )

    memory.save_context(
        inputs={"input": "What about caching?"},
        outputs={"output": "Redis would work well. Add it to the architecture doc."},
    )

    summary = memory.load_memory_variables({})
    print(f"Custom summary:\n{summary['history']}")


# ============================================================================
# Return Messages Format Example
# ============================================================================


def message_format_example():
    """Example returning summary as SystemMessage objects."""
    memory = MemoryLayerConversationSummaryMemory(
        session_id="message_format_example",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        return_messages=True,  # Return as SystemMessage
    )

    memory.save_context(
        inputs={"input": "I need help debugging a race condition"},
        outputs={
            "output": "Race conditions can be tricky. Try adding locks or "
            "using thread-safe data structures."
        },
    )

    result = memory.load_memory_variables({})
    messages = result["history"]

    print(f"Number of messages: {len(messages)}")
    if messages:
        msg = messages[0]
        print(f"Message type: {type(msg).__name__}")
        print(f"Content: {msg.content[:100]}...")


# ============================================================================
# ConversationChain Example
# ============================================================================


def conversation_chain_example():
    """Example using summary memory with ConversationChain.

    Summary memory is especially useful for long conversations
    where the full history would exceed the LLM's context window.
    """
    memory = MemoryLayerConversationSummaryMemory(
        session_id="long_conversation_session",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        max_tokens=500,
    )

    # Example usage with ConversationChain:
    #
    # from langchain.chains import ConversationChain
    # from langchain_openai import ChatOpenAI
    #
    # llm = ChatOpenAI(model="gpt-4")
    # chain = ConversationChain(
    #     llm=llm,
    #     memory=memory,
    # )
    #
    # # Even after 100 turns, memory provides a concise summary
    # for i in range(100):
    #     chain.run(f"Message {i}: some content")
    #
    # # The LLM sees a summary, not 100 raw messages

    print("ConversationChain with summary memory example ready")
    print(f"Max summary tokens: {memory.max_tokens}")


# ============================================================================
# Include Sources Example
# ============================================================================


def include_sources_example():
    """Example showing source memory IDs in the summary.

    When include_sources is True, the summary includes references
    to the original memories used to generate it.
    """
    memory = MemoryLayerConversationSummaryMemory(
        session_id="sources_example",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        include_sources=True,
    )

    memory.save_context(
        inputs={"input": "Tell me about your API rate limits"},
        outputs={
            "output": "Our API has a rate limit of 100 requests per minute. "
            "Enterprise plans have higher limits."
        },
    )

    summary = memory.load_memory_variables({})
    print(f"Summary with sources:\n{summary['history']}")


# ============================================================================
# Compare Buffer vs Summary Example
# ============================================================================


def compare_memory_types_example():
    """Compare full buffer memory vs summary memory.

    This example shows the difference between storing full conversation
    history vs using AI-generated summaries.
    """
    from memorylayer_langchain import MemoryLayerMemory

    session_id = "comparison_session"

    # Buffer memory - stores everything
    buffer_memory = MemoryLayerMemory(
        session_id=f"{session_id}_buffer",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
    )

    # Summary memory - stores summaries
    summary_memory = MemoryLayerConversationSummaryMemory(
        session_id=f"{session_id}_summary",
        base_url=MEMORYLAYER_BASE_URL,
        api_key=MEMORYLAYER_API_KEY,
        workspace_id=MEMORYLAYER_WORKSPACE_ID,
        max_tokens=200,
    )

    # Same conversation in both
    conversations = [
        ("What's machine learning?", "ML is a subset of AI..."),
        ("How is it different from deep learning?", "Deep learning uses neural..."),
        ("What are some applications?", "Applications include image recognition..."),
    ]

    for user_input, ai_response in conversations:
        buffer_memory.save_context(
            inputs={"input": user_input},
            outputs={"output": ai_response},
        )
        summary_memory.save_context(
            inputs={"input": user_input},
            outputs={"output": ai_response},
        )

    buffer_result = buffer_memory.load_memory_variables({})
    summary_result = summary_memory.load_memory_variables({})

    print("Buffer Memory (full history):")
    print(f"  Length: {len(buffer_result['history'])} chars")
    print(f"  Content: {buffer_result['history'][:100]}...")
    print()
    print("Summary Memory (AI summary):")
    print(f"  Length: {len(summary_result['history'])} chars")
    print(f"  Content: {summary_result['history'][:100]}...")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Summary Memory Examples with MemoryLayer")
    print("=" * 60)

    print("\n1. Basic Summary Example")
    print("-" * 40)
    basic_summary_example()

    print("\n2. Custom Prompt Example")
    print("-" * 40)
    custom_prompt_example()

    print("\n3. Message Format Example")
    print("-" * 40)
    message_format_example()

    print("\n4. ConversationChain Example")
    print("-" * 40)
    conversation_chain_example()

    print("\n5. Include Sources Example")
    print("-" * 40)
    include_sources_example()

    print("\n6. Compare Memory Types Example")
    print("-" * 40)
    compare_memory_types_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
