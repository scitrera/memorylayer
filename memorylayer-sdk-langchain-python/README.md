# MemoryLayer LangChain Integration

LangChain memory integration for [MemoryLayer.ai](https://memorylayer.ai) - Persistent memory for AI agents.

## Installation

```bash
pip install memorylayer-langchain
```

## Overview

This package provides LangChain-compatible memory classes that use MemoryLayer as the backend, giving you:

- **Persistent Memory** - Memory survives across agent runs and application restarts
- **Stable API** - Consistent interface regardless of LangChain version changes
- **Rich Memory Types** - Semantic, episodic, procedural, and working memory support
- **Drop-in Replacement** - Works with LCEL and legacy chains
- **Session Isolation** - Multiple conversations tracked independently

## Quick Start

### LCEL Chains (Recommended)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from memorylayer_langchain import MemoryLayerChatMessageHistory

# Create a chat chain with message history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

# Wrap with persistent history
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ),
    input_messages_key="input",
    history_messages_key="history",
)

# Use with any session - history persists automatically
response = chain_with_history.invoke(
    {"input": "Hello! My name is Alice."},
    config={"configurable": {"session_id": "user_alice_session_1"}},
)

# Later, even after restart, Alice's context is remembered
response = chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user_alice_session_1"}},
)
```

### Legacy Chains

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from memorylayer_langchain import MemoryLayerMemory

# Create persistent memory
memory = MemoryLayerMemory(
    session_id="customer_support_session_1",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)

# Use with ConversationChain
llm = ChatOpenAI(model="gpt-4")
chain = ConversationChain(llm=llm, memory=memory)

# Conversation persists across chain invocations and restarts
chain.run("Hi, I need help with my order #12345")
chain.run("It hasn't arrived yet")
```

## Features

### MemoryLayerChatMessageHistory

Drop-in replacement for LangChain chat history, designed for LCEL chains.

```python
from memorylayer_langchain import MemoryLayerChatMessageHistory

history = MemoryLayerChatMessageHistory(
    session_id="conversation_1",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)

# Add messages
history.add_user_message("Hello!")
history.add_ai_message("Hi there! How can I help?")

# Retrieve all messages
messages = history.messages
for msg in messages:
    print(f"{msg.type}: {msg.content}")

# Clear history
history.clear()
```

### MemoryLayerMemory

LangChain BaseMemory implementation for legacy chains. Drop-in replacement for `ConversationBufferMemory`.

```python
from memorylayer_langchain import MemoryLayerMemory

memory = MemoryLayerMemory(
    session_id="user_123_conversation",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
    return_messages=False,  # True for message objects
    human_prefix="User",
    ai_prefix="Assistant",
)

# Save conversation turn
memory.save_context(
    inputs={"input": "What's Python?"},
    outputs={"output": "Python is a programming language."},
)

# Load memory variables
history = memory.load_memory_variables({})
print(history["history"])
# Output: User: What's Python?
#         Assistant: Python is a programming language.
```

### MemoryLayerConversationSummaryMemory

Returns AI-generated summaries instead of full conversation history. Useful for long conversations that would exceed context windows.

```python
from memorylayer_langchain import MemoryLayerConversationSummaryMemory

memory = MemoryLayerConversationSummaryMemory(
    session_id="long_conversation",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
    max_tokens=500,
    summary_prompt="Summarize the key points from this conversation.",
)

# After many conversation turns...
summary = memory.load_memory_variables({})
print(summary["history"])  # Concise AI-generated summary
```

## Configuration Options

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `session_id` | Unique identifier for the conversation session | Required |
| `base_url` | MemoryLayer API base URL | `http://localhost:61001` |
| `api_key` | API key for authentication | `None` |
| `workspace_id` | Workspace ID for multi-tenant isolation | `None` |
| `timeout` | Request timeout in seconds | `30.0` |
| `memory_tags` | Additional tags for stored memories | `[]` |

### MemoryLayerMemory Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `memory_key` | Key for memory variables | `"history"` |
| `return_messages` | Return message objects vs string | `False` |
| `human_prefix` | Prefix for human messages | `"Human"` |
| `ai_prefix` | Prefix for AI messages | `"AI"` |
| `input_key` | Custom input key | `None` |
| `output_key` | Custom output key | `None` |

### MemoryLayerConversationSummaryMemory Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_tokens` | Maximum tokens in summary | `500` |
| `summary_prompt` | Custom summarization prompt | Built-in |
| `include_sources` | Include source memory IDs | `False` |

## Advanced Usage

### Custom Memory Tags

Tag messages for cross-session filtering and organization:

```python
history = MemoryLayerChatMessageHistory(
    session_id="support_ticket_456",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
    memory_tags=["customer:enterprise", "topic:billing", "priority:high"],
)
```

### Multi-Session Management

Track multiple conversations independently within the same workspace:

```python
# Session for user 1
user1_history = MemoryLayerChatMessageHistory(
    session_id="user_1_main",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)

# Session for user 2 - completely isolated
user2_history = MemoryLayerChatMessageHistory(
    session_id="user_2_main",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)
```

### Streaming Support

LCEL chains with RunnableWithMessageHistory support streaming natively:

```python
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_123",
    ),
    input_messages_key="input",
    history_messages_key="history",
)

# Stream the response
for chunk in chain_with_history.stream(
    {"input": "Tell me a story"},
    config={"configurable": {"session_id": "story_session"}},
):
    print(chunk.content, end="", flush=True)
```

### Custom Input/Output Keys

Match your chain's key names:

```python
memory = MemoryLayerMemory(
    session_id="qa_session",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
    input_key="question",
    output_key="answer",
    memory_key="chat_history",
)

memory.save_context(
    inputs={"question": "What is Python?"},
    outputs={"answer": "Python is a programming language."},
)

result = memory.load_memory_variables({})
print(result["chat_history"])
```

### Synchronous Client

For direct API access without LangChain abstractions:

```python
from memorylayer_langchain import SyncMemoryLayerClient, sync_client

# Using context manager
with sync_client(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
) as client:
    # Store a memory
    memory = client.remember(
        content="User prefers Python for backend development",
        type="semantic",
        importance=0.8,
        tags=["preferences", "programming"],
    )

    # Search memories
    results = client.recall(
        query="what programming language does the user prefer?",
        limit=5,
    )

    # Get a summary
    reflection = client.reflect(
        query="summarize user's technology preferences",
        max_tokens=300,
    )
```

## Migration from LangChain Memory

### From ConversationBufferMemory

```python
# Before (LangChain built-in - not persistent)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# After (MemoryLayer - persistent)
from memorylayer_langchain import MemoryLayerMemory
memory = MemoryLayerMemory(
    session_id="my_session",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)
```

### From ConversationSummaryMemory

```python
# Before (LangChain built-in - not persistent)
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm)

# After (MemoryLayer - persistent)
from memorylayer_langchain import MemoryLayerConversationSummaryMemory
memory = MemoryLayerConversationSummaryMemory(
    session_id="my_session",
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123",
)
```

## Why MemoryLayer?

### Problem: LangChain Memory Doesn't Persist

Standard LangChain memory is lost when your application restarts:

```python
# LangChain's built-in memory
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
chain.run("My name is Alice")  # Memory stored in RAM

# Application restarts... memory is gone!
```

### Solution: MemoryLayer Provides True Persistence

```python
# MemoryLayer integration
memory = MemoryLayerMemory(session_id="alice_session", ...)
chain = ConversationChain(llm=llm, memory=memory)
chain.run("My name is Alice")  # Memory stored in MemoryLayer

# Application restarts... memory is preserved!
memory2 = MemoryLayerMemory(session_id="alice_session", ...)
# Alice's conversation history is still available
```

### Additional Benefits

- **Stable API** - LangChain memory interfaces change frequently. MemoryLayer provides a stable abstraction.
- **Cross-Platform** - Access the same memories from Python, TypeScript, or any HTTP client.
- **Rich Memory Types** - Beyond simple chat history: semantic, episodic, procedural memories with relationships.
- **Built-in Search** - Semantic search across all stored memories.
- **Reflection** - AI-powered synthesis and summarization of memories.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/memorylayer_langchain

# Linting
ruff check src/memorylayer_langchain
```

## Examples

See the `examples/` directory for complete working examples:

- `lcel_example.py` - Modern LCEL chains with RunnableWithMessageHistory
- `legacy_chain_example.py` - Legacy ConversationChain integration
- `summary_memory_example.py` - Conversation summary memory usage

## License

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.

## Links

- [MemoryLayer Documentation](https://docs.memorylayer.ai)
- [LangChain Documentation](https://python.langchain.com)
- [GitHub](https://github.com/scitrera/memorylayer)
- [Homepage](https://memorylayer.ai)
