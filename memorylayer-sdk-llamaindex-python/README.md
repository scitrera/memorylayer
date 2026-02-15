# MemoryLayer.ai LlamaIndex Integration

LlamaIndex integration for [MemoryLayer.ai](https://memorylayer.ai) - Persistent memory for LlamaIndex applications.

## Installation

```bash
pip install memorylayer-llamaindex
```

## Quick Start

```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from memorylayer_llamaindex import MemoryLayerChatStore

# Create the chat store connected to MemoryLayer
chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_123"
)

# Create ChatMemoryBuffer with persistent storage
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_alice",
    token_limit=3000
)

# Store messages - they persist across application restarts!
memory.put(ChatMessage(role=MessageRole.USER, content="Hello! I'm learning Python."))
memory.put(ChatMessage(
    role=MessageRole.ASSISTANT,
    content="Great! Python is a wonderful language. What would you like to learn about?"
))

# Retrieve conversation history
history = memory.get()
for msg in history:
    print(f"[{msg.role.value}]: {msg.content}")
```

## Features

- **LlamaIndex Native** - Implements `BaseChatStore` interface for seamless integration
- **Persistent Memory** - Chat history survives application restarts
- **Multi-Session Support** - Isolated conversations per user/session via chat keys
- **Async Support** - Full async/await API for high-performance applications
- **Agent Compatible** - Works with LlamaIndex agents and chat engines
- **Type-safe** - Full type hints with Pydantic models

## Core Integration

### ChatMemoryBuffer

The primary integration point is `MemoryLayerChatStore` with LlamaIndex's `ChatMemoryBuffer`:

```python
from llama_index.core.memory import ChatMemoryBuffer
from memorylayer_llamaindex import MemoryLayerChatStore

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_demo"
)

# Each user/session gets isolated memory via chat_store_key
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_session_123",
    token_limit=4000
)

# Messages are automatically persisted
memory.put(ChatMessage(role=MessageRole.USER, content="Remember this fact"))
memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="I'll remember that!"))

# Later, even after restart, retrieve full history
history = memory.get()
```

### SimpleChatEngine

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_demo"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="chat_session_1",
    token_limit=4000
)

llm = OpenAI(model="gpt-4o-mini")
chat_engine = SimpleChatEngine.from_defaults(
    memory=memory,
    llm=llm,
    system_prompt="You are a helpful assistant with persistent memory."
)

# Chat with persistent memory
response = chat_engine.chat("Hello! I'm Sarah, a data scientist.")
print(response)

# Later sessions will remember the conversation
```

### FunctionAgent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_agents"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="agent_session_1",
    token_limit=8000
)

# Define tools
def get_time() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = [FunctionTool.from_defaults(fn=get_time)]

# Create agent with persistent memory
llm = OpenAI(model="gpt-4o-mini")
agent = FunctionAgent(tools=tools, llm=llm)
ctx = Context(agent)

# Run agent - memory persists across interactions
response = await agent.run("What time is it?", ctx=ctx, memory=memory)
```

## Direct ChatStore Operations

For fine-grained control, use `MemoryLayerChatStore` directly:

### Set Messages

```python
# Replace all messages for a key
chat_store.set_messages("user_123", [
    ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=MessageRole.USER, content="Hello!"),
    ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
])
```

### Get Messages

```python
# Retrieve all messages for a key
messages = chat_store.get_messages("user_123")
for msg in messages:
    print(f"[{msg.role.value}]: {msg.content}")
```

### Add Message

```python
# Add a single message
chat_store.add_message(
    "user_123",
    ChatMessage(role=MessageRole.USER, content="New message")
)
```

### Delete Messages

```python
# Delete all messages for a key
deleted = chat_store.delete_messages("user_123")

# Delete a specific message by index
deleted_msg = chat_store.delete_message("user_123", idx=2)

# Delete the last message
last_msg = chat_store.delete_last_message("user_123")
```

### Get Keys

```python
# List all chat keys in the store
keys = chat_store.get_keys()
print(f"Active conversations: {keys}")
```

## Async Operations

All operations have async equivalents for high-performance applications:

```python
import asyncio

async def async_example():
    chat_store = MemoryLayerChatStore(
        base_url="http://localhost:61001",
        api_key="your-api-key",
        workspace_id="ws_demo"
    )

    # Async set messages
    await chat_store.aset_messages("async_user", [
        ChatMessage(role=MessageRole.USER, content="Async hello!"),
        ChatMessage(role=MessageRole.ASSISTANT, content="Async hi!")
    ])

    # Async add message
    await chat_store.async_add_message(
        "async_user",
        ChatMessage(role=MessageRole.USER, content="Follow-up")
    )

    # Async get messages
    messages = await chat_store.aget_messages("async_user")

    # Async delete
    await chat_store.adelete_messages("async_user")

    # Async get keys
    keys = await chat_store.aget_keys()

asyncio.run(async_example())
```

## Multi-Session Conversations

Each `chat_store_key` maintains isolated conversation history:

```python
chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_demo"
)

# Separate memories for different users
alice_memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_alice"
)

bob_memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_bob"
)

# Alice's conversation
alice_memory.put(ChatMessage(role=MessageRole.USER, content="What's the weather?"))

# Bob's conversation (completely separate)
bob_memory.put(ChatMessage(role=MessageRole.USER, content="Help with Python code"))

# Each retrieves only their own history
alice_history = alice_memory.get()  # Only Alice's messages
bob_history = bob_memory.get()      # Only Bob's messages
```

## Persistence Across Restarts

Messages stored via `MemoryLayerChatStore` persist in MemoryLayer's database:

```python
# Session 1: Store messages
chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_demo"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="persistent_session"
)

memory.put(ChatMessage(role=MessageRole.USER, content="My favorite color is blue."))
memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Got it! I'll remember that."))

# === Application Restart ===

# Session 2: Messages are still there!
chat_store2 = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    api_key="your-api-key",
    workspace_id="ws_demo"
)

memory2 = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store2,
    chat_store_key="persistent_session"
)

history = memory2.get()
print(f"Retrieved {len(history)} messages from previous session")
```

## Utility Functions

The package exports utility functions for custom integrations:

### Message Conversion

```python
from memorylayer_llamaindex import (
    chat_message_to_memory_payload,
    memory_to_chat_message,
    message_role_to_string,
    string_to_message_role
)

# Convert ChatMessage to MemoryLayer payload
msg = ChatMessage.from_str("Hello!", role=MessageRole.USER)
payload = chat_message_to_memory_payload(msg, key="user_123", index=0)
# payload = {"content": "Hello!", "type": "episodic", "tags": [...], ...}

# Convert MemoryLayer memory back to ChatMessage
memory = {"content": "Hello!", "metadata": {"role": "user", "message_index": 0}}
chat_msg = memory_to_chat_message(memory)

# Role conversions
role_str = message_role_to_string(MessageRole.ASSISTANT)  # "assistant"
role = string_to_message_role("user")  # MessageRole.USER
```

### Helper Functions

```python
from memorylayer_llamaindex import get_message_index, get_chat_key, CHAT_KEY_TAG_PREFIX

# Extract metadata from memories
memory = {"metadata": {"message_index": 5, "chat_key": "user_123"}}
index = get_message_index(memory)  # 5
key = get_chat_key(memory)         # "user_123"

# Tag prefix constant
print(CHAT_KEY_TAG_PREFIX)  # "llamaindex_chat_key:"
```

## API Reference

### MemoryLayerChatStore

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:61001"` | MemoryLayer API base URL |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `workspace_id` | `str \| None` | `None` | Workspace ID for operations |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

### Sync Methods

| Method | Description |
|--------|-------------|
| `set_messages(key, messages)` | Replace all messages for a key |
| `get_messages(key)` | Get all messages for a key |
| `add_message(key, message)` | Add a single message |
| `delete_messages(key)` | Delete all messages for a key |
| `delete_message(key, idx)` | Delete message at index |
| `delete_last_message(key)` | Delete the last message |
| `get_keys()` | Get all chat keys |

### Async Methods

| Method | Description |
|--------|-------------|
| `aset_messages(key, messages)` | Async set messages |
| `aget_messages(key)` | Async get messages |
| `async_add_message(key, message)` | Async add message |
| `adelete_messages(key)` | Async delete all messages |
| `adelete_message(key, idx)` | Async delete at index |
| `adelete_last_message(key)` | Async delete last message |
| `aget_keys()` | Async get all keys |

## Examples

See the [examples directory](./examples/) for complete working examples:

- **[basic_chat_memory.py](./examples/basic_chat_memory.py)** - Basic ChatMemoryBuffer usage, multi-session support, persistence, direct store operations
- **[agent_with_memory.py](./examples/agent_with_memory.py)** - Agent integration, multi-turn conversations, async patterns, context retrieval

## Development

### Install Development Dependencies

```bash
cd oss/memorylayer-sdk-llamaindex-python
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Type Checking

```bash
mypy src/memorylayer_llamaindex
```

### Linting

```bash
ruff check src/memorylayer_llamaindex
ruff format src/memorylayer_llamaindex
```

## Requirements

- Python 3.12+
- `memorylayer-client` - MemoryLayer Python SDK
- `llama-index-core>=0.10.0` - LlamaIndex core library

## License

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.

## Links

- [MemoryLayer Documentation](https://docs.memorylayer.ai)
- [MemoryLayer API Reference](https://api.memorylayer.ai/docs)
- [LlamaIndex Documentation](https://docs.llamaindex.ai)
- [GitHub Repository](https://github.com/scitrera/memorylayer)
- [Homepage](https://memorylayer.ai)
