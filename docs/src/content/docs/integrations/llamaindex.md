---
title: LlamaIndex Integration
description: Persistent chat store for LlamaIndex applications
sidebar:
  order: 4
---

The MemoryLayer LlamaIndex integration (`memorylayer-llamaindex`) provides a persistent `BaseChatStore` implementation for LlamaIndex applications.

## Installation

```bash
pip install memorylayer-llamaindex
```

## Quick Start

```python
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from memorylayer_llamaindex import MemoryLayerChatStore

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    workspace_id="ws_123"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="user_alice",
    token_limit=3000
)

# Messages persist across application restarts
memory.put(ChatMessage(role=MessageRole.USER, content="Hello!"))
memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"))

history = memory.get()
```

## Features

- **LlamaIndex Native** — Implements `BaseChatStore` interface
- **Persistent Memory** — Chat history survives application restarts
- **Multi-Session Support** — Isolated conversations per user/session
- **Async Support** — Full async/await API
- **Agent Compatible** — Works with LlamaIndex agents and chat engines

## SimpleChatEngine

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    workspace_id="ws_demo"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="chat_session_1",
    token_limit=4000
)

chat_engine = SimpleChatEngine.from_defaults(
    memory=memory,
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant with persistent memory."
)

response = chat_engine.chat("Hello! I'm Sarah.")
```

## FunctionAgent

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

chat_store = MemoryLayerChatStore(
    base_url="http://localhost:61001",
    workspace_id="ws_agents"
)

memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="agent_session_1",
    token_limit=8000
)

agent = FunctionAgent(tools=[...], llm=llm)
response = await agent.run("What time is it?", ctx=ctx, memory=memory)
```

## Direct ChatStore Operations

```python
# Set messages
chat_store.set_messages("user_123", [
    ChatMessage(role=MessageRole.USER, content="Hello!"),
    ChatMessage(role=MessageRole.ASSISTANT, content="Hi!")
])

# Get messages
messages = chat_store.get_messages("user_123")

# Add single message
chat_store.add_message("user_123",
    ChatMessage(role=MessageRole.USER, content="New message"))

# Delete messages
chat_store.delete_messages("user_123")

# List all keys
keys = chat_store.get_keys()
```

## Async Operations

All operations have async equivalents:

```python
await chat_store.aset_messages("user", messages)
await chat_store.aget_messages("user")
await chat_store.async_add_message("user", message)
await chat_store.adelete_messages("user")
await chat_store.aget_keys()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:61001"` | MemoryLayer API URL |
| `api_key` | `str \| None` | `None` | API key |
| `workspace_id` | `str \| None` | `None` | Workspace ID |
| `timeout` | `float` | `30.0` | Request timeout |

## Persistence Across Restarts

```python
# Session 1: Store messages
chat_store = MemoryLayerChatStore(base_url="http://localhost:61001")
memory = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store,
    chat_store_key="persistent_session"
)
memory.put(ChatMessage(role=MessageRole.USER, content="Remember this."))

# === Application Restart ===

# Session 2: Messages are still there
chat_store2 = MemoryLayerChatStore(base_url="http://localhost:61001")
memory2 = ChatMemoryBuffer.from_defaults(
    chat_store=chat_store2,
    chat_store_key="persistent_session"
)
history = memory2.get()  # Contains previous messages
```

## Requirements

- Python 3.12+
- `memorylayer-client` — MemoryLayer Python SDK
- `llama-index-core>=0.10.0` — LlamaIndex core library
