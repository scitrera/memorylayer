---
title: LangChain Integration
description: Persistent memory for LangChain applications
sidebar:
  order: 3
---

The MemoryLayer LangChain integration (`memorylayer-langchain`) provides persistent, cross-session memory for LangChain applications.

## Installation

```bash
pip install memorylayer-langchain
```

## Why MemoryLayer for LangChain?

Standard LangChain memory is **lost when your application restarts**. MemoryLayer provides true persistence — memory survives across sessions, restarts, and deployments.

| Feature | LangChain Built-in | MemoryLayer |
|---------|-------------------|-------------|
| Persistence | In-memory only | Database-backed |
| Cross-session | No | Yes |
| Cross-platform | No | Python, TypeScript, HTTP |
| Memory types | Chat history | Episodic, semantic, procedural |
| Relationships | No | Typed relationship graph |
| Semantic search | No | Vector similarity |

## Quick Start (LCEL Chains)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from memorylayer_langchain import MemoryLayerChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=lambda session_id: MemoryLayerChatMessageHistory(
        session_id=session_id,
        base_url="http://localhost:61001",
        workspace_id="ws_123",
    ),
    input_messages_key="input",
    history_messages_key="history",
)

# Use with any session - history persists automatically
response = chain_with_history.invoke(
    {"input": "Hello! My name is Alice."},
    config={"configurable": {"session_id": "user_alice"}},
)
```

## Legacy Chains

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from memorylayer_langchain import MemoryLayerMemory

memory = MemoryLayerMemory(
    session_id="customer_support_session_1",
    base_url="http://localhost:61001",
    workspace_id="ws_123",
)

llm = ChatOpenAI(model="gpt-4")
chain = ConversationChain(llm=llm, memory=memory)

chain.run("Hi, I need help with my order #12345")
```

## Memory Classes

### MemoryLayerChatMessageHistory

Drop-in replacement for LangChain chat history, designed for LCEL chains.

```python
history = MemoryLayerChatMessageHistory(
    session_id="conversation_1",
    base_url="http://localhost:61001",
    workspace_id="ws_123",
)

history.add_user_message("Hello!")
history.add_ai_message("Hi there!")
messages = history.messages
history.clear()
```

### MemoryLayerMemory

LangChain `BaseMemory` implementation for legacy chains. Drop-in replacement for `ConversationBufferMemory`.

### MemoryLayerConversationSummaryMemory

Returns AI-generated summaries instead of full conversation history. Useful for long conversations.

```python
from memorylayer_langchain import MemoryLayerConversationSummaryMemory

memory = MemoryLayerConversationSummaryMemory(
    session_id="long_conversation",
    base_url="http://localhost:61001",
    workspace_id="ws_123",
    max_tokens=500,
)
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `session_id` | Unique session identifier | Required |
| `base_url` | MemoryLayer API URL | `http://localhost:61001` |
| `api_key` | API key for authentication | `None` |
| `workspace_id` | Workspace ID | `None` |
| `timeout` | Request timeout in seconds | `30.0` |
| `memory_tags` | Additional tags for stored memories | `[]` |

## Migration from LangChain Memory

```python
# Before (not persistent)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory()

# After (persistent)
from memorylayer_langchain import MemoryLayerMemory
memory = MemoryLayerMemory(
    session_id="my_session",
    base_url="http://localhost:61001",
    workspace_id="ws_123",
)
```
