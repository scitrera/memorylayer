---
title: Python SDK
description: MemoryLayer Python SDK - async client library for memory operations
sidebar:
  order: 1
  label: Overview
---

The MemoryLayer Python SDK (`memorylayer-client`) provides an async Python client for interacting with the MemoryLayer server.

## Installation

```bash
pip install memorylayer-client
```

## Features

- **Async/Await** — Full async support with context managers
- **Type-Safe** — Complete type hints with Pydantic models
- **Memory Operations** — Remember, recall, reflect, forget, decay
- **Relationship Graph** — Link memories with typed relationships
- **Session Management** — Working memory with TTL and commit
- **Batch Operations** — Bulk create, update, delete
- **Error Handling** — Comprehensive typed exception hierarchy

## Quick Start

```python
from memorylayer import MemoryLayerClient, MemoryType

async with MemoryLayerClient(
    base_url="http://localhost:61001",
    workspace_id="my-workspace"
) as client:
    # Store a memory
    memory = await client.remember(
        content="User prefers Python for backend development",
        type=MemoryType.SEMANTIC,
        importance=0.8,
        tags=["preferences", "programming"]
    )

    # Search memories
    results = await client.recall(
        query="what programming language does the user prefer?",
        limit=5
    )

    for memory in results.memories:
        print(f"{memory.content} (relevance: {memory.importance})")

    # Synthesize memories
    reflection = await client.reflect(
        query="summarize user's technology preferences"
    )
    print(reflection.reflection)
```

## Configuration

```python
client = MemoryLayerClient(
    base_url="http://localhost:61001",  # Server URL
    api_key="your-api-key",             # Optional for local dev
    workspace_id="my-workspace",        # Default workspace
    session_id="sess_123",              # Optional active session
    timeout=30.0                        # Request timeout in seconds
)
```

## Next Steps

- [Quick Start Guide](/sdk-python/quickstart/) — Step-by-step tutorial
- [API Reference](/sdk-python/api-reference/) — Complete method reference
