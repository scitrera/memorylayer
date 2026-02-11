---
title: Python Quick Start
description: Step-by-step guide to using the MemoryLayer Python SDK
sidebar:
  order: 2
  label: Quick Start
---

This guide walks through the core operations of the MemoryLayer Python SDK.

## Prerequisites

- Python 3.12+
- MemoryLayer server running (`memorylayer serve`)

## Install the SDK

```bash
pip install memorylayer-client
```

## Connect to the Server

```python
from memorylayer import MemoryLayerClient

client = MemoryLayerClient(
    base_url="http://localhost:61001",
    workspace_id="my-workspace"
)
```

## Remember (Store Memories)

```python
from memorylayer import MemoryType, MemorySubtype

# Basic storage
memory = await client.remember(
    content="User prefers FastAPI over Flask",
    type=MemoryType.SEMANTIC,
)

# With full options
memory = await client.remember(
    content="User prefers FastAPI over Flask",
    type=MemoryType.SEMANTIC,
    subtype=MemorySubtype.PREFERENCE,
    importance=0.8,
    tags=["preferences", "frameworks"],
    metadata={"source": "conversation"}
)
```

## Recall (Search Memories)

```python
from memorylayer import RecallMode, SearchTolerance

# Simple search
results = await client.recall(
    query="what frameworks does the user prefer?",
    limit=5
)

# Advanced search with filters
results = await client.recall(
    query="what frameworks does the user prefer?",
    types=[MemoryType.SEMANTIC],
    mode=RecallMode.RAG,
    limit=10,
    min_relevance=0.7,
    tolerance=SearchTolerance.MODERATE
)

for memory in results.memories:
    print(f"{memory.content} (relevance: {memory.importance})")
```

## Reflect (Synthesize Insights)

```python
reflection = await client.reflect(
    query="summarize everything about the user's development workflow",
    max_tokens=500,
    include_sources=True
)

print(reflection.reflection)
```

## Associate (Link Memories)

```python
from memorylayer import RelationshipType

association = await client.associate(
    source_id="mem_problem_123",
    target_id="mem_solution_456",
    relationship=RelationshipType.SOLVES,
    strength=0.9
)
```

## Sessions (Working Memory)

Sessions provide temporary working memory that persists across API calls and can be committed to long-term storage.

```python
# Create a session
session = await client.create_session(
    ttl_seconds=3600,
    workspace_id="my-workspace"
)

# Store working memory
await client.set_context(
    session.id,
    "current_task",
    {"description": "Debugging auth", "file": "auth.py"}
)

# Retrieve working memory
context = await client.get_context(session.id, ["current_task"])

# Extend session TTL
await client.touch_session(session.id)

# Commit to long-term storage
result = await client.commit_session(
    session.id,
    min_importance=0.5,
    deduplicate=True
)
print(f"Created {result['memories_created']} memories")

# Delete session
await client.delete_session(session.id)
```

## Session Briefing

Get a summary of recent activity:

```python
briefing = await client.get_briefing(lookback_hours=24)
print(briefing.recent_activity)
```

## Workspace Management

```python
# Create workspace
workspace = await client.create_workspace("my-project")

# Get workspace schema
schema = await client.get_workspace_schema("ws_123")
print(schema["relationship_types"])
print(schema["memory_subtypes"])
```

## Batch Operations

```python
results = await client.batch_memories([
    {"type": "create", "data": {"content": "Memory 1", "importance": 0.7}},
    {"type": "create", "data": {"content": "Memory 2", "importance": 0.8}},
    {"type": "delete", "data": {"memory_id": "mem_old", "hard": False}}
])
print(f"Successful: {results['successful']}, Failed: {results['failed']}")
```

## Error Handling

```python
from memorylayer import (
    MemoryLayerError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError
)

try:
    memory = await client.get_memory("mem_123")
except NotFoundError:
    print("Memory not found")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded")
except ServerError as e:
    print(f"Server error: {e.status_code}")
```

## Context Environment

The context environment provides a server-side Python sandbox for executing code, querying LLMs, and running autonomous reasoning loops over memories.

### Setup

Context environments are session-scoped. Create a session first:

```python
session = await client.create_session(workspace_id="my-workspace")
```

### Load and Analyze Memories

```python
# Load memories into the sandbox
await client.context_load(
    var="project_memories",
    query="project architecture decisions",
    limit=50
)

# Run code against loaded memories
result = await client.context_exec("""
decisions = [m for m in project_memories if m.get('subtype') == 'decision']
summary = f"Found {len(decisions)} architecture decisions"
""", result_var="summary")

print(result["result"])
```

### Query LLM with Sandbox Context

```python
# Ask the server-side LLM to analyze sandbox data
answer = await client.context_query(
    prompt="What are the key architecture decisions and their rationale?",
    variables=["decisions"]
)
print(answer["response"])
```

### Inspect Sandbox State

```python
# View all variables
state = await client.context_inspect()
print(state["variables"])

# Inspect a specific variable
detail = await client.context_inspect(variable="decisions", preview_chars=500)
print(detail["preview"])
```

### Autonomous Reasoning (RLM)

Run a Recursive Language Model loop that iteratively reasons over memories:

```python
result = await client.context_rlm(
    goal="Identify recurring error patterns and recommend fixes",
    memory_query="errors bugs fixes",
    max_iterations=10,
    detail_level="standard"
)
print(result["result"])
print(f"Completed in {result['iterations']} iterations")
```

### Cleanup

```python
await client.context_cleanup()
```

## Additional Operations

```python
# Decay a memory's importance
decayed = await client.decay("mem_123", decay_rate=0.1)

# Trace memory provenance
trace = await client.trace_memory("mem_123")
print(trace["chain"])

# Soft delete (archive)
await client.forget("mem_123")
```
