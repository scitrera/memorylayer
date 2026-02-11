# MemoryLayer Python SDK

Python SDK for [MemoryLayer.ai](https://memorylayer.ai) - Memory infrastructure for AI agents.

## Installation

```bash
pip install memorylayer-client
```

## Quick Start

```python
from memorylayer import MemoryLayerClient, MemoryType

async with MemoryLayerClient(
    base_url="http://localhost:61001",
    api_key="your-api-key",  # Optional for local development
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

## Features

- **Simple, Pythonic API** - Async/await support with context managers
- **Type-safe** - Full type hints with Pydantic models
- **Memory Operations** - Remember, recall, reflect, forget, decay
- **Relationship Graph** - Link memories with typed relationships
- **Session Management** - Working memory with TTL and commit
- **Batch Operations** - Bulk create, update, delete
- **Error Handling** - Comprehensive exception hierarchy

## Core Operations

### Remember (Store Memory)

```python
memory = await client.remember(
    content="User prefers FastAPI over Flask",
    type=MemoryType.SEMANTIC,
    subtype=MemorySubtype.PREFERENCE,
    importance=0.8,
    tags=["preferences", "frameworks"],
    metadata={"source": "conversation"}
)
```

### Recall (Search Memories)

```python
from memorylayer import RecallMode, SearchTolerance

results = await client.recall(
    query="what frameworks does the user prefer?",
    types=[MemoryType.SEMANTIC],
    mode=RecallMode.RAG,  # or RecallMode.LLM, RecallMode.HYBRID
    limit=10,
    min_relevance=0.7,
    tolerance=SearchTolerance.MODERATE
)
```

### Reflect (Synthesize Memories)

```python
reflection = await client.reflect(
    query="summarize everything about the user's development workflow",
    max_tokens=500,
    include_sources=True
)

print(reflection.reflection)
```

### Associate (Link Memories)

```python
from memorylayer import RelationshipType

association = await client.associate(
    source_id="mem_problem_123",
    target_id="mem_solution_456",
    relationship=RelationshipType.SOLVES,
    strength=0.9
)
```

### Decay (Reduce Importance)

```python
# Reduce memory importance over time
decayed = await client.decay("mem_123", decay_rate=0.1)
```

### Trace (Memory Provenance)

```python
# Get memory origin and association chain
trace = await client.trace_memory("mem_123")
print(trace["chain"])
```

### Batch Operations

```python
# Perform multiple operations in one request
results = await client.batch_memories([
    {"type": "create", "data": {"content": "Memory 1", "importance": 0.7}},
    {"type": "create", "data": {"content": "Memory 2", "importance": 0.8}},
    {"type": "delete", "data": {"memory_id": "mem_old", "hard": False}}
])
print(f"Successful: {results['successful']}, Failed: {results['failed']}")
```

## Session Management

Sessions provide working memory with TTL that can be committed to long-term storage.

```python
# Create session (auto-creates workspace if needed)
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

# Commit working memory to long-term storage
result = await client.commit_session(
    session.id,
    min_importance=0.5,
    deduplicate=True
)
print(f"Created {result['memories_created']} memories")

# Delete session
await client.delete_session(session.id)
```

### Session Briefing

```python
briefing = await client.get_briefing(lookback_hours=24)
print(briefing.recent_activity_summary)
```

## Workspace Management

```python
# Create workspace
workspace = await client.create_workspace("my-project")

# Get workspace
workspace = await client.get_workspace("ws_123")

# Update workspace
workspace = await client.update_workspace(
    "ws_123",
    name="New Name",
    settings={"key": "value"}
)

# Get workspace schema (relationship types, memory subtypes)
schema = await client.get_workspace_schema("ws_123")
print(schema["relationship_types"])
```

## Memory Types

### Cognitive Types

- **Episodic** - Specific events/interactions
- **Semantic** - Facts, concepts, relationships
- **Procedural** - How to do things
- **Working** - Current task context (session-scoped)

### Domain Subtypes

- **Solution** - Working fixes to problems
- **Problem** - Issues encountered
- **CodePattern** - Reusable patterns
- **Fix** - Bug fixes with context
- **Error** - Error patterns and resolutions
- **Workflow** - Process knowledge
- **Preference** - User/project preferences
- **Decision** - Architectural decisions

## Relationship Types

Link memories with typed relationships:

```python
from memorylayer import RelationshipType

# Causal
RelationshipType.CAUSES
RelationshipType.TRIGGERS
RelationshipType.LEADS_TO
RelationshipType.PREVENTS

# Solution
RelationshipType.SOLVES
RelationshipType.ADDRESSES
RelationshipType.IMPROVES

# Learning
RelationshipType.BUILDS_ON
RelationshipType.CONTRADICTS
RelationshipType.SUPERSEDES

# ... and more (26 total)
```

## Error Handling

```python
from memorylayer import (
    MemoryLayerError,
    AuthenticationError,
    AuthorizationError,
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
except AuthorizationError:
    print("Access denied")
except RateLimitError:
    print("Rate limit exceeded")
except ServerError as e:
    print(f"Server error: {e.status_code}")
```

## Configuration

```python
client = MemoryLayerClient(
    base_url="http://localhost:61001",  # Default
    api_key="your-api-key",             # Optional for local dev
    workspace_id="my-workspace",        # Default workspace
    session_id="sess_123",              # Optional active session
    timeout=30.0                        # Request timeout in seconds
)
```

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Type Checking

```bash
mypy src/memorylayer
```

### Linting

```bash
ruff check src/memorylayer
ruff format src/memorylayer
```

## License

Apache 2.0 License - see LICENSE file for details.

## Links

- [Documentation](https://docs.memorylayer.ai)
- [GitHub](https://github.com/scitrera/memorylayer)
- [Homepage](https://memorylayer.ai)
