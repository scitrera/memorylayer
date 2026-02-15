# MemoryLayer.ai Python SDK

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
    mode=RecallMode.RAG,  # Active mode: vector similarity + graph traversal
    limit=10,
    min_relevance=0.7,
    tolerance=SearchTolerance.MODERATE,
    include_associations=True  # Include related memories via graph traversal
)

# Note: LLM and Hybrid modes are deprecated. Use Context Environment's
# context_rlm() for LLM-powered analysis instead.
```

### Reflect (Synthesize Memories)

```python
reflection = await client.reflect(
    query="summarize everything about the user's development workflow",
    detail_level="standard",  # "brief", "standard", or "detailed"
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
print(briefing.recent_activity)
```

## Context Environment

The Context Environment provides server-side Python execution for advanced memory analysis. Execute code, load memories into variables, and use LLM-powered reasoning.

**Note:** Context Environment operations require an active session. Call `set_session()` first.

### Execute Python Code

```python
# Set active session
client.set_session(session.id)

# Execute code in sandbox
await client.context_exec("import pandas as pd")
await client.context_exec("data = [1, 2, 3, 4, 5]")

# Execute and get result
result = await client.context_exec("sum(data)")
print(result["result"])  # 15
```

### Load Memories into Sandbox

```python
# Load memories as a variable
await client.context_load(
    var="preferences",
    query="user preferences",
    limit=20,
    min_relevance=0.7
)

# Inspect loaded data
state = await client.context_inspect("preferences")
print(state["type"], state["preview"])
```

### Query with LLM

```python
# Ask LLM to analyze sandbox variables
result = await client.context_query(
    prompt="Summarize the user's preferences and find patterns",
    variables=["preferences"]
)
print(result["response"])
```

### Recursive Language Model (RLM)

```python
# Run autonomous reasoning loop
result = await client.context_rlm(
    goal="Analyze coding preferences and identify contradictions",
    memory_query="coding preferences",
    max_iterations=10,
    detail_level="detailed"
)
print(result["result"])
```

### Inject Values

```python
# Inject data into sandbox
await client.context_inject(
    key="config",
    value={"debug": True, "max_retries": 3}
)
```

### Status and Cleanup

```python
# Check sandbox status
status = await client.context_status()
print(f"Variables: {status['variable_count']}")

# Checkpoint state (for enterprise persistence)
await client.context_checkpoint()

# Clean up sandbox
await client.context_cleanup()
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
- **Code Pattern** - Reusable patterns
- **Fix** - Bug fixes with context
- **Error** - Error patterns and resolutions
- **Workflow** - Process knowledge
- **Preference** - User/project preferences
- **Decision** - Architectural decisions
- **Directive** - User instructions/constraints

## Relationship Types

Link memories with typed relationships organized into 11 categories. The SDK supports 60+ relationship types:

```python
from memorylayer import RelationshipType

# Causal (4 types)
RelationshipType.CAUSES
RelationshipType.TRIGGERS
RelationshipType.LEADS_TO
RelationshipType.PREVENTS

# Solution (4 types)
RelationshipType.SOLVES
RelationshipType.ADDRESSES
RelationshipType.ALTERNATIVE_TO
RelationshipType.IMPROVES

# Learning (4 types)
RelationshipType.BUILDS_ON
RelationshipType.CONTRADICTS
RelationshipType.CONFIRMS
RelationshipType.SUPERSEDES

# Similarity (3 types)
RelationshipType.SIMILAR_TO
RelationshipType.VARIANT_OF
RelationshipType.RELATED_TO

# Workflow (4 types)
RelationshipType.FOLLOWS
RelationshipType.DEPENDS_ON
RelationshipType.ENABLES
RelationshipType.BLOCKS

# Quality (3 types)
RelationshipType.EFFECTIVE_FOR
RelationshipType.PREFERRED_OVER
RelationshipType.DEPRECATED_BY

# Context (4 types)
RelationshipType.OCCURS_IN
RelationshipType.APPLIES_TO
RelationshipType.WORKS_WITH
RelationshipType.REQUIRES

# ... and more categories (11 total)
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
except ValidationError as e:
    print(f"Validation error: {e}")
except RateLimitError:
    print("Rate limit exceeded")
except ServerError as e:
    print(f"Server error: {e.status_code}")
except MemoryLayerError as e:
    print(f"MemoryLayer error: {e}")
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

Apache 2.0 License -- see [LICENSE](../LICENSE) for details.

## Links

- [Documentation](https://docs.memorylayer.ai)
- [GitHub](https://github.com/scitrera/memorylayer)
- [Homepage](https://memorylayer.ai)
