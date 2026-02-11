---
title: Python API Reference
description: Complete API reference for the MemoryLayer Python SDK
sidebar:
  order: 3
  label: API Reference
---

## MemoryLayerClient

The main client class for interacting with MemoryLayer.

### Constructor

```python
MemoryLayerClient(
    base_url: str = "http://localhost:61001",
    api_key: str | None = None,
    workspace_id: str | None = None,
    session_id: str | None = None,
    timeout: float = 30.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:61001"` | Server URL |
| `api_key` | `str \| None` | `None` | API key for authentication |
| `workspace_id` | `str \| None` | `None` | Default workspace ID |
| `session_id` | `str \| None` | `None` | Active session ID |
| `timeout` | `float` | `30.0` | Request timeout in seconds |

### Memory Operations

#### remember()

Store a new memory.

```python
async def remember(
    content: str,
    type: MemoryType | str | None = None,
    subtype: MemorySubtype | str | None = None,
    importance: float = 0.5,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    space_id: str | None = None,
) -> Memory
```

#### recall()

Search memories by semantic query.

```python
async def recall(
    query: str,
    types: list[MemoryType] | None = None,
    subtypes: list[MemorySubtype] | None = None,
    tags: list[str] | None = None,
    mode: RecallMode | None = None,
    limit: int = 10,
    min_relevance: float | None = None,
    recency_weight: float | None = None,
    tolerance: SearchTolerance | None = None,
    include_associations: bool | None = None,
    traverse_depth: int | None = None,
    max_expansion: int | None = None,
) -> RecallResult
```

#### reflect()

Synthesize insights across memories.

```python
async def reflect(
    query: str,
    max_tokens: int = 500,
    include_sources: bool = True,
) -> ReflectResult
```

#### get_memory()

Get a specific memory by ID.

```python
async def get_memory(memory_id: str) -> Memory
```

#### update_memory()

Update a memory's properties.

```python
async def update_memory(
    memory_id: str,
    content: str | None = None,
    importance: float | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> Memory
```

#### forget()

Delete or archive a memory.

```python
async def forget(memory_id: str, hard: bool = False) -> bool
```

#### decay()

Reduce a memory's importance.

```python
async def decay(memory_id: str, decay_rate: float = 0.1) -> Memory
```

#### trace_memory()

Get memory provenance and association chain.

```python
async def trace_memory(memory_id: str) -> dict
```

#### batch_memories()

Perform multiple operations in one request.

```python
async def batch_memories(operations: list[dict]) -> dict
```

### Association Operations

#### associate()

Create a relationship between memories.

```python
async def associate(
    source_id: str,
    target_id: str,
    relationship: RelationshipType,
    strength: float = 0.5,
    metadata: dict | None = None,
) -> Association
```

#### get_associations()

Get associations for a memory.

```python
async def get_associations(
    memory_id: str,
    direction: str = "both"
) -> list[Association]
```

### Session Operations

#### create_session()

```python
async def create_session(
    ttl_seconds: int = 3600,
    workspace_id: str | None = None,
    context_id: str | None = None,
    auto_set_session: bool = True,
) -> Session
```

#### get_session()

```python
async def get_session(session_id: str) -> Session
```

#### set_session()

```python
async def set_session(session_id: str) -> None
```

#### get_session_id()

```python
async def get_session_id() -> str | None
```

#### clear_session()

```python
async def clear_session() -> None
```

#### set_context()

Store key-value pair in session working memory.

```python
async def set_context(
    session_id: str,
    key: str,
    value: Any,
    ttl_seconds: int | None = None,
) -> None
```

#### get_context()

Retrieve values from session working memory.

```python
async def get_context(
    session_id: str,
    keys: list[str],
) -> dict
```

#### commit_session()

Commit working memory to long-term storage.

```python
async def commit_session(
    session_id: str,
    min_importance: float = 0.5,
    deduplicate: bool = True,
    categories: list[str] | None = None,
    max_memories: int = 50,
) -> dict
```

#### touch_session()

Extend session TTL.

```python
async def touch_session(session_id: str) -> dict
```

#### delete_session()

```python
async def delete_session(session_id: str) -> bool
```

### Workspace Operations

#### create_workspace()

```python
async def create_workspace(name: str, **kwargs) -> Workspace
```

#### get_workspace()

```python
async def get_workspace(workspace_id: str) -> Workspace
```

#### update_workspace()

```python
async def update_workspace(
    workspace_id: str,
    name: str | None = None,
    settings: dict | None = None,
) -> Workspace
```

#### get_workspace_schema()

```python
async def get_workspace_schema(workspace_id: str) -> dict
```

### Briefing

#### get_briefing()

```python
async def get_briefing(
    lookback_hours: int = 24,
) -> SessionBriefing
```

### Context Environment Operations

#### context_exec()

```python
async def context_exec(
    code: str,
    result_var: str | None = None,
    return_result: bool = True,
    max_return_chars: int = 10_000,
) -> dict[str, Any]
```

#### context_inspect()

```python
async def context_inspect(
    variable: str | None = None,
    preview_chars: int = 200,
) -> dict[str, Any]
```

#### context_load()

```python
async def context_load(
    var: str,
    query: str,
    limit: int = 50,
    types: list[MemoryType] | None = None,
    tags: list[str] | None = None,
    min_relevance: float | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]
```

#### context_inject()

```python
async def context_inject(
    key: str,
    value: str,
    parse_json: bool = False,
) -> dict[str, Any]
```

#### context_query()

```python
async def context_query(
    prompt: str,
    variables: list[str],
    max_context_chars: int | None = None,
    result_var: str | None = None,
) -> dict[str, Any]
```

#### context_rlm()

```python
async def context_rlm(
    goal: str,
    memory_query: str | None = None,
    memory_limit: int = 100,
    max_iterations: int = 10,
    variables: list[str] | None = None,
    result_var: str | None = None,
    detail_level: str = "standard",
) -> dict[str, Any]
```

#### context_status()

```python
async def context_status() -> dict[str, Any]
```

#### context_checkpoint()

```python
async def context_checkpoint() -> None
```

#### context_cleanup()

```python
async def context_cleanup() -> None
```

## Types

### MemoryType

```python
class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
```

### MemorySubtype

```python
class MemorySubtype(str, Enum):
    SOLUTION = "solution"
    PROBLEM = "problem"
    CODE_PATTERN = "code_pattern"
    FIX = "fix"
    ERROR = "error"
    WORKFLOW = "workflow"
    PREFERENCE = "preference"
    DECISION = "decision"
    PROFILE = "profile"
    ENTITY = "entity"
    EVENT = "event"
    DIRECTIVE = "directive"
```

### RecallMode

```python
class RecallMode(str, Enum):
    RAG = "rag"
    LLM = "llm"
    HYBRID = "hybrid"
```

### SearchTolerance

```python
class SearchTolerance(str, Enum):
    LOOSE = "loose"
    MODERATE = "moderate"
    STRICT = "strict"
```

### RelationshipType

60+ relationship types organized into 11 categories:

- **Hierarchical**: `parent_of`, `child_of`, `part_of`, `contains`, `instance_of`, `subtype_of`
- **Causal**: `causes`, `triggers`, `leads_to`, `prevents`, and more
- **Temporal**: `precedes`, `concurrent_with`, `follows_temporally`
- **Similarity**: `similar_to`, `variant_of`, `related_to`, `analogous_to`
- **Learning**: `builds_on`, `contradicts`, `confirms`, `supersedes`, and more
- **Refinement**: `refines`, `abstracts`, `specializes`, `generalizes`
- **Reference**: `references`, `referenced_by`
- **Solution**: `solves`, `addresses`, `alternative_to`, `improves`, and more
- **Context**: `occurs_in`, `applies_to`, `works_with`, `requires`, and more
- **Workflow**: `follows`, `depends_on`, `enables`, `blocks`, and more
- **Quality**: `effective_for`, `preferred_over`, `deprecated_by`, and more

Use `get_workspace_schema()` to list all available relationship types.

## Exceptions

```python
MemoryLayerError          # Base exception
├── AuthenticationError   # 401 - Invalid API key
├── AuthorizationError    # 403 - Access denied
├── NotFoundError         # 404 - Resource not found
├── ValidationError       # 422 - Invalid request
├── RateLimitError        # 429 - Rate limit exceeded
└── ServerError           # 500+ - Server error
```

All exceptions have these properties:

| Property | Type | Description |
|----------|------|-------------|
| `message` | `str` | Error message |
| `status_code` | `int` | HTTP status code |
