---
title: Context Environment
description: Server-side sandbox for code execution, LLM queries, and autonomous reasoning over memories
sidebar:
  order: 5
  label: Context Environment
---

The Context Environment provides a **server-side Python sandbox** tied to sessions. It separates the **variable space** (data stored in the sandbox) from the **token space** (what the LLM processes), solving the "context rot" problem where LLM performance degrades as contexts grow.

Key capabilities:

- Execute Python code in a sandboxed environment
- Load memories from the store into sandbox variables
- Query an LLM with sandbox variables as context
- Run autonomous reasoning loops (RLM) over memories

## Two Usage Modes

### Classic RLM

The server orchestrates an iterative reasoning loop. You provide a goal, the server-side LLM generates code, executes it, evaluates progress, and repeats until the goal is achieved. Use the `context_rlm()` operation.

This is based on the Recursive Language Model (RLM) pattern introduced in [Recursive Language Models](https://arxiv.org/abs/2512.24601), where the LLM works iteratively through a code execution sandbox rather than processing the full context in a single prompt.

### Inverted RLM (Agent-Driven)

The calling LLM (e.g., Claude Code via MCP tools) is the orchestrator. It uses the individual context operations (`exec`, `inspect`, `load`, `query`) to drive the reasoning loop itself. The LLM decides what code to run, examines results, and iterates — the server provides the sandbox infrastructure.

This is the default mode when using MemoryLayer with Claude Code via MCP tools.

## Getting Started

**Requirement:** An active session. Context environments are session-scoped.

### Python

```python
# Start a session
session = await client.create_session(workspace_id="my-project")

# Load memories into the sandbox
await client.context_load(
    var="memories",
    query="authentication patterns",
    limit=50
)

# Execute code to analyze them
result = await client.context_exec("""
auth_memories = [m for m in memories if 'auth' in m.get('tags', [])]
summary = f"Found {len(auth_memories)} auth-related memories"
""", result_var="summary")

# Query LLM with sandbox context
answer = await client.context_query(
    prompt="What authentication patterns have been used in this project?",
    variables=["auth_memories"]
)
```

## Operations Reference

| Operation | Description |
|-----------|-------------|
| `context_exec` | Execute Python code in the sandbox |
| `context_inspect` | View sandbox variables and state |
| `context_load` | Load memories into a sandbox variable via semantic search |
| `context_inject` | Inject a value directly into the sandbox |
| `context_query` | Ask the server-side LLM using sandbox variables as context |
| `context_rlm` | Run an autonomous Recursive Language Model reasoning loop |
| `context_status` | Get sandbox status and resource usage |
| `context_checkpoint` | Checkpoint state for persistence hooks |
| `context_cleanup` | Clean up and remove the sandbox (REST API only, not available as MCP tool) |

## Loading Memories

`context_load` runs a semantic recall against the memory store and injects the results into a sandbox variable as a list of dicts. Each dict contains:

- `id` — Memory identifier
- `content` — The memory text
- `type` — Memory type (episodic, semantic, procedural, working)
- `importance` — Importance score (0-1)
- `tags` — List of tags
- `created_at` — Creation timestamp
- `metadata` — Additional metadata

```python
await client.context_load(
    var="recent_errors",
    query="error patterns in production",
    limit=100,
    min_relevance=0.3,
    tags=["error"]
)
```

## Code Execution

The sandbox provides a persistent Python environment within a session. State (variables, imports, function definitions) persists across `context_exec` calls.

### Executor Backends

Two backends are available:

- **`smolagents`** (default) — Full Python execution with controlled imports. Available imports: `math`, `json`, `datetime`, `collections`, `itertools`, `random`, `re`, `statistics`, `functools`.
- **`restricted`** — AST-based whitelist executor. No imports allowed. Suitable for strict sandboxing requirements.

### Rate Limiting

Execution is rate-limited via soft and hard caps to prevent abuse. Soft caps generate warnings; hard caps reject requests.

```python
result = await client.context_exec("""
import json
from collections import Counter

tag_counts = Counter()
for m in memories:
    for tag in m.get('tags', []):
        tag_counts[tag] += 1

top_tags = tag_counts.most_common(10)
""", result_var="top_tags")
```

## LLM Queries

`context_query` sends sandbox variables as context to the server-side LLM. The LLM reasons over datasets loaded into the sandbox without consuming the client's context window.

```python
answer = await client.context_query(
    prompt="Summarize the top error patterns and suggest preventive measures",
    variables=["recent_errors", "top_tags"],
    result_var="analysis"
)
```

## Classic RLM

The autonomous reasoning loop works as follows:

1. Optionally pre-load memories via the `memory_query` parameter
2. Each iteration: the server-side LLM generates Python code, executes it in the sandbox, and evaluates progress
3. The loop continues until the goal is achieved, max iterations are reached, or timeout occurs
4. Returns a synthesis with an iteration trace

### Python

```python
result = await client.context_rlm(
    goal="Identify the most common error patterns and their resolutions",
    memory_query="errors and fixes",
    memory_limit=100,
    max_iterations=10,
    detail_level="standard"
)
print(result["result"])
```

### MCP Tools

When using via MCP server (Claude Code, Claude Desktop):

```json
{
  "goal": "Identify the most common error patterns and their resolutions",
  "memory_query": "errors and fixes",
  "memory_limit": 100,
  "max_iterations": 10,
  "detail_level": "standard"
}
```

## Inverted RLM (Agent-Driven)

When using Claude Code or another orchestrating LLM, the agent drives the reasoning loop directly using individual operations:

1. **Load** data with `context_load`
2. **Inspect** with `context_inspect` to understand what is available
3. **Analyze** by writing and running code with `context_exec`
4. **Ask** the server-side LLM questions with `context_query`
5. **Iterate** until satisfied

The sandbox state survives LLM context compaction. After a context reset, the agent can call `context_inspect` to re-orient with its existing sandbox variables and resume work without data loss.

## Persistence and Lifecycle

- Sandboxes are tied to sessions and cleaned up automatically when sessions end
- Use `context_checkpoint` to fire persistence hooks (enterprise deployments can save sandbox state to durable storage)
- Use `context_cleanup` to explicitly clean up and release sandbox resources

```python
# Checkpoint during a long session
await client.context_checkpoint()

# Explicit cleanup
await client.context_cleanup()
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_CONTEXT_EXECUTOR` | `smolagents` | Executor backend (`smolagents` or `restricted`) |
| `MEMORYLAYER_CONTEXT_MAX_EXEC_SECONDS` | `30` | Timeout per execution |
| `MEMORYLAYER_CONTEXT_MAX_OUTPUT_CHARS` | `50000` | Max captured output characters |
| `MEMORYLAYER_CONTEXT_QUERY_MAX_TOKENS` | `4096` | Max tokens for LLM queries |
| `MEMORYLAYER_CONTEXT_MAX_MEMORY_BYTES` | `268435456` | Memory limit per sandbox (256 MB) |
| `MEMORYLAYER_CONTEXT_RLM_MAX_ITERATIONS` | `10` | Global RLM iteration cap |
| `MEMORYLAYER_CONTEXT_RLM_MAX_EXEC_SECONDS` | `120` | Total RLM timeout |
