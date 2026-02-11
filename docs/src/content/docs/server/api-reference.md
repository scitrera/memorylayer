---
title: REST API Reference
description: Complete REST API reference for the MemoryLayer server
sidebar:
  order: 3
  label: REST API Reference
---

The MemoryLayer server exposes a RESTful API at `http://localhost:61001` by default. All endpoints accept and return JSON.

## Authentication

For local development, no authentication is required. For production deployments, include an API key in the `Authorization` header:

```
Authorization: Bearer your-api-key
```

## Memory Operations

### POST /v1/memories

Store a new memory.

**Request Body:**

```json
{
  "content": "User prefers Python for backend development",
  "type": "semantic",
  "subtype": "preference",
  "importance": 0.8,
  "tags": ["preferences", "programming"],
  "metadata": {"source": "conversation"},
  "workspace_id": "my-workspace"
}
```

**Response:** `201 Created`

```json
{
  "id": "mem_abc123",
  "content": "User prefers Python for backend development",
  "type": "semantic",
  "subtype": "preference",
  "importance": 0.8,
  "tags": ["preferences", "programming"],
  "created_at": "2026-01-15T10:30:00Z"
}
```

### POST /v1/memories/recall

Search memories by semantic query.

**Request Body:**

```json
{
  "query": "What programming languages does the user prefer?",
  "workspace_id": "my-workspace",
  "limit": 10,
  "min_relevance": 0.5,
  "types": ["semantic"],
  "tags": ["preferences"],
  "mode": "rag"
}
```

**Response:** `200 OK`

```json
{
  "memories": [
    {
      "id": "mem_abc123",
      "content": "User prefers Python for backend development",
      "type": "semantic",
      "importance": 0.8,
      "relevance": 0.92,
      "tags": ["preferences", "programming"]
    }
  ],
  "total_count": 1,
  "mode_used": "rag",
  "search_latency_ms": 42.5
}
```

### POST /v1/memories/reflect

Synthesize insights across memories.

**Request Body:**

```json
{
  "query": "Summarize user's technology preferences",
  "workspace_id": "my-workspace",
  "detail_level": "overview",
  "include_sources": true,
  "depth": 2,
  "types": ["semantic"],
  "tags": ["preferences"]
}
```

**Response:** `200 OK`

```json
{
  "reflection": "Based on stored memories, the user prefers...",
  "source_memories": ["mem_abc123", "mem_def456"],
  "confidence": 0.85
}
```

### GET /v1/memories/{memory_id}

Get a specific memory by ID.

### PUT /v1/memories/{memory_id}

Update a memory's content, importance, tags, or metadata.

### DELETE /v1/memories/{memory_id}

Delete a memory (soft delete by default).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hard` | boolean | `false` | Permanently delete instead of archiving |

### POST /v1/memories/{memory_id}/decay

Apply decay to reduce a memory's importance.

**Request Body:**

```json
{
  "decay_rate": 0.1
}
```

### POST /v1/memories/batch

Perform multiple memory operations in a single request.

**Request Body:**

```json
{
  "operations": [
    {"type": "create", "data": {"content": "Memory 1", "importance": 0.7}},
    {"type": "create", "data": {"content": "Memory 2", "importance": 0.8}},
    {"type": "delete", "data": {"memory_id": "mem_old", "hard": false}}
  ],
  "workspace_id": "my-workspace"
}
```

## Associations

### POST /v1/memories/{memory_id}/associate

Create a relationship between two memories.

**Request Body:**

```json
{
  "target_id": "mem_solution_456",
  "relationship": "solves",
  "strength": 0.9,
  "workspace_id": "my-workspace"
}
```

### GET /v1/memories/{memory_id}/associations

Get all associations for a memory.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | string | `both` | `outgoing`, `incoming`, or `both` |

### POST /v1/memories/{memory_id}/traverse

Multi-hop graph traversal.

**Request Body:**

```json
{
  "relationship_types": ["causes", "triggers", "leads_to"],
  "max_depth": 3,
  "direction": "both",
  "min_strength": 0.5,
  "workspace_id": "my-workspace"
}
```

## Sessions

### POST /v1/sessions

Create a new session.

**Request Body:**

```json
{
  "workspace_id": "my-workspace",
  "ttl_seconds": 3600,
  "metadata": {"task": "debugging"}
}
```

### POST /v1/sessions/{session_id}/memory

Store a key-value pair in session working memory.

**Request Body:**

```json
{
  "key": "current_task",
  "value": {"description": "Debugging auth", "file": "auth.py"},
  "ttl_seconds": 3600
}
```

### GET /v1/sessions/{session_id}/memory

Retrieve a value from session working memory.

### POST /v1/sessions/{session_id}/commit

Commit session working memory to long-term storage.

**Request Body:**

```json
{
  "min_importance": 0.5,
  "deduplicate": true,
  "categories": ["task", "findings"],
  "max_memories": 50
}
```

### POST /v1/sessions/{session_id}/touch

Extend session TTL.

### DELETE /v1/sessions/{session_id}

Delete a session.

## Workspaces

### POST /v1/workspaces

Create a new workspace.

### GET /v1/workspaces/{workspace_id}

Get workspace details.

### PUT /v1/workspaces/{workspace_id}

Update workspace settings.

### GET /v1/workspaces/{workspace_id}/schema

Get workspace schema (relationship types, memory subtypes).

## Briefing

### GET /v1/sessions/briefing

Get a session briefing with recent activity summary. Workspace is resolved from authentication context headers. No query parameters required.

## Contradictions

### GET /v1/workspaces/{workspace_id}/contradictions

List contradictions in a workspace.

### POST /v1/contradictions/{contradiction_id}/resolve

Resolve a contradiction.

## Context Environment

All context endpoints require an `X-Session-ID` header to identify the session sandbox.

### POST /v1/context/execute

Execute Python code in the session sandbox. State persists across calls.

**Request Body:**

```json
{
  "code": "result = sum(x['importance'] for x in memories)",
  "result_var": "total_importance",
  "return_result": true,
  "max_return_chars": 10000
}
```

**Response:** `200 OK`

```json
{
  "output": "",
  "result": "4.2",
  "error": null,
  "variables_changed": ["result", "total_importance"]
}
```

### POST /v1/context/inspect

Inspect sandbox variables. Omit `variable` for overview of all variables.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variable` | string | — | Specific variable to inspect (omit for overview) |
| `preview_chars` | integer | `200` | Characters to include in value previews |

**Example:**

```
POST /v1/context/inspect?variable=memories&preview_chars=200
```

**Response (specific variable):**

```json
{
  "variable": "memories",
  "type": "list",
  "preview": "[{'id': 'mem_abc', 'content': 'User prefers...'}]",
  "size_bytes": 4096
}
```

**Response (all variables):**

```json
{
  "variable_count": 3,
  "variables": {"memories": "list (50 items)", "summary": "str", "total": "float"},
  "total_size_bytes": 8192
}
```

### POST /v1/context/load

Load memories into sandbox via semantic search.

**Request Body:**

```json
{
  "var": "auth_memories",
  "query": "authentication patterns",
  "limit": 50,
  "types": ["semantic", "procedural"],
  "tags": ["auth"],
  "min_relevance": 0.5,
  "include_embeddings": false
}
```

**Response:** `200 OK`

```json
{
  "count": 12,
  "variable": "auth_memories",
  "query": "authentication patterns",
  "total_available": 25
}
```

### POST /v1/context/inject

Inject a value directly into the sandbox.

**Request Body:**

```json
{
  "key": "config",
  "value": "{\"threshold\": 0.7, \"max_items\": 100}",
  "parse_json": true
}
```

**Response:** `200 OK`

```json
{
  "variable": "config",
  "type": "dict",
  "preview": "{'threshold': 0.7, 'max_items': 100}"
}
```

### POST /v1/context/query

Query the server-side LLM with sandbox variables as context.

**Request Body:**

```json
{
  "prompt": "What are the common error patterns in this data?",
  "variables": ["error_memories"],
  "max_context_chars": 50000,
  "result_var": "analysis"
}
```

**Response:** `200 OK`

```json
{
  "response": "Based on the error memories, there are three common patterns...",
  "variables_used": ["error_memories"],
  "result_var": "analysis"
}
```

### POST /v1/context/rlm

Run a Recursive Language Model reasoning loop.

**Request Body:**

```json
{
  "goal": "Identify the most common error patterns and their resolutions",
  "memory_query": "errors and fixes",
  "memory_limit": 100,
  "max_iterations": 10,
  "variables": ["existing_analysis"],
  "result_var": "findings",
  "detail_level": "standard"
}
```

**Response:** `200 OK`

```json
{
  "result": "Analysis found 3 recurring error patterns...",
  "iterations": 4,
  "goal_achieved": true,
  "trace": [
    {
      "iteration": 1,
      "generated_code": "...",
      "exec_output": "...",
      "variables_changed": ["grouped_errors"],
      "evaluation": "CONTINUE",
      "action": "continue"
    }
  ]
}
```

### GET /v1/context/status

Get sandbox status.

**Response:** `200 OK`

```json
{
  "exists": true,
  "variable_count": 5,
  "variables": ["memories", "summary", "config"],
  "total_size_bytes": 16384,
  "memory_limit_bytes": 268435456,
  "metadata": {
    "created_at": "2026-01-15T10:30:00Z",
    "exec_count": 12,
    "total_operations": 20,
    "last_exec_at": "2026-01-15T10:45:00Z"
  }
}
```

### POST /v1/context/checkpoint

Checkpoint sandbox state for persistence. Fires persistence hooks for enterprise deployments.

**Response:** `204 No Content`

### DELETE /v1/context/cleanup

Remove session sandbox.

**Response:** `204 No Content`

## Health

### GET /health

Health check endpoint. Returns `200 OK` when the server is running.

### GET /health/ready

Readiness check with service status.
