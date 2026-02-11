---
title: Workspaces
description: Organize memories with workspace isolation
sidebar:
  order: 2
  label: Workspaces
---

## Workspaces

Workspaces provide **tenant isolation** for memories. Each workspace has completely independent:

- Memory store with its own vector indices
- Relationship graph
- Session history
- Configuration and settings

### Creating Workspaces

```python
workspace = await client.create_workspace("my-project")
```

```typescript
const workspace = await client.createWorkspace("My Project", {
  embedding_model: "text-embedding-3-small",
  default_importance: 0.5,
});
```

### Using Workspaces

Set the workspace at client initialization:

```python
client = MemoryLayerClient(
    base_url="http://localhost:61001",
    workspace_id="my-project"
)
```

All subsequent operations are scoped to that workspace.

### Workspace Isolation

Memories in different workspaces are completely separate:

```
Workspace: "frontend-app"          Workspace: "backend-api"
├── "Uses React 18"                ├── "Uses FastAPI"
├── "Prefers Tailwind CSS"         ├── "PostgreSQL for persistence"
└── "Dark mode by default"         └── "JWT authentication"
```

### Common Patterns

| Pattern | Workspace Strategy |
|---------|-------------------|
| Per-project | One workspace per git repository |
| Per-user | One workspace per user account |
| Per-environment | Separate dev/staging/prod workspaces |
| Per-agent | Each agent instance gets its own workspace |

### Auto-Detection (MCP Server)

The MCP server automatically detects the workspace from your project:

1. Git remote origin URL → extracts repo name
2. Git root directory name
3. Current working directory name

```
~/code/my-app/        → workspace: "my-app"
~/code/api-server/    → workspace: "api-server"
```

### Workspace Schema

Each workspace has a schema defining available relationship types and memory subtypes:

```python
schema = await client.get_workspace_schema("ws_123")
print(schema["relationship_types"])   # Available relationship types
print(schema["memory_subtypes"])      # Available subtypes
```

## Organizing Memories

Within a workspace, use **tags** for fine-grained classification of memories:

```python
memory = await client.remember(
    content="JWT tokens expire after 1 hour",
    tags=["auth", "jwt", "security"]
)

# Filter by tags during recall
results = await client.recall(
    query="authentication setup",
    tags=["auth"]
)
```

Tags are flexible, per-memory labels that support AND-logic filtering during recall.
