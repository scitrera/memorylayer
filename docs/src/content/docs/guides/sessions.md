---
title: Sessions & Working Memory
description: Manage temporary context with sessions and commit to long-term storage
sidebar:
  order: 3
  label: Sessions & Working Memory
---

Sessions provide **working memory** — temporary context that persists across API calls within a session and can optionally be committed to long-term storage.

## How Sessions Work

```
Session Start
    ↓
Store working memory (key-value pairs)
    ↓
Working memory persists across requests
    ↓
Session End → optionally commit to long-term storage
```

## Creating a Session

### Python

```python
session = await client.create_session(
    ttl_seconds=3600,
    workspace_id="my-workspace"
)
print(f"Session ID: {session.id}")
```

### TypeScript

```typescript
const { session, briefing } = await client.createSession({
  workspaceId: "my-workspace",
  ttlSeconds: 3600,
  briefing: true,  // Get briefing on session start
});
```

## Working Memory

Store and retrieve key-value pairs within a session:

### Python

```python
# Store working memory
await client.set_context(
    session.id,
    "current_task",
    {"description": "Debugging auth", "file": "auth.py"}
)

# Retrieve working memory
context = await client.get_context(session.id, ["current_task"])
```

### TypeScript

```typescript
// Store working memory
await client.setWorkingMemory(session.id, "current_task", {
  description: "Debugging auth",
  file: "auth.py",
});

// Retrieve working memory
const memory = await client.getWorkingMemory(session.id, "current_task");
```

### MCP Tools

When using via MCP server (Claude Code, Claude Desktop), use the session tools:

```json
// memory_session_start — Start a session
{ "workspace_id": "my-project", "ttl_seconds": 3600 }

// memory_session_status — Check session status
{}

// memory_session_commit — Commit working memory to long-term storage
{ "importance_threshold": 0.5 }

// memory_session_end — End session
{ "commit": true }
```

## Session Lifecycle

### Extend TTL

Keep a session alive:

```python
await client.touch_session(session.id)
```

### Commit to Long-Term Storage

Extract important working memory items and store them as permanent memories:

```python
result = await client.commit_session(
    session_id,
    min_importance=0.5,
    deduplicate=True
)
print(f"Created {result['memories_created']} long-term memories")
```

### End a Session

```python
await client.delete_session(session.id)
```

## Session Briefings

Get a summary of recent activity when starting a new session:

```python
briefing = await client.get_briefing(lookback_hours=24)
print(briefing.recent_activity_summary)
```

```typescript
const briefing = await client.getBriefing(24, true);
console.log(briefing.recent_activity_summary);
console.log(briefing.open_threads);
```

## Use Cases

### Agent Task Tracking

```python
# Start of task
await client.set_context(session.id, "task", {
    "goal": "Fix authentication bug",
    "files_modified": [],
    "decisions": []
})

# During work
task = await client.get_context(session.id, ["task"])
task["files_modified"].append("auth.py")
await client.set_context(session.id, "task", task)

# End of task — commit learnings
await client.commit_session(session.id)
```

### Claude Code Sessions

The MCP server's session tools are designed for Claude Code workflows:

1. `memory_session_start` — Initialize at session start
2. `memory_remember` — Store decisions and learnings during work
3. `memory_session_end` with `commit: true` — Persist working memory at session end

### Multi-Turn Conversations

Sessions keep context across multiple API calls without re-fetching:

```python
session = await client.create_session(ttl_seconds=7200)

# Turn 1
await client.set_context(session.id, "user_name", "Alice")
await client.set_context(session.id, "topic", "Python debugging")

# Turn 2 (different API call, same session)
name = await client.get_context(session.id, ["user_name"])
# Returns "Alice" — context preserved
```
