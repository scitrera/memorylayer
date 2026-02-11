---
title: Memory Types
description: Understanding cognitive memory types and domain subtypes in MemoryLayer
sidebar:
  order: 1
  label: Memory Types
---

MemoryLayer uses a two-layer classification system for memories, inspired by cognitive science and developer workflow research.

## Cognitive Types

Every memory has a **cognitive type** that describes how it is structured and retained.

### Episodic Memory

**What happened** — specific events and interactions.

- Decays naturally over time
- Timestamped and contextual
- Best for tracking what occurred during sessions

```python
await client.remember(
    content="Discovered a race condition in the payment service during load testing on Feb 3",
    type=MemoryType.EPISODIC,
    importance=0.8,
    tags=["payment", "bug", "load-testing"]
)
```

### Semantic Memory

**What is true** — facts, concepts, and relationships.

- Permanent until explicitly modified or superseded
- Forms the backbone of the knowledge graph
- Best for preferences, decisions, and factual knowledge

```python
await client.remember(
    content="The project uses PostgreSQL 16 with pgvector for embeddings",
    type=MemoryType.SEMANTIC,
    importance=0.7,
    tags=["database", "architecture"]
)
```

### Procedural Memory

**How to do things** — processes, patterns, and solutions.

- Permanent and reusable
- Often linked to problem memories via `SOLVES` relationships
- Best for solutions, workflows, and code patterns

```python
await client.remember(
    content="To fix CORS issues: add allowed origins to FastAPI middleware via CORSMiddleware",
    type=MemoryType.PROCEDURAL,
    importance=0.8,
    tags=["cors", "fastapi", "fix"]
)
```

### Working Memory

**Current context** — what's happening right now.

- Session-scoped with TTL
- Automatically expires when the session ends
- Can be committed to long-term storage (episodic/semantic) via `commit_session()`
- Best for tracking current task state

```python
await client.set_context(
    session_id,
    "current_task",
    {"description": "Debugging auth flow", "file": "auth.py", "line": 42}
)
```

## Domain Subtypes

An optional second classification that describes **what** the memory is about. Subtypes work across all cognitive types.

| Subtype | Description | Typical Cognitive Type |
|---------|-------------|----------------------|
| **solution** | Working fixes to problems | Procedural |
| **problem** | Issues encountered | Episodic |
| **code_pattern** | Reusable code patterns | Procedural |
| **fix** | Bug fixes with context | Procedural |
| **error** | Error patterns and resolutions | Semantic |
| **workflow** | Process knowledge | Procedural |
| **preference** | User or project preferences | Semantic |
| **decision** | Architectural decisions | Semantic |
| **profile** | Person or entity profiles | Semantic |
| **entity** | Named entities (people, places, things) | Semantic |
| **event** | Significant events or milestones | Episodic |
| **directive** | User instructions and constraints | Semantic |

### Combining Types and Subtypes

```python
# A decision (semantic + decision)
await client.remember(
    content="Chose SQLite over PostgreSQL for the OSS version for zero-config deployment",
    type=MemoryType.SEMANTIC,
    subtype=MemorySubtype.DECISION,
    importance=0.9,
    tags=["database", "architecture", "oss"]
)

# A bug fix (procedural + fix)
await client.remember(
    content="Fixed memory leak in WebSocket handler by closing connections in finally block",
    type=MemoryType.PROCEDURAL,
    subtype=MemorySubtype.FIX,
    importance=0.8,
    tags=["websocket", "memory-leak", "bug"]
)
```

## Importance Scoring

Every memory has an importance score between 0.0 and 1.0:

| Score | Level | Examples |
|-------|-------|---------|
| **0.9–1.0** | Critical | Security vulnerabilities, breaking changes, core architecture decisions |
| **0.7–0.8** | High | Bug fixes, API changes, important patterns |
| **0.5–0.6** | Standard | Preferences, general knowledge, minor solutions |
| **0.3–0.4** | Low | Observations, temporary workarounds |
| **0.0–0.2** | Ephemeral | Debugging traces, session context |

### Decay

Memories can have their importance reduced over time using the `decay()` operation:

```python
# Reduce importance by 10%
await client.decay("mem_123", decay_rate=0.1)
```

This is useful for episodic memories that become less relevant over time, while keeping semantic and procedural memories stable.

## Choosing the Right Type

| Scenario | Cognitive Type | Subtype | Importance |
|----------|---------------|---------|------------|
| "User prefers dark mode" | Semantic | Preference | 0.6 |
| "Fixed CORS by adding middleware" | Procedural | Fix | 0.8 |
| "Decided to use JWT for auth" | Semantic | Decision | 0.9 |
| "Encountered timeout in CI pipeline" | Episodic | Problem | 0.5 |
| "Deploy process: git push → CI → staging → prod" | Procedural | Workflow | 0.7 |
| "ECONNREFUSED means Redis isn't running" | Semantic | Error | 0.6 |
