---
title: Core Concepts
description: Understand the fundamental concepts of MemoryLayer
sidebar:
  order: 3
---

## Memory Types

MemoryLayer provides **two orthogonal classification systems** for memories.

### Cognitive Types

Every memory has a cognitive type that describes how it is structured:

| Type | Description | Retention | Example |
|------|-------------|-----------|---------|
| **Episodic** | Specific events or interactions | Decays over time | "User asked about Python logging on Jan 15" |
| **Semantic** | Facts, concepts, relationships | Permanent until modified | "User prefers TypeScript over JavaScript" |
| **Procedural** | How to do things | Permanent | "To deploy, run `npm run deploy`" |
| **Working** | Current task context | Session-scoped | "Currently debugging auth.py line 42" |

### Domain Subtypes

An optional classification that describes *what* the memory is about:

| Subtype | Description | Example |
|---------|-------------|---------|
| **solution** | Working fixes to problems | "Fixed CORS by adding allowed origins to middleware" |
| **problem** | Issues encountered | "Auth tokens expire silently without refresh" |
| **code_pattern** | Reusable patterns | "Use dependency injection for database connections" |
| **fix** | Bug fixes with context | "Fixed race condition in user sync by adding mutex" |
| **error** | Error patterns and resolutions | "ECONNREFUSED means Redis isn't running" |
| **workflow** | Process knowledge | "Always run migrations before deploying" |
| **preference** | User or project preferences | "Use 2-space indentation in this project" |
| **decision** | Architectural decisions | "Chose PostgreSQL over MongoDB for ACID compliance" |
| **profile** | Person or entity profiles | "Alice is the backend lead, prefers Go" |
| **entity** | Named entities (people, places, things) | "Project Atlas is the internal data pipeline" |
| **event** | Significant events or milestones | "Launched v2.0 on March 15" |
| **directive** | User instructions and constraints | "Always use TypeScript strict mode" |

## Importance Scoring

Every memory has an importance score between 0.0 and 1.0 that influences retrieval priority and decay behavior:

| Score Range | Level | Use For |
|-------------|-------|---------|
| **0.9–1.0** | Critical | Security fixes, breaking changes, core architecture |
| **0.7–0.8** | High | Bug fixes, API changes, important patterns |
| **0.5–0.6** | Standard | Preferences, general knowledge |
| **0.3–0.4** | Low | Observations, temporary workarounds |
| **0.0–0.2** | Ephemeral | Debugging traces, session context |

## Workspaces

Workspaces provide **tenant isolation** for memories. Each workspace has its own:

- Memory store with independent vector indices
- Relationship graph
- Session history
- Configuration and settings

Typical usage: one workspace per project, per user, or per environment.

```python
# All operations are scoped to a workspace
client = MemoryLayerClient(
    base_url="http://localhost:61001",
    workspace_id="my-project"
)
```

## Sessions

Sessions provide **working memory** — temporary context that persists within a session and can optionally be committed to long-term storage.

```python
# Start a session
session = await client.create_session(
    workspace_id="my-project",
    ttl_seconds=3600
)

# Store working memory
await client.set_context(session.id, "current_task", {
    "description": "Debugging authentication",
    "file": "auth.py"
})

# Later: commit important items to long-term storage
await client.commit_session(session.id, min_importance=0.5)
```

## Associations (Knowledge Graph)

Memories can be linked with **typed relationships** to form a knowledge graph. This enables multi-hop queries that vector similarity alone cannot answer.

### Relationship Categories

Typed relationships organized across categories:

| Category | Example Relationships | Use Case |
|----------|----------------------|----------|
| **Hierarchical** | `parent_of`, `child_of`, `part_of`, `contains` | "What is this part of?" |
| **Causal** | `causes`, `triggers`, `leads_to`, `prevents` | "What caused this error?" |
| **Temporal** | `precedes`, `concurrent_with`, `follows_temporally` | "What happened when?" |
| **Similarity** | `similar_to`, `variant_of`, `related_to` | "What else is relevant?" |
| **Learning** | `builds_on`, `contradicts`, `confirms`, `supersedes` | "Is this still accurate?" |
| **Refinement** | `refines`, `abstracts`, `specializes`, `generalizes` | "How does this relate?" |
| **Reference** | `references`, `referenced_by` | "What references this?" |
| **Solution** | `solves`, `addresses`, `alternative_to`, `improves` | "What fixes this problem?" |
| **Context** | `occurs_in`, `applies_to`, `works_with`, `requires` | "Where does this apply?" |
| **Workflow** | `follows`, `depends_on`, `enables`, `blocks` | "What's the sequence?" |
| **Quality** | `effective_for`, `preferred_over`, `deprecated_by` | "What's the best approach?" |

### Example

```python
from memorylayer import RelationshipType

# Link a problem to its solution
await client.associate(
    source_id="mem_problem_123",
    target_id="mem_solution_456",
    relationship=RelationshipType.SOLVES,
    strength=0.9
)

# Get associations for a memory
associations = await client.get_associations("mem_problem_123", direction="both")
```

## Context Environment

The Context Environment provides a **server-side Python sandbox** for working with memories programmatically. It separates the variable space (data in the sandbox) from the token space (what the LLM processes), enabling analysis of large memory sets without consuming the client's context window.

### Key Operations

| Operation | Description                                                                    |
|-----------|--------------------------------------------------------------------------------|
| **Load** | Load memories into sandbox variables via semantic search                       |
| **Execute** | Run Python code against sandbox data                                           |
| **Query** | Ask the server-side LLM using sandbox variables as context                     |
| **RLM** | Run an autonomous recursive language model session |
| **Inspect** | View sandbox state (useful after LLM context resets)                           |

### Two Reasoning Modes

**Classic RLM** — The server autonomously orchestrates an iterative loop: generate code → execute → evaluate → repeat until the goal is achieved.

**Inverted RLM** — The calling LLM (e.g., Claude Code) drives the loop using individual sandbox operations. The sandbox state survives LLM context compaction, allowing the agent to resume analysis after a context reset by calling `inspect`.

See the [Context Environment Guide](/guides/context-environment/) for detailed usage.

## Recall Modes

When searching memories, you can choose a retrieval strategy:

| Mode | Description | Best For |
|------|-------------|----------|
| **RAG** | Vector similarity search | Fast, precise recall |
| **LLM** | LLM-powered semantic search | Understanding intent |
| **Hybrid** | Combination of both | Best overall quality |

## Core Operations

| Operation | Description |
|-----------|-------------|
| **Remember** | Store a new memory |
| **Recall** | Search memories by semantic query |
| **Reflect** | Synthesize insights across memories |
| **Forget** | Delete or archive a memory |
| **Associate** | Link memories with typed relationships |
| **Decay** | Reduce a memory's importance over time |
| **Trace** | Get a memory's provenance and association chain |
