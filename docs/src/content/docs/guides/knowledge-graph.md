---
title: Knowledge Graph
description: Link memories with typed relationships for multi-hop queries
sidebar:
  order: 4
  label: Knowledge Graph
---

MemoryLayer's knowledge graph connects memories with **typed relationships**, enabling queries that vector similarity search alone cannot answer.

## Why a Knowledge Graph?

Vector search finds memories with similar content. But some questions require following chains of relationships:

- "What caused this error?" → follow `CAUSES` → `LEADS_TO` chains
- "What solutions have we tried?" → follow `SOLVES` → `ALTERNATIVE_TO` paths
- "Is this still the recommended approach?" → check `SUPERSEDES` → `DEPRECATED_BY`

## Creating Associations

### Python

```python
from memorylayer import RelationshipType

await client.associate(
    source_id="mem_problem_123",
    target_id="mem_solution_456",
    relationship=RelationshipType.SOLVES,
    strength=0.9
)
```

### TypeScript

```typescript
import { RelationshipType } from "@scitrera/memorylayer-sdk";

await client.associate(
  "mem-problem-123",
  "mem-solution-456",
  RelationshipType.SOLVES,
  0.9
);
```

## Relationship Types

MemoryLayer provides typed relationship types organized into categories. Here are the most commonly used:

### Hierarchical — "What's the structure?"

| Relationship | Meaning |
|-------------|---------|
| `parent_of` | A is the parent of B |
| `child_of` | A is a child of B |
| `part_of` | A is part of B |
| `contains` | A contains B |

### Causal — "What led to what?"

| Relationship | Meaning |
|-------------|---------|
| `causes` | A directly causes B |
| `triggers` | A triggers B to happen |
| `leads_to` | A eventually leads to B |
| `prevents` | A prevents B from occurring |

### Solution — "What fixes what?"

| Relationship | Meaning |
|-------------|---------|
| `solves` | A is a solution for B |
| `addresses` | A partially addresses B |
| `alternative_to` | A is an alternative to B |
| `improves` | A is an improvement on B |

### Context — "Where does this apply?"

| Relationship | Meaning |
|-------------|---------|
| `occurs_in` | A happens in context B |
| `applies_to` | A is relevant to B |
| `works_with` | A works together with B |
| `requires` | A requires B |

### Learning — "How does knowledge evolve?"

| Relationship | Meaning |
|-------------|---------|
| `builds_on` | A builds on the knowledge in B |
| `contradicts` | A contradicts B |
| `confirms` | A confirms/validates B |
| `supersedes` | A replaces B (B is outdated) |

### Similarity — "What's related?"

| Relationship | Meaning |
|-------------|---------|
| `similar_to` | A is similar to B |
| `variant_of` | A is a variant/version of B |
| `related_to` | A is generally related to B |

### Workflow — "What's the sequence?"

| Relationship | Meaning |
|-------------|---------|
| `follows` | A comes after B |
| `depends_on` | A depends on B |
| `enables` | A enables B to happen |
| `blocks` | A blocks/prevents B |

### Quality — "What's the best approach?"

| Relationship | Meaning |
|-------------|---------|
| `effective_for` | A is effective for B |
| `preferred_over` | A is preferred over B |
| `deprecated_by` | A is deprecated in favor of B |

Additional categories include **Temporal** (`precedes`, `concurrent_with`), **Refinement** (`refines`, `abstracts`, `specializes`), and **Reference** (`references`, `referenced_by`). See the [Relationship Types Reference](/reference/relationships/) for the complete list.

## Graph Traversal

### Getting Associations

```python
# Get all associations for a memory
associations = await client.get_associations("mem_123", direction="both")

for assoc in associations:
    print(f"{assoc.relationship}: {assoc.target_id} (strength: {assoc.strength})")
```

### Multi-Hop Traversal

Follow chains of relationships across multiple memories:

```python
# Graph traversal is available via the TypeScript SDK's traverseGraph() method
# or the REST API: POST /v1/memories/{memory_id}/traverse
# The Python SDK provides get_associations() for direct relationship queries:
associations = await client.get_associations("mem_problem_123", direction="both")
```

```typescript
const result = await client.traverseGraph("mem-123", {
  relationshipTypes: [RelationshipType.CAUSES, RelationshipType.LEADS_TO],
  maxDepth: 3,
  direction: "both",
  minStrength: 0.5,
});
```

### MCP Tool

```json
{
  "start_memory_id": "mem_abc123",
  "relationship_types": ["causes", "triggers", "leads_to"],
  "max_depth": 3,
  "direction": "both",
  "max_paths": 50
}
```

## Example: Debugging Chain

```
[Retry timeout error]
    ├──CAUSED_BY──> [Connection pool exhaustion]
    │                   ├──SOLVED_BY──> [Increased pool size to 50]
    │                   └──SUPERSEDED_BY──> [Implemented circuit breaker]
    └──OCCURS_IN──> [Payment service under load]
```

Building this graph:

```python
# Store the memories
error = await client.remember("Retry timeout errors in payment service", ...)
cause = await client.remember("Connection pool exhaustion under load", ...)
fix1 = await client.remember("Increased connection pool size to 50", ...)
fix2 = await client.remember("Implemented circuit breaker pattern", ...)

# Create the relationships
await client.associate(cause.id, error.id, RelationshipType.CAUSES, 0.9)
await client.associate(cause.id, fix1.id, RelationshipType.SOLVES, 0.7)
await client.associate(fix1.id, fix2.id, RelationshipType.SUPERSEDES, 0.9)
```

Now querying "What happened with retry errors?" traverses the graph and returns the full causal chain from problem to final solution.

## Auditing

Find contradictions and inconsistencies in your knowledge graph:

```python
# Auditing is available via the MCP tool `memory_audit`
# or the REST API: GET /v1/workspaces/{workspace_id}/contradictions
```

```json
// MCP tool
{
  "memory_id": "mem_abc123",
  "auto_resolve": false
}
```

The audit checks for:
- Contradicting memories (A contradicts B)
- Superseded information that's still being referenced
- Circular dependency chains
