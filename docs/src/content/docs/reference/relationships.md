---
title: Relationship Types Reference
description: Complete reference for relationship types in the knowledge graph
sidebar:
  order: 3
  label: Relationship Types
---

MemoryLayer provides 60+ typed relationships organized into 11 categories for connecting memories in a knowledge graph. The categories below show the most commonly used relationships — use `get_workspace_schema()` to list all available types.

## Hierarchical Relationships

Define parent-child and containment structures.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `parent_of` | A → B | A is parent of B | "Module is parent of function" |
| `child_of` | A → B | A is child of B | "Function is child of module" |
| `part_of` | A → B | A is part of B | "Auth middleware is part of API gateway" |
| `contains` | A → B | A contains B | "Config contains database settings" |
| `instance_of` | A → B | A is instance of B | "This error is instance of TimeoutError" |
| `subtype_of` | A → B | A is a subtype of B | "JWT auth is subtype of token auth" |

## Causal Relationships

Track cause-and-effect chains.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `causes` | A → B | A directly causes B | "Memory leak causes OOM crash" |
| `triggers` | A → B | A triggers B to happen | "Config change triggers restart" |
| `leads_to` | A → B | A eventually leads to B | "Tech debt leads to refactoring" |
| `prevents` | A → B | A prevents B from occurring | "Rate limiting prevents DDoS" |

## Solution Relationships

Connect problems to their solutions.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `solves` | A → B | A is a solution for B | "Circuit breaker solves cascade failure" |
| `addresses` | A → B | A partially addresses B | "Retry logic addresses transient errors" |
| `alternative_to` | A ↔ B | A is an alternative to B | "Redis is alternative to Memcached" |
| `improves` | A → B | A is an improvement on B | "Connection pooling improves raw connections" |

## Context Relationships

Describe where and when things apply.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `occurs_in` | A → B | A happens in context B | "Timeout occurs in payment service" |
| `applies_to` | A → B | A is relevant to B | "CORS config applies to API gateway" |
| `works_with` | A ↔ B | A works together with B | "FastAPI works with SQLAlchemy" |
| `requires` | A → B | A requires B | "Deployment requires Docker" |

## Learning Relationships

Track how knowledge evolves over time.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `builds_on` | A → B | A builds on knowledge in B | "V2 API builds on V1 patterns" |
| `contradicts` | A ↔ B | A contradicts B | "New benchmark contradicts old results" |
| `confirms` | A → B | A confirms/validates B | "Load test confirms performance fix" |
| `supersedes` | A → B | A replaces B (B is outdated) | "JWT auth supersedes session cookies" |

## Similarity Relationships

Connect related content.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `similar_to` | A ↔ B | A is similar to B | "Auth bug is similar to previous auth issue" |
| `variant_of` | A → B | A is a variant/version of B | "Retry with backoff is variant of simple retry" |
| `related_to` | A ↔ B | A is generally related to B | "Caching is related to performance" |

## Workflow Relationships

Define sequences and dependencies.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `follows` | A → B | A comes after B | "Deploy follows testing" |
| `depends_on` | A → B | A depends on B | "API depends on database migration" |
| `enables` | A → B | A enables B to happen | "Auth service enables user management" |
| `blocks` | A → B | A blocks/prevents B | "Broken CI blocks deployment" |

## Temporal Relationships

Track time-based ordering.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `precedes` | A → B | A happens before B | "Design precedes implementation" |
| `concurrent_with` | A ↔ B | A happens at the same time as B | "Frontend refactor concurrent with API migration" |
| `follows_temporally` | A → B | A happens after B | "Deployment follows testing" |

## Refinement Relationships

Track how knowledge is refined.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `refines` | A → B | A refines/improves on B | "V2 config refines V1 approach" |
| `abstracts` | A → B | A is an abstraction of B | "Design doc abstracts implementation" |
| `specializes` | A → B | A specializes B for a use case | "Redis cache specializes generic cache" |
| `generalizes` | A → B | A generalizes B | "Error handling pattern generalizes retry logic" |

## Reference Relationships

Track citations and references.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `references` | A → B | A references B | "Bug report references stack trace" |
| `referenced_by` | A → B | A is referenced by B | "API docs referenced by integration guide" |

## Quality Relationships

Express preferences and deprecations.

| Relationship | Direction | Description | Example |
|-------------|-----------|-------------|---------|
| `effective_for` | A → B | A is effective for B | "Caching is effective for read-heavy loads" |
| `preferred_over` | A → B | A is preferred over B | "TypeScript preferred over JavaScript" |
| `deprecated_by` | A → B | A is deprecated in favor of B | "REST deprecated by GraphQL (for this use case)" |

:::note
The tables above show representative relationships from each category. The full ontology contains 60+ relationship types. Use `get_workspace_schema()` to list all available types for your workspace.
:::

## Usage

### Python

```python
from memorylayer import RelationshipType

await client.associate(
    source_id="mem_123",
    target_id="mem_456",
    relationship=RelationshipType.SOLVES,
    strength=0.9
)
```

### TypeScript

```typescript
import { RelationshipType } from "@scitrera/memorylayer-sdk";

await client.associate(
  "mem-123",
  "mem-456",
  RelationshipType.SOLVES,
  0.9
);
```

### MCP

```json
{
  "source_id": "mem_123",
  "target_id": "mem_456",
  "relationship": "solves",
  "strength": 0.9
}
```

## Strength

Association strength is a float between 0.0 and 1.0:

| Range | Meaning |
|-------|---------|
| 0.8–1.0 | Strong, verified relationship |
| 0.5–0.7 | Moderate confidence |
| 0.1–0.4 | Weak or speculative |
