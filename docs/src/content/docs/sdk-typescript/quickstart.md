---
title: TypeScript Quick Start
description: Step-by-step guide to using the MemoryLayer TypeScript SDK
sidebar:
  order: 2
  label: Quick Start
---

This guide walks through the core operations of the MemoryLayer TypeScript SDK.

## Prerequisites

- Node.js 18+
- MemoryLayer server running (`memorylayer serve`)

## Install the SDK

```bash
npm install @scitrera/memorylayer-sdk
```

## Connect to the Server

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace",
});
```

## Remember (Store Memories)

```typescript
import { MemoryType, MemorySubtype } from "@scitrera/memorylayer-sdk";

// Basic storage
const memory = await client.remember("Fixed authentication bug in login flow");

// With options
const memory = await client.remember(
  "Implemented retry logic with exponential backoff",
  {
    type: MemoryType.PROCEDURAL,
    subtype: MemorySubtype.CODE_PATTERN,
    importance: 0.9,
    tags: ["retry", "error-handling"],
    metadata: {
      file: "src/api/client.ts",
      author: "alice@example.com",
    },
  }
);
```

## Recall (Search Memories)

```typescript
import { RecallMode, DetailLevel } from "@scitrera/memorylayer-sdk";

// Simple recall
const result = await client.recall("How do we handle retries?");

// Advanced recall with filters
const result = await client.recall("authentication patterns", {
  types: [MemoryType.PROCEDURAL, MemoryType.SEMANTIC],
  tags: ["auth"],
  limit: 10,
  minRelevance: 0.6,
  includeAssociations: true,
  mode: RecallMode.RAG,
  detailLevel: DetailLevel.FULL,
});
```

## Reflect (Synthesize Insights)

```typescript
const reflection = await client.reflect(
  "What patterns have we learned about error handling?",
  {
    detailLevel: DetailLevel.FULL,
    depth: 3,
    types: [MemoryType.PROCEDURAL],
    includeSources: true,
  }
);

console.log(reflection.reflection);
console.log(`Based on ${reflection.source_memories.length} memories`);
```

## Memory Management

```typescript
// Get a specific memory
const memory = await client.getMemory("mem-123");

// Update a memory
const updated = await client.updateMemory("mem-123", {
  importance: 0.95,
  tags: ["critical", "security"],
});

// Soft delete (archive)
await client.forget("mem-123");

// Hard delete (permanent)
await client.forget("mem-123", true);

// Apply decay
const decayed = await client.decay("mem-123", 0.1);

// Trace provenance
const trace = await client.traceMemory("mem-123");
```

## Associations (Knowledge Graph)

```typescript
import { RELATIONSHIP_TYPES } from "@scitrera/memorylayer-sdk";

// Create a relationship
const association = await client.associate(
  "mem-problem-123",
  "mem-solution-456",
  RELATIONSHIP_TYPES.SOLVES,
  0.9
);

// Get all associations for a memory
const associations = await client.getAssociations("mem-123", "both");

// Traverse the knowledge graph
const result = await client.traverseGraph("mem-123", {
  relationshipTypes: [RELATIONSHIP_TYPES.CAUSES, RELATIONSHIP_TYPES.LEADS_TO],
  maxDepth: 3,
  direction: "both",
  minStrength: 0.5,
});
```

## Session Management

```typescript
// Create a session
const { session, briefing } = await client.createSession({
  workspaceId: "my-workspace",
  ttlSeconds: 3600,
  briefing: true,
});

// Session ID is automatically included in subsequent requests
client.setSession(session.id);

// Store working memory
await client.setWorkingMemory(session.id, "current_task", {
  description: "Debugging auth",
  file: "auth.py",
});

// Retrieve working memory
const memory = await client.getWorkingMemory(session.id, "current_task");

// Extend session TTL
await client.touchSession(session.id);

// Commit to long-term storage
const commitResult = await client.commitSession(session.id, {
  minImportance: 0.5,
  deduplicate: true,
});
console.log(`Created ${commitResult.memories_created} memories`);

// Delete session
await client.deleteSession(session.id);

// Clear session from client
client.clearSession();
```

## Workspace Management

```typescript
// Create a workspace
const workspace = await client.createWorkspace("My Project", {
  embedding_model: "text-embedding-3-small",
  default_importance: 0.5,
});

// Get workspace schema
const schema = await client.getWorkspaceSchema("ws-123");
console.log(schema.relationship_types);
console.log(schema.memory_subtypes);

```

## Batch Operations

```typescript
const result = await client.batchMemories([
  { action: "create", memory: { content: "Memory 1", importance: 0.7 } },
  { action: "create", memory: { content: "Memory 2", importance: 0.8 } },
  { action: "delete", memory_id: "mem-old", hard: false },
]);

console.log(`Successful: ${result.successful}, Failed: ${result.failed}`);
```

## Error Handling

```typescript
import {
  MemoryLayerError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ValidationError,
  RateLimitError,
} from "@scitrera/memorylayer-sdk";

try {
  await client.remember("test");
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error("Invalid API key");
  } else if (error instanceof NotFoundError) {
    console.error("Resource not found");
  } else if (error instanceof ValidationError) {
    console.error("Validation failed:", error.details);
  } else if (error instanceof RateLimitError) {
    console.error(`Rate limited. Retry after ${error.retryAfter}s`);
  } else if (error instanceof MemoryLayerError) {
    console.error(`Error ${error.statusCode}: ${error.message}`);
  }
}
```
