# MemoryLayer TypeScript SDK

TypeScript/JavaScript SDK for [MemoryLayer.ai](https://memorylayer.ai) - memory infrastructure for AI agents.

## Installation

```bash
npm install @scitrera/memorylayer-sdk
```

## Quick Start

```typescript
import { MemoryLayerClient, MemoryType } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",  // Optional, this is the default
  apiKey: "your-api-key",             // Optional for local development
  workspaceId: "my-workspace",
});

// Store a memory
const memory = await client.remember("User prefers dark mode", {
  type: MemoryType.SEMANTIC,
  importance: 0.8,
  tags: ["preference", "ui"],
});

// Recall memories
const result = await client.recall("What are the user's UI preferences?", {
  limit: 5,
  minRelevance: 0.7,
});

console.log(result.memories);
```

## Features

- **Full TypeScript Support** - Complete type definitions included
- **Memory Operations** - Remember, recall, reflect, forget, decay
- **Relationship Graph** - Link memories with 60+ relationship types organized into 11 categories
- **Session Management** - Working memory with TTL and commit
- **Context Environment** - Server-side Python sandbox for data operations
- **Batch Operations** - Bulk create, update, delete
- **Graph Traversal** - Multi-hop relationship queries
- **Error Handling** - Typed exception hierarchy

## Core Memory Operations

### Remember (Store)

```typescript
// Basic storage
const memory = await client.remember("Fixed authentication bug in login flow");

// With options
const memory = await client.remember("Implemented retry logic with exponential backoff", {
  type: MemoryType.PROCEDURAL,
  subtype: MemorySubtype.CODE_PATTERN,
  importance: 0.9,
  tags: ["retry", "error-handling"],
  metadata: {
    file: "src/api/client.ts",
    author: "alice@example.com",
  },
});
```

### Recall (Retrieve)

```typescript
// Simple recall
const result = await client.recall("How do we handle retries?");

// Advanced recall with filters
const result = await client.recall("authentication patterns", {
  types: [MemoryType.PROCEDURAL, MemoryType.SEMANTIC],
  tags: ["auth"],
  limit: 10,
  minRelevance: 0.6,
  includeAssociations: true,
  createdAfter: new Date("2024-01-01"),
  mode: RecallMode.RAG,        // RAG is the active mode (vector + graph)
  detailLevel: DetailLevel.FULL, // or ABSTRACT, OVERVIEW
});
```

### Reflect (Synthesize)

```typescript
const reflection = await client.reflect(
  "What patterns have we learned about error handling?",
  {
    detailLevel: "full",  // "abstract", "overview", or "full"
    depth: 3,
    types: [MemoryType.PROCEDURAL],
    includeSources: true,
  }
);

console.log(reflection.reflection);
console.log(`Based on ${reflection.source_memories.length} memories`);
```

### Memory Management

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

// Apply decay to reduce importance
const decayed = await client.decay("mem-123", 0.1);

// Trace memory provenance
const trace = await client.traceMemory("mem-123");
```

### Batch Operations

```typescript
const result = await client.batchMemories([
  { action: "create", memory: { content: "Memory 1", importance: 0.7 } },
  { action: "create", memory: { content: "Memory 2", importance: 0.8 } },
  { action: "delete", memory_id: "mem-old", hard: false },
]);

console.log(`Successful: ${result.successful}, Failed: ${result.failed}`);
```

## Associations

```typescript
import { RELATIONSHIP_TYPES } from "@scitrera/memorylayer-sdk";

// Create relationships between memories
const association = await client.associate(
  "mem-problem-123",
  "mem-solution-456",
  RELATIONSHIP_TYPES.SOLVES,
  0.9
);

// Or use the full options interface
const association = await client.createAssociation({
  sourceId: "mem-problem-123",
  targetId: "mem-solution-456",
  relationship: RELATIONSHIP_TYPES.SOLVES,
  strength: 0.9,
  metadata: { verified: true },
});

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

Sessions provide working memory with TTL that can be committed to long-term storage.

```typescript
// Create a session (auto-sets session ID for subsequent requests)
const { session, briefing } = await client.createSession({
  workspaceId: "my-workspace",
  ttlSeconds: 3600,
  briefing: true,  // Get briefing on session start
});

// Session ID is automatically included in subsequent requests
// Or manually manage it:
client.setSession(session.id);
console.log(client.getSessionId());
client.clearSession();

// Store working memory on server
await client.setWorkingMemory(session.id, "current_task", {
  description: "Debugging auth",
  file: "auth.py"
});

// Retrieve working memory
const memory = await client.getWorkingMemory(session.id, "current_task");

// Extend session TTL
const updated = await client.touchSession(session.id);

// Commit working memory to long-term storage
const commitResult = await client.commitSession(session.id, {
  minImportance: 0.5,
  deduplicate: true,
  maxMemories: 50,
});
console.log(`Created ${commitResult.memories_created} memories`);

// Delete session
await client.deleteSession(session.id);

// Get briefing of recent activity
const briefing = await client.getBriefing(24, true);
console.log(briefing.recent_activity_summary);
console.log(briefing.open_threads);
```

## Workspace Management

```typescript
// Create a workspace
const workspace = await client.createWorkspace("My Project", {
  embedding_model: "text-embedding-3-small",
  default_importance: 0.5,
});

// Get workspace details
const workspace = await client.getWorkspace("ws-123");

// Update workspace
const updated = await client.updateWorkspace("ws-123", {
  name: "New Name",
  settings: { key: "value" },
});

// Get workspace schema (relationship types, memory subtypes)
const schema = await client.getWorkspaceSchema("ws-123");
console.log(schema.relationship_types);
console.log(schema.memory_subtypes);
```

## Types & Enums

### MemoryType

```typescript
enum MemoryType {
  EPISODIC = "episodic",     // Events and experiences
  SEMANTIC = "semantic",     // Facts and knowledge
  PROCEDURAL = "procedural", // How-to knowledge
  WORKING = "working",       // Temporary context
}
```

### MemorySubtype

```typescript
enum MemorySubtype {
  SOLUTION = "solution",
  PROBLEM = "problem",
  CODE_PATTERN = "code_pattern",
  FIX = "fix",
  ERROR = "error",
  WORKFLOW = "workflow",
  PREFERENCE = "preference",
  DECISION = "decision",
  DIRECTIVE = "directive",
  PROFILE = "profile",
  ENTITY = "entity",
  EVENT = "event",
}
```

### RecallMode

```typescript
enum RecallMode {
  RAG = "rag",       // Active: Vector similarity + graph traversal
  LLM = "llm",       // Deprecated: LLM-powered semantic search
  HYBRID = "hybrid", // Deprecated: Combination of both
}
```

### RelationshipType

The SDK exports `RELATIONSHIP_TYPES` as a constants object (not an enum). RelationshipType is defined as `type string`. The server supports 60+ relationship types organized into 11 categories. Most commonly used:

- **Hierarchical**: `parent_of`, `child_of`, `part_of`, `contains`, `instance_of`, `subtype_of`
- **Causal**: `causes`, `triggers`, `leads_to`, `prevents`
- **Solution**: `solves`, `addresses`, `alternative_to`, `improves`
- **Context**: `occurs_in`, `applies_to`, `works_with`, `requires`
- **Learning**: `builds_on`, `contradicts`, `confirms`, `supersedes`
- **Similarity**: `similar_to`, `variant_of`, `related_to`
- **Workflow**: `follows`, `depends_on`, `enables`, `blocks`
- **Quality**: `effective_for`, `preferred_over`, `deprecated_by`
- **Temporal**: `precedes`, `concurrent_with`, `follows_temporally`
- **Refinement**: `refines`, `abstracts`, `specializes`, `generalizes`
- **Reference**: `references`, `referenced_by`

Import with:
```typescript
import { RELATIONSHIP_TYPES } from "@scitrera/memorylayer-sdk";
```

Use `getWorkspaceSchema()` to list all available types.

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
  } else if (error instanceof AuthorizationError) {
    console.error("Access denied");
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

## Context Environment

The TypeScript SDK provides access to the server-side Python sandbox for data operations:

```typescript
// Execute Python code
const result = await client.contextExec("import pandas as pd; df = pd.DataFrame({'a': [1,2,3]})");

// Inspect sandbox variables
const vars = await client.contextInspect({ variable: "df" });

// Load memories into sandbox
await client.contextLoad("memories_var", "error handling patterns", { limit: 50 });

// Inject data into sandbox
await client.contextInject("config", JSON.stringify({ key: "value" }), { parseJson: true });

// Query data using LLM
const answer = await client.contextQuery("What is the average value?", ["df"]);

// Run Reason-Learn-Memory loop
const synthesis = await client.contextRlm("Analyze error patterns", {
  memoryQuery: "errors and fixes",
  maxIterations: 10,
  detailLevel: "detailed"
});

// Check sandbox status
const status = await client.contextStatus();

// Checkpoint sandbox state
await client.contextCheckpoint();

// Clean up sandbox
await client.contextCleanup();
```

## Configuration

```typescript
const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001", // Default
  apiKey: process.env.MEMORYLAYER_API_KEY,
  workspaceId: process.env.MEMORYLAYER_WORKSPACE_ID,
  sessionId: "optional-session-id",  // Auto-include in requests
  timeout: 30000, // Request timeout in ms (default: 30000)
});
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test
```

## TypeScript Support

This SDK is written in TypeScript and provides full type definitions out of the box. No need for `@types/*` packages.

## License

Apache 2.0 License

## Links

- [Documentation](https://docs.memorylayer.ai)
- [GitHub](https://github.com/scitrera/memorylayer)
- [Homepage](https://memorylayer.ai)
