---
title: TypeScript API Reference
description: Complete API reference for the MemoryLayer TypeScript SDK
sidebar:
  order: 3
  label: API Reference
---

## MemoryLayerClient

### Constructor

```typescript
new MemoryLayerClient(options: ClientConfig)
```

```typescript
interface ClientConfig {
  baseUrl?: string;      // Default: "http://localhost:61001"
  apiKey?: string;       // Optional API key
  workspaceId?: string;  // Default workspace
  sessionId?: string;    // Auto-include in requests
  timeout?: number;      // Request timeout in ms (default: 30000)
}
```

### Memory Operations

#### remember()

```typescript
async remember(
  content: string,
  options?: RememberOptions
): Promise<Memory>
```

```typescript
interface RememberOptions {
  type?: MemoryType;
  subtype?: MemorySubtype;
  importance?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}
```

#### recall()

```typescript
async recall(
  query: string,
  options?: RecallOptions
): Promise<RecallResult>
```

```typescript
interface RecallOptions {
  types?: MemoryType[];
  tags?: string[];
  limit?: number;
  minRelevance?: number;
  includeAssociations?: boolean;
  createdAfter?: Date;
  createdBefore?: Date;
  mode?: RecallMode;
  detailLevel?: DetailLevel;
}
```

#### reflect()

```typescript
async reflect(
  query: string,
  options?: ReflectOptions
): Promise<ReflectionResult>
```

```typescript
interface ReflectOptions {
  workspaceId?: string;
  detailLevel?: DetailLevel;
  includeSources?: boolean;
  depth?: number;
  types?: MemoryType[];
  subtypes?: MemorySubtype[];
  tags?: string[];
  contextId?: string;
}
```

#### getMemory()

```typescript
async getMemory(memoryId: string): Promise<Memory>
```

#### updateMemory()

```typescript
async updateMemory(
  memoryId: string,
  updates: MemoryUpdate
): Promise<Memory>
```

#### forget()

```typescript
async forget(memoryId: string, hard?: boolean): Promise<void>
```

#### decay()

```typescript
async decay(memoryId: string, decayRate: number): Promise<Memory>
```

#### traceMemory()

```typescript
async traceMemory(memoryId: string): Promise<MemoryTrace>
```

#### batchMemories()

```typescript
async batchMemories(
  operations: BatchOperation[]
): Promise<BatchResult>
```

### Association Operations

#### associate()

```typescript
async associate(
  sourceId: string,
  targetId: string,
  relationship: RelationshipType,
  strength?: number
): Promise<Association>
```

#### createAssociation()

```typescript
async createAssociation(
  options: AssociationCreateOptions
): Promise<Association>
```

#### getAssociations()

```typescript
async getAssociations(
  memoryId: string,
  direction?: "outgoing" | "incoming" | "both"
): Promise<Association[]>
```

#### traverseGraph()

```typescript
async traverseGraph(
  memoryId: string,
  options?: TraverseOptions
): Promise<GraphTraversalResult>
```

```typescript
interface TraverseOptions {
  relationshipTypes?: RelationshipType[];
  maxDepth?: number;
  direction?: "outgoing" | "incoming" | "both";
  minStrength?: number;
}
```

### Session Operations

#### createSession()

```typescript
async createSession(
  options?: CreateSessionOptions
): Promise<{ session: Session; briefing?: SessionBriefing }>
```

#### getSession()

```typescript
async getSession(sessionId: string): Promise<Session>
```

#### setSession() / getSessionId() / clearSession()

```typescript
setSession(sessionId: string): void
getSessionId(): string | undefined
clearSession(): void
```

#### setWorkingMemory()

```typescript
async setWorkingMemory(
  sessionId: string,
  key: string,
  value: unknown
): Promise<void>
```

#### getWorkingMemory()

```typescript
async getWorkingMemory(
  sessionId: string,
  key?: string
): Promise<Record<string, unknown>>
```

#### commitSession()

```typescript
async commitSession(
  sessionId: string,
  options?: CommitOptions
): Promise<CommitResult>
```

#### touchSession()

```typescript
async touchSession(sessionId: string): Promise<Session>
```

#### deleteSession()

```typescript
async deleteSession(sessionId: string): Promise<void>
```

### Workspace Operations

#### createWorkspace()

```typescript
async createWorkspace(
  name: string,
  settings?: Record<string, unknown>
): Promise<Workspace>
```

#### getWorkspace()

```typescript
async getWorkspace(workspaceId: string): Promise<Workspace>
```

#### updateWorkspace()

```typescript
async updateWorkspace(
  workspaceId: string,
  updates: WorkspaceUpdate
): Promise<Workspace>
```

#### getWorkspaceSchema()

```typescript
async getWorkspaceSchema(
  workspaceId: string
): Promise<WorkspaceSchema>
```

### Briefing

#### getBriefing()

```typescript
async getBriefing(
  lookbackHours?: number,
  includeContradictions?: boolean
): Promise<SessionBriefing>
```

### Context Environment Operations

#### contextExec()

```typescript
async contextExec(code: string, options?: ContextExecOptions): Promise<ContextExecResult>
```

#### contextInspect()

```typescript
async contextInspect(options?: ContextInspectOptions): Promise<ContextInspectResult>
```

#### contextLoad()

```typescript
async contextLoad(varName: string, query: string, options?: ContextLoadOptions): Promise<ContextLoadResult>
```

#### contextInject()

```typescript
async contextInject(key: string, value: unknown, options?: ContextInjectOptions): Promise<ContextInjectResult>
```

#### contextQuery()

```typescript
async contextQuery(prompt: string, variables: string[], options?: ContextQueryOptions): Promise<ContextQueryResult>
```

#### contextRlm()

```typescript
async contextRlm(goal: string, options?: ContextRlmOptions): Promise<ContextRlmResult>
```

#### contextStatus()

```typescript
async contextStatus(): Promise<ContextStatusResult>
```

#### contextCheckpoint()

```typescript
async contextCheckpoint(): Promise<void>
```

#### contextCleanup()

```typescript
async contextCleanup(): Promise<void>
```

## Enums

### MemoryType

```typescript
enum MemoryType {
  EPISODIC = "episodic",
  SEMANTIC = "semantic",
  PROCEDURAL = "procedural",
  WORKING = "working",
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
  PROFILE = "profile",
  ENTITY = "entity",
  EVENT = "event",
  DIRECTIVE = "directive",
}
```

### RecallMode

```typescript
enum RecallMode {
  RAG = "rag",
  LLM = "llm",
  HYBRID = "hybrid",
}
```

### DetailLevel

```typescript
enum DetailLevel {
  ABSTRACT = "abstract",
  OVERVIEW = "overview",
  FULL = "full",
}
```

### RelationshipType

```typescript
type RelationshipType = string;

const RELATIONSHIP_TYPES = {
  // Causal
  CAUSES: "causes",
  TRIGGERS: "triggers",
  LEADS_TO: "leads_to",
  PREVENTS: "prevents",
  // Solution
  SOLVES: "solves",
  ADDRESSES: "addresses",
  ALTERNATIVE_TO: "alternative_to",
  IMPROVES: "improves",
  // Context
  OCCURS_IN: "occurs_in",
  APPLIES_TO: "applies_to",
  WORKS_WITH: "works_with",
  REQUIRES: "requires",
  // Learning
  BUILDS_ON: "builds_on",
  CONTRADICTS: "contradicts",
  CONFIRMS: "confirms",
  SUPERSEDES: "supersedes",
  // Similarity
  SIMILAR_TO: "similar_to",
  VARIANT_OF: "variant_of",
  RELATED_TO: "related_to",
  // Workflow
  FOLLOWS: "follows",
  DEPENDS_ON: "depends_on",
  ENABLES: "enables",
  BLOCKS: "blocks",
  // Quality
  EFFECTIVE_FOR: "effective_for",
  PREFERRED_OVER: "preferred_over",
  DEPRECATED_BY: "deprecated_by",
  // Hierarchical
  PART_OF: "part_of",
  CONTAINS: "contains",
  INSTANCE_OF: "instance_of",
  SUBTYPE_OF: "subtype_of",
  // Temporal
  PRECEDES: "precedes",
  CONCURRENT_WITH: "concurrent_with",
} as const;
```

60+ relationship types in 11 categories:

- **Hierarchical**: `parent_of`, `child_of`, `part_of`, `contains`, `instance_of`, `subtype_of`
- **Causal**: `causes`, `triggers`, `leads_to`, `prevents`, and more
- **Temporal**: `precedes`, `concurrent_with`, `follows_temporally`
- **Similarity**: `similar_to`, `variant_of`, `related_to`, `analogous_to`
- **Learning**: `builds_on`, `contradicts`, `confirms`, `supersedes`, and more
- **Refinement**: `refines`, `abstracts`, `specializes`, `generalizes`
- **Reference**: `references`, `referenced_by`
- **Solution**: `solves`, `addresses`, `alternative_to`, `improves`, and more
- **Context**: `occurs_in`, `applies_to`, `works_with`, `requires`, and more
- **Workflow**: `follows`, `depends_on`, `enables`, `blocks`, and more
- **Quality**: `effective_for`, `preferred_over`, `deprecated_by`, and more

Use `getWorkspaceSchema()` to list all available relationship types.

## Exceptions

```typescript
MemoryLayerError          // Base error class
├── AuthenticationError   // 401 - Invalid API key
├── AuthorizationError    // 403 - Access denied
├── NotFoundError         // 404 - Resource not found
├── ValidationError       // 400 - Invalid request (has .details)
└── RateLimitError        // 429 - Rate limited (has .retryAfter)
```

All errors have these properties:

| Property | Type | Description |
|----------|------|-------------|
| `message` | `string` | Error message |
| `statusCode` | `number` | HTTP status code |
