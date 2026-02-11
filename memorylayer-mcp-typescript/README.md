# @scitrera/memorylayer-mcp-server

TypeScript MCP (Model Context Protocol) server for [MemoryLayer.ai](https://memorylayer.ai).

Provides 16 memory tools for LLM agents to store, recall, synthesize, and manage information across sessions.

## Installation

```bash
npm install @scitrera/memorylayer-mcp-server
```

## Quick Start

### As a Standalone MCP Server

```bash
# Set environment variables
export MEMORYLAYER_URL=http://localhost:61001
export MEMORYLAYER_WORKSPACE_ID=my-workspace

# Run the server
npx memorylayer-mcp
```

### Claude Code Configuration (Recommended)

> **Detailed setup guide:** See [CLAUDE_CODE_SETUP.md](../docs/CLAUDE_CODE_SETUP.md) for step-by-step instructions.

Claude Code runs MCP servers from the project directory, so our server auto-detects the workspace from your git repo or folder name. Add `.mcp.json` to your project root:

```json
{
  "mcpServers": {
    "memorylayer": {
      "command": "npx",
      "args": ["@scitrera/memorylayer-mcp-server"],
      "env": {
        "MEMORYLAYER_URL": "http://localhost:61001"
      }
    }
  }
}
```

**Auto-workspace detection**: The server uses your git repo name (or directory name) as the workspace ID. Each project gets isolated memory storage automatically.

**Override options:**
```json
{
  "mcpServers": {
    "memorylayer": {
      "command": "npx",
      "args": ["@scitrera/memorylayer-mcp-server"],
      "env": {
        "MEMORYLAYER_URL": "http://localhost:61001",
        "MEMORYLAYER_WORKSPACE_ID": "${WORKSPACE_ID:-my-project}"
      }
    }
  }
}
```

Or via CLI:
```bash
claude mcp add --transport stdio memorylayer \
  --env MEMORYLAYER_URL=http://localhost:61001 \
  -- npx @scitrera/memorylayer-mcp-server
```

### Claude Desktop Configuration

Add to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "memorylayer": {
      "command": "npx",
      "args": ["@scitrera/memorylayer-mcp-server"],
      "env": {
        "MEMORYLAYER_URL": "http://localhost:61001",
        "MEMORYLAYER_WORKSPACE_ID": "my-project"
      }
    }
  }
}
```

**Note**: Claude Desktop doesn't change directories per-project, so you should set `MEMORYLAYER_WORKSPACE_ID` explicitly for each project entry.

### Programmatic Usage

```typescript
import { MemoryLayerClient, createServer } from "@scitrera/memorylayer-mcp-server";

// Create client (wraps the @scitrera/memorylayer-sdk)
const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace",
  apiKey: "optional-api-key"
});

// Create MCP server
const server = await createServer(client);

// Run server on stdio transport
await server.run();
```

## Available Tools

### Core Memory Tools (5)

#### 1. `memory_remember`
Store a new memory for later recall.

```typescript
{
  content: "User prefers TypeScript for new projects",
  type: "semantic",        // episodic, semantic, procedural, working
  importance: 0.8,         // 0.0 - 1.0
  tags: ["preference", "typescript"],
  subtype: "Preference"    // Optional domain classification
}
```

#### 2. `memory_recall`
Search memories by semantic query.

```typescript
{
  query: "What are the user's coding preferences?",
  limit: 10,
  min_relevance: 0.5,
  types: ["semantic"],     // Optional filter
  tags: ["preference"]     // Optional filter (AND logic)
}
```

#### 3. `memory_reflect`
Synthesize insights across multiple memories.

```typescript
{
  query: "What patterns have we seen with database performance?",
  max_tokens: 500,
  include_sources: true,
  depth: 2                // Association traversal depth
}
```

#### 4. `memory_forget`
Delete or decay outdated information.

```typescript
{
  memory_id: "mem_abc123",
  reason: "Outdated information",
  hard: false             // true = permanent delete
}
```

#### 5. `memory_associate`
Link memories with typed relationships.

```typescript
{
  source_id: "mem_problem",
  target_id: "mem_solution",
  relationship: "solves",  // 60+ relationship types available
  strength: 0.9           // 0.0 - 1.0
}
```

### Extended Memory Tools (4)

#### 6. `memory_briefing`
Get a session briefing with recent context.

```typescript
{
  lookback_hours: 24,
  include_contradictions: true
}
```

#### 7. `memory_statistics`
Get workspace analytics and memory usage.

```typescript
{
  include_breakdown: true  // Include breakdown by type/subtype
}
```

#### 8. `memory_graph_query`
Multi-hop graph traversal for causal chains.

```typescript
{
  start_memory_id: "mem_abc123",
  relationship_types: ["causes", "triggers"],
  max_depth: 3,
  direction: "both",      // outgoing, incoming, both
  max_paths: 50
}
```

#### 9. `memory_audit`
Find contradictions and inconsistencies.

```typescript
{
  memory_id: "mem_abc123",  // Optional - omit to audit entire workspace
  auto_resolve: false       // Auto-prefer newer contradicting memories
}
```

### Session Management Tools (3)

These tools enable working memory that persists across tool calls within a session.

#### 10. `memory_session_start`
Start a new session for working memory tracking.

```typescript
{
  metadata: { task: "debugging" }  // Optional metadata
}
```

#### 11. `memory_session_end`
End the current session and optionally commit working memory.

```typescript
{
  commit: true,               // Commit to long-term storage
  importance_threshold: 0.5   // Min importance for extracted memories
}
```

#### 12. `memory_session_status`
Get current session status including working memory summary.

```typescript
{}  // No parameters required
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORYLAYER_URL` | Base URL for MemoryLayer API | `http://localhost:61001` |
| `MEMORYLAYER_API_KEY` | API key for authentication | (none) |
| `MEMORYLAYER_WORKSPACE_ID` | Workspace ID (overrides auto-detection) | (auto-detected) |
| `MEMORYLAYER_AUTO_WORKSPACE` | Set to `false` to disable auto-detection | `true` |

## Memory Types

- **Episodic**: Specific events/interactions
- **Semantic**: Facts, concepts, relationships
- **Procedural**: How-to knowledge
- **Working**: Current task context (session-scoped)

## Relationship Types (60+)

Organized into 11 categories:

**Hierarchical**: parent_of, child_of, part_of, contains, instance_of, subtype_of
**Causal**: causes, triggers, leads_to, prevents
**Temporal**: precedes, concurrent_with, follows_temporally
**Similarity**: similar_to, variant_of, related_to
**Learning**: builds_on, contradicts, confirms, supersedes
**Refinement**: refines, abstracts, specializes, generalizes
**Reference**: references, referenced_by
**Solution**: solves, addresses, alternative_to, improves
**Context**: occurs_in, applies_to, works_with, requires
**Workflow**: follows, depends_on, enables, blocks
**Quality**: effective_for, preferred_over, deprecated_by

## Architecture

The MCP server wraps the `@scitrera/memorylayer-sdk` TypeScript SDK, providing an MCP-compatible interface for LLM agents.

```
memorylayer-mcp-typescript/
├── src/
│   ├── types.ts       # TypeScript types for MCP tools
│   ├── tools.ts       # MCP tool definitions (12 tools)
│   ├── client.ts      # Wrapper around @scitrera/memorylayer-sdk
│   ├── session.ts     # Local session state management
│   ├── handlers.ts    # Tool handler implementations
│   ├── server.ts      # MCP server using @modelcontextprotocol/sdk
│   └── index.ts       # Main exports
├── bin/
│   └── memorylayer-mcp.ts  # CLI entry point
├── package.json
├── tsconfig.json
└── README.md
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run dev

# Run locally
npm start
```

## Using the SDK Client Directly

The MCP server's client is a thin wrapper around the TypeScript SDK. For direct SDK usage without MCP, install `@scitrera/memorylayer-sdk`:

```bash
npm install @scitrera/memorylayer-sdk
```

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace"
});

const memory = await client.remember("Important fact", {
  type: "semantic",
  importance: 0.8
});
```

## License

Apache 2.0 License - see LICENSE file for details.

## Links

- [MemoryLayer.ai](https://memorylayer.ai)
- [Documentation](https://docs.memorylayer.ai)
- [TypeScript SDK](https://www.npmjs.com/package/@scitrera/memorylayer-sdk)
- [Model Context Protocol](https://modelcontextprotocol.io)
