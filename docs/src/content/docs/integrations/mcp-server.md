---
title: MCP Server
description: Model Context Protocol server for Claude and other LLMs
sidebar:
  order: 2
  label: MCP Server
---

The MemoryLayer MCP server (`@scitrera/memorylayer-mcp-server`) provides 21 memory tools for LLM agents to store, recall, synthesize, and manage information across sessions via the [Model Context Protocol](https://modelcontextprotocol.io).

## Installation

```bash
npm install @scitrera/memorylayer-mcp-server
```

## Quick Start

```bash
# Set environment variables
export MEMORYLAYER_URL=http://localhost:61001
export MEMORYLAYER_WORKSPACE_ID=my-workspace

# Run the server
npx memorylayer-mcp
```

## Configuration for Claude Code

Add `.mcp.json` to your project root:

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

Or via CLI:

```bash
claude mcp add --transport stdio memorylayer \
  --env MEMORYLAYER_URL=http://localhost:61001 \
  -- npx @scitrera/memorylayer-mcp-server
```

## Configuration for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

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

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORYLAYER_URL` | Base URL for MemoryLayer API | `http://localhost:61001` |
| `MEMORYLAYER_API_KEY` | API key for authentication | — |
| `MEMORYLAYER_WORKSPACE_ID` | Workspace ID (overrides auto-detection) | auto-detected |
| `MEMORYLAYER_AUTO_WORKSPACE` | Set to `false` to disable auto-detection | `true` |
| `MEMORYLAYER_SESSION_MODE` | Set to `false` to disable session/working memory | `true` |
| `MEMORYLAYER_TOOL_PROFILE` | Tool profile: `cc` (default), `full`, or `minimal` | `cc` |
| `MEMORYLAYER_AUTO_START_SESSION` | Set to `false` to disable auto-starting session | `true` |

## Available Tools

### Core Memory Tools (9 tools)

| Tool | Description |
|------|-------------|
| `memory_remember` | Store a new memory for later recall |
| `memory_recall` | Search memories by semantic query |
| `memory_reflect` | Synthesize insights across multiple memories |
| `memory_forget` | Delete or decay outdated information |
| `memory_associate` | Link memories with typed relationships |
| `memory_briefing` | Get a session briefing with recent context |
| `memory_statistics` | Get workspace analytics and memory usage |
| `memory_graph_query` | Multi-hop graph traversal for causal chains |
| `memory_audit` | Find contradictions and inconsistencies |

### Session Management Tools (4 tools)

| Tool | Description |
|------|-------------|
| `memory_session_start` | Start a session for working memory tracking |
| `memory_session_end` | End session, optionally commit to long-term storage |
| `memory_session_commit` | Commit working memory without ending session |
| `memory_session_status` | Check current session state |

### Context Environment Tools (8 tools)

| Tool | Description |
|------|-------------|
| `memory_context_exec` | Execute Python code in server-side sandbox |
| `memory_context_inspect` | Inspect variables in server-side sandbox |
| `memory_context_load` | Load memories into sandbox variable via semantic search |
| `memory_context_inject` | Inject a value directly into sandbox as named variable |
| `memory_context_query` | Query server-side LLM using sandbox variables as context |
| `memory_context_rlm` | Run Recursive Language Model (RLM) loop on server |
| `memory_context_status` | Get status of server-side context environment |
| `memory_context_checkpoint` | Checkpoint sandbox state for persistence |

## Tool Examples

### memory_remember

```json
{
  "content": "User prefers TypeScript for new projects",
  "type": "semantic",
  "importance": 0.8,
  "tags": ["preference", "typescript"],
  "subtype": "preference"
}
```

### memory_recall

```json
{
  "query": "What are the user's coding preferences?",
  "limit": 10,
  "min_relevance": 0.5,
  "types": ["semantic"],
  "tags": ["preference"]
}
```

### memory_reflect

```json
{
  "query": "What patterns have we seen with database performance?",
  "detail_level": "overview",
  "include_sources": true,
  "depth": 2
}
```

### memory_associate

```json
{
  "source_id": "mem_problem",
  "target_id": "mem_solution",
  "relationship": "solves",
  "strength": 0.9
}
```

### memory_graph_query

```json
{
  "start_memory_id": "mem_abc123",
  "relationship_types": ["causes", "triggers"],
  "max_depth": 3,
  "direction": "both"
}
```

## Workspace Detection

The server determines your workspace in this order:

1. **`MEMORYLAYER_WORKSPACE_ID`** environment variable (explicit override)
2. **Git remote origin** — extracts repo name from remote URL
3. **Git root directory** — uses the root folder name
4. **Current directory** — falls back to the current working directory name

## Programmatic Usage

```typescript
import { MemoryLayerClient, createServer } from "@scitrera/memorylayer-mcp-server";

const client = new MemoryLayerClient({
  baseUrl: "http://localhost:61001",
  workspaceId: "my-workspace",
});

const server = await createServer(client);
await server.run();
```
