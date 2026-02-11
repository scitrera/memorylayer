---
title: Claude Code Integration
description: Use MemoryLayer with Claude Code for persistent agent memory
sidebar:
  order: 1
  label: Claude Code
---

The MemoryLayer plugin for Claude Code provides persistent memory that survives context window compaction, session briefings, and automatic memory triggers.

## What It Does

- **PreCompact Memory Capture** — Automatically saves important information before context window compaction
- **Session Briefings** — Recalls relevant context at session start
- **Automatic Memory Triggers** — Suggests storing memories after git commits and significant events
- **Knowledge Graph** — Links related memories with typed relationships
- **Context Sandbox** — Server-side Python sandbox that persists through compaction

## Prerequisites

The plugin requires a running MemoryLayer server:

```bash
pip install memorylayer-server[local]
memorylayer serve
```

The server runs on `http://localhost:61001` by default. See [Configuration](/server/configuration/) for embedding and LLM provider setup.

## Installation

First, add the MemoryLayer marketplace, then install the plugin:

```bash
# Add the marketplace (one-time setup)
claude plugin marketplace add https://github.com/scitrera/memorylayer.git

# Install the plugin
claude plugin install memorylayer@memorylayer.ai
```

This installs the MCP server, hooks, and slash commands in one step.

### Verify Setup

1. Start Claude Code in your project
2. Run `/mcp` to check the server is connected
3. Ask Claude to remember something: "Remember that this project uses SQLite"

### Manual MCP Setup (Alternative)

If you prefer to configure just the MCP server without the plugin's hooks and slash commands:

```bash
claude mcp add --transport stdio --scope user memorylayer \
  -- npx @scitrera/memorylayer-mcp-server
```

Or add `.mcp.json` to your project root:

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

## What the Plugin Provides

### Hooks

The plugin includes hooks that fire automatically at key moments:

| Hook | Trigger | Behavior |
|------|---------|----------|
| `PreCompact` | Before context compaction | Stores important information, checkpoints sandbox state, and commits working memory before context is lost |
| `SessionStart` | Session begins | Loads context briefing, directives, and starts server-side session |
| `Stop` | Session ends | Commits working memory to long-term storage and ends the server session |
| `PreToolUse` | Before Task/Edit/Write | Injects recalled context relevant to the operation |
| `PostToolUse` | After Bash/Edit/Write/Task | Captures outcomes (commit summaries, file changes, agent results) |
| `UserPromptSubmit` | On user input | Recalls relevant memories for recall/review/implement prompts |

### Slash Commands

| Command | Description |
|---------|-------------|
| `/memorylayer-status` | Check connection and workspace info |
| `/memorylayer-setup` | Guided setup and troubleshooting |
| `/memorylayer-remember <content>` | Quick memory storage |
| `/memorylayer-recall <query>` | Quick memory search |

## Context Compaction Protection

Without MemoryLayer, when the context window fills up, Claude forgets everything that gets truncated. With the plugin installed, the `PreCompact` hook fires automatically before compaction to store important information — no manual configuration needed.

## Workspace Isolation

Each project automatically gets its own workspace based on your git repository or directory name. Memories are isolated per-project by default. To override:

```bash
export MEMORYLAYER_WORKSPACE_ID="my-custom-workspace"
```

## Teaching Claude to Use Memory

Add instructions to your project's `CLAUDE.md`:

```markdown
## Memory Protocol

You have MemoryLayer MCP tools. Use them proactively:

**Session Start**: Call `memory_briefing` for context, then `memory_recall` for the current task.

**Auto-Store** (without being asked):
- Bug fixes → problem + solution (procedural, importance 0.8)
- Architecture decisions → decision + rationale (semantic, importance 0.9)
- Patterns discovered → pattern + when to use (procedural, importance 0.7)

**Tags**: Always include component names and technologies.
```

## Available Tools

Once connected, Claude Code will have access to all MemoryLayer MCP tools. See the [MCP Server Integration](/integrations/mcp-server/) for the complete tool list.
