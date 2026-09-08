# MemoryLayer Codex Plugin

Codex plugin for MemoryLayer persistent project memory. It provides:

- MemoryLayer MCP server configuration (`memorylayer-mcp`)
- A Codex skill that tells Codex when to recall/store durable context
- Shared environment variables with the Claude Code plugin

## Prerequisites

Run a MemoryLayer server, usually at `http://localhost:61001`, and make sure `memorylayer-mcp` is on `PATH`.

```bash
export MEMORYLAYER_URL="http://localhost:61001"
export MEMORYLAYER_API_KEY=""
export MEMORYLAYER_WORKSPACE_ID=""
```

`MEMORYLAYER_API_KEY` and `MEMORYLAYER_WORKSPACE_ID` are optional for local/open deployments.

## Layout

- `.codex-plugin/plugin.json` — Codex plugin manifest
- `.mcp.json` — MemoryLayer MCP server config
- `skills/memorylayer/SKILL.md` — Codex guidance for using MemoryLayer memory

## Relationship To Claude Code Plugin

`../memorylayer-cc-plugin` has Claude Code hooks and slash commands. This plugin intentionally avoids Claude-specific hooks and uses Codex-native skills plus MCP configuration instead.
