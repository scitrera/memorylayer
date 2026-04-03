# MemoryLayer OpenCode Plugin

Persistent memory for [OpenCode](https://github.com/sst/opencode) sessions. Gives your AI coding agent long-term memory that survives context compaction and persists across sessions.

## What It Does

This plugin integrates [MemoryLayer](https://memorylayer.ai) into OpenCode via two mechanisms:

1. **MCP Tools** (21 tools) — The MemoryLayer MCP server gives the LLM direct access to remember, recall, reflect, associate, and manage memories through the standard Model Context Protocol.

2. **Proactive Hooks** — The plugin automatically:
   - Injects workspace briefing and user directives at session start
   - Recalls relevant memories when you ask preference/convention questions
   - Captures tool observations (file edits, searches, commands) as working memory
   - Commits working memory before context compaction
   - Cleans up sessions on exit

## Quick Start

### 1. Install the MemoryLayer server

```bash
pip install memorylayer-server
memorylayer serve
```

Or with Docker:

```bash
docker run -d -p 61001:61001 -v memorylayer-data:/data scitrera/memorylayer-server
```

### 2. Configure OpenCode

Add to your `opencode.json`:

```json
{
  "mcp": {
    "memorylayer": {
      "type": "local",
      "command": ["npx", "@scitrera/memorylayer-mcp-server"],
      "environment": {
        "MEMORYLAYER_URL": "{env:MEMORYLAYER_URL}",
        "MEMORYLAYER_API_KEY": "{env:MEMORYLAYER_API_KEY}"
      },
      "enabled": true
    }
  },
  "plugin": ["@scitrera/memorylayer-opencode-plugin"]
}
```

### 3. Set environment variables (optional)

```bash
export MEMORYLAYER_URL=http://localhost:61001  # default
export MEMORYLAYER_API_KEY=your-key            # optional, for authenticated servers
```

### 4. Verify

Start OpenCode and run `/memorylayer-status` to check the connection.

## Slash Commands

| Command | Description |
|---------|-------------|
| `/memorylayer-remember <content>` | Store a memory with auto-detected type and importance |
| `/memorylayer-recall <query>` | Search memories by semantic query |
| `/memorylayer-status` | Check connection status and workspace info |
| `/memorylayer-setup` | Guided setup and verification |

## How Hooks Work

### Session Start (`experimental.chat.system.transform`)
On first interaction, loads workspace briefing, user directives, and any existing sandbox state into the system prompt.

### User Messages (`chat.message`)
Detects 5 pattern categories in user messages and performs targeted recall:
- **Preference**: "which X should we use", "what's our convention"
- **Recall**: "remember", "what did we", "remind me"
- **Analysis**: "review", "analyze", "status"
- **Implementation**: "implement", "build", "fix"
- **Error**: "error", "bug", "broken"

### Tool Execution (`tool.execute.before` / `tool.execute.after`)
- **Before**: Injects recalled context for edit/write and task/delegation tools
- **After**: Silently captures observations (files, facts, concepts, intent) as working memory

### Context Compaction (`experimental.session.compacting`)
Commits working memory to long-term storage and checkpoints server-side sandbox state before the context window is trimmed.

### Shell Environment (`shell.env`)
Propagates `MEMORYLAYER_URL` and `MEMORYLAYER_API_KEY` to shell commands.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORYLAYER_URL` | `http://localhost:61001` | MemoryLayer server URL |
| `MEMORYLAYER_API_KEY` | — | API key (optional for local servers) |
| `MEMORYLAYER_WORKSPACE_ID` | auto-detected | Workspace identifier (auto-detected from git repo name) |

### Workspace Auto-Detection

The workspace ID is automatically detected from:
1. Git remote origin repository name
2. Git root directory name
3. Current working directory name

Override with the `MEMORYLAYER_WORKSPACE_ID` environment variable.

## Memory Types

| Type | Description | Importance |
|------|-------------|------------|
| `semantic` | Facts, concepts, knowledge | 0.5-0.9 |
| `procedural` | How-to, solutions, patterns | 0.5-0.8 |
| `episodic` | Events, what happened | 0.5-0.7 |
| `working` | Current task context (auto-expires) | 0.3-0.6 |

### Common Subtypes

`directive`, `decision`, `fix`, `solution`, `code_pattern`, `error`, `workflow`, `preference`, `problem`

## Architecture

```
OpenCode
├── MCP Server (@scitrera/memorylayer-mcp-server)
│   └── 21 memory tools (remember, recall, reflect, etc.)
│
├── Plugin Hooks (@scitrera/memorylayer-opencode-plugin)
│   ├── system.transform  → session briefing injection
│   ├── chat.message      → pattern-based recall
│   ├── tool.before       → pre-tool context injection
│   ├── tool.after        → observation capture
│   ├── session.compacting → working memory commit
│   └── shell.env         → env var propagation
│
└── MemoryLayer Server (memorylayer-server)
    ├── Memory storage (SQLite + vector search)
    ├── Knowledge graph (associations)
    ├── Working memory (sessions)
    └── Context sandbox (Python execution)
```

## Troubleshooting

### Server not reachable
```bash
# Check if server is running
curl http://localhost:61001/health

# Start the server
memorylayer serve

# Or with Docker
docker run -d -p 61001:61001 -v memorylayer-data:/data scitrera/memorylayer-server
```

### MCP tools not available
Verify `opencode.json` has the `mcp.memorylayer` configuration. Restart OpenCode after configuration changes.

### Plugin not loading
Ensure the plugin is listed in `opencode.json`:
```json
{
  "plugin": ["@scitrera/memorylayer-opencode-plugin"]
}
```

## License

Apache-2.0
