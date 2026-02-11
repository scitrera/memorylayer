# MemoryLayer Plugin for Claude Code

Persistent memory for Claude Code - never lose context to compaction again.

## What It Does

This plugin integrates [MemoryLayer](https://memorylayer.ai) with Claude Code to provide:

- **PreCompact Memory Capture** - Automatically saves important information before context window compaction
- **Session Briefings** - Recalls relevant context at session start
- **Automatic Memory Triggers** - Suggests storing memories after git commits and significant events
- **Knowledge Graph** - Links related memories with typed relationships

## Installation

### From Claude Code

```bash
claude plugin add memorylayer
```

### Manual Installation

Clone or copy this plugin to your Claude Code plugins directory:

```bash
# Global plugins
cp -r memorylayer-cc-plugin ~/.claude/plugins/memorylayer

# Or project-local
cp -r memorylayer-cc-plugin .claude/plugins/memorylayer
```

## Prerequisites

### MemoryLayer Server

The plugin requires a running MemoryLayer server. For local development:

```bash
# Install and run the server
pip install memorylayer-server
memorylayer serve
```

The server runs on `http://localhost:61001` by default.

### Configuration

Set environment variables to customize:

```bash
# Custom server URL
export MEMORYLAYER_URL="http://your-server:61001"

# Custom workspace (default: auto-detected from git repo)
export MEMORYLAYER_WORKSPACE_ID="my-project"

# Disable session/working memory (default: enabled)
export MEMORYLAYER_SESSION_MODE="false"
```

## What Gets Installed

### MCP Server

The plugin configures the `@scitrera/memorylayer-mcp-server` which provides these tools:

#### Core Memory Tools

| Tool | Description |
|------|-------------|
| `memory_remember` | Store new memories |
| `memory_recall` | Search memories by query |
| `memory_reflect` | Synthesize insights across memories |
| `memory_forget` | Delete outdated information |
| `memory_associate` | Link related memories |
| `memory_briefing` | Get session context summary |
| `memory_statistics` | View workspace analytics |
| `memory_graph_query` | Traverse memory relationships |
| `memory_audit` | Find contradictions |

#### Session Management Tools

| Tool | Description |
|------|-------------|
| `memory_session_start` | Initialize session and server-side sandbox |
| `memory_session_end` | End session, optionally commit to long-term storage |
| `memory_session_commit` | Checkpoint working memory without ending session |
| `memory_session_status` | Check current session state |

#### Context Environment Tools (Server-Side Sandbox)

These tools provide a persistent Python execution environment on the server. Sandbox state survives context compaction — call `memory_context_inspect` after compaction to re-orient.

| Tool | Description |
|------|-------------|
| `memory_context_exec` | Execute Python code in the sandbox |
| `memory_context_inspect` | Inspect sandbox variables (call after compaction) |
| `memory_context_load` | Load memories into a sandbox variable via search |
| `memory_context_inject` | Inject a value into the sandbox |
| `memory_context_query` | Ask the server LLM using sandbox context |
| `memory_context_rlm` | Run a Recursive Language Model (RLM) loop |
| `memory_context_status` | Get sandbox environment status |
| `memory_context_checkpoint` | Persist sandbox state for enterprise hooks |

Working memory persists within a session and survives context compaction. At session end, important context is automatically extracted to long-term memory.

### Slash Commands

| Command | Description |
|---------|-------------|
| `/memorylayer-status` | Check connection and workspace info |
| `/memorylayer-setup` | Guided setup and troubleshooting |
| `/memorylayer-remember <content>` | Quick memory storage |
| `/memorylayer-recall <query>` | Quick memory search |

### Hooks

| Hook | Trigger | Behavior |
|------|---------|----------|
| `PreCompact` | Before context compaction | **Critical** - Stores important information, checkpoints sandbox state, and commits working memory before context is lost |
| `SessionStart` | Session begins | Loads context briefing, directives, and starts server-side session for workspace resolution |
| `Stop` | Session ends | Commits working memory to long-term storage and ends the server session |
| `PreToolUse` | Before Task/Edit/Write | Injects recalled context relevant to the operation (query-aware dedup) |
| `PostToolUse` | After Bash/Edit/Write/Task | Captures outcomes (commit summaries, file changes, new file creation, agent results) |
| `UserPromptSubmit` | On user input | Recalls relevant memories for recall/review/implement/error-related prompts |

## How It Works

### Context Compaction Protection

Without MemoryLayer:
```
[Long conversation with decisions, fixes, learnings]
     ↓ context window full
[COMPACTION - history truncated]
     ↓
[Claude forgets everything not in remaining context]
```

With MemoryLayer:
```
[Long conversation with decisions, fixes, learnings]
     ↓ context window full
[PreCompact hook fires]
     ↓
[Claude stores key information to MemoryLayer]
     ↓
[COMPACTION - history truncated]
     ↓
[Claude can recall stored memories when needed]
```

### Workspace Isolation

Each project automatically gets its own workspace based on:
1. Git repository name (from remote origin)
2. Git root directory name
3. Current working directory name

This means memories are isolated per-project by default.

## Usage Tips

### Explicit Memory Commands

You can always ask Claude directly:

- "Remember that we decided to use PostgreSQL for the database"
- "What do you remember about the authentication system?"
- "Store this bug fix for future reference"

### Memory Types

- **episodic** - Events, what happened
- **semantic** - Facts, concepts, knowledge
- **procedural** - How-to, solutions, patterns
- **working** - Current task context (auto-expires)

### Importance Levels

- **0.9** - Critical decisions, breaking changes
- **0.7-0.8** - Bug fixes, architecture decisions
- **0.5-0.6** - General knowledge, minor features
- **0.3-0.4** - Temporary notes

## Troubleshooting

### MCP Server Not Connecting

1. Check the server is running:
   ```bash
   curl http://localhost:61001/health
   ```

2. Check environment variable:
   ```bash
   echo $MEMORYLAYER_URL
   ```

### Hooks Not Firing

1. Verify plugin is installed:
   ```bash
   claude plugin list
   ```

2. Check hooks are loaded:
   ```bash
   claude hooks list
   ```

### Wrong Workspace

The workspace is auto-detected from the git repo or directory name. To override:

```bash
export MEMORYLAYER_WORKSPACE_ID="my-custom-workspace"
```

## Links

- [MemoryLayer Documentation](https://memorylayer.ai/docs)
- [MCP Server README](../memorylayer-mcp-typescript/README.md)
- [Hook Configuration Details](../docs/CLAUDE_CODE_HOOKS.md)
- [CLAUDE.md Templates](../docs/CLAUDE_MD_TEMPLATES.md)

## License

MIT
