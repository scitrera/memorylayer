# /memorylayer-setup

Guided setup for MemoryLayer configuration.

## Usage

```
/memorylayer-setup
```

## Behavior

Run the setup skill to help the user configure MemoryLayer:

1. **Check server connection** - Verify MCP server is reachable
2. **Verify MCP tools are available** - List available memory tools
3. **Configure permissions** - Guide user to allow all memorylayer tools
4. **Test memory operations** - Store and recall a test memory
5. **Confirm workspace detection** - Show detected workspace name
6. **Verify hooks are active** - Confirm PreCompact, SessionStart, Stop hooks

### Permission Configuration

To avoid permission prompts for every MCP tool call, add this wildcard to your settings:

**Option A: Project settings** (`.claude/settings.local.json`):
```json
{
  "permissions": {
    "allow": [
      "mcp__plugin_memorylayer_memorylayer__*"
    ]
  }
}
```

**Option B: Global settings** (`~/.claude.json`):
```json
{
  "projects": {
    "/path/to/your/project": {
      "allowedTools": ["mcp__plugin_memorylayer_memorylayer__*"]
    }
  }
}
```

**Option C: CLI flag** (per-session):
```bash
claude --allowedTools "mcp__plugin_memorylayer_memorylayer__*"
```

The wildcard `*` allows all memorylayer tools: remember, recall, reflect, associate, forget, context_*, and session_*.

## When to Use

- First time using MemoryLayer in a project
- After installation issues
- When memories aren't being stored/recalled properly
- To verify configuration is correct
- When you want to reduce permission prompts
