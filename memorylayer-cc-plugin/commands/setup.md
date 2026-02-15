# /memorylayer-setup

Automated setup and verification for MemoryLayer.

## Usage

```
/memorylayer-setup
```

## Behavior

This command performs a fully automated setup sequence. Each step is executed, not just described:

1. **Check server** — Runs `curl -sf http://localhost:61001/health`. If unreachable, offers to install and start: `pip install memorylayer-server && memorylayer serve &`
2. **Auto-configure permissions** — Reads `.claude/settings.local.json`, merges `mcp__plugin_memorylayer_memorylayer__*` into the permissions allow list, writes back. This eliminates per-tool permission prompts.
3. **Verify MCP tools** — Calls `memory_briefing` to confirm tools are connected and the server is responding.
4. **Smoke test** — Stores a test memory, recalls it, then forgets it to verify the full read/write cycle.
5. **Verify hooks** — Reads `~/.memorylayer/hook-state.json` to confirm SessionStart hook fired.
6. **Status summary** — Prints server URL, workspace, session, permission status, tool count, and active hooks.

### Permission Configuration

The setup command automatically configures permissions by merging this into `.claude/settings.local.json`:

```json
{
  "permissions": {
    "allow": [
      "mcp__plugin_memorylayer_memorylayer__*"
    ]
  }
}
```

This wildcard allows all memorylayer MCP tools without individual prompts.

For manual configuration alternatives:

**Global settings** (`~/.claude.json`):
```json
{
  "projects": {
    "/path/to/your/project": {
      "allowedTools": ["mcp__plugin_memorylayer_memorylayer__*"]
    }
  }
}
```

**CLI flag** (per-session):
```bash
claude --allowedTools "mcp__plugin_memorylayer_memorylayer__*"
```

## When to Use

- First time using MemoryLayer in a project
- After installation or upgrade
- When memories aren't being stored/recalled properly
- To verify configuration is correct
- When encountering permission prompts for memory tools
