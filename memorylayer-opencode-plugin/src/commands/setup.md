# /memorylayer-setup

Automated setup and verification for MemoryLayer with OpenCode.

## Usage

```
/memorylayer-setup
```

## Behavior

This command performs a fully automated setup sequence:

1. **Check server** -- Run `curl -sf http://localhost:61001/health`. If unreachable, offer to install and start: `pip install memorylayer-server && memorylayer serve &`

2. **Verify MCP tools** -- Call `memory_briefing` to confirm tools are connected and the server is responding.

3. **Smoke test** -- Store a test memory, recall it, then forget it to verify the full read/write cycle.

4. **Verify opencode.json** -- Check that the MCP server and plugin are configured in `opencode.json`:

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

5. **Status summary** -- Print server URL, workspace, session, tool count, and active hooks.

## When to Use

- First time using MemoryLayer with OpenCode
- After installation or upgrade
- When memories aren't being stored/recalled properly
- To verify configuration is correct
