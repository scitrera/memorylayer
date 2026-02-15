# MemoryLayer Setup Skill

Automated setup and verification for MemoryLayer.

## Activation

This skill activates when users ask about:
- Setting up MemoryLayer
- Configuring memory settings
- Troubleshooting memory issues

## Instructions

When this skill is activated, perform the following steps automatically. Do not just describe the steps — execute them.

### Step 1: Check Server Connection

Run a health check against the MemoryLayer server:

```bash
curl -sf http://localhost:61001/health
```

**If the server is unreachable**, offer to install and start it:

```bash
pip install memorylayer-server
memorylayer serve &
```

Wait a few seconds after starting, then re-check health.

### Step 2: Auto-Configure Permissions

Check if MemoryLayer tool permissions are already configured. Read `.claude/settings.local.json` (create the file if it doesn't exist). If the permissions allow list does not include the memorylayer wildcard, add it:

```json
{
  "permissions": {
    "allow": [
      "mcp__plugin_memorylayer_memorylayer__*"
    ]
  }
}
```

**Important**: If `.claude/settings.local.json` already exists, read it first and merge the permission into the existing `allow` array. Do not overwrite other settings.

### Step 3: Verify MCP Tools

Call `memory_briefing` to confirm the MCP tools are connected and working. This simultaneously:
- Verifies the MCP server is running
- Verifies the SDK can communicate with the MemoryLayer server
- Returns workspace stats

If this fails, check that the plugin is installed (`/plugin list`) and that the server is running (Step 1).

### Step 4: Run Smoke Test

Perform a store-recall-forget cycle to verify end-to-end functionality:

1. **Store**: `memory_remember` with content "MemoryLayer setup verification test" and tags ["setup-test"]
2. **Recall**: `memory_recall` with query "setup verification test" — verify the test memory is returned
3. **Forget**: `memory_forget` with the memory ID from step 1 — clean up the test memory

If any step fails, report the error and suggest checking server logs.

### Step 5: Verify Hooks

Check that the SessionStart hook fired by reading the hook state file:

```bash
cat ~/.memorylayer/hook-state.json 2>/dev/null || echo "No hook state found"
```

The file should contain a `sessionId` field, indicating the SessionStart hook ran when this session began. If the file doesn't exist or has no sessionId, the hooks may not be configured — verify the plugin is installed.

### Step 6: Print Status Summary

Summarize the setup results:

```
MemoryLayer Setup Complete
--------------------------
Server:      http://localhost:61001 (healthy)
Workspace:   <workspace name from briefing>
Session:     <session ID from hook state>
Permissions: Configured in .claude/settings.local.json
Tools:       <count> MCP tools available
Hooks:       SessionStart, PostToolUse, PreCompact, Stop
```

## Troubleshooting Guide

### "Connection refused" errors
- Server not running: `memorylayer serve`
- Wrong URL: Check `MEMORYLAYER_URL` environment variable
- Port conflict: Try `MEMORYLAYER_PORT=61002 memorylayer serve`

### "Workspace not found"
- Workspace is auto-created on first use
- Check workspace name matches expectations via `memory_briefing`

### Hooks not firing
- Verify plugin is installed: check `.mcp.json` exists
- Verify hooks.json is present in plugin directory
- Restart Claude Code session after plugin installation

### Memories not persisting
- Check server logs for errors: `memorylayer serve` (foreground)
- Verify storage path is writable
- Check disk space

### Permission prompts on every tool call
- Run this setup skill again — Step 2 auto-configures permissions
- Or manually add `mcp__plugin_memorylayer_memorylayer__*` to `.claude/settings.local.json`

## Success Criteria

Setup is complete when all of the following pass:
- [x] Server responds to health check
- [x] Permissions configured (no prompts for memory tools)
- [x] MCP tools connected (memory_briefing works)
- [x] Smoke test passes (store, recall, forget)
- [x] Hooks active (sessionId in hook state)
- [x] Status summary printed
