# MemoryLayer Setup Skill

Help users configure MemoryLayer for their project.

## Activation

This skill activates when users ask about:
- Setting up MemoryLayer
- Configuring memory settings
- Troubleshooting memory issues

## Instructions

When this skill is activated, help the user with MemoryLayer setup:

### 1. Check Server Connection

First, verify the MemoryLayer server is accessible:

```bash
curl -s http://localhost:61001/health
```

If this fails, guide the user to start the server:

```bash
pip install memorylayer-server
memorylayer serve
```

### 2. Verify MCP Tools

Check that the MCP tools are available by attempting to use `memory_briefing`.

### 3. Check Workspace Detection

The workspace should be auto-detected from the current directory. Verify with:

```bash
# Should show the workspace name in server logs
# Or check via the briefing response
```

### 4. Test Memory Operations

Perform a simple test:

1. Store a test memory:
   ```
   memory_remember: "Test memory for setup verification"
   ```

2. Recall it:
   ```
   memory_recall: "test setup verification"
   ```

3. Clean up:
   ```
   memory_forget: <memory_id>
   ```

### 5. Optional: Custom Configuration

If the user needs custom configuration, guide them to set environment variables:

```bash
# Custom server URL
export MEMORYLAYER_URL="http://custom-server:61001"

# Custom workspace (overrides auto-detection)
export MEMORYLAYER_WORKSPACE_ID="my-workspace"
```

### 6. Verify Hooks

Confirm the hooks are working:

- **SessionStart**: Should have triggered when this session began
- **PreCompact**: Will trigger when context gets full
- **Stop**: Will trigger when ending the session

## Troubleshooting Guide

### "Connection refused" errors
- Server not running: `memorylayer serve`
- Wrong URL: Check `MEMORYLAYER_URL`

### "Workspace not found"
- Workspace is auto-created on first use
- Check workspace name matches expectations

### Hooks not firing
- Verify plugin is installed: `claude plugin list`
- Check hooks configuration: `claude hooks list`

### Memories not persisting
- Check server logs for errors
- Verify storage path is writable

## Success Criteria

Setup is complete when:
- [ ] Server responds to health check
- [ ] MCP tools are available
- [ ] Test memory can be stored and recalled
- [ ] Workspace is correctly identified
