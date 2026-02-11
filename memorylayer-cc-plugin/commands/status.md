# /memorylayer-status

Check MemoryLayer connection status and workspace information.

## Usage

```
/memorylayer-status
```

## Behavior

When this command is invoked:

1. **Check Server Connection**
   - Use `memory_briefing` to verify the MCP server is connected
   - Report connection status

2. **Display Workspace Info**
   - Show the current workspace name (auto-detected from directory/git)
   - Show memory statistics from the briefing

3. **Report Hook Status**
   - Confirm which hooks are active (PreCompact, SessionStart, Stop)

## Response Format

Provide a concise status report:

```
MemoryLayer Status
==================
Server: Connected (http://localhost:61001)
Workspace: my-project (auto-detected from git)

Statistics:
- Total memories: 42
- Recent (24h): 5
- Associations: 18

Hooks Active:
- PreCompact: Yes (preserves context before compaction)
- SessionStart: Yes (loads context at start)
- Stop: Yes (prompts for final memories)

All systems operational.
```

If there are issues, report them clearly:

```
MemoryLayer Status
==================
Server: NOT CONNECTED

Troubleshooting:
1. Start the server: memorylayer serve
2. Check MEMORYLAYER_URL environment variable
3. Run /memorylayer-setup for guided configuration
```
