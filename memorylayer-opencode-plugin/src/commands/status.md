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

3. **Report Plugin Status**
   - Confirm which hooks are active

## Response Format

```
MemoryLayer Status
==================
Server: Connected (http://localhost:61001)
Workspace: my-project (auto-detected from git)

Statistics:
- Total memories: 42
- Recent (24h): 5
- Associations: 18

Plugin Hooks Active:
- system.transform: Yes (injects briefing at session start)
- chat.message: Yes (pattern-based recall)
- tool.before/after: Yes (context injection + observation capture)
- session.compacting: Yes (preserves context before compaction)

All systems operational.
```

If there are issues:

```
MemoryLayer Status
==================
Server: NOT CONNECTED

Troubleshooting:
1. Start the server: memorylayer serve
2. Check MEMORYLAYER_URL environment variable
3. Run /memorylayer-setup for guided configuration
```
