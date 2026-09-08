---
name: memorylayer
description: Use when Codex should recall or store durable project context with MemoryLayer, preserve decisions across sessions, use MemoryLayer MCP tools, or troubleshoot MemoryLayer setup.
---

# MemoryLayer For Codex

Use MemoryLayer as Codex's durable project memory when prior decisions, conventions, preferences, task state, or cross-session knowledge would change the answer.

## Behavior

- Prefer workspace-scoped memories. Respect `MEMORYLAYER_WORKSPACE_ID` when it is set; otherwise let the MCP server auto-detect the workspace from the repo.
- Before broad reviews, architecture work, implementation on an unfamiliar subsystem, or recall requests, use `memory_recall` / `memory_briefing` when the MemoryLayer MCP tools are available.
- Store only durable, non-secret information: user directives, decisions with rationale, solved bugs, architecture tradeoffs, reusable workflows, and important task state.
- Do not store credentials, API keys, private tokens, or raw confidential payloads. Summarize sensitive context at a safe abstraction level.
- When the MCP tools are unavailable, say MemoryLayer is not connected and continue from local repository context.

## Suggested Tool Use

- `memory_briefing`: start of a large task or review.
- `memory_recall`: preferences, prior decisions, subsystem history, or repeated bug/error work.
- `memory_remember`: durable result after a decision, bug fix, benchmark result, architecture conclusion, or user directive.
- `memory_session_start`, `memory_session_commit`, `memory_session_end`: long-running agent work where working memory should survive handoffs.
- `memory_context_*`: server-side sandbox analysis that should survive context compaction or resume after a long investigation.

## Response Discipline

When you use MemoryLayer, mention the memory-backed context only when it materially affects the answer. Keep normal Codex repository inspection and tests as the source of truth for current code behavior.
