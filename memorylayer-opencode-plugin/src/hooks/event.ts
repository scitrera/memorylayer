/**
 * Event hook for MemoryLayer OpenCode plugin.
 *
 * Handles context compaction by preserving critical memory state
 * and committing working memory before context is lost.
 */

import { getClient, checkHealth } from "../shared/client.js";

/**
 * Handle session compacting — commit working memory and checkpoint sandbox.
 *
 * OpenCode's `experimental.session.compacting` hook is the equivalent
 * of Claude Code's PreCompact. We commit working memory and return
 * context strings that survive the compaction.
 */
export async function handleCompacting(_sessionID: string): Promise<string[]> {
  const context: string[] = [];

  const healthy = await checkHealth();
  if (!healthy) return context;

  const client = getClient();
  const clientSessionId = client.getSessionId();

  if (clientSessionId) {
    // Commit working memory to long-term storage before compaction
    try {
      await client.commitSession(clientSessionId, { importance_threshold: 0.3 });
      context.push(
        "[MemoryLayer] Working memory committed to long-term storage before compaction. " +
        "Use `memory_recall` to retrieve prior context. " +
        "Use `memory_context_inspect` to check server-side sandbox variables."
      );
    } catch {
      // Commit may fail, but we should still try to preserve context
    }

    // Checkpoint sandbox state if active
    try {
      const status = await client.contextStatus() as { exists?: boolean; variable_count?: number };
      if (status.exists && (status.variable_count ?? 0) > 0) {
        await client.contextCheckpoint();
        context.push(
          "[MemoryLayer] Server-side sandbox state checkpointed. " +
          "Variables persist across compaction — use `memory_context_inspect` to re-orient."
        );
      }
    } catch {
      // Sandbox checkpoint is best-effort
    }
  }

  // Always include recovery instructions
  if (context.length === 0) {
    context.push(
      "[MemoryLayer] Context compaction occurred. Use `memory_recall` to retrieve " +
      "prior context and `memory_context_inspect` to check sandbox state."
    );
  }

  return context;
}
