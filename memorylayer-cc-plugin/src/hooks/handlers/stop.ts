/**
 * Stop hook handler
 * Commits session and persists working memory at session end
 */

import type { HookOutput } from "../types.js";
import { getClient, checkHealth } from "../client.js";

/**
 * Handle Stop event -- commit and end the server session.
 *
 * Stop hooks cannot inject additional context (Claude Code is shutting down),
 * so this handler performs side effects only: committing working memory
 * to long-term storage and ending the server session.
 */
export async function handleStop(): Promise<HookOutput> {
  const healthy = await checkHealth();
  if (!healthy) {
    return { success: true };
  }

  // getClient() syncs session ID from env/hook-state via resolveSessionId
  const client = getClient();
  const sessionId = client.getSessionId();
  if (!sessionId) {
    console.error("[stop] no session ID available, skipping commit/end");
    return { success: true };
  }

  try {

    // Commit working memory to long-term storage
    try {
      await client.commitSession(sessionId, { importance_threshold: 0.5 });
    } catch {
      // Commit may fail if session expired or not supported
    }

    // End the server session
    try {
      await client.endSession(sessionId, {
        commit: true,
        importance_threshold: 0.5,
      });
    } catch {
      // Session may already be ended
    }
  } catch {
    // Best-effort -- don't fail the hook on shutdown
  }

  return { success: true };
}
