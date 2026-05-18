/**
 * Session lifecycle hooks for MemoryLayer OpenCode plugin.
 *
 * Handles session initialization (briefing, directives, sandbox state)
 * and session teardown (commit working memory, end session).
 */

import type { Memory } from "@scitrera/memorylayer-mcp-server";
import { getClient, checkHealth } from "../shared/client.js";
import { formatSessionStart } from "../shared/formatters.js";
import { markRecallDone, resetRecallStatus, updateSessionInfo } from "../shared/state.js";

/**
 * Check for existing sandbox state from a prior session or pre-compaction.
 */
async function checkSandboxState(client: ReturnType<typeof getClient>): Promise<Record<string, unknown> | null> {
  try {
    const status = await client.contextStatus() as { exists?: boolean; variable_count?: number };
    if (status.exists && (status.variable_count ?? 0) > 0) {
      return await client.contextInspect({});
    }
  } catch {
    // Context environment may not be available
  }
  return null;
}

/**
 * Initialize a MemoryLayer session and return formatted context for system prompt injection.
 *
 * Called from the `experimental.chat.system.transform` hook on the first message
 * of a session, or from the `event` hook on session start.
 */
export async function initializeSession(topic?: string): Promise<string | null> {
  const healthy = await checkHealth();
  if (!healthy) {
    return "MemoryLayer server not reachable. Memory features unavailable this session.";
  }

  try {
    const client = getClient();

    // Start server session
    let sessionId: string | undefined;
    try {
      const sessionResult = await client.startSession({ ttl_seconds: 3600 });
      sessionId = sessionResult.session_id;
      const workspaceId = client.getWorkspaceId();
      if (workspaceId) {
        updateSessionInfo(workspaceId, sessionId);
      }
    } catch {
      // Session start failed, continue without session management
    }

    // Reset recall status for new session
    resetRecallStatus();

    // Run briefing, directive recall, and sandbox check in parallel
    const [briefingResult, directiveResult, sandboxResult] = await Promise.allSettled([
      client.getBriefing({ limit: 10, includeMemories: false }),
      client.recall({
        query: "user directives and preferences",
        subtypes: ["directive", "preference"],
        limit: 10,
      }),
      checkSandboxState(client),
    ]);

    const briefing = briefingResult.status === "fulfilled" ? briefingResult.value : null;
    const directives: Memory[] =
      directiveResult.status === "fulfilled"
        ? directiveResult.value.memories
        : [];
    const sandboxState = sandboxResult.status === "fulfilled" ? sandboxResult.value : null;

    // If there's a topic, recall for it too
    let topicRecall = null;
    if (topic) {
      try {
        topicRecall = await client.recall({ query: topic, limit: 10, detail_level: "abstract" });
        markRecallDone(topic);
      } catch {
        // Topic recall is optional
      }
    }

    return formatSessionStart(briefing, directives, topicRecall, topic, sandboxState);
  } catch (error) {
    return `MemoryLayer session error: ${error instanceof Error ? error.message : error}`;
  }
}

/**
 * Finalize the MemoryLayer session — commit working memory and end session.
 *
 * Called when the OpenCode session ends. Best-effort: never throws.
 */
export async function finalizeSession(): Promise<void> {
  const healthy = await checkHealth();
  if (!healthy) return;

  const client = getClient();
  const sessionId = client.getSessionId();
  if (!sessionId) return;

  try {
    // Commit working memory to long-term storage
    try {
      await client.commitSession(sessionId, { importance_threshold: 0.5 });
    } catch {
      // Commit may fail if session expired
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
    // Best-effort — don't fail on shutdown
  }
}
