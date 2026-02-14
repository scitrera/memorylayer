/**
 * PostToolUse hook handler
 * Silently captures tool observations as working memory
 */

import type { HookInput, HookOutput } from "../types.js";
import { formatStorageGuidance } from "../formatters.js";
import { shouldSkipTool, buildObservation, type ObservationData } from "../observation.js";
import { getClient } from "../client.js";
import { getCurrentPrompt } from "../state.js";

/**
 * Store observation asynchronously (fire-and-forget)
 */
async function storeObservationAsync(obs: ObservationData): Promise<void> {
  // getClient() syncs session ID from env/hook-state via resolveSessionId
  const client = getClient();
  const sessionId = client.getSessionId();

  if (!sessionId) return;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3000);

  try {
    await client.setWorkingMemory(sessionId, `obs_${obs.contentHash}`, {
      type: obs.type,
      title: obs.title,
      tool: obs.toolName,
      files_read: obs.filesRead,
      files_modified: obs.filesModified,
      facts: obs.facts,
      concepts: obs.concepts,
      intent: obs.intent,
      summary: obs.summary,
      captured_at: new Date().toISOString(),
    });
  } catch {
    // Silent failure - never block tool execution
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Check if Bash output indicates significant action
 */
function isSignificantBashOutput(input: HookInput): boolean {
  const command = input.tool_input?.command as string | undefined;
  if (!command) return false;

  // Git commits
  if (/git\s+commit/i.test(command)) return true;

  // Build commands with errors
  if (/npm\s+run\s+build|cargo\s+build|make\b|tsc\b/i.test(command)) {
    const output = input.tool_output || "";
    return /error|fail/i.test(output);
  }

  return false;
}

/**
 * Handle PostToolUse event
 */
export async function handlePostToolUse(input: HookInput): Promise<HookOutput> {
  const toolName = input.tool_name;

  if (!toolName) {
    return { success: true };
  }

  // Skip tools that shouldn't be captured
  if (shouldSkipTool(toolName)) {
    return { success: true };
  }

  // Build observation from tool usage
  const currentPrompt = getCurrentPrompt();
  const obs = buildObservation(input, currentPrompt);

  // If observation is empty/no-op, skip
  if (!obs) {
    return { success: true };
  }

  // Fire-and-forget storage (do NOT await)
  storeObservationAsync(obs).catch(() => {});

  // Keep significant-event guidance for git commits and build errors
  if (toolName === "Bash" && isSignificantBashOutput(input)) {
    const command = input.tool_input?.command as string || "";

    if (/git\s+commit/i.test(command)) {
      return {
        success: true,
        additionalContext: formatStorageGuidance("Bash", true),
      };
    }

    if (/build|tsc|make/i.test(command)) {
      return {
        success: true,
        additionalContext: "Build had errors. Consider storing the issue with `memory_remember` (subtype: error) for future reference.",
      };
    }
  }

  // Silent capture - no context injection
  return { success: true };
}
