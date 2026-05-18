/**
 * Tool execution hooks for MemoryLayer OpenCode plugin.
 *
 * Before-tool: Injects relevant memory context for write/delegation tools.
 * After-tool: Silently captures tool observations as working memory.
 */

import { getClient, checkHealth } from "../shared/client.js";
import { formatRecallResult, formatStorageGuidance } from "../shared/formatters.js";
import {
  shouldSkipTool,
  buildObservation,
  type ObservationData,
} from "../shared/observation.js";
import {
  wasQueryRecalledThisTurn,
  markRecallDone,
  getCurrentTopic,
  getCurrentPrompt,
} from "../shared/state.js";

// ---------------------------------------------------------------------------
// Before-tool hook
// ---------------------------------------------------------------------------

/**
 * Handle pre-tool execution for task/delegation tools.
 * Returns context text to prepend to tool description, or null.
 */
async function handleTaskTool(toolArgs: Record<string, unknown>): Promise<string | null> {
  const taskPrompt = (toolArgs.prompt || toolArgs.description || "") as string;
  if (!taskPrompt) {
    return "RECALL-FIRST RULE: Consider using `memory_recall` before delegating to subagent. Subagents cannot access MemoryLayer.";
  }

  const query = taskPrompt.substring(0, 100);
  if (wasQueryRecalledThisTurn(query)) {
    return "Recall already done for this topic. Include relevant memories in subagent prompt.";
  }

  const healthy = await checkHealth();
  if (!healthy) return null;

  try {
    const client = getClient();
    const result = await client.recall({ query, limit: 5 });
    markRecallDone(query);

    if (result.memories.length === 0) {
      return "No relevant memories found for this task. Proceeding with delegation.";
    }

    const recallOutput = formatRecallResult(result, query);
    return `INCLUDE IN SUBAGENT PROMPT - Relevant context from memory:\n\n${recallOutput}`;
  } catch {
    return "Memory recall failed. Consider manual recall before delegation.";
  }
}

/**
 * Handle pre-tool execution for edit/write tools.
 * Returns context text, or null.
 */
async function handleEditWriteTool(toolArgs: Record<string, unknown>): Promise<string | null> {
  const filePath = (toolArgs.file_path || toolArgs.path || toolArgs.file || "") as string;
  if (!filePath) return null;

  const filename = filePath.split("/").pop() || filePath;
  const topic = getCurrentTopic();
  const query = topic ? `${filename} ${topic}` : `${filename} patterns solutions`;

  if (wasQueryRecalledThisTurn(query)) return null;

  const healthy = await checkHealth();
  if (!healthy) return null;

  try {
    const client = getClient();
    const result = await client.recall({ query, limit: 3 });

    if (result.memories.length === 0) return null;

    markRecallDone(query);
    return `Relevant context for ${filename}:\n\n${formatRecallResult(result, filename)}`;
  } catch {
    return null;
  }
}

/**
 * Handle before-tool execution.
 *
 * In OpenCode's hook system, this is called via "tool.execute.before" with
 * (input, output) where we can mutate output.args. We use the return value
 * pattern here and let the entry point handle injection.
 *
 * Returns additional context text, or null.
 */
export async function handleToolBefore(
  toolName: string,
  toolArgs: Record<string, unknown>
): Promise<string | null> {
  switch (toolName) {
    case "task":
      return handleTaskTool(toolArgs);

    case "edit":
    case "write":
    case "multiedit":
    case "apply_patch":
      return handleEditWriteTool(toolArgs);

    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// After-tool hook
// ---------------------------------------------------------------------------

/**
 * Store observation asynchronously (fire-and-forget)
 */
async function storeObservationAsync(obs: ObservationData): Promise<void> {
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
    // Silent failure — never block tool execution
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Check if bash output indicates a significant action
 */
function isSignificantBashOutput(
  toolArgs: Record<string, unknown>,
  toolOutput: string
): boolean {
  const command = (toolArgs.command || "") as string;

  // Git commits
  if (/git\s+commit/i.test(command)) return true;

  // Build commands with errors
  if (/npm\s+run\s+build|cargo\s+build|make\b|tsc\b|bun\s+build/i.test(command)) {
    return /error|fail/i.test(toolOutput);
  }

  return false;
}

/**
 * Handle after-tool execution — capture observations as working memory.
 *
 * Returns additional context guidance for significant events, or null.
 */
export async function handleToolAfter(
  toolName: string,
  toolArgs: Record<string, unknown>,
  toolOutput: string
): Promise<string | null> {
  if (shouldSkipTool(toolName)) return null;

  const currentPrompt = getCurrentPrompt();
  const obs = buildObservation(toolName, toolArgs, toolOutput, currentPrompt);

  if (!obs) return null;

  // Fire-and-forget storage
  storeObservationAsync(obs).catch(() => {});

  // Return guidance for significant events
  if (toolName === "bash" && isSignificantBashOutput(toolArgs, toolOutput)) {
    const command = (toolArgs.command || "") as string;

    if (/git\s+commit/i.test(command)) {
      return formatStorageGuidance("bash", true);
    }

    if (/build|tsc|make|bun\s+build/i.test(command)) {
      return "Build had errors. Consider storing the issue with `memory_remember` (subtype: error) for future reference.";
    }
  }

  return null;
}
