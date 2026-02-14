/**
 * UserPromptSubmit hook handler
 * Detects patterns in user prompts and provides recall guidance
 */

import type { HookInput, HookOutput } from "../types.js";
import { getClient, checkHealth } from "../client.js";
import { formatRecallResult } from "../formatters.js";
import { wasRecallDoneThisTurn, markRecallDone, setCurrentTopic, setCurrentPrompt } from "../state.js";

/** Pattern categories and their recall queries */
const PATTERNS = {
  recall: {
    regex: /\b(remember|recall|what did we|how did we|remind me|what was the|what were the|what do you (know|remember))\b/i,
    queries: ["context", "history", "decision"],
  },
  analysis: {
    regex: /\b(review|assess|analyze|evaluate|audit|investigate|explore|research|compare|status|state of|readiness|gap analysis)\b/i,
    queries: ["status", "assessment", "gaps", "problems"],
  },
  implementation: {
    regex: /\b(implement|build|create|add|fix|refactor|update|change|modify)\b/i,
    queries: ["patterns", "solutions", "issues"],
  },
  error: {
    regex: /\b(error|bug|issue|broken|failing|crash|exception|doesn't work|not working)\b/i,
    queries: ["fix", "error", "solution"],
  },
};

/**
 * Detect which pattern category matches the prompt
 */
function detectPattern(prompt: string): keyof typeof PATTERNS | null {
  for (const [category, config] of Object.entries(PATTERNS)) {
    if (config.regex.test(prompt)) {
      return category as keyof typeof PATTERNS;
    }
  }
  return null;
}

/**
 * Extract key terms from the prompt for recall query
 */
function extractKeyTerms(prompt: string): string {
  // Remove common words and extract meaningful terms
  const stopWords = new Set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this",
    "that", "these", "those", "i", "you", "we", "they", "it",
    "my", "your", "our", "their", "its", "please", "help", "me",
    "want", "need", "like", "to", "for", "with", "on", "in", "at",
  ]);

  const words = prompt.toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 2 && !stopWords.has(w));

  // Take first 5 meaningful words
  return words.slice(0, 5).join(" ");
}

/**
 * Handle UserPromptSubmit event
 */
export async function handleUserPromptSubmit(input: HookInput): Promise<HookOutput> {
  const prompt = input.user_prompt;
  if (!prompt) {
    return { success: true };
  }

  // Persist the user's prompt for intent detection in PostToolUse
  setCurrentPrompt(prompt);

  // Skip if recall was already done this turn
  if (wasRecallDoneThisTurn()) {
    return { success: true };
  }

  // Check server health
  const healthy = await checkHealth();
  if (!healthy) {
    return { success: true };
  }

  // Store the user's topic for cross-hook context (Edit/Write recall)
  const keyTerms = extractKeyTerms(prompt);
  if (keyTerms) {
    setCurrentTopic(keyTerms);
  }

  // Detect pattern category
  const pattern = detectPattern(prompt);
  if (!pattern) {
    return { success: true };
  }

  try {
    // Build recall query from prompt + pattern-specific terms
    const patternQueries = PATTERNS[pattern].queries;
    const query = `${keyTerms} ${patternQueries[0]}`.trim();

    // Perform recall using the same client as MCP tools
    const client = getClient();
    const result = await client.recall({ query, limit: 5, mode: "rag" });
    markRecallDone(query);

    if (result.memories.length === 0) {
      return { success: true };
    }

    // Format guidance based on pattern
    const guidance = getPatternGuidance(pattern);
    const recallOutput = formatRecallResult(result, query);

    return {
      success: true,
      additionalContext: `${guidance}\n\n${recallOutput}`,
    };
  } catch {
    // Don't fail the hook on recall errors
    return { success: true };
  }
}

/**
 * Get guidance text for each pattern type
 */
function getPatternGuidance(pattern: keyof typeof PATTERNS): string {
  const guidance: Record<string, string> = {
    recall: "Recalled memories matching this request:",
    analysis: "This request involves analysis/research. Relevant prior context found:",
    implementation: "This request involves implementation. Relevant patterns/solutions found:",
    error: "This request mentions an error/bug. Relevant prior fixes found:",
  };
  return guidance[pattern] || "Relevant memories found:";
}
