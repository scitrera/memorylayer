/**
 * User message hook for MemoryLayer OpenCode plugin.
 *
 * Detects patterns in user messages and performs intelligent recall
 * to inject relevant memories into the conversation context.
 */

import { getClient, checkHealth } from "../shared/client.js";
import { formatRecallResult } from "../shared/formatters.js";
import {
  wasRecallDoneThisTurn,
  markRecallDone,
  resetRecallStatus,
  setCurrentTopic,
  setCurrentPrompt,
} from "../shared/state.js";

/** Pattern categories and their recall queries */
const PATTERNS = {
  preference: {
    regex: /\b(which\s+\w+\s+should|should\s+we\s+use|what\s+do\s+we\s+use|how\s+do\s+we|what'?s\s+our|preferred|convention|do\s+we\s+have\s+a|what'?s\s+the\s+(default|standard|preferred))\b/i,
    queries: ["directive", "preference", "convention"],
  },
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

  return words.slice(0, 5).join(" ");
}

/**
 * Get guidance text for each pattern type
 */
function getPatternGuidance(pattern: keyof typeof PATTERNS): string {
  const guidance: Record<string, string> = {
    preference: "This is a preference/convention question. Relevant directives and decisions found:",
    recall: "Recalled memories matching this request:",
    analysis: "This request involves analysis/research. Relevant prior context found:",
    implementation: "This request involves implementation. Relevant patterns/solutions found:",
    error: "This request mentions an error/bug. Relevant prior fixes found:",
  };
  return guidance[pattern] || "Relevant memories found:";
}

/**
 * Extract user message text from message parts.
 */
export function extractMessageText(parts: Array<{ type: string; text?: string }>): string {
  return parts
    .filter(p => p.type === "text" && p.text)
    .map(p => p.text!)
    .join("\n");
}

/**
 * Handle a new user message — detect patterns and recall relevant memories.
 *
 * Returns additional context text to inject, or null if no recall needed.
 */
export async function handleUserMessage(messageText: string): Promise<string | null> {
  if (!messageText) return null;

  // Reset recall status for new user turn
  resetRecallStatus();

  // Persist the user's prompt for intent detection in tool hooks
  setCurrentPrompt(messageText);

  // Skip if recall was already done this turn
  if (wasRecallDoneThisTurn()) return null;

  // Check server health
  const healthy = await checkHealth();
  if (!healthy) return null;

  // Store the user's topic for cross-hook context
  const keyTerms = extractKeyTerms(messageText);
  if (keyTerms) {
    setCurrentTopic(keyTerms);
  }

  // Detect pattern category
  const pattern = detectPattern(messageText);
  if (!pattern) return null;

  try {
    const patternQueries = PATTERNS[pattern].queries;
    const query = `${keyTerms} ${patternQueries[0]}`.trim();

    const client = getClient();
    const result = await client.recall({ query, limit: 10 });
    markRecallDone(query);

    if (result.memories.length === 0) return null;

    const guidance = getPatternGuidance(pattern);
    const recallOutput = formatRecallResult(result, query);

    return `${guidance}\n\n${recallOutput}`;
  } catch {
    return null;
  }
}
