/**
 * Format API responses for injection into Claude's context.
 *
 * Uses the same types returned by MemoryLayerClient (the MCP adapter),
 * so hooks and MCP tools share the same data contracts.
 */

import type { Memory, RecallResult, ToolResponse } from "../types.js";

/**
 * Format a single memory for display
 */
function formatMemory(memory: Memory, index: number): string {
  const lines: string[] = [];

  const typeStr = memory.subtype
    ? `${memory.type}/${memory.subtype}`
    : memory.type;
  const relevanceStr = memory.relevance_score
    ? ` (relevance: ${(memory.relevance_score * 100).toFixed(0)}%)`
    : "";

  lines.push(`${index + 1}. [${typeStr}]${relevanceStr}`);
  lines.push(`   ${memory.content}`);

  const tags = memory.tags ?? [];
  if (tags.length > 0) {
    lines.push(`   Tags: ${tags.join(", ")}`);
  }

  return lines.join("\n");
}

/**
 * Format recall results for context injection
 */
export function formatRecallResult(result: RecallResult, query: string): string {
  if (result.memories.length === 0) {
    return `No memories found matching "${query}".`;
  }

  const totalCount = result.total_count ?? result.memories.length;
  const lines: string[] = [
    `Found ${totalCount} memories for "${query}" (showing ${result.memories.length}):`,
    "",
  ];

  for (let i = 0; i < result.memories.length; i++) {
    lines.push(formatMemory(result.memories[i], i));
    if (i < result.memories.length - 1) {
      lines.push("");
    }
  }

  return lines.join("\n");
}

/**
 * Format briefing for context injection.
 * Accepts the ToolResponse from MemoryLayerClient.getBriefing().
 */
export function formatBriefing(briefing: ToolResponse): string {
  if (!briefing) {
    return "=== Workspace Briefing ===\n\nNo workspace data available.";
  }

  const totalMemories = (briefing.total_memories as number) ?? 0;
  const activeTopics = (briefing.active_topics as string[]) ?? [];
  const memoryTypes = (briefing.memory_types as Record<string, number>) ?? {};
  const recentActivity = (briefing.recent_activity as Array<{
    timestamp?: string;
    summary?: string;
    memories_created?: number;
  }>) ?? [];

  const lines: string[] = [
    "=== Workspace Briefing ===",
    "",
    `Total memories: ${totalMemories}`,
  ];

  if (activeTopics.length > 0) {
    lines.push(`Active topics: ${activeTopics.join(", ")}`);
  }

  // Memory type breakdown
  const types = Object.entries(memoryTypes);
  if (types.length > 0) {
    lines.push("");
    lines.push("Memory types:");
    for (const [type, count] of types) {
      lines.push(`  - ${type}: ${count}`);
    }
  }

  // Recent activity
  if (recentActivity.length > 0) {
    lines.push("");
    lines.push("Recent activity:");
    for (const activity of recentActivity.slice(0, 3)) {
      const date = new Date(activity.timestamp ?? new Date().toISOString()).toLocaleDateString();
      lines.push(`  - ${date}: ${activity.summary ?? ""} (${activity.memories_created ?? 0} memories)`);
    }
  }

  return lines.join("\n");
}

/**
 * Format directive memories specially (high importance user instructions)
 */
export function formatDirectives(memories: Memory[]): string {
  const directives = memories.filter(m => m.subtype === "directive");

  if (directives.length === 0) {
    return "";
  }

  const lines: string[] = [
    "=== User Directives (must follow) ===",
    "",
  ];

  for (const directive of directives) {
    lines.push(`• ${directive.content}`);
  }

  return lines.join("\n");
}

/**
 * Format sandbox state for context injection (post-compaction recovery)
 */
export function formatSandboxState(inspectResult: Record<string, unknown>): string {
  const variableCount = (inspectResult.variable_count as number) ?? 0;
  const variables = (inspectResult.variables as Record<string, { type: string; preview: string }>) ?? {};

  const lines: string[] = [
    "=== Existing Sandbox State (server-side) ===",
    "",
    `The server-side sandbox has ${variableCount} variable(s) from a prior session or before context compaction.`,
    "These variables are live and available for memory_context_exec, memory_context_query, and memory_context_rlm.",
    "",
  ];

  for (const [name, info] of Object.entries(variables)) {
    lines.push(`  ${name} (${info.type}): ${info.preview}`);
  }

  lines.push("");
  lines.push("Use `memory_context_inspect` for detailed variable inspection.");

  return lines.join("\n");
}

/**
 * Format combined SessionStart output
 */
export function formatSessionStart(
  briefing: ToolResponse | null,
  directives: Memory[],
  topicRecall: RecallResult | null,
  topic?: string,
  sandboxState?: Record<string, unknown> | null
): string {
  const sections: string[] = [];

  // Briefing first
  if (briefing) {
    sections.push(formatBriefing(briefing));
  }

  // Directives (high priority)
  const directiveSection = formatDirectives(directives);
  if (directiveSection) {
    sections.push(directiveSection);
  }

  // Topic-specific recall
  if (topicRecall && topicRecall.memories.length > 0 && topic) {
    sections.push(formatRecallResult(topicRecall, topic));
  }

  // Existing sandbox state (post-compaction or resumed session)
  if (sandboxState) {
    sections.push(formatSandboxState(sandboxState));
  }

  // Add session guidance
  const guidance = `=== Session Guidance ===

Storing Memories:
- \`memory_remember\`: store to long-term memory with type, subtype, and importance
- \`memory_session_commit\`: checkpoint working memory mid-session (without ending it)

Importance Guide: directives/preferences → 0.9 | decisions/architecture → 0.7-0.8 | fixes/solutions → 0.7 | patterns → 0.5-0.6
Types: semantic (facts), procedural (how-to), episodic (events), working (current context, auto-expires)
Subtypes: directive, decision, fix, solution, code_pattern, error, workflow, preference, problem

Context Environment (sandbox survives compaction):
- Load + analyze memories server-side: \`memory_context_load\` → \`memory_context_query\` or \`memory_context_rlm\`
- Run computations on loaded data: \`memory_context_exec\` (Python sandbox)
- After compaction, call \`memory_context_inspect\` to re-orient with existing sandbox variables`;
  sections.push(guidance);

  if (sections.length === 1) {
    // Only guidance, no prior context
    return "MemoryLayer: No prior context found for this session.\n\n" + guidance;
  }

  return sections.join("\n\n");
}

/**
 * Format guidance for storing memories after tool use
 */
export function formatStorageGuidance(toolName: string, isSignificant: boolean): string {
  if (!isSignificant) {
    return "";
  }

  const guidance: Record<string, string> = {
    Task: "Consider storing exploration/research findings with `memory_remember` (subtype: decision/problem/entity).",
    Bash: "If this was a significant git commit, test result, or build output, store with `memory_remember` (subtype: workflow).",
    Edit: "If this edit completes a milestone, store with `memory_remember` (type: working, tags: [active-file]).",
    Write: "If this file represents a significant deliverable, store with `memory_remember` (type: working, tags: [active-file]).",
  };

  return guidance[toolName] || "";
}