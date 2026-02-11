/**
 * PostToolUse hook handler
 * Provides guidance on storing significant results to memory
 */

import type { HookInput, HookOutput } from "../types.js";
import { formatStorageGuidance } from "../formatters.js";

/**
 * Check if Task output indicates significant findings
 */
function isSignificantTaskOutput(output: string): boolean {
  // Look for indicators of substantial analysis
  const significantPatterns = [
    /found|discovered|identified|analyzed|assessed/i,
    /architecture|pattern|structure|design/i,
    /problem|issue|gap|error|bug/i,
    /solution|fix|recommendation|suggestion/i,
    /files?:|components?:|services?:/i,
  ];

  return significantPatterns.some(p => p.test(output));
}

/**
 * Check if Bash output indicates significant action
 */
function isSignificantBashOutput(input: HookInput): boolean {
  const command = input.tool_input?.command as string | undefined;
  if (!command) return false;

  // Git commits
  if (/git\s+commit/i.test(command)) return true;

  // Test runs with failures or important results
  if (/npm\s+(run\s+)?test|pytest|cargo\s+test|go\s+test/i.test(command)) {
    const output = input.tool_output || "";
    // Only significant if there are failures or notable results
    return /fail|error|passed.*\d+/i.test(output);
  }

  // Build commands
  if (/npm\s+run\s+build|cargo\s+build|make\b|tsc\b/i.test(command)) {
    const output = input.tool_output || "";
    // Only significant if there were errors
    return /error|fail/i.test(output);
  }

  return false;
}

/**
 * Check if Edit/Write represents a significant milestone
 */
function isSignificantEdit(input: HookInput): boolean {
  const filePath = input.tool_input?.file_path as string | undefined;
  if (!filePath) return false;

  // New file creation (Write tool) is noteworthy
  if (input.tool_name === "Write") return true;

  // Config/manifest/CI files are significant
  const configPattern = /(package\.json|tsconfig[\w.]*\.json|\.env|docker-compose|Makefile|pyproject\.toml|Cargo\.toml|go\.(mod|sum)|\.github\/|\.gitlab-ci|Dockerfile)/i;
  if (configPattern.test(filePath)) return true;

  // Large edits (>10 net new lines added)
  const oldStr = input.tool_input?.old_string as string | undefined;
  const newStr = input.tool_input?.new_string as string | undefined;
  if (oldStr && newStr) {
    const addedLines = newStr.split("\n").length - oldStr.split("\n").length;
    if (addedLines > 10) return true;
  }

  return false;
}

/**
 * Handle PostToolUse for Task tool
 */
function handleTaskTool(input: HookInput): HookOutput {
  const output = input.tool_output || "";
  const agentType = input.tool_input?.subagent_type as string | undefined;

  // Exploration/research agents should always store findings
  const isExploreAgent = agentType && /explore|research|architect|analyst/i.test(agentType);

  if (isExploreAgent || isSignificantTaskOutput(output)) {
    return {
      success: true,
      additionalContext: `MANDATORY STORAGE: This ${isExploreAgent ? "exploration/research" : "task"} produced findings that should persist.\n\nUse \`memory_remember\` with:\n- importance: 0.7-0.9 based on significance\n- subtype: "decision" for assessments, "problem" for issues found, "entity" for inventories\n- tags: relevant to project/component`,
    };
  }

  return { success: true };
}

/**
 * Handle PostToolUse for Bash tool
 */
function handleBashTool(input: HookInput): HookOutput {
  if (isSignificantBashOutput(input)) {
    const command = input.tool_input?.command as string || "";

    if (/git\s+commit/i.test(command)) {
      return {
        success: true,
        additionalContext: formatStorageGuidance("Bash", true),
      };
    }

    if (/test/i.test(command)) {
      return {
        success: true,
        additionalContext: "Test run had notable results. Consider storing with `memory_remember` (type: working, subtype: workflow, tags: [test, result]).",
      };
    }

    if (/build|tsc|make/i.test(command)) {
      return {
        success: true,
        additionalContext: "Build had errors. Consider storing the issue with `memory_remember` (subtype: error) for future reference.",
      };
    }
  }

  return { success: true };
}

/**
 * Handle PostToolUse for Edit/Write tools
 */
function handleEditWriteTool(input: HookInput): HookOutput {
  if (isSignificantEdit(input)) {
    return {
      success: true,
      additionalContext: formatStorageGuidance("Edit", true),
    };
  }

  // Minimal guidance - don't clutter context for routine edits
  return { success: true };
}

/**
 * Handle PostToolUse event
 */
export async function handlePostToolUse(input: HookInput): Promise<HookOutput> {
  const toolName = input.tool_name;

  if (!toolName) {
    return { success: true };
  }

  switch (toolName) {
    case "Task":
      return handleTaskTool(input);

    case "Bash":
      return handleBashTool(input);

    case "Edit":
    case "Write":
      return handleEditWriteTool(input);

    default:
      return { success: true };
  }
}
