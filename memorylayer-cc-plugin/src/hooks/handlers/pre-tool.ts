/**
 * PreToolUse hook handler
 * Provides recall guidance before tool execution
 */

import type {HookInput, HookOutput} from "../types.js";
import {getClient, checkHealth} from "../client.js";
import {formatRecallResult} from "../formatters.js";
import {wasQueryRecalledThisTurn, markRecallDone, getCurrentTopic} from "../state.js";

/**
 * Handle PreToolUse for Task tool (subagent delegation)
 */
async function handleTaskTool(input: HookInput): Promise<HookOutput> {
    // Extract topic from Task prompt if available
    const taskPrompt = input.tool_input?.prompt as string | undefined;
    if (!taskPrompt) {
        return {
            success: true,
            additionalContext: "RECALL-FIRST RULE: Consider using `memory_recall` before delegating to subagent. Subagents cannot access MemoryLayer.",
        };
    }

    // Use first 100 chars of task prompt as query
    const query = taskPrompt.substring(0, 100);

    // Check if this specific query was already recalled (allow different queries)
    if (wasQueryRecalledThisTurn(query)) {
        return {
            success: true,
            additionalContext: "Recall already done for this topic. Include relevant memories in subagent prompt.",
        };
    }

    // Check server health
    const healthy = await checkHealth();
    if (!healthy) {
        return {success: true};
    }

    try {
        const client = getClient();
        const result = await client.recall({query, limit: 5});
        markRecallDone(query);

        if (result.memories.length === 0) {
            return {
                success: true,
                additionalContext: "No relevant memories found for this task. Proceeding with delegation.",
            };
        }

        const recallOutput = formatRecallResult(result, query);
        return {
            success: true,
            additionalContext: `INCLUDE IN SUBAGENT PROMPT - Relevant context from memory:\n\n${recallOutput}`,
        };
    } catch {
        return {
            success: true,
            additionalContext: "Memory recall failed. Consider manual recall before delegation.",
        };
    }
}

/**
 * Handle PreToolUse for Edit/Write tools
 */
async function handleEditWriteTool(input: HookInput): Promise<HookOutput> {
    // Only provide guidance for non-trivial edits
    const filePath = input.tool_input?.file_path as string | undefined;
    if (!filePath) {
        return {success: true};
    }

    // Build query from filename + user's current topic (if available)
    const filename = filePath.split("/").pop() || filePath;
    const topic = getCurrentTopic();
    const query = topic ? `${filename} ${topic}` : `${filename} patterns solutions`;

    // Skip if this specific query was already recalled
    if (wasQueryRecalledThisTurn(query)) {
        return {success: true};
    }

    // Check server health
    const healthy = await checkHealth();
    if (!healthy) {
        return {success: true};
    }

    try {
        const client = getClient();
        const result = await client.recall({query, limit: 3});

        if (result.memories.length === 0) {
            return {success: true};
        }

        markRecallDone(query);
        const recallOutput = formatRecallResult(result, filename);

        return {
            success: true,
            additionalContext: `Relevant context for ${filename}:\n\n${recallOutput}`,
        };
    } catch {
        return {success: true};
    }
}

/**
 * Handle PreToolUse event
 */
export async function handlePreToolUse(input: HookInput): Promise<HookOutput> {
    const toolName = input.tool_name;

    if (!toolName) {
        return {success: true};
    }

    switch (toolName) {
        case "Task":
            return handleTaskTool(input);

        case "Edit":
        case "Write":
            return handleEditWriteTool(input);

        default:
            return {success: true};
    }
}
