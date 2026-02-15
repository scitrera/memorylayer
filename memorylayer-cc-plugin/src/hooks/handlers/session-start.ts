/**
 * SessionStart hook handler
 * Retrieves briefing and relevant memories at session start
 */

import type { Memory } from "@scitrera/memorylayer-mcp-server";
import type {HookInput, HookOutput} from "../types.js";
import {getClient, checkHealth} from "../client.js";
import {formatSessionStart} from "../formatters.js";
import {markRecallDone} from "../state.js";

/**
 * Extract a topic from the user's first message in transcript
 */
function extractTopic(input: HookInput): string | undefined {
    if (!input.transcript || input.transcript.length === 0) {
        return undefined;
    }

    // Find first user messageno
    const userMsg = input.transcript.find(m => m.role === "user");
    if (!userMsg) {
        return undefined;
    }

    // Use first 100 chars as topic query (truncate for API efficiency)
    const content = userMsg.content.trim();
    if (content.length > 100) {
        return content.substring(0, 100);
    }
    return content || undefined;
}

/**
 * Check for existing sandbox state from a prior session or pre-compaction.
 * Returns the inspect result if a sandbox exists, null otherwise.
 */
async function checkSandboxState(client: ReturnType<typeof getClient>): Promise<Record<string, unknown> | null> {
    try {
        const status = await client.contextStatus() as { exists?: boolean; variable_count?: number };
        if (status.exists && (status.variable_count ?? 0) > 0) {
            // Sandbox has variables — fetch the overview
            return await client.contextInspect({});
        }
    } catch {
        // Context environment may not be available, that's fine
    }
    return null;
}

/**
 * Handle SessionStart event
 */
export async function handleSessionStart(input: HookInput): Promise<HookOutput> {
    // Check if server is reachable
    const healthy = await checkHealth();
    if (!healthy) {
        return {
            success: true,
            additionalContext: "MemoryLayer server not reachable. Memory features unavailable this session.",
        };
    }

    try {
        const client = getClient();

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

        // If there's a topic in the transcript, recall for it too
        const topic = extractTopic(input);
        let topicRecall = null;

        if (topic) {
            try {
                topicRecall = await client.recall({query: topic, limit: 10, detail_level: "abstract",});
                markRecallDone(topic);
            } catch {
                // Topic recall is optional, continue without it
            }
        }

        // Format the combined output
        const context = formatSessionStart(briefing, directives, topicRecall, topic, sandboxState);

        return {
            success: true,
            additionalContext: context,
        };
    } catch (error) {
        return {
            success: false,
            error: `SessionStart handler error: ${error instanceof Error ? error.message : error}`,
        };
    }
}
