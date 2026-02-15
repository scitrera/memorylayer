#!/usr/bin/env node
// noinspection ES6PreferShortImport

/**
 * CLI entry point for MemoryLayer Claude Code hooks
 *
 * Usage:
 *   memorylayer-hook <hook-type>
 *
 * Reads HookInput JSON from stdin, writes HookOutput JSON to stdout.
 *
 * Hook types:
 *   SessionStart     - Called at session start
 *   UserPromptSubmit - Called when user submits a prompt
 *   PreToolUse       - Called before a tool is used
 *   PostToolUse      - Called after a tool is used
 *   PreCompact       - Called before context compaction
 *   Stop             - Called when session ends
 */

import {appendFileSync, readFileSync, existsSync, unlinkSync} from "fs";
import {join} from "path";
import {tmpdir} from "os";
import {HookEvent, type HookInput, type HookOutput} from "../src/hooks/types.js";
import {handleSessionStart} from "../src/hooks/handlers/session-start.js";
import {handleUserPromptSubmit} from "../src/hooks/handlers/user-prompt.js";
import {handlePreToolUse} from "../src/hooks/handlers/pre-tool.js";
import {handlePostToolUse} from "../src/hooks/handlers/post-tool.js";
import {handleStop} from "../src/hooks/handlers/stop.js";
import {resetRecallStatus, updateSessionInfo} from "../src/hooks/state.js";
import {getClient} from "../src/hooks/client.js";

/**
 * Read all input from stdin
 */
async function readStdin(): Promise<string> {
    const chunks: Buffer[] = [];

    return new Promise((resolve, reject) => {
        process.stdin.on("data", (chunk) => chunks.push(chunk));
        process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
        process.stdin.on("error", reject);

        // Timeout after 5 seconds if no input
        setTimeout(() => {
            if (chunks.length === 0) {
                resolve("{}");
            }
        }, 5000);
    });
}

/**
 * Parse hook type from command line
 */
function parseHookType(arg: string | undefined): HookEvent | null {
    if (!arg) return null;

    const normalized = arg.toLowerCase();

    switch (normalized) {
        case "sessionstart":
            return HookEvent.SessionStart;
        case "userpromptsubmit":
            return HookEvent.UserPromptSubmit;
        case "pretooluse":
            return HookEvent.PreToolUse;
        case "posttooluse":
            return HookEvent.PostToolUse;
        case "precompact":
            return HookEvent.PreCompact;
        case "stop":
            return HookEvent.Stop;
        default:
            return null;
    }
}

/** Hooks that support hookSpecificOutput with additionalContext */
const HOOKS_WITH_ADDITIONAL_CONTEXT = new Set([
    HookEvent.SessionStart,  // Works empirically even though not in schema
    HookEvent.PreToolUse,
    HookEvent.PostToolUse,
    HookEvent.UserPromptSubmit,
]);

/**
 * Build properly formatted hook output
 */
function buildOutput(
    hookType: HookEvent,
    additionalContext?: string,
    block = false,
    reason?: string
): Record<string, unknown> {
    const output: Record<string, unknown> = {
        continue: !block,  // Always continue unless blocking
    };

    // For PreToolUse, we can block
    if (hookType === HookEvent.PreToolUse && block) {
        output.continue = false;
        output.decision = "block";
        output.reason = reason || "Blocked by MemoryLayer hook";
    }

    // Only add hookSpecificOutput for hooks that support it
    // SessionStart, Stop, PreCompact do NOT support hookSpecificOutput
    if (additionalContext && HOOKS_WITH_ADDITIONAL_CONTEXT.has(hookType)) {
        output.hookSpecificOutput = {
            hookEventName: hookType,
            additionalContext,
        };
    }

    return output;
}

/**
 * Build error output
 */
function buildErrorOutput(hookType: HookEvent, error: string): HookOutput {
    return {
        hookSpecificOutput: {
            hookEventName: hookType,
            additionalContext: `MemoryLayer hook error: ${error}`,
        },
    };
}

/** Internal result from handlers */
interface HandlerResult {
    additionalContext?: string;
    block?: boolean;
    reason?: string;
    error?: string;
}

/**
 * Convert legacy HookOutput to HandlerResult
 */
function legacyToResult(output: HookOutput): HandlerResult {
    // Handle new format (has hookSpecificOutput)
    if (output.hookSpecificOutput) {
        return {
            additionalContext: output.hookSpecificOutput.additionalContext,
            block: output.decision === "block",
            reason: output.reason,
        };
    }
    // Handle legacy format (has additionalContext at top level)
    return {
        additionalContext: (output as any).additionalContext,
        block: (output as any).block,
        reason: (output as any).blockReason,
        error: output.error,
    };
}

/**
 * Dispatch to appropriate handler
 */
async function dispatch(hookType: HookEvent, input: HookInput): Promise<HandlerResult> {
    switch (hookType) {
        case HookEvent.SessionStart: {
            // Reset recall status at session start
            resetRecallStatus();

            const envFile = process.env.CLAUDE_ENV_FILE;

            // Try to read the session from the MCP server's handoff file
            const handoffDir = join(tmpdir(), "memorylayer");
            const handoffFile = join(handoffDir, `session-${process.ppid}.json`);

            let sessionId: string | undefined;
            let workspaceId: string | undefined;

            try {
                if (existsSync(handoffFile)) {
                    const data = JSON.parse(readFileSync(handoffFile, "utf-8"));
                    sessionId = data.sessionId;
                    workspaceId = data.workspaceId;
                    // Clean up the handoff file
                    unlinkSync(handoffFile);
                }
            } catch {
                // Handoff file not available, fall through
            }

            // Fallback: if no handoff file, create session directly (backward compat)
            if (!sessionId) {
                try {
                    const client = getClient();
                    workspaceId = client.getWorkspaceId();
                    const sessionResult = await client.startSession({ttl_seconds: 3600});
                    sessionId = sessionResult.session_id;
                    console.error(`[session-start] created hook session=${sessionId} workspace=${workspaceId} (direct, no handoff)`);
                } catch {
                    // Ignore - memory features will still work via MCP tools
                    console.error("[session-start] failed to create hook session (server unreachable?)");
                }
            } else {
                console.error(`[session-start] got hook session=${sessionId} workspace=${workspaceId} (from handoff file)`);
            }

            // Write to CLAUDE_ENV_FILE so subsequent hook processes get the session.
            // Each Claude Code session has its own env file, providing session isolation.
            if (envFile) {
                try {
                    if (workspaceId) {
                        appendFileSync(envFile, `export MEMORYLAYER_WORKSPACE_ID="${workspaceId}"\n`);
                    }
                    if (sessionId) {
                        appendFileSync(envFile, `export MEMORYLAYER_SESSION_ID="${sessionId}"\n`);
                    }
                    console.error(`[session-start] wrote to CLAUDE_ENV_FILE=${envFile} sessionId=${sessionId || "(none)"} workspaceId=${workspaceId || "(none)"}`);
                } catch (err) {
                    console.error(`[session-start] failed to write CLAUDE_ENV_FILE=${envFile}:`, err);
                }
            } else {
                console.error("[session-start] CLAUDE_ENV_FILE not set -- env propagation unavailable, falling back to hook-state.json only");
            }

            // Save to hook state as fallback (singleton -- will be superseded by env propagation)
            if (workspaceId) {
                updateSessionInfo(workspaceId, sessionId);
            }

            return legacyToResult(await handleSessionStart(input));
        }

        case HookEvent.UserPromptSubmit:
            // Reset recall status for new user turn
            resetRecallStatus();
            return legacyToResult(await handleUserPromptSubmit(input));

        case HookEvent.PreToolUse:
            return legacyToResult(await handlePreToolUse(input));

        case HookEvent.PostToolUse:
            return legacyToResult(await handlePostToolUse(input));

        case HookEvent.PreCompact:
            // PreCompact: Can't inject instructions at this point, just continue
            return {};

        case HookEvent.Stop:
            // Stop: commit working memory and end session (side effects only, no context injection)
            return legacyToResult(await handleStop());

        default:
            return {};
    }
}

/**
 * Main entry point
 */
async function main() {
    const hookTypeArg = process.argv[2];
    const hookType = parseHookType(hookTypeArg);

    if (!hookType) {
        const output: HookOutput = {
            error: `Invalid or missing hook type. Usage: memorylayer-hook <hook-type>\nValid types: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PreCompact, Stop`,
        };
        console.log(JSON.stringify(output));
        process.exit(1);
    }

    try {
        // Read input from stdin
        const stdinData = await readStdin();
        let input: HookInput;

        try {
            input = stdinData.trim() ? JSON.parse(stdinData) as HookInput : {hook_type: hookType};
        } catch {
            input = {hook_type: hookType};
        }

        // Add hook type to input
        input.hook_type = hookType;

        // Dispatch to handler
        const result = await dispatch(hookType, input);

        // Handle errors from handlers
        if (result.error) {
            const output = buildErrorOutput(hookType, result.error);
            console.log(JSON.stringify(output));
            return;
        }

        // Build properly formatted output
        const output = buildOutput(hookType, result.additionalContext, result.block, result.reason);

        // Write output to stdout
        console.log(JSON.stringify(output));

    } catch (error) {
        const output = buildErrorOutput(hookType, error instanceof Error ? error.message : String(error));
        console.log(JSON.stringify(output));
        process.exit(1);
    }
}

// Run
main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
