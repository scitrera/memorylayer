/**
 * MemoryLayer OpenCode Plugin
 *
 * Provides persistent memory for OpenCode sessions via hooks that:
 * - Inject workspace briefing and directives at session start
 * - Recall relevant memories when users ask questions
 * - Capture tool observations as working memory
 * - Commit working memory before context compaction
 * - Clean up sessions on exit
 *
 * The plugin works alongside the MemoryLayer MCP server which provides
 * the full suite of 21+ memory tools to the LLM.
 *
 * @module @scitrera/memorylayer-opencode-plugin
 */

import type { MemoryLayerHooks, PluginInput, PluginOptions } from "./shared/types.js";
import { setPluginDirectory } from "./shared/client.js";
import { initializeSession, finalizeSession } from "./hooks/session.js";
import { handleUserMessage, extractMessageText } from "./hooks/message.js";
import { handleToolBefore, handleToolAfter } from "./hooks/tool.js";
import { handleCompacting } from "./hooks/event.js";

// Re-export types for downstream consumers (e.g., enterprise plugins)
export type { MemoryLayerHooks, PluginInput, PluginOptions, Part, Model, HookState } from "./shared/types.js";

/** Track whether session has been initialized */
let sessionInitialized = false;

/**
 * MemoryLayer plugin for OpenCode.
 *
 * Usage in opencode.json:
 * ```json
 * {
 *   "plugin": ["@scitrera/memorylayer-opencode-plugin"]
 * }
 * ```
 */
export default async function memorylayerPlugin(
  ctx: PluginInput,
  _options?: PluginOptions
): Promise<MemoryLayerHooks> {
  // Initialize client with workspace detection from plugin context
  setPluginDirectory(ctx.worktree || ctx.directory);

  const hooks: MemoryLayerHooks = {
    /**
     * System prompt transform — inject MemoryLayer context on first interaction.
     *
     * This hook modifies the system prompt to include workspace briefing,
     * directives, and session guidance. It runs once per session (on first call)
     * and injects the formatted context into the system prompt array.
     */
    "experimental.chat.system.transform": async (_input, output) => {
      if (!sessionInitialized) {
        sessionInitialized = true;
        const context = await initializeSession();
        if (context) {
          output.system.push(context);
        }
      }
    },

    /**
     * User message hook — detect patterns and recall relevant memories.
     *
     * When a user sends a message matching known patterns (preference questions,
     * recall requests, implementation tasks, error reports), this hook performs
     * a targeted recall and injects the results as additional message parts.
     */
    "chat.message": async (_input, output) => {
      const messageText = extractMessageText(output.parts);
      if (!messageText) return;

      const context = await handleUserMessage(messageText);
      if (context) {
        output.parts.push({
          type: "text",
          text: `\n\n<memory-context>\n${context}\n</memory-context>`,
        });
      }
    },

    /**
     * Pre-tool hook — inject relevant context before tool execution.
     *
     * For write/edit tools: recalls context relevant to the file being modified.
     * For task/delegation tools: recalls and suggests including context in subagent prompts.
     */
    "tool.execute.before": async (input, output) => {
      const context = await handleToolBefore(input.tool, output.args);
      if (context) {
        // Inject context as metadata that the LLM can see
        // OpenCode passes args to the tool — we add a _memorylayer_context field
        // that tools can optionally use, and it appears in the tool call metadata
        (output.args as Record<string, unknown>)._memorylayer_context = context;
      }
    },

    /**
     * Post-tool hook — capture tool observations as working memory.
     *
     * Silently captures structured observations (files read/modified, facts,
     * concepts, intent) and stores them as working memory. Fire-and-forget
     * to avoid blocking tool execution.
     *
     * For significant events (git commits, build errors), injects guidance
     * suggesting the user store important information.
     */
    "tool.execute.after": async (input, output) => {
      const guidance = await handleToolAfter(
        input.tool,
        input.args,
        output.output
      );
      if (guidance) {
        // Append guidance to tool output so the LLM sees it
        output.output = output.output + `\n\n${guidance}`;
      }
    },

    /**
     * Compaction hook — preserve memory state before context window is trimmed.
     *
     * Commits working memory to long-term storage and checkpoints the
     * server-side sandbox so state survives context compaction.
     */
    "experimental.session.compacting": async (input, output) => {
      const context = await handleCompacting(input.sessionID);
      output.context.push(...context);
    },

    /**
     * Shell environment hook — inject MemoryLayer env vars into shell commands.
     */
    "shell.env": async (_input, output) => {
      if (process.env.MEMORYLAYER_URL) {
        output.env.MEMORYLAYER_URL = process.env.MEMORYLAYER_URL;
      }
      if (process.env.MEMORYLAYER_API_KEY) {
        output.env.MEMORYLAYER_API_KEY = process.env.MEMORYLAYER_API_KEY;
      }
    },

    /**
     * Command hook — handle memorylayer slash commands.
     */
    "command.execute.before": async (input, output) => {
      const cmd = input.command;

      if (cmd === "memorylayer-remember") {
        output.parts.push({
          type: "text",
          text: `Use the \`memory_remember\` tool to store the following: ${input.arguments}\n\nAuto-detect appropriate type, subtype, importance, and tags from the content.`,
        });
      } else if (cmd === "memorylayer-recall") {
        output.parts.push({
          type: "text",
          text: `Use the \`memory_recall\` tool to search for: ${input.arguments}\n\nDisplay results with relevance scores and key metadata.`,
        });
      } else if (cmd === "memorylayer-status") {
        output.parts.push({
          type: "text",
          text: "Check MemoryLayer connection status: use `memory_briefing` to verify the MCP server is connected, then report server URL, workspace, memory statistics, and active session info.",
        });
      } else if (cmd === "memorylayer-setup") {
        output.parts.push({
          type: "text",
          text: [
            "Run MemoryLayer setup verification:",
            "1. Check server health (curl http://localhost:61001/health)",
            "2. Verify MCP tools are connected (call memory_briefing)",
            "3. Smoke test: store a test memory, recall it, then forget it",
            "4. Report: server URL, workspace, tool count, connection status",
            "",
            "If server is not running, suggest: pip install memorylayer-server && memorylayer serve",
          ].join("\n"),
        });
      }
    },
  };

  // Register cleanup on process exit
  const cleanup = () => {
    finalizeSession().catch(() => {});
  };
  process.on("beforeExit", cleanup);
  process.on("SIGTERM", cleanup);
  process.on("SIGINT", cleanup);

  return hooks;
}

// Also export the plugin as a PluginModule shape
export const id = "memorylayer";
export const server = memorylayerPlugin;
