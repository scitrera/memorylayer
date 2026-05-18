/**
 * Type definitions for MemoryLayer OpenCode plugin.
 *
 * OpenCode plugins receive a PluginInput context and return a Hooks object.
 * Hooks follow an (input, output) => Promise<void> pattern where you mutate
 * the output object to inject changes.
 */

import type { Memory, RecallResult, ToolResponse } from "@scitrera/memorylayer-mcp-server";

// Re-export MCP server types used by formatters
export type { Memory, RecallResult, ToolResponse };

// ---------------------------------------------------------------------------
// OpenCode Plugin Types (based on @opencode-ai/plugin v1.3.x)
// Defined here to avoid hard dependency on @opencode-ai/plugin at runtime.
// ---------------------------------------------------------------------------

/** Context provided to the plugin server function */
export interface PluginInput {
  /** OpenCode SDK client */
  client: unknown;
  /** Project metadata */
  project: Record<string, unknown>;
  /** Current working directory */
  directory: string;
  /** Git worktree root */
  worktree: string;
  /** OpenCode server URL */
  serverUrl: URL;
  /** Bun shell instance */
  $: unknown;
}

/** Optional plugin configuration from opencode.json */
export type PluginOptions = Record<string, unknown>;

/** Message part for context injection */
export interface Part {
  type: "text";
  text: string;
}

/** Model descriptor */
export interface Model {
  providerID: string;
  modelID: string;
}

/**
 * Hook definitions returned by the plugin.
 * Only the hooks we actually implement are typed here.
 */
export interface MemoryLayerHooks {
  /** Listen to OpenCode events (session lifecycle, etc.) */
  event?: (input: { event: OpenCodeEvent }) => Promise<void>;

  /** Modify the system prompt to inject memory context */
  "experimental.chat.system.transform"?: (
    input: { sessionID?: string; model: Model },
    output: { system: string[] },
  ) => Promise<void>;

  /** Hook into user messages to perform recall */
  "chat.message"?: (
    input: {
      sessionID: string;
      agent?: string;
      model?: { providerID: string; modelID: string };
      messageID?: string;
      variant?: string;
    },
    output: { message: UserMessage; parts: Part[] },
  ) => Promise<void>;

  /** Modify tool arguments before execution */
  "tool.execute.before"?: (
    input: { tool: string; sessionID: string; callID: string },
    output: { args: Record<string, unknown> },
  ) => Promise<void>;

  /** Capture tool output after execution */
  "tool.execute.after"?: (
    input: { tool: string; sessionID: string; callID: string; args: Record<string, unknown> },
    output: { title: string; output: string; metadata: Record<string, unknown> },
  ) => Promise<void>;

  /** Inject context before context compaction */
  "experimental.session.compacting"?: (
    input: { sessionID: string },
    output: { context: string[]; prompt?: string },
  ) => Promise<void>;

  /** Inject environment variables into shell commands */
  "shell.env"?: (
    input: { cwd: string; sessionID?: string; callID?: string },
    output: { env: Record<string, string> },
  ) => Promise<void>;

  /** Register custom tools */
  tool?: Record<string, unknown>;

  /** Intercept slash commands */
  "command.execute.before"?: (
    input: { command: string; sessionID: string; arguments: string },
    output: { parts: Part[] },
  ) => Promise<void>;
}

/** OpenCode event structure */
export interface OpenCodeEvent {
  type: string;
  properties: Record<string, unknown>;
}

/** User message structure */
export interface UserMessage {
  role: "user";
  parts: Part[];
}

// ---------------------------------------------------------------------------
// Hook State (shared with CC plugin pattern)
// ---------------------------------------------------------------------------

/** State tracked between hook invocations */
export interface HookState {
  /** Current session ID */
  sessionId?: string;
  /** Workspace ID being used */
  workspaceId?: string;
  /** Whether recall has been done this conversation turn */
  recallDoneThisTurn: boolean;
  /** Timestamp of last recall */
  lastRecallAt?: string;
  /** Query used for last recall */
  lastRecallQuery?: string;
  /** Queries already performed this turn (for query-aware dedup) */
  recallQueriesThisTurn?: string[];
  /** User's current topic from most recent prompt */
  currentTopic?: string;
  /** User's current prompt text for intent detection */
  currentPrompt?: string;
}
