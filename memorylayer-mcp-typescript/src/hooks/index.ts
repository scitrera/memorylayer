/**
 * MemoryLayer Claude Code hooks
 *
 * This module provides TypeScript-based hooks for Claude Code integration.
 * Hooks inject memory context into Claude's processing at key points,
 * using the same MemoryLayerClient as the MCP tools.
 */

// Types
export {
  HookEvent,
  type HookInput,
  type HookOutput,
  type TranscriptMessage,
  type HookState,
} from "./types.js";

// State management
export {
  readHookState,
  writeHookState,
  markRecallDone,
  wasRecallDoneThisTurn,
  resetRecallStatus,
  updateSessionInfo,
  getWorkspaceId,
  getSessionId,
} from "./state.js";

// Client (same MemoryLayerClient used by MCP tools)
export {
  getClient,
  checkHealth,
} from "./client.js";

// Formatters
export {
  formatRecallResult,
  formatBriefing,
  formatDirectives,
  formatSessionStart,
  formatSandboxState,
  formatStorageGuidance,
} from "./formatters.js";

// Handlers
export { handleSessionStart } from "./handlers/session-start.js";
export { handleUserPromptSubmit } from "./handlers/user-prompt.js";
export { handlePreToolUse } from "./handlers/pre-tool.js";
export { handlePostToolUse } from "./handlers/post-tool.js";
export { handleStop } from "./handlers/stop.js";