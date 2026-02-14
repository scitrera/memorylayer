/**
 * Main exports for @scitrera/memorylayer-mcp-server
 */

export { MCPServer, createServer } from "./server.js";
export type { MCPServerOptions } from "./server.js";
export { MemoryLayerClient } from "./client.js";
export type { ClientOptions } from "./client.js";
export { MCPToolHandlers } from "./handlers.js";
export {
  TOOLS,
  SESSION_TOOLS,
  TOOL_PROFILES,
  TOOL_NAMES,
  DEFAULT_PROFILE,
  getToolsForProfile,
  isToolEnabled,
  // Legacy exports (deprecated)
  CORE_TOOLS,
  EXTENDED_TOOLS,
} from "./tools.js";
export type { ToolProfile, ToolDefinition } from "./tools.js";
export { SessionManager } from "./session.js";
export type { SessionState, SessionConfig, WorkingMemoryEntry } from "./session.js";
export * from "./types.js";
export { detectWorkspaceId } from "./workspace.js";
