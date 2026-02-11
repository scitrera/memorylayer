#!/usr/bin/env node
/**
 * CLI entry point for MemoryLayer MCP server
 *
 * Usage:
 *   memorylayer-mcp
 *
 * Environment variables:
 *   MEMORYLAYER_URL - Base URL for MemoryLayer API (default: http://localhost:61001)
 *   MEMORYLAYER_API_KEY - API key for authentication (optional)
 *   MEMORYLAYER_WORKSPACE_ID - Workspace ID (default: auto-detected from directory)
 *   MEMORYLAYER_AUTO_WORKSPACE - Set to "false" to disable auto-detection
 *   MEMORYLAYER_SESSION_MODE - Set to "false" to disable session/working memory (default: true)
 */

import { MemoryLayerClient } from "../src/client.js";
import { createServer } from "../src/server.js";
import { detectWorkspaceId } from "../src/workspace.js";

async function main() {
  // Parse environment variables
  const baseUrl = process.env.MEMORYLAYER_URL || "http://localhost:61001";
  const apiKey = process.env.MEMORYLAYER_API_KEY;
  const autoWorkspace = process.env.MEMORYLAYER_AUTO_WORKSPACE !== "false";
  const sessionMode = process.env.MEMORYLAYER_SESSION_MODE !== "false";

  // Log working directory for debugging workspace detection issues
  console.error(`MCP server starting in directory: ${process.cwd()}`);

  // Determine workspace ID
  let workspaceId: string;
  if (process.env.MEMORYLAYER_WORKSPACE_ID) {
    workspaceId = process.env.MEMORYLAYER_WORKSPACE_ID;
    console.error(`Using explicit workspace from env: ${workspaceId}`);
  } else if (autoWorkspace) {
    workspaceId = detectWorkspaceId();
    console.error(`Auto-detected workspace: ${workspaceId}`);
  } else {
    workspaceId = "default";
    console.error(`Using default workspace: ${workspaceId}`);
  }

  // Create client
  const client = new MemoryLayerClient({
    baseUrl,
    apiKey,
    workspaceId
  });

  // Create and run server
  const server = await createServer(client, { workspaceId, sessionMode });

  console.error("MemoryLayer MCP Server Manifest:", JSON.stringify(server.getManifest(), null, 2));

  await server.run();
}

// Handle errors and signals
process.on("SIGINT", () => {
  console.error("Received SIGINT, shutting down gracefully");
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.error("Received SIGTERM, shutting down gracefully");
  process.exit(0);
});

process.on("uncaughtException", (error) => {
  console.error("Uncaught exception:", error);
  process.exit(1);
});

process.on("unhandledRejection", (reason) => {
  console.error("Unhandled rejection:", reason);
  process.exit(1);
});

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
