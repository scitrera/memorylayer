/**
 * Hook client - provides access to the MemoryLayerClient for hook operations.
 *
 * Hooks use the exact same MemoryLayerClient as MCP tools, just with a shorter
 * timeout since hooks need to respond quickly.
 */

import { MemoryLayerClient } from "../client.js";
import { detectWorkspaceId } from "../workspace.js";

/** Singleton client instance for hooks */
let clientInstance: MemoryLayerClient | null = null;

/**
 * Get or create the singleton MemoryLayerClient instance.
 * This is the same client class used by MCP tools.
 */
export function getClient(): MemoryLayerClient {
  if (!clientInstance) {
    let workspaceId = process.env.MEMORYLAYER_WORKSPACE_ID;
    if (!workspaceId) {
      try {
        workspaceId = detectWorkspaceId();
      } catch {
        workspaceId = "_default";
      }
    }

    clientInstance = new MemoryLayerClient({
      baseUrl: process.env.MEMORYLAYER_URL,
      apiKey: process.env.MEMORYLAYER_API_KEY,
      workspaceId,
      timeout: 5000, // Shorter timeout for hooks
    });
  }
  return clientInstance;
}

/**
 * Check if the MemoryLayer server is reachable
 */
export async function checkHealth(): Promise<boolean> {
  try {
    await getClient().getBriefing(false);
    return true;
  } catch {
    return false;
  }
}