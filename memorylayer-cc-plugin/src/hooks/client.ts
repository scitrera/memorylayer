/**
 * Hook client - provides access to the MemoryLayerClient for hook operations.
 *
 * Hooks use the exact same MemoryLayerClient as MCP tools, just with a shorter
 * timeout since hooks need to respond quickly.
 */

import { MemoryLayerClient, detectWorkspaceId } from "@scitrera/memorylayer-mcp-server";
import { resolveSessionId } from "./state.js";

/** Singleton client instance for hooks */
let clientInstance: MemoryLayerClient | null = null;

/**
 * Get or create the singleton MemoryLayerClient instance.
 * This is the same client class used by MCP tools.
 *
 * On each call, syncs the session ID via resolveSessionId() so that
 * hooks running after SessionStart send the X-Session-ID header for
 * correct workspace resolution on the server.
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

  // Sync session ID on every call — resolveSessionId prefers env (cheap)
  // and only falls back to hook-state.json disk read if env is unset.
  const sessionId = resolveSessionId("client");
  if (sessionId && clientInstance.getSessionId() !== sessionId) {
    clientInstance.setSessionId(sessionId);
  }

  return clientInstance;
}

/**
 * Check if the MemoryLayer server is reachable
 */
export async function checkHealth(): Promise<boolean> {
  try {
    await getClient().getBriefing({ limit: 1, includeMemories: false });
    return true;
  } catch {
    return false;
  }
}
