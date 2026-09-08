/**
 * Memory capability registration for OpenClaw.
 *
 * OpenClaw's memory model expects four pieces (see
 * `openclaw/plugin-sdk/memory-host-core`):
 *   - `runtime`            — supplies a `MemorySearchManager` that backs
 *                            `memory_search` / `memory_get` tool calls and
 *                            the backend config used by the flush process.
 *   - `flushPlanResolver`  — returns config telling OpenClaw when / how to
 *                            compact the transcript into memories.
 *   - `promptBuilder`      — emits static prompt fragments that teach the
 *                            model how to use memory tools.
 *   - `publicArtifacts`    — surfaces stored memories in UI artifacts.
 *
 * Memory search and reads are supplied by the MemoryLayer-backed manager.
 */

import type {
  MemoryPluginCapability,
  MemoryPluginPublicArtifact,
} from "openclaw/plugin-sdk/memory-host-core";

import type { MemoryPluginRuntime } from "./host-types.js";

import type { ConfiguredMemoryLayer } from "./client.js";
import type { PluginConfig } from "./config.js";
import { buildMemorySearchManager } from "./search-manager.js";

export interface BuildMemoryCapabilityParams {
  getConnection: () => Promise<ConfiguredMemoryLayer>;
  config: PluginConfig;
}

export function buildMemoryCapability(
  params: BuildMemoryCapabilityParams,
): MemoryPluginCapability {
  return {
    runtime: buildRuntime(params),
    flushPlanResolver: () => null,
    promptBuilder: () => [],
    publicArtifacts: {
      listArtifacts: async () => [] satisfies MemoryPluginPublicArtifact[],
    },
  };
}

function buildRuntime(params: BuildMemoryCapabilityParams): MemoryPluginRuntime {
  // Per-agent MemorySearchManager cache: same agent gets the same manager
  // back across calls; agent isolation is preserved by keying on agentId.
  const managerCache = new Map<string, ReturnType<typeof buildMemorySearchManager>>();

  return {
    getMemorySearchManager: async ({ agentId }) => {
      let manager = managerCache.get(agentId);
      if (manager === undefined) {
        manager = buildMemorySearchManager(params.getConnection, params.config);
        managerCache.set(agentId, manager);
      }
      return { manager };
    },
    resolveMemoryBackendConfig: () => ({
      backend: "builtin" as const,
    }),
    closeMemorySearchManager: async ({ agentId }) => {
      const manager = managerCache.get(agentId);
      if (manager === undefined) return;
      managerCache.delete(agentId);
      await manager.close?.();
    },
    closeAllMemorySearchManagers: async () => {
      const managers = Array.from(managerCache.values());
      managerCache.clear();
      await Promise.all(managers.map((m) => m.close?.()));
    },
  };
}

// Keep references reachable so future MemorySearchManager / publicArtifacts
// implementations don't have to refactor the build signature.
export type { ConfiguredMemoryLayer, PluginConfig };
