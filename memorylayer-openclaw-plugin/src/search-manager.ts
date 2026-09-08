/**
 * MemorySearchManager backed by MemoryLayer.ai.
 *
 * OpenClaw calls into this for `memory_search` tool invocations and other
 * memory queries. We translate to MemoryLayer's `recall()` / `getMemory()`
 * and map results into the SDK's expected shape.
 *
 * Synthetic paths
 * ---------------
 * MemoryLayer memories don't have a filesystem-style path; the OpenClaw
 * `MemorySearchResult` shape requires one. We emit `memorylayer://{id}` as
 * the stable identifier — `readFile()` recognises that prefix and rehydrates
 * via `getMemory()`.
 */

import type {
  MemoryEmbeddingProbeResult,
  MemoryProviderStatus,
  MemoryReadResult,
  MemorySearchManager,
  MemorySearchResult,
} from "./host-types.js";
import type { Memory } from "@scitrera/memorylayer-sdk";

import type { ConfiguredMemoryLayer } from "./client.js";
import type { PluginConfig } from "./config.js";

const PATH_PREFIX = "memorylayer://";
const PROVIDER_NAME = "memorylayer";

export function buildMemorySearchManager(
  getConnection: () => Promise<ConfiguredMemoryLayer>,
  config: PluginConfig,
): MemorySearchManager {
  return {
    async search(query, opts) {
      const { client } = await getConnection();
      const result = await client.recall(query, {
        limit: opts?.maxResults,
        minRelevance: opts?.minScore,
        workspaceId: config.defaultWorkspaceId,
      });
      return result.memories.map(memoryToSearchResult);
    },

    async readFile({ relPath }) {
      if (!relPath.startsWith(PATH_PREFIX)) {
        // OpenClaw may pass real workspace paths for the file-backed memory
        // model — workclaw stores everything in MemoryLayer, so an unknown
        // path is not an error; we just have nothing to return.
        return { text: "", path: relPath } satisfies MemoryReadResult;
      }
      const memoryId = relPath.slice(PATH_PREFIX.length);
      const { client } = await getConnection();
      const memory = await client.getMemory(memoryId);
      return { text: memory.content, path: relPath } satisfies MemoryReadResult;
    },

    status() {
      return {
        backend: "builtin",
        provider: PROVIDER_NAME,
        workspaceDir: config.defaultWorkspaceId ?? "",
        sources: ["memory"],
      } satisfies MemoryProviderStatus;
    },

    async probeEmbeddingAvailability(): Promise<MemoryEmbeddingProbeResult> {
      // Embeddings are managed server-side by MemoryLayer. We can't usefully
      // distinguish "no embedding model installed" from "transport down" from
      // here, so we report ok and let real recall calls surface errors.
      return { ok: true, checked: true };
    },

    async probeVectorAvailability() {
      return true;
    },

    async close() {
      const { aether } = await getConnection();
      // AetherClient exposes a disconnect/close API — guard with optional
      // chaining since some flows may have already torn it down.
      const maybeClose = (aether as unknown as { close?: () => Promise<void> | void }).close;
      if (typeof maybeClose === "function") {
        await maybeClose.call(aether);
      }
    },
  };
}

function memoryToSearchResult(memory: Memory): MemorySearchResult {
  const lineCount = countLines(memory.content);
  return {
    path: `${PATH_PREFIX}${memory.id}`,
    startLine: 1,
    endLine: Math.max(1, lineCount),
    // TODO(phase3): MemoryLayer doesn't yet surface per-recall relevance in
    // the typed Memory shape — use `importance` as a stable-but-static proxy
    // until the SDK exposes the actual recall score.
    score: memory.importance,
    snippet: memory.content,
    source: "memory",
    citation: memory.id,
  };
}

function countLines(text: string): number {
  if (!text) return 0;
  let count = 1;
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) === 10 /* \n */) count++;
  }
  return count;
}
