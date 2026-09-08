import { describe, it, expect, vi } from "vitest";

import { buildMemorySearchManager } from "../src/search-manager.js";
import type { ConfiguredMemoryLayer } from "../src/client.js";
import type { PluginConfig } from "../src/config.js";

const baseConfig: PluginConfig = {
  aetherAddress: "localhost:50099",
  aetherWorkspace: "ws-prod",
  aetherImplementation: "openclaw-plugin",
  aetherSpecifier: "default",
  memorylayerTargetTopic: "sv::memorylayer::default",
  memorylayerBaseUrl: "http://memorylayer.aether-relay",
  memorylayerTimeoutMs: 30_000,
  tenantId: "acme",
  principalType: "user",
  principalId: "alice",
  defaultWorkspaceId: "ws-1",
  skillsSyncEnabled: false,
};

function buildFakeConnection(stub: {
  recall?: (query: string, opts?: unknown) => Promise<unknown>;
  getMemory?: (id: string) => Promise<unknown>;
}): () => Promise<ConfiguredMemoryLayer> {
  const client = {
    recall: stub.recall ?? vi.fn(),
    getMemory: stub.getMemory ?? vi.fn(),
  };
  const conn = { client, aether: {} } as unknown as ConfiguredMemoryLayer;
  return async () => conn;
}

describe("MemorySearchManager", () => {
  it("maps recall result memories to MemorySearchResult shape", async () => {
    const recall = vi.fn().mockResolvedValue({
      memories: [
        {
          id: "mem-1",
          content: "first memory line\nsecond line",
          importance: 0.82,
        },
        {
          id: "mem-2",
          content: "one-liner",
          importance: 0.41,
        },
      ],
    });
    const sm = buildMemorySearchManager(
      buildFakeConnection({ recall }),
      baseConfig,
    );

    const results = await sm.search("what does alice prefer?", { maxResults: 5 });

    expect(recall).toHaveBeenCalledWith("what does alice prefer?", {
      limit: 5,
      minRelevance: undefined,
      workspaceId: "ws-1",
    });
    expect(results).toHaveLength(2);
    expect(results[0]).toEqual({
      path: "memorylayer://mem-1",
      startLine: 1,
      endLine: 2,
      score: 0.82,
      snippet: "first memory line\nsecond line",
      source: "memory",
      citation: "mem-1",
    });
    expect(results[1]?.endLine).toBe(1);
    expect(results[1]?.path).toBe("memorylayer://mem-2");
  });

  it("forwards minScore as minRelevance", async () => {
    const recall = vi.fn().mockResolvedValue({ memories: [] });
    const sm = buildMemorySearchManager(buildFakeConnection({ recall }), baseConfig);
    await sm.search("q", { minScore: 0.7 });
    expect(recall).toHaveBeenCalledWith("q", expect.objectContaining({ minRelevance: 0.7 }));
  });

  it("readFile rehydrates a memorylayer:// path via getMemory", async () => {
    const getMemory = vi.fn().mockResolvedValue({ id: "mem-1", content: "the body" });
    const sm = buildMemorySearchManager(buildFakeConnection({ getMemory }), baseConfig);
    const out = await sm.readFile({ relPath: "memorylayer://mem-1" });
    expect(getMemory).toHaveBeenCalledWith("mem-1");
    expect(out).toEqual({ text: "the body", path: "memorylayer://mem-1" });
  });

  it("readFile returns empty for unknown path prefix", async () => {
    const getMemory = vi.fn();
    const sm = buildMemorySearchManager(buildFakeConnection({ getMemory }), baseConfig);
    const out = await sm.readFile({ relPath: "/some/other/path.md" });
    expect(getMemory).not.toHaveBeenCalled();
    expect(out).toEqual({ text: "", path: "/some/other/path.md" });
  });

  it("status reports builtin/memorylayer backend with default workspace", () => {
    const sm = buildMemorySearchManager(buildFakeConnection({}), baseConfig);
    const status = sm.status();
    expect(status.backend).toBe("builtin");
    expect(status.provider).toBe("memorylayer");
    expect(status.workspaceDir).toBe("ws-1");
    expect(status.sources).toEqual(["memory"]);
  });

  it("probe methods return optimistic availability", async () => {
    const sm = buildMemorySearchManager(buildFakeConnection({}), baseConfig);
    expect(await sm.probeEmbeddingAvailability()).toEqual({ ok: true, checked: true });
    expect(await sm.probeVectorAvailability()).toBe(true);
  });
});
