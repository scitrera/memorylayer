import { beforeEach, describe, expect, it, vi } from "vitest";

import { MemoryLayerClient } from "../src/client.js";
import { RpgNamespace } from "../src/rpg.js";

global.fetch = vi.fn();

function okJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}

describe("RPG namespace", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
      workspaceId: "ws-123",
    });
  });

  it("is available and maps camelCase sync inputs to the API contract", async () => {
    (global.fetch as any).mockResolvedValueOnce(
      okJson({ nodes_created: 1, edges_created: 1, sync_time_ms: 2 }),
    );

    expect(client.rpg).toBeInstanceOf(RpgNamespace);
    await client.rpg.sync(
      [{
        nodeId: "src/main.ts",
        nodeType: "rpg_file",
        path: "src/main.ts",
        name: "main.ts",
        parentId: "src",
      }],
      [{ sourceId: "src", targetId: "src/main.ts", relationship: "contains" }],
      { fullSync: true, sourceCommit: "abc123", contextId: "rpg-task-42" },
    );

    const [url, init] = (global.fetch as any).mock.calls[0];
    expect(url).toBe("http://localhost:61001/v1/rpg/sync");
    const body = JSON.parse(init.body);
    expect(body.nodes[0]).toMatchObject({
      node_id: "src/main.ts",
      node_type: "rpg_file",
      parent_id: "src",
    });
    expect(body.edges[0]).toMatchObject({
      source_id: "src",
      target_id: "src/main.ts",
      relationship: "contains",
      strength: 1,
    });
    expect(body.full_sync).toBe(true);
    expect(body.context_id).toBe("rpg-task-42");
  });

  it("encodes subgraph filters as query parameters", async () => {
    (global.fetch as any).mockResolvedValueOnce(
      okJson({ nodes: [], edges: [], root_id: "src", depth: 2, total_nodes: 0, total_edges: 0 }),
    );

    await client.rpg.getSubgraph({
      nodeId: "src",
      depth: 2,
      nodeTypes: ["rpg_file", "rpg_class"],
      relationshipTypes: ["contains", "imports"],
    });

    const [rawUrl] = (global.fetch as any).mock.calls[0];
    const url = new URL(rawUrl);
    expect(url.pathname).toBe("/v1/rpg/subgraph");
    expect(url.searchParams.get("node_id")).toBe("src");
    expect(url.searchParams.get("node_types")).toBe("rpg_file,rpg_class");
    expect(url.searchParams.get("relationship_types")).toBe("contains,imports");
  });

  it("serializes delete requests and propagates OBO authority", async () => {
    (global.fetch as any)
      .mockResolvedValueOnce(okJson({ deleted: 1, not_found: 0 }))
      .mockResolvedValueOnce(okJson({ workspace_id: "ws-123", has_rpg: true }));

    await client.rpg.deleteNodes(["src/main.ts"], "rpg-task-42");
    const [, deleteInit] = (global.fetch as any).mock.calls[0];
    expect(JSON.parse(deleteInit.body)).toEqual({
      node_ids: ["src/main.ts"],
      context_id: "rpg-task-42",
    });

    await client.rpg.withAuthority({
      grantId: "g-rpg",
      subject: { type: "user", id: "alice" },
    }).getStatus();
    const [, statusInit] = (global.fetch as any).mock.calls[1];
    expect(statusInit.headers["X-Aether-Grant-ID"]).toBe("g-rpg");
    expect(statusInit.headers["X-Aether-Subject-ID"]).toBe("alice");
  });
});
