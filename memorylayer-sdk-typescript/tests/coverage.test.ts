import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import { EnterpriseRequiredError, NotFoundError } from "../src/errors.js";
import type {
  ApiToken,
  ApiTokenWithSecret,
  Entity,
  KBArticle,
  Knowledgebase,
} from "../src/types.js";

global.fetch = vi.fn();

function mockOkJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}
function mockOk204() {
  return { ok: true, status: 204, json: async () => undefined };
}
function mockErr(status: number, body: unknown = {}) {
  return { ok: false, status, json: async () => body, headers: { get: () => null } };
}

function lastCall(): [string, RequestInit] {
  const calls = (global.fetch as ReturnType<typeof vi.fn>).mock.calls;
  return calls[calls.length - 1] as [string, RequestInit];
}

const mockToken: ApiToken = {
  id: "tok-1",
  name: "ci-token",
  principal_type: "User",
  workspace_patterns: ["*"],
  scopes: ["*"],
  created_at: "2024-01-01T00:00:00Z",
  expires_at: null,
  revoked: false,
};

const mockEntity: Entity = {
  id: "ent-1",
  workspace_id: "ws-1",
  entity_type: "person",
  canonical_name: "Alice",
  normalized_name: "alice",
  aliases: ["Ali"],
  confidence: 1.0,
  provenance: {},
  representative_memory_id: null,
  status: "active",
  merged_into: null,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const mockMemory = {
  id: "mem-1",
  workspace_id: "ws-1",
  tenant_id: "t-1",
  context_id: "_default",
  content: "hello",
  content_hash: "h1",
  type: "episodic",
  importance: 0.5,
  tags: [],
  metadata: {},
  access_count: 0,
  decay_factor: 1.0,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const mockThread = {
  id: "thr-1",
  workspace_id: "ws-1",
  tenant_id: "t-1",
  context_id: "_default",
  metadata: {},
  message_count: 0,
  last_decomposed_index: 0,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

describe("Tokens SDK coverage", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("listTokens GET /v1/tokens", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ tokens: [mockToken] }));
    const tokens = await client.listTokens();
    expect(tokens).toHaveLength(1);
    expect(tokens[0].name).toBe("ci-token");
    const [url] = lastCall();
    expect(url).toContain("/v1/tokens");
  });

  it("listTokens passes include_revoked", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ tokens: [] }));
    await client.listTokens(true);
    expect(lastCall()[0]).toContain("include_revoked=true");
  });

  it("createToken POST /v1/tokens returns plaintext token", async () => {
    const created: ApiTokenWithSecret = { ...mockToken, token: "secret-plaintext" };
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson(created));
    const result = await client.createToken({ name: "ci-token", expiresInDays: 30 });
    expect(result.token).toBe("secret-plaintext");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/tokens");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.name).toBe("ci-token");
    expect(body.expires_in_days).toBe(30);
  });

  it("getToken GET /v1/tokens/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson(mockToken));
    const result = await client.getToken("tok-1");
    expect(result.id).toBe("tok-1");
    expect(lastCall()[0]).toContain("/v1/tokens/tok-1");
  });

  it("deleteToken DELETE /v1/tokens/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.deleteToken("tok-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/tokens/tok-1");
    expect(init.method).toBe("DELETE");
  });

  it("revokeToken POST /v1/tokens/:id/revoke", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.revokeToken("tok-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/tokens/tok-1/revoke");
    expect(init.method).toBe("POST");
  });
});

describe("listMemories (browse)", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("GET /v1/memories returns memories + total_count", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ memories: [mockMemory], total_count: 1 }),
    );
    const result = await client.listMemories();
    expect(result.total_count).toBe(1);
    expect(result.memories[0].id).toBe("mem-1");
    expect(lastCall()[0]).toContain("/v1/memories");
  });

  it("passes limit/offset/type/tag/context_id query params", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ memories: [], total_count: 0 }));
    await client.listMemories({ limit: 25, offset: 50, type: "semantic", tag: "x", contextId: "ctx-1" });
    const [url] = lastCall();
    expect(url).toContain("limit=25");
    expect(url).toContain("offset=50");
    expect(url).toContain("type=semantic");
    expect(url).toContain("tag=x");
    expect(url).toContain("context_id=ctx-1");
  });
});

describe("Entity registry ops", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("listEntities GET /v1/entities", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ entities: [mockEntity], total_count: 1 }));
    const result = await client.listEntities({ status: "active", limit: 10 });
    expect(result.entities[0].canonical_name).toBe("Alice");
    const [url] = lastCall();
    expect(url).toContain("/v1/entities");
    expect(url).toContain("status=active");
  });

  it("getEntity GET /v1/entities/:id returns entity", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ entity: mockEntity }));
    const result = await client.getEntity("ent-1");
    expect(result.id).toBe("ent-1");
    expect(lastCall()[0]).toContain("/v1/entities/ent-1");
  });

  it("resolveEntity GET /v1/entities/resolve with name", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ resolution: { entity: mockEntity, matched_via: "exact", score: 1.0 } }),
    );
    const result = await client.resolveEntity("Alice", { entityType: "person" });
    expect(result.resolution.matched_via).toBe("exact");
    const [url] = lastCall();
    expect(url).toContain("/v1/entities/resolve");
    expect(url).toContain("name=Alice");
    expect(url).toContain("entity_type=person");
  });

  it("mergeEntities POST /v1/entities/merge", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ entity: mockEntity }));
    const result = await client.mergeEntities({ sourceId: "ent-2", targetId: "ent-1", reason: "dup" });
    expect(result.id).toBe("ent-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/entities/merge");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.source_id).toBe("ent-2");
    expect(body.reason).toBe("dup");
  });

  it("maps 501 to EnterpriseRequiredError when registry disabled", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockErr(501, { detail: "registry off" }));
    await expect(client.listEntities()).rejects.toBeInstanceOf(EnterpriseRequiredError);
  });
});

describe("Association update/delete", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("updateAssociation PATCH endpoint with strength/metadata", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.updateAssociation("mem-1", "assoc-1", { strength: 0.8, metadata: { k: "v" } });
    const [url, init] = lastCall();
    expect(url).toContain("/v1/memories/mem-1/associations/assoc-1");
    expect(init.method).toBe("PATCH");
    const body = JSON.parse(init.body as string);
    expect(body.strength).toBe(0.8);
    expect(body.metadata).toEqual({ k: "v" });
  });

  it("deleteAssociation DELETE endpoint", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.deleteAssociation("mem-1", "assoc-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/memories/mem-1/associations/assoc-1");
    expect(init.method).toBe("DELETE");
  });
});

describe("Recall extra params", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("threads offset/event_after/event_before/time_order/include_global* into body", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ memories: [], mode_used: "rag", search_latency_ms: 1, total_count: 0, query_tokens: 1 }),
    );
    await client.recall("q", {
      offset: 5,
      eventAfter: new Date("2024-01-01T00:00:00Z"),
      eventBefore: new Date("2024-02-01T00:00:00Z"),
      timeOrder: "desc",
      includeGlobal: false,
      includeGlobalUser: true,
    });
    const [, init] = lastCall();
    const body = JSON.parse(init.body as string);
    expect(body.offset).toBe(5);
    expect(body.event_after).toBe("2024-01-01T00:00:00.000Z");
    expect(body.event_before).toBe("2024-02-01T00:00:00.000Z");
    expect(body.time_order).toBe("desc");
    expect(body.include_global).toBe(false);
    expect(body.include_global_user).toBe(true);
  });
});

describe("Contexts deleteContext", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("DELETE /v1/workspaces/:wid/contexts/:cid", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.deleteContext("ctx-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/workspaces/ws-1/contexts/ctx-1");
    expect(init.method).toBe("DELETE");
  });

  it("DELETE uses explicit workspaceId over client default", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.deleteContext("ctx-1", "ws-other");
    const [url] = lastCall();
    expect(url).toContain("/v1/workspaces/ws-other/contexts/ctx-1");
  });
});

describe("Chat: user threads / update / delete message", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("listUserThreads GET /v1/threads/user/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ threads: [mockThread], total_count: 1 }));
    const result = await client.listUserThreads("user-1", { ownership: "user", limit: 10 });
    expect(result).toHaveLength(1);
    const [url] = lastCall();
    expect(url).toContain("/v1/threads/user/user-1");
    expect(url).toContain("ownership=user");
  });

  it("updateThread PUT /v1/threads/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ thread: { ...mockThread, title: "New" } }));
    const result = await client.updateThread("thr-1", { title: "New" });
    expect(result.title).toBe("New");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/threads/thr-1");
    expect(init.method).toBe("PUT");
    const body = JSON.parse(init.body as string);
    expect(body.title).toBe("New");
  });

  it("deleteMessage DELETE /v1/threads/:id/messages/:mid", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
    await client.deleteMessage("thr-1", "msg-1");
    const [url, init] = lastCall();
    expect(url).toContain("/v1/threads/thr-1/messages/msg-1");
    expect(init.method).toBe("DELETE");
  });
});

describe("Knowledgebase namespace", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  const mockKb: Knowledgebase = {
    workspace_id: "ws-1",
    article_count: 3,
    community_count: 2,
    generated_at: "2024-01-01T00:00:00Z",
    stats: null,
  };
  const mockArticle: KBArticle = {
    id: "index",
    article_type: "index",
    title: "Index",
    content_md: "# Index",
    metadata: {},
    generated_at: "2024-01-01T00:00:00Z",
  };

  it("kb namespace defined", () => {
    expect(client.kb).toBeDefined();
  });

  it("kb.get GET /v1/knowledgebase", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson(mockKb));
    const result = await client.kb.get();
    expect(result.article_count).toBe(3);
    expect(lastCall()[0]).toContain("/v1/knowledgebase");
  });

  it("kb.generate POST /v1/knowledgebase/generate with options", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson(mockKb));
    await client.kb.generate({ maxCommunities: 10, regenerate: true });
    const [url, init] = lastCall();
    expect(url).toContain("/v1/knowledgebase/generate");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.max_communities).toBe(10);
    expect(body.regenerate).toBe(true);
  });

  it("kb.listArticles GET /v1/knowledgebase/articles", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson({ articles: [mockArticle], total: 1 }));
    const result = await client.kb.listArticles({ articleType: "index" });
    expect(result).toHaveLength(1);
    const [url] = lastCall();
    expect(url).toContain("/v1/knowledgebase/articles");
    expect(url).toContain("article_type=index");
  });

  it("kb.getArticle GET /v1/knowledgebase/articles/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOkJson(mockArticle));
    const result = await client.kb.getArticle("index");
    expect(result.id).toBe("index");
    expect(lastCall()[0]).toContain("/v1/knowledgebase/articles/index");
  });

  it("kb.exportVault GET /v1/knowledgebase/export as arraybuffer", async () => {
    const buf = new ArrayBuffer(8);
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ok: true,
      status: 200,
      arrayBuffer: async () => buf,
    });
    const result = await client.kb.exportVault();
    expect(result).toBe(buf);
    expect(lastCall()[0]).toContain("/v1/knowledgebase/export");
  });

  it("kb.getGraphAnalysis GET /v1/knowledgebase/graph", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({
        analysis: {
          snapshot: { workspace_id: "ws-1", node_count: 1, edge_count: 0, includes_rpg: false },
          communities: [],
          central_nodes: [],
          bridges: [],
          stats: { node_count: 1, edge_count: 0, community_count: 0, density: 0, avg_degree: 0, max_degree: 0, god_node_count: 0 },
        },
        cached: false,
      }),
    );
    const result = await client.kb.getGraphAnalysis();
    expect(result?.snapshot.node_count).toBe(1);
    expect(lastCall()[0]).toContain("/v1/knowledgebase/graph");
  });

  it("kb.getCommunity GET /v1/knowledgebase/graph/communities/:id", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ id: 0, memory_ids: ["m1"], size: 1, cohesion_score: 0.5, central_node_ids: [], label: "Topic" }),
    );
    const result = await client.kb.getCommunity(0);
    expect(result.label).toBe("Topic");
    expect(lastCall()[0]).toContain("/v1/knowledgebase/graph/communities/0");
  });

  it("kb.getArticle 404 surfaces NotFoundError", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockErr(404, { detail: "missing" }));
    await expect(client.kb.getArticle("nope")).rejects.toBeInstanceOf(NotFoundError);
  });
});

describe("MCP server import/export", () => {
  let client: MemoryLayerClient;
  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    vi.clearAllMocks();
  });

  it("import POST /v1/mcp-servers/import", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ imported: 2, updated: 0, skipped: 0, errors: [] }),
    );
    const result = await client.mcpServers.import({
      mcpServers: { pg: { command: "npx", args: ["pg"] }, gh: { url: "https://x", type: "http" } },
    });
    expect(result.imported).toBe(2);
    const [url, init] = lastCall();
    expect(url).toContain("/v1/mcp-servers/import");
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.mcpServers.pg.command).toBe("npx");
    // Workspace flows via the X-Workspace-ID header (client-level), matching
    // the existing namespace request convention.
    expect((init.headers as Record<string, string>)["X-Workspace-ID"]).toBe("ws-1");
  });

  it("export GET /v1/mcp-servers/export returns .mcp.json doc", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ mcpServers: { pg: { command: "npx", args: ["pg"] } } }),
    );
    const result = await client.mcpServers.export();
    expect(result.mcpServers.pg.command).toBe("npx");
    expect(lastCall()[0]).toContain("/v1/mcp-servers/export");
  });
});
