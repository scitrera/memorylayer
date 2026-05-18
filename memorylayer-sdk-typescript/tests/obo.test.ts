import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import type { AuthorityContext } from "../src/types.js";

global.fetch = vi.fn();

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

const mockRecall = {
  memories: [mockMemory],
  mode_used: "rag",
  search_latency_ms: 10,
  total_count: 1,
  query_tokens: 3,
};

function mockOkJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}

describe("OBO / actingFor", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
      workspaceId: "ws-default",
    });
    vi.clearAllMocks();
  });

  it("actingFor() adds X-Aether-* headers on remember", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    const alice = client.actingFor({ grantId: "g_abc", subject: { type: "user", id: "alice" } });
    await alice.remember("hello", { workspaceId: "ws-1" });

    const [, init] = (global.fetch as any).mock.calls[0];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_abc");
    expect(init.headers["X-Aether-Authority-Mode"]).toBe("on_behalf_of");
    expect(init.headers["X-Aether-Subject-Type"]).toBe("user");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("alice");
  });

  it("actingFor() adds X-Aether-* headers on recall", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson(mockRecall));

    const alice = client.actingFor({ grantId: "g_abc", subject: { type: "user", id: "alice" } });
    await alice.recall("query", { workspaceId: "ws-1" });

    const [, init] = (global.fetch as any).mock.calls[0];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_abc");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("alice");
  });

  it("forWorkspace() sets workspaceId on inner calls", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    const alice = client.actingFor({ grantId: "g_abc", subject: { type: "user", id: "alice" } });
    const aliceWork = alice.forWorkspace("ws-work");
    await aliceWork.remember("hello");

    const [, init] = (global.fetch as any).mock.calls[0];
    const body = JSON.parse(init.body);
    expect(body.workspace_id).toBe("ws-work");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("alice");
  });

  it("concurrent actingFor() for 3 subjects — no cross-talk", async () => {
    (global.fetch as any)
      .mockResolvedValueOnce(mockOkJson({ memory: mockMemory }))
      .mockResolvedValueOnce(mockOkJson({ memory: mockMemory }))
      .mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    const alice = client.actingFor({ grantId: "g_alice", subject: { type: "user", id: "alice" } });
    const bob = client.actingFor({ grantId: "g_bob", subject: { type: "user", id: "bob" } });
    const carol = client.actingFor({ grantId: "g_carol", subject: { type: "user", id: "carol" } });

    await Promise.all([
      alice.remember("a"),
      bob.remember("b"),
      carol.remember("c"),
    ]);

    const calls = (global.fetch as any).mock.calls as [string, RequestInit & { headers: Record<string, string> }][];
    const subjects = calls.map(([, init]) => (init.headers as Record<string, string>)["X-Aether-Subject-ID"]);
    expect(subjects).toContain("alice");
    expect(subjects).toContain("bob");
    expect(subjects).toContain("carol");

    // Each call must use its own grant — no subject should appear more than once
    expect(new Set(subjects).size).toBe(3);

    const grantIds = calls.map(([, init]) => (init.headers as Record<string, string>)["X-Aether-Grant-ID"]);
    expect(new Set(grantIds).size).toBe(3);
  });

  it("per-call authority option propagates headers", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    const authority: AuthorityContext = { grantId: "g_direct", subject: { type: "user", id: "dave" } };
    await client.remember("hello", { authority, workspaceId: "ws-1" });

    const [, init] = (global.fetch as any).mock.calls[0];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_direct");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("dave");
  });

  it("defaultAuthority on ClientConfig is used as fallback", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    const defaultAuthority: AuthorityContext = {
      grantId: "g_default",
      subject: { type: "service", id: "svc-1" },
    };
    const clientWithDefault = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "key",
      defaultAuthority,
    });
    await clientWithDefault.remember("hello");

    const [, init] = (global.fetch as any).mock.calls[0];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_default");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("svc-1");
  });

  it("backward compat: calls without authority produce no X-Aether-* headers", async () => {
    (global.fetch as any).mockResolvedValueOnce(mockOkJson({ memory: mockMemory }));

    await client.remember("hello");

    const [, init] = (global.fetch as any).mock.calls[0];
    expect(init.headers["X-Aether-Grant-ID"]).toBeUndefined();
    expect(init.headers["X-Aether-Authority-Mode"]).toBeUndefined();
    expect(init.headers["X-Aether-Subject-Type"]).toBeUndefined();
    expect(init.headers["X-Aether-Subject-ID"]).toBeUndefined();
  });
});
