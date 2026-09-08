import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import { MemoryLayerError, NotFoundError } from "../src/errors.js";
import { paginate, collectPages } from "../src/utils.js";

global.fetch = vi.fn();

function okJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}
function errResp(status: number, retryAfter?: string) {
  return {
    ok: false,
    status,
    json: async () => ({ detail: "boom" }),
    headers: { get: (name: string) => (name === "Retry-After" && retryAfter ? retryAfter : null) },
  };
}

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

describe("retry-with-backoff", () => {
  beforeEach(() => vi.clearAllMocks());

  it("retries on 5xx then succeeds", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 1,
    });
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(errResp(503))
      .mockResolvedValueOnce(okJson({ memory: mockMemory }));

    const result = await client.getMemory("mem-1");
    expect(result.id).toBe("mem-1");
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(2);
  });

  it("retries on 429 and honors Retry-After header", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 100000, // huge — proves Retry-After (small) is used instead
    });
    const sleepSpy = vi.spyOn(globalThis, "setTimeout");
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(errResp(429, "0")) // Retry-After: 0s
      .mockResolvedValueOnce(okJson({ memory: mockMemory }));

    const result = await client.getMemory("mem-1");
    expect(result.id).toBe("mem-1");
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(2);

    // The backoff sleep must have used the Retry-After value (0ms), NOT the
    // configured base delay (100000ms). Find a setTimeout call with delay 0.
    const usedRetryAfter = sleepSpy.mock.calls.some(
      (c) => c[1] === 0,
    );
    expect(usedRetryAfter).toBe(true);
    sleepSpy.mockRestore();
  });

  it("does not retry non-retryable 404 (fails fast)", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 1,
    });
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(errResp(404));
    await expect(client.getMemory("nope")).rejects.toBeInstanceOf(NotFoundError);
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
  });

  it("gives up after maxRetries and throws", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 1,
      maxRetries: 2,
    });
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(errResp(500))
      .mockResolvedValueOnce(errResp(500))
      .mockResolvedValueOnce(errResp(500));
    await expect(client.getMemory("mem-1")).rejects.toBeInstanceOf(MemoryLayerError);
    // 1 initial + 2 retries = 3 attempts
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(3);
  });

  it("retries on network error then succeeds", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 1,
    });
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error("ECONNRESET"))
      .mockResolvedValueOnce(okJson({ memory: mockMemory }));
    const result = await client.getMemory("mem-1");
    expect(result.id).toBe("mem-1");
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(2);
  });

  it("does not retry POST on 5xx (non-idempotent)", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      retryBaseDelay: 1,
      maxRetries: 3,
    });
    // remember() uses POST /v1/memories — must NOT be retried on 503
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(errResp(503));
    await expect(
      client.remember("hello world")
    ).rejects.toBeInstanceOf(MemoryLayerError);
    // Exactly 1 call — no retries for POST
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
  });

  it("maxRetries=0 disables retries", async () => {
    const client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "k",
      workspaceId: "ws-1",
      maxRetries: 0,
    });
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(errResp(503));
    await expect(client.getMemory("mem-1")).rejects.toBeInstanceOf(MemoryLayerError);
    expect((global.fetch as ReturnType<typeof vi.fn>).mock.calls).toHaveLength(1);
  });
});

describe("pagination helper", () => {
  it("paginate yields all items across pages and stops on short page", async () => {
    const pages: Record<number, number[]> = {
      0: [1, 2],
      2: [3, 4],
      4: [5], // short page -> stop
    };
    const fetchPage = vi.fn(async ({ offset }: { limit: number; offset: number }) => pages[offset] ?? []);
    const out: number[] = [];
    for await (const item of paginate(fetchPage, 2)) {
      out.push(item);
    }
    expect(out).toEqual([1, 2, 3, 4, 5]);
    expect(fetchPage).toHaveBeenCalledTimes(3);
  });

  it("paginate stops on empty first page", async () => {
    const fetchPage = vi.fn(async () => [] as number[]);
    const out: number[] = [];
    for await (const item of paginate(fetchPage, 10)) out.push(item);
    expect(out).toEqual([]);
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it("collectPages gathers everything eagerly", async () => {
    const pages: Record<number, string[]> = { 0: ["a", "b"], 2: ["c"] };
    const fetchPage = async ({ offset }: { limit: number; offset: number }) => pages[offset] ?? [];
    const all = await collectPages(fetchPage, 2);
    expect(all).toEqual(["a", "b", "c"]);
  });

  it("paginates a real list endpoint (listMemories) via paginate()", async () => {
    global.fetch = vi.fn();
    const client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", apiKey: "k", workspaceId: "ws-1" });
    const m = (id: string) => ({ ...mockMemory, id });
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(okJson({ memories: [m("1"), m("2")], total_count: 3 }))
      .mockResolvedValueOnce(okJson({ memories: [m("3")], total_count: 3 }));

    const ids: string[] = [];
    for await (const mem of paginate(async ({ limit, offset }) => (await client.listMemories({ limit, offset })).memories, 2)) {
      ids.push(mem.id);
    }
    expect(ids).toEqual(["1", "2", "3"]);
  });
});
