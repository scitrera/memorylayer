import { describe, it, expect, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";

// Don't mock global.fetch here — we want to assert the custom impl wins.

describe("MemoryLayerClient custom fetch transport", () => {
  it("uses the injected fetch instead of globalThis.fetch", async () => {
    const customFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ recalled: [], total_count: 0 }),
    } as unknown as Response);

    const client = new MemoryLayerClient({
      baseUrl: "http://aether-transport.local",
      apiKey: "test-key",
      workspaceId: "ws-1",
      fetch: customFetch as unknown as typeof fetch,
    });

    await client.recall("anything");

    expect(customFetch).toHaveBeenCalledTimes(1);
    const [url, init] = customFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://aether-transport.local/v1/memories/recall");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>)["Authorization"]).toBe("Bearer test-key");
  });

  it("falls back to globalThis.fetch when no custom impl is provided", async () => {
    const globalFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ recalled: [], total_count: 0 }),
    } as unknown as Response);
    const previous = global.fetch;
    global.fetch = globalFetch as unknown as typeof fetch;

    try {
      const client = new MemoryLayerClient({
        baseUrl: "http://default-transport.local",
        apiKey: "test-key",
      });
      await client.recall("anything");
      expect(globalFetch).toHaveBeenCalledTimes(1);
    } finally {
      global.fetch = previous;
    }
  });

  it("does not call the global fetch when a custom impl is provided", async () => {
    const customFetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ recalled: [] }),
    } as unknown as Response);
    const globalFetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    } as unknown as Response);
    const previous = global.fetch;
    global.fetch = globalFetchSpy as unknown as typeof fetch;

    try {
      const client = new MemoryLayerClient({
        baseUrl: "http://x.local",
        apiKey: "k",
        fetch: customFetch as unknown as typeof fetch,
      });
      await client.recall("q");
      expect(customFetch).toHaveBeenCalledTimes(1);
      expect(globalFetchSpy).not.toHaveBeenCalled();
    } finally {
      global.fetch = previous;
    }
  });
});
