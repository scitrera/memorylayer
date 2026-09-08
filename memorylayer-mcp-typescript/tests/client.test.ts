/**
 * Tests for the MCP-compatible MemoryLayerClient adapter.
 *
 * Focus: the chat-thread helpers must hit the canonical server router
 * prefix ``/v1/threads`` (NOT the legacy ``/v1/chat/threads``, which 404s).
 * These helpers go through ``rawRequest`` which uses the global ``fetch``,
 * so we stub ``fetch`` and assert the URL the client requested.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { MemoryLayerClient } from "../src/client.js";

describe("MemoryLayerClient chat-thread paths", () => {
  let client: MemoryLayerClient;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001" });
    fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({}),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const lastUrl = (): string => String(fetchMock.mock.calls.at(-1)?.[0]);
  const lastMethod = (): string =>
    String((fetchMock.mock.calls.at(-1)?.[1] as { method?: string })?.method);

  it("chatThreadCreate POSTs to /v1/threads", async () => {
    await client.chatThreadCreate({ title: "hi" });
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads");
    expect(lastMethod()).toBe("POST");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadAppend POSTs to /v1/threads/{id}/messages", async () => {
    await client.chatThreadAppend("t1", [{ role: "user", content: "x" }]);
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads/t1/messages");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadGet GETs /v1/threads/{id}", async () => {
    await client.chatThreadGet("t1");
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads/t1");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadGet appends query params under /v1/threads/{id}", async () => {
    await client.chatThreadGet("t1", { limit: 5, order: "desc" });
    expect(lastUrl()).toContain("http://localhost:61001/v1/threads/t1?");
    expect(lastUrl()).toContain("limit=5");
    expect(lastUrl()).toContain("order=desc");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadList GETs /v1/threads", async () => {
    await client.chatThreadList();
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadDecompose POSTs to /v1/threads/{id}/decompose", async () => {
    await client.chatThreadDecompose("t1");
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads/t1/decompose");
    expect(lastUrl()).not.toContain("/chat/");
  });

  it("chatThreadDelete DELETEs /v1/threads/{id}", async () => {
    await client.chatThreadDelete("t1");
    expect(lastUrl()).toBe("http://localhost:61001/v1/threads/t1");
    expect(lastMethod()).toBe("DELETE");
    expect(lastUrl()).not.toContain("/chat/");
  });
});
