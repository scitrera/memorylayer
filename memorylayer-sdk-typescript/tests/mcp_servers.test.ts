import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import type { McpServer } from "../src/mcp_servers.js";

global.fetch = vi.fn();

const mockStdioServer: McpServer = {
  id: "mcp_postgres001",
  tenant_id: "_default",
  workspace_id: "ws-1",
  name: "postgres-mcp",
  transport: "stdio",
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-postgres"],
  env: { DATABASE_URL: "postgres://localhost/mydb" },
  headers: {},
  metadata: {},
  source_mode: "server",
  manifest_hash: "h1",
  enabled: true,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const mockHttpServer: McpServer = {
  id: "mcp_github001",
  tenant_id: "_default",
  workspace_id: "ws-1",
  name: "github-mcp",
  transport: "http",
  args: [],
  env: {},
  url: "https://mcp.example.com/github",
  headers: { Authorization: "Bearer token123" },
  metadata: {},
  source_mode: "server",
  manifest_hash: "h2",
  enabled: true,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

function mockOkJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}
function mockOk204() {
  return { ok: true, status: 204, json: async () => undefined };
}

describe("client.mcpServers namespace", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
      workspaceId: "ws-1",
    });
    vi.clearAllMocks();
  });

  it("client.mcpServers is defined", () => {
    expect(client.mcpServers).toBeDefined();
  });

  describe("list", () => {
    it("GET /v1/mcp-servers and returns mcp_servers array", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_servers: [mockStdioServer] }),
      );
      const result = await client.mcpServers.list();
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe("postgres-mcp");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/mcp-servers");
    });

    it("passes workspace_id query param", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_servers: [] }),
      );
      await client.mcpServers.list({ workspaceId: "ws-override" });
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("workspace_id=ws-override");
    });

    it("passes transport filter query param", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_servers: [] }),
      );
      await client.mcpServers.list({ transport: "stdio" });
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("transport=stdio");
    });

    it("passes enabled filter query param", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_servers: [] }),
      );
      await client.mcpServers.list({ enabled: false });
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("enabled=false");
    });
  });

  describe("get", () => {
    it("GET /v1/mcp-servers/:id and returns mcp_server", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: mockStdioServer }),
      );
      const result = await client.mcpServers.get("mcp_postgres001");
      expect(result.id).toBe("mcp_postgres001");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/mcp-servers/mcp_postgres001");
    });
  });

  describe("create", () => {
    it("POST /v1/mcp-servers with stdio body", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: mockStdioServer }),
      );
      const result = await client.mcpServers.create({
        name: "postgres-mcp",
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-postgres"],
        env: { DATABASE_URL: "postgres://localhost/mydb" },
      });
      expect(result.id).toBe("mcp_postgres001");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/mcp-servers");
      expect((init as { method: string }).method).toBe("POST");
      const body = JSON.parse(init.body as string);
      expect(body.name).toBe("postgres-mcp");
      expect(body.transport).toBe("stdio");
      expect(body.command).toBe("npx");
    });

    it("POST /v1/mcp-servers with http body", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: mockHttpServer }),
      );
      const result = await client.mcpServers.create({
        name: "github-mcp",
        transport: "http",
        url: "https://mcp.example.com/github",
        headers: { Authorization: "Bearer token123" },
      });
      expect(result.id).toBe("mcp_github001");
      const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      const body = JSON.parse(init.body as string);
      expect(body.transport).toBe("http");
      expect(body.url).toBe("https://mcp.example.com/github");
    });
  });

  describe("update", () => {
    it("PATCH /v1/mcp-servers/:id with updates", async () => {
      const updated = { ...mockStdioServer, enabled: false, description: "Updated" };
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: updated }),
      );
      const result = await client.mcpServers.update("mcp_postgres001", {
        enabled: false,
        description: "Updated",
      });
      expect(result.enabled).toBe(false);
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/mcp-servers/mcp_postgres001");
      expect((init as { method: string }).method).toBe("PATCH");
      const body = JSON.parse(init.body as string);
      expect(body.enabled).toBe(false);
      expect(body.description).toBe("Updated");
    });
  });

  describe("delete", () => {
    it("DELETE /v1/mcp-servers/:id", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
      await client.mcpServers.delete("mcp_postgres001");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/mcp-servers/mcp_postgres001");
      expect((init as { method: string }).method).toBe("DELETE");
    });
  });

  describe("resolve", () => {
    it("POST /v1/mcp-servers/resolve with name", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: mockStdioServer }),
      );
      const result = await client.mcpServers.resolve({ name: "postgres-mcp" });
      expect(result?.name).toBe("postgres-mcp");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/mcp-servers/resolve");
      const body = JSON.parse(init.body as string);
      expect(body.name).toBe("postgres-mcp");
    });

    it("POST /v1/mcp-servers/resolve returns null when not found", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ mcp_server: null }),
      );
      const result = await client.mcpServers.resolve({ name: "unknown" });
      expect(result).toBeNull();
    });
  });
});

describe("OboProxy.mcpServers", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
    });
    vi.clearAllMocks();
  });

  it("actingFor() exposes .mcpServers namespace", () => {
    const proxy = client.actingFor({ grantId: "g1", subject: { type: "user", id: "alice" } });
    expect(proxy.mcpServers).toBeDefined();
  });

  it("proxy.mcpServers.list sends OBO headers", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ mcp_servers: [mockStdioServer] }),
    );
    const proxy = client.actingFor({ grantId: "g_alice", subject: { type: "user", id: "alice" } });
    await proxy.mcpServers.list();
    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit & { headers: Record<string, string> }];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_alice");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("alice");
  });

  it("proxy.mcpServers.create sends OBO headers", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ mcp_server: mockStdioServer }),
    );
    const proxy = client.actingFor({ grantId: "g_bob", subject: { type: "user", id: "bob" } });
    await proxy.mcpServers.create({ name: "test", transport: "stdio" });
    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit & { headers: Record<string, string> }];
    expect(init.headers["X-Aether-Subject-ID"]).toBe("bob");
  });
});
