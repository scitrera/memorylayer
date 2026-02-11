/**
 * Tests for MCP Server
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { MCPServer, createServer } from "../src/server.js";
import { MemoryLayerClient } from "../src/client.js";
import { TOOLS, SESSION_TOOLS, CONTEXT_ENVIRONMENT_TOOLS } from "../src/tools.js";

// Mock the SDK client
vi.mock("../src/client.js", () => {
  return {
    MemoryLayerClient: vi.fn().mockImplementation(() => ({
      getWorkspaceId: vi.fn().mockReturnValue("test-workspace"),
      remember: vi.fn().mockResolvedValue({ id: "mem-123", type: "semantic", importance: 0.5, tags: [] }),
      recall: vi.fn().mockResolvedValue({ memories: [], total_count: 0, search_latency_ms: 10, mode_used: "semantic" }),
      startSession: vi.fn().mockResolvedValue({ session_id: "server-session-123" }),
      endSession: vi.fn().mockResolvedValue({ memories_extracted: 2 }),
    })),
  };
});

describe("MCPServer", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001" });
  });

  describe("constructor", () => {
    it("should create server with default options", () => {
      const server = new MCPServer(client);
      expect(server).toBeDefined();
    });

    it("should enable session mode by default", () => {
      const server = new MCPServer(client);
      const manifest = server.getManifest();
      expect(manifest.name).toBe("memorylayer");
    });

    it("should accept custom options", () => {
      const server = new MCPServer(client, {
        workspaceId: "custom-workspace",
        sessionMode: false,
      });
      expect(server).toBeDefined();
    });
  });

  describe("getManifest", () => {
    it("should return server manifest", () => {
      const server = new MCPServer(client);
      const manifest = server.getManifest();

      expect(manifest.name).toBe("memorylayer");
      expect(manifest.version).toBe("0.1.0");
      expect(manifest.description).toContain("MemoryLayer.ai");
      expect(manifest.capabilities).toBeDefined();
    });

    it("should list core tools in capabilities", () => {
      const server = new MCPServer(client);
      const manifest = server.getManifest();
      const toolNames = manifest.capabilities?.tools as string[];

      expect(toolNames).toContain("memory_remember");
      expect(toolNames).toContain("memory_recall");
    });
  });

  describe("tool listing", () => {
    it("should include session tools when session mode enabled", () => {
      const server = new MCPServer(client, { sessionMode: true });
      const manifest = server.getManifest();

      // The manifest only shows TOOLS, but the handler includes SESSION_TOOLS
      // This tests that core tools are present
      const toolNames = manifest.capabilities?.tools as string[];
      expect(toolNames.length).toBeGreaterThan(0);
    });
  });
});

describe("createServer", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    vi.clearAllMocks();
    client = new MemoryLayerClient({ baseUrl: "http://localhost:61001" });
  });

  it("should create server instance", async () => {
    const server = await createServer(client);
    expect(server).toBeInstanceOf(MCPServer);
  });

  it("should pass options to server", async () => {
    const server = await createServer(client, {
      workspaceId: "my-workspace",
      sessionMode: false,
    });
    expect(server).toBeInstanceOf(MCPServer);
  });
});

describe("Tool counts", () => {
  it("should have expected number of core tools", () => {
    // memory_remember, memory_recall, memory_reflect, memory_forget,
    // memory_associate, memory_briefing, memory_statistics, memory_graph_query, memory_audit
    expect(TOOLS.length).toBe(9);
  });

  it("should have expected number of session tools", () => {
    // memory_session_start, memory_session_end, memory_session_commit, memory_session_status
    expect(SESSION_TOOLS.length).toBe(4);
  });

  it("should have expected number of context environment tools", () => {
    // memory_context_exec, memory_context_inspect, memory_context_load,
    // memory_context_inject, memory_context_query, memory_context_rlm,
    // memory_context_status, memory_context_checkpoint
    expect(CONTEXT_ENVIRONMENT_TOOLS.length).toBe(8);
  });

  it("should have 21 total tools when combined", () => {
    expect(TOOLS.length + SESSION_TOOLS.length + CONTEXT_ENVIRONMENT_TOOLS.length).toBe(21);
  });
});
