import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import { MemoryType, RecallMode, RelationshipType } from "../src/types.js";
import { AuthenticationError, NotFoundError, ValidationError } from "../src/errors.js";

// Mock fetch globally
global.fetch = vi.fn();

describe("MemoryLayerClient", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
      workspaceId: "ws-123",
    });
    vi.clearAllMocks();
  });

  describe("remember", () => {
    it("should create a memory", async () => {
      const mockMemory = {
        id: "mem-123",
        workspace_id: "ws-123",
        tenant_id: "tenant-123",
        context_id: "_default",
        content: "Test memory",
        content_hash: "hash123",
        type: MemoryType.EPISODIC,
        importance: 0.5,
        tags: [],
        metadata: {},
        access_count: 0,
        decay_factor: 1.0,
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ memory: mockMemory }),
      });

      const result = await client.remember("Test memory");

      expect(result).toEqual(mockMemory);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({
            "Authorization": "Bearer test-key",
            "X-Workspace-ID": "ws-123",
            "Content-Type": "application/json",
          }),
        })
      );
    });

    it("should create a memory with options", async () => {
      const mockMemory = {
        id: "mem-123",
        workspace_id: "ws-123",
        tenant_id: "tenant-123",
        context_id: "_default",
        content: "Test memory",
        content_hash: "hash123",
        type: MemoryType.SEMANTIC,
        subtype: "solution",
        importance: 0.8,
        tags: ["test"],
        metadata: { key: "value" },
        access_count: 0,
        decay_factor: 1.0,
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ memory: mockMemory }),
      });

      const result = await client.remember("Test memory", {
        type: MemoryType.SEMANTIC,
        importance: 0.8,
        tags: ["test"],
        metadata: { key: "value" },
      });

      expect(result).toEqual(mockMemory);
    });
  });

  describe("recall", () => {
    it("should recall memories", async () => {
      const mockResult = {
        memories: [
          {
            id: "mem-123",
            workspace_id: "ws-123",
            tenant_id: "tenant-123",
            context_id: "_default",
            content: "Test memory",
            content_hash: "hash123",
            type: MemoryType.EPISODIC,
            importance: 0.5,
            tags: [],
            metadata: {},
            access_count: 1,
            decay_factor: 1.0,
            created_at: "2024-01-01T00:00:00Z",
            updated_at: "2024-01-01T00:00:00Z",
          },
        ],
        mode_used: RecallMode.RAG,
        search_latency_ms: 50,
        total_count: 1,
        query_tokens: 10,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const result = await client.recall("test query");

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/recall",
        expect.objectContaining({
          method: "POST",
        })
      );
    });

    it("should recall memories with options", async () => {
      const mockResult = {
        memories: [],
        mode_used: RecallMode.HYBRID,
        search_latency_ms: 75,
        total_count: 0,
        query_tokens: 5,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const result = await client.recall("test query", {
        types: [MemoryType.SEMANTIC],
        mode: RecallMode.HYBRID,
        limit: 5,
        minRelevance: 0.7,
      });

      expect(result).toEqual(mockResult);
    });
  });

  describe("reflect", () => {
    it("should reflect on memories", async () => {
      const mockResult = {
        reflection: "Test reflection",
        source_memories: [],
        tokens_processed: 100,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const result = await client.reflect("What did I learn?");

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/reflect",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });

  describe("associate", () => {
    it("should create an association", async () => {
      const mockAssociation = {
        id: "assoc-123",
        workspace_id: "ws-123",
        source_id: "mem-1",
        target_id: "mem-2",
        relationship: "solution",
        strength: 0.8,
        metadata: {},
        created_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ association: mockAssociation }),
      });

      const result = await client.associate(
        "mem-1",
        "mem-2",
        "solution",
        0.8
      );

      expect(result).toEqual(mockAssociation);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/mem-1/associate",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });

  describe("error handling", () => {
    it("should throw AuthenticationError on 401", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ message: "Invalid API key" }),
      });

      await expect(client.remember("test")).rejects.toThrow(AuthenticationError);
    });

    it("should throw NotFoundError on 404", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ message: "Memory not found" }),
      });

      await expect(client.getMemory("mem-123")).rejects.toThrow(NotFoundError);
    });

    it("should throw ValidationError on 400", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ message: "Invalid request" }),
      });

      await expect(client.remember("")).rejects.toThrow(ValidationError);
    });
  });

  describe("sessions", () => {
    it("should create a session", async () => {
      const mockSession = {
        id: "sess-123",
        workspace_id: "ws-123",
        tenant_id: "tenant-123",
        context_id: "_default",
        working_memory: {},
        metadata: {},
        expires_at: "2024-01-01T01:00:00Z",
        created_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ session: mockSession }),
      });

      const result = await client.createSession();

      expect(result.session).toEqual(mockSession);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/sessions",
        expect.objectContaining({
          method: "POST",
        })
      );
    });

    it("should set working memory", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      await client.setWorkingMemory("sess-123", "key", "value");

      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/sessions/sess-123/memory",
        expect.objectContaining({
          method: "POST",
        })
      );
    });

    it("should get working memory", async () => {
      const mockMemory = { key: "value" };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockMemory,
      });

      const result = await client.getWorkingMemory("sess-123", "key");

      expect(result).toEqual(mockMemory);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/sessions/sess-123/memory?key=key",
        expect.objectContaining({
          method: "GET",
        })
      );
    });

    it("should commit session", async () => {
      const mockCommitResponse = {
        session_id: "sess-123",
        memories_extracted: 5,
        memories_deduplicated: 1,
        memories_created: 4,
        breakdown: { semantic: 2, episodic: 2 },
        extraction_time_ms: 150,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockCommitResponse,
      });

      const result = await client.commitSession("sess-123");

      expect(result).toEqual(mockCommitResponse);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/sessions/sess-123/commit",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });

  describe("workspaces", () => {
    it("should create a workspace", async () => {
      const mockWorkspace = {
        id: "ws-456",
        tenant_id: "tenant-123",
        name: "Test Workspace",
        settings: {},
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ workspace: mockWorkspace }),
      });

      const result = await client.createWorkspace("Test Workspace");

      expect(result).toEqual(mockWorkspace);
    });

    it("should list workspaces filtered by tag", async () => {
      const mockWorkspaces = [
        {
          id: "ws-kb",
          tenant_id: "tenant-123",
          name: "KB",
          settings: {},
          tags: ["knowledge", "topic:finance"],
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ workspaces: mockWorkspaces }),
      });

      const result = await client.listWorkspaces({ tags: ["knowledge", "topic:finance"], match: "all" });

      expect(result).toEqual(mockWorkspaces);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/workspaces?tags=knowledge&tags=topic%3Afinance&match=all",
        expect.objectContaining({ method: "GET" })
      );
    });

    it("should list all workspaces when no filter is given", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ workspaces: [] }),
      });

      await client.listWorkspaces();

      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/workspaces",
        expect.objectContaining({ method: "GET" })
      );
    });

    it("should create a context", async () => {
      const mockContext = {
        id: "ctx-123",
        workspace_id: "ws-123",
        tenant_id: "tenant-123",
        name: "Test Context",
        settings: {},
        created_at: "2024-01-01T00:00:00Z",
        updated_at: "2024-01-01T00:00:00Z",
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ context: mockContext }),
      });

      const result = await client.createContext("Test Context");

      expect(result).toEqual(mockContext);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/workspaces/ws-123/contexts",
        expect.objectContaining({
          method: "POST",
        })
      );
    });

    it("should list contexts", async () => {
      const mockContexts = [
        {
          id: "ctx-123",
          workspace_id: "ws-123",
          tenant_id: "tenant-123",
          name: "Default",
          settings: {},
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ contexts: mockContexts }),
      });

      const result = await client.listContexts();

      expect(result).toEqual(mockContexts);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/workspaces/ws-123/contexts",
        expect.objectContaining({
          method: "GET",
        })
      );
    });
  });

  describe("graph traversal", () => {
    it("should traverse graph", async () => {
      const mockResult = {
        paths: [
          {
            nodes: ["mem-1", "mem-2"],
            edges: [],
            total_strength: 0.8,
            depth: 1,
          },
        ],
        total_paths: 1,
        unique_nodes: ["mem-1", "mem-2"],
        query_latency_ms: 25,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const result = await client.traverseGraph("mem-1", { maxDepth: 2 });

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/mem-1/traverse",
        expect.objectContaining({
          method: "POST",
        })
      );
      const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      expect(body).toEqual({ workspace_id: "ws-123", max_depth: 2 });
    });

    it("should send only fields the traverse endpoint accepts", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ paths: [], total_paths: 0, unique_nodes: [], query_latency_ms: 1 }),
      });

      await client.traverseGraph("mem-1", {
        relationshipTypes: ["causes"],
        relationshipCategories: ["causal" as any],
        maxDepth: 3,
        direction: "outgoing",
        minStrength: 0.4,
        maxPaths: 10,
        maxNodes: 20,
        workspaceId: "ws-other",
      });

      const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      expect(body).toEqual({
        workspace_id: "ws-other",
        relationship_types: ["causes"],
        max_depth: 3,
        direction: "outgoing",
        min_strength: 0.4,
      });
    });
  });

  describe("batch operations", () => {
    it("should batch memories", async () => {
      const mockResult = {
        total_operations: 3,
        successful: 3,
        failed: 0,
        results: [
          { index: 0, type: "create", status: "success", memory_id: "mem-1", error: null },
          { index: 1, type: "update", status: "success", memory_id: "mem-2", error: null },
          { index: 2, type: "delete", status: "success", memory_id: "mem-3", error: null },
        ],
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const operations = [
        { op: "create" as const, content: "Test 1", importance: 0.7, tags: ["a"] },
        { op: "update" as const, memory_id: "mem-2", tags: ["reviewed"], pinned: true },
        { op: "delete" as const, memory_id: "mem-3", hard: false },
      ];
      const result = await client.batchMemories(operations);
      const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      expect(body).toEqual({ operations });

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/batch",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });
  describe("touchSession", () => {
    it("returns the new expiration and sends no body by default", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ expires_at: "2026-01-01T01:00:00+00:00" }),
      });

      const result = await client.touchSession("sess-1");

      expect(result).toEqual({ expires_at: "2026-01-01T01:00:00+00:00" });
      const [url, init] = (global.fetch as any).mock.calls[0];
      expect(url).toBe("http://localhost:61001/v1/sessions/sess-1/touch");
      expect(init.method).toBe("POST");
      expect(init.body).toBeUndefined();
    });

    it("passes extendSeconds as the extend_seconds query parameter", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ expires_at: "2026-01-01T02:00:00+00:00" }),
      });

      await client.touchSession("sess-1", 7200);

      const [url, init] = (global.fetch as any).mock.calls[0];
      expect(url).toBe("http://localhost:61001/v1/sessions/sess-1/touch?extend_seconds=7200");
      expect(init.body).toBeUndefined();
    });
  });

  describe("chat threads", () => {
    const thread = {
      id: "thr-1",
      workspace_id: "_user_chat",
      tenant_id: "_default",
      context_id: "_default",
      metadata: {},
      message_count: 0,
      last_decomposed_index: 0,
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
      ownership: "user",
    };

    it("createThread forwards ownership, scope, parent thread and idle action", async () => {
      (global.fetch as any).mockResolvedValueOnce({ ok: true, status: 201, json: async () => ({ thread }) });

      const result = await client.createThread({
        workspaceId: "ws-app",
        title: "Child",
        scope: "office",
        ownership: "user",
        parentThread: "thr-parent",
        idleAction: "hide",
      });

      expect(result.id).toBe("thr-1");
      const body = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      expect(body).toEqual({
        workspace_id: "_user_chat",
        title: "Child",
        scope: "office",
        ownership: "user",
        parent_thread: "thr-parent",
        idle_action: "hide",
      });
    });

    it("createThread keeps the workspace for workspace-owned threads and by default", async () => {
      (global.fetch as any).mockResolvedValue({ ok: true, status: 201, json: async () => ({ thread }) });

      await client.createThread({ workspaceId: "ws-app", ownership: "workspace" });
      await client.createThread();

      const first = JSON.parse((global.fetch as any).mock.calls[0][1].body);
      const second = JSON.parse((global.fetch as any).mock.calls[1][1].body);
      expect(first).toEqual({ workspace_id: "ws-app", ownership: "workspace" });
      expect(second).toEqual({ workspace_id: "ws-123" });
    });

    it("listThreads sends scope, ownership, parent and hidden filters", async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ threads: [thread], total_count: 1 }),
      });

      const threads = await client.listThreads({
        scopeFilter: "web",
        ownershipFilter: "workspace",
        parentThread: "thr-parent",
        includeHidden: true,
        limit: 10,
      });

      expect(threads).toHaveLength(1);
      const url = new URL((global.fetch as any).mock.calls[0][0]);
      expect(url.pathname).toBe("/v1/threads");
      expect(Object.fromEntries(url.searchParams)).toEqual({
        workspace_id: "ws-123",
        limit: "10",
        scope_filter: "web",
        ownership_filter: "workspace",
        parent_thread: "thr-parent",
        include_hidden: "true",
      });
    });
  });
});
