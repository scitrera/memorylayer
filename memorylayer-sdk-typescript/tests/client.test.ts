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
            "X-API-Key": "test-key",
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
        relationship: RelationshipType.SOLVES,
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
        RelationshipType.SOLVES,
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
        "http://localhost:61001/v1/associations/traverse",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });

  describe("batch operations", () => {
    it("should batch memories", async () => {
      const mockResult = {
        results: [
          { index: 0, success: true, memory: { id: "mem-1" } },
          { index: 1, success: true, memory: { id: "mem-2" } },
        ],
        total_processed: 2,
        successful: 2,
        failed: 0,
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResult,
      });

      const result = await client.batchMemories([
        { action: "create", memory: { content: "Test 1" } },
        { action: "create", memory: { content: "Test 2" } },
      ]);

      expect(result).toEqual(mockResult);
      expect(global.fetch).toHaveBeenCalledWith(
        "http://localhost:61001/v1/memories/batch",
        expect.objectContaining({
          method: "POST",
        })
      );
    });
  });
});
