import { describe, it, expect, beforeEach, vi } from "vitest";
import { MemoryLayerClient } from "../src/client.js";
import type { Skill, SkillFile } from "../src/skills.js";

global.fetch = vi.fn();

const mockSkill: Skill = {
  id: "skl_abc123",
  tenant_id: "_default",
  workspace_id: "ws-1",
  name: "pdf-extraction",
  description: "Extract text and tables from PDF files",
  version: "0.1.0",
  body: "## Instructions\n\nUse this skill to extract PDF content.",
  metadata: {},
  source_mode: "server",
  manifest_hash: "h1",
  bundle_hash: "h2",
  enabled: true,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

const mockSkillFile: SkillFile = {
  id: "sklf_xyz456",
  skill_id: "skl_abc123",
  path: "scripts/extract.py",
  kind: "script",
  content: "# extract.py\nprint('hello')\n",
  content_hash: "h3",
  size_bytes: 30,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
};

function mockOkJson(data: unknown) {
  return { ok: true, status: 200, json: async () => data };
}
function mockOk204() {
  return { ok: true, status: 204, json: async () => undefined };
}

describe("client.skills namespace", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
      workspaceId: "ws-1",
    });
    vi.clearAllMocks();
  });

  it("client.skills is defined", () => {
    expect(client.skills).toBeDefined();
  });

  describe("list", () => {
    it("GET /v1/skills and returns skills array", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skills: [mockSkill] }),
      );
      const result = await client.skills.list();
      expect(result).toHaveLength(1);
      expect(result[0].name).toBe("pdf-extraction");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills");
    });

    it("passes workspace_id query param", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skills: [] }),
      );
      await client.skills.list({ workspaceId: "ws-override" });
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("workspace_id=ws-override");
    });

    it("passes include_shadowed query param", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skills: [] }),
      );
      await client.skills.list({ includeShadowed: true });
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("include_shadowed=true");
    });
  });

  describe("get", () => {
    it("GET /v1/skills/:id and returns skill", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skill: mockSkill }),
      );
      const result = await client.skills.get("skl_abc123");
      expect(result.id).toBe("skl_abc123");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills/skl_abc123");
    });
  });

  describe("getManifest", () => {
    it("GET /v1/skills/:id/manifest", async () => {
      const manifestText = "---\nname: pdf-extraction\n---\n## Instructions";
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson(manifestText),
      );
      const result = await client.skills.getManifest("skl_abc123");
      expect(result).toBe(manifestText);
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills/skl_abc123/manifest");
    });
  });

  describe("listFiles", () => {
    it("GET /v1/skills/:id/files", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ files: [mockSkillFile] }),
      );
      const result = await client.skills.listFiles("skl_abc123");
      expect(result).toHaveLength(1);
      expect(result[0].path).toBe("scripts/extract.py");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills/skl_abc123/files");
    });
  });

  describe("getFile", () => {
    it("GET /v1/skills/:id/files/:path", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ file: mockSkillFile }),
      );
      const result = await client.skills.getFile("skl_abc123", "scripts/extract.py");
      expect(result.kind).toBe("script");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills/skl_abc123/files/");
    });
  });

  describe("save", () => {
    it("POST /v1/skills with manifest body", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skill: mockSkill }),
      );
      const result = await client.skills.save({
        name: "pdf-extraction",
        description: "Extract text and tables from PDF files",
        body: "## Instructions",
      });
      expect(result.id).toBe("skl_abc123");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/skills");
      expect((init as { method: string }).method).toBe("POST");
      const body = JSON.parse(init.body as string);
      expect(body.name).toBe("pdf-extraction");
      expect(body.source_mode).toBe("server");
    });
  });

  describe("delete", () => {
    it("DELETE /v1/skills/:id", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
      await client.skills.delete("skl_abc123");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/skills/skl_abc123");
      expect((init as { method: string }).method).toBe("DELETE");
    });
  });

  describe("resolve", () => {
    it("POST /v1/skills/resolve with name", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skill: mockSkill }),
      );
      const result = await client.skills.resolve({ name: "pdf-extraction" });
      expect(result?.name).toBe("pdf-extraction");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      expect(url).toContain("/v1/skills/resolve");
      const body = JSON.parse(init.body as string);
      expect(body.name).toBe("pdf-extraction");
    });

    it("POST /v1/skills/resolve returns null when not found", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skill: null }),
      );
      const result = await client.skills.resolve({ name: "unknown" });
      expect(result).toBeNull();
    });
  });

  describe("pull", () => {
    it("fetches skill and files", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill }))
        .mockResolvedValueOnce(mockOkJson({ files: [mockSkillFile] }));
      const result = await client.skills.pull("skl_abc123");
      expect(result.skill.id).toBe("skl_abc123");
      expect(result.files).toHaveLength(1);
    });
  });

  describe("push", () => {
    it("delegates to save", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkJson({ skill: mockSkill }),
      );
      const result = await client.skills.push({
        name: "pdf-extraction",
        description: "desc",
        body: "## body",
      });
      expect(result.id).toBe("skl_abc123");
    });
  });
});

describe("OboProxy.skills", () => {
  let client: MemoryLayerClient;

  beforeEach(() => {
    client = new MemoryLayerClient({
      baseUrl: "http://localhost:61001",
      apiKey: "test-key",
    });
    vi.clearAllMocks();
  });

  it("actingFor() exposes .skills namespace", () => {
    const proxy = client.actingFor({ grantId: "g1", subject: { type: "user", id: "alice" } });
    expect(proxy.skills).toBeDefined();
  });

  it("proxy.skills.list sends OBO headers", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ skills: [mockSkill] }),
    );
    const proxy = client.actingFor({ grantId: "g_alice", subject: { type: "user", id: "alice" } });
    await proxy.skills.list();
    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit & { headers: Record<string, string> }];
    expect(init.headers["X-Aether-Grant-ID"]).toBe("g_alice");
    expect(init.headers["X-Aether-Subject-ID"]).toBe("alice");
  });

  it("proxy.skills.save sends OBO headers", async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      mockOkJson({ skill: mockSkill }),
    );
    const proxy = client.actingFor({ grantId: "g_bob", subject: { type: "user", id: "bob" } });
    await proxy.skills.save({ name: "test", description: "desc", body: "body" });
    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit & { headers: Record<string, string> }];
    expect(init.headers["X-Aether-Subject-ID"]).toBe("bob");
  });
});
