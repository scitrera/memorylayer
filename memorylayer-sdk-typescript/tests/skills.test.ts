import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as nodePath from "node:path";
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
  return { ok: true, status: 200, json: async () => data, text: async () => JSON.stringify(data) };
}
function mockOkText(text: string) {
  return {
    ok: true,
    status: 200,
    text: async () => text,
    json: async () => {
      throw new SyntaxError("Unexpected non-JSON response");
    },
  };
}
function mockOk204() {
  return { ok: true, status: 204, json: async () => undefined, text: async () => "" };
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
    it("GET /v1/skills/:id/manifest returns raw markdown text (not parsed)", async () => {
      const manifestText = "---\nname: pdf-extraction\nversion: 0.1.0\n---\n## Instructions";
      // Server responds with text/markdown; json() would throw. mockOkText's
      // json() throws to prove we never call it for this endpoint.
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        mockOkText(manifestText),
      );
      const result = await client.skills.getManifest("skl_abc123");
      expect(result).toBe(manifestText);
      expect(typeof result).toBe("string");
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
    it("GET /v1/skills/:id/files/:path returns content from the non-JSON file endpoint", async () => {
      const fileContent = "# extract.py\nprint('hello')\n";
      // First fetch: raw file bytes (text/x-python). Second fetch: listFiles
      // metadata (JSON), merged onto the returned SkillFile.
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkText(fileContent))
        .mockResolvedValueOnce(mockOkJson({ files: [mockSkillFile] }));
      const result = await client.skills.getFile("skl_abc123", "scripts/extract.py");
      expect(result.content).toBe(fileContent);
      expect(result.kind).toBe("script");
      expect(result.path).toBe("scripts/extract.py");
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      expect(url).toContain("/v1/skills/skl_abc123/files/scripts/extract.py");
    });
  });

  describe("save", () => {
    // save() now does create-vs-update detection (resolve) + per-file upserts
    // + a final re-fetch (get). Each test mocks that fetch sequence explicitly.

    function calls() {
      return (global.fetch as ReturnType<typeof vi.fn>).mock.calls as Array<
        [string, RequestInit]
      >;
    }

    it("CREATE: resolve(null) -> POST manifest MINUS files -> re-fetch", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: null })) // resolve: not found
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // POST create
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })); // get re-fetch

      const result = await client.skills.save({
        name: "pdf-extraction",
        description: "Extract text and tables from PDF files",
        body: "## Instructions",
        files: [],
      });

      expect(result.id).toBe("skl_abc123");
      // call[0] = resolve, call[1] = POST create
      const [resolveUrl] = calls()[0];
      expect(resolveUrl).toContain("/v1/skills/resolve");
      const [createUrl, createInit] = calls()[1];
      expect(createUrl).toMatch(/\/v1\/skills$/);
      expect(createInit.method).toBe("POST");
      const body = JSON.parse(createInit.body as string);
      expect(body.name).toBe("pdf-extraction");
      expect(body.source_mode).toBe("server");
      expect(body.files).toBeUndefined(); // files MUST NOT ride on the create body
    });

    it("CREATE with files: POST minus files + one base64 file PUT per file", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: null })) // resolve
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // POST create
        .mockResolvedValueOnce(mockOkJson({ path: "scripts/hello.py" })) // PUT file
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })); // get re-fetch

      const fileContent = "print('hello')\n";
      const result = await client.skills.save({
        name: "pdf-extraction",
        description: "desc",
        body: "## body",
        files: [{ path: "scripts/hello.py", content: fileContent }],
      });
      expect(result.id).toBe("skl_abc123");

      // The create body must omit files.
      const createBody = JSON.parse(calls()[1][1].body as string);
      expect(createBody.files).toBeUndefined();

      // call[2] = the file PUT with base64 content.
      const [fileUrl, fileInit] = calls()[2];
      expect(fileUrl).toContain("/v1/skills/skl_abc123/files/scripts/hello.py");
      expect(fileInit.method).toBe("PUT");
      const filePut = JSON.parse(fileInit.body as string);
      const expectedB64 = Buffer.from(fileContent, "utf-8").toString("base64");
      expect(filePut.content_b64).toBe(expectedB64);
      // Round-trips back to the original content.
      expect(Buffer.from(filePut.content_b64, "base64").toString("utf-8")).toBe(
        fileContent,
      );
    });

    it("UPDATE: existing skill -> PUT manifest + upsert kept/new files; DELETEs dropped file (true reconcile)", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      const serverFiles: SkillFile[] = [
        { ...mockSkillFile, path: "scripts/keep.py" },
        { ...mockSkillFile, id: "sklf_old", path: "scripts/gone.py" },
      ];
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // resolve: found
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // PUT update
        .mockResolvedValueOnce(mockOkJson({ files: serverFiles })) // listFiles (reconcile)
        .mockResolvedValueOnce(mockOk204()) // DELETE scripts/gone.py
        .mockResolvedValueOnce(mockOkJson({ path: "scripts/keep.py" })) // PUT keep.py
        .mockResolvedValueOnce(mockOkJson({ path: "scripts/new.py" })) // PUT new.py
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })); // get re-fetch

      await client.skills.save({
        name: "pdf-extraction",
        description: "desc",
        body: "## body",
        files: [
          { path: "scripts/keep.py", content: "keep" },
          { path: "scripts/new.py", content: "new" },
        ],
      });

      // call[1] = PUT update on the existing id (no files in body).
      const [updateUrl, updateInit] = calls()[1];
      expect(updateUrl).toContain("/v1/skills/skl_abc123");
      expect(updateInit.method).toBe("PUT");
      const updateBody = JSON.parse(updateInit.body as string);
      expect(updateBody.files).toBeUndefined();

      // The dropped file is DELETEd server-side (true reconcile), and only it.
      const deleteFileCalls = calls().filter(
        ([u, i]) => i.method === "DELETE" && u.includes("/files/"),
      );
      expect(deleteFileCalls).toHaveLength(1);
      expect(deleteFileCalls[0][0]).toContain(
        "/v1/skills/skl_abc123/files/scripts/gone.py",
      );

      // Both manifest files upserted via PUT; neither kept/new file is deleted.
      const putFileUrls = calls()
        .filter(([u, i]) => i.method === "PUT" && u.includes("/files/"))
        .map(([u]) => u);
      expect(putFileUrls.some((u) => u.includes("scripts/keep.py"))).toBe(true);
      expect(putFileUrls.some((u) => u.includes("scripts/new.py"))).toBe(true);

      // No more "can't delete" warning — the reconcile actually prunes.
      expect(warnSpy).not.toHaveBeenCalled();
      warnSpy.mockRestore();
    });

    it("PARTIAL FAILURE: file PUT rejects -> save() rejects with a clear error (never resolves silently)", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: null })) // resolve
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // POST create
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ detail: "disk full" }),
          text: async () => '{"detail":"disk full"}',
        }); // PUT file fails

      await expect(
        client.skills.save({
          name: "pdf-extraction",
          description: "desc",
          body: "## body",
          files: [{ path: "scripts/hello.py", content: "x" }],
        }),
      ).rejects.toThrow(/saved but file scripts\/hello\.py failed/);
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

  describe("deleteFile", () => {
    it("DELETE /v1/skills/:id/files/:path (segment-encoded, tolerates empty 204)", async () => {
      (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(mockOk204());
      await client.skills.deleteFile("skl_abc123", "scripts/a b.py");
      const [url, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit];
      // Per-segment encodeURIComponent preserves the "/" separator.
      expect(url).toContain("/v1/skills/skl_abc123/files/scripts/a%20b.py");
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
    it("delegates to save (resolve -> create -> re-fetch)", async () => {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skill: null })) // resolve
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // POST create
        .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })); // get re-fetch
      const result = await client.skills.push({
        name: "pdf-extraction",
        description: "desc",
        body: "## body",
      });
      expect(result.id).toBe("skl_abc123");
    });
  });

  describe("materialize", () => {
    // Each materialize run does: list -> per skill (getManifest + listFiles).
    // We only assert the list() call's query string here, so a single skill
    // with no files keeps the fetch sequence short.
    function mockMaterializeFetches(): void {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skills: [mockSkill] })) // list
        .mockResolvedValueOnce(mockOkText("---\nname: pdf-extraction\n---\nbody")) // getManifest (text/markdown)
        .mockResolvedValueOnce(mockOkJson({ files: [] })); // listFiles
    }

    function firstUrl(): string {
      const [url] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, unknown];
      return url;
    }

    it("does NOT send an enabled filter by default (writes all skills)", async () => {
      mockMaterializeFetches();
      await client.skills.materialize("/tmp/skills-test-default", { workspaceId: "ws-1" });
      expect(firstUrl()).not.toContain("enabled=");
    });

    it("threads enabled:true into the internal list() call", async () => {
      mockMaterializeFetches();
      await client.skills.materialize("/tmp/skills-test-enabled", {
        workspaceId: "ws-1",
        enabled: true,
      });
      expect(firstUrl()).toContain("enabled=true");
    });

    it("threads enabled:false into the internal list() call", async () => {
      mockMaterializeFetches();
      await client.skills.materialize("/tmp/skills-test-disabled", {
        workspaceId: "ws-1",
        enabled: false,
      });
      expect(firstUrl()).toContain("enabled=false");
    });
  });

  describe("materialize (on-disk)", () => {
    let tmpDir: string;

    beforeEach(async () => {
      tmpDir = await fs.mkdtemp(nodePath.join(os.tmpdir(), "skills-materialize-"));
    });

    afterEach(async () => {
      await fs.rm(tmpDir, { recursive: true, force: true });
    });

    it("writes SKILL.md (markdown text) + file content fetched from the file endpoint", async () => {
      const manifestText =
        "---\nname: pdf-extraction\ndescription: Extract\nversion: 0.1.0\n---\n## Instructions\n";
      const fileContent = "# extract.py\nprint('hello')\n";
      // Full materialize sequence for one skill with one file:
      //   list(json) -> getManifest(text) -> listFiles(json) -> content GET(text)
      // listFiles is called exactly ONCE (not once-per-file); content is fetched
      // via _fetchFileContent (a bare content GET, no extra listFiles).
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skills: [mockSkill] })) // list
        .mockResolvedValueOnce(mockOkText(manifestText)) // getManifest
        .mockResolvedValueOnce(mockOkJson({ files: [mockSkillFile] })) // listFiles (once)
        .mockResolvedValueOnce(mockOkText(fileContent)); // _fetchFileContent (content GET)

      await client.skills.materialize(tmpDir, { workspaceId: "ws-1", enabled: true });

      const skillMd = await fs.readFile(
        nodePath.join(tmpDir, "pdf-extraction", "SKILL.md"),
        "utf8",
      );
      expect(skillMd).toBe(manifestText);
      expect(skillMd).toContain("name: pdf-extraction");

      const written = await fs.readFile(
        nodePath.join(tmpDir, "pdf-extraction", "scripts", "extract.py"),
        "utf8",
      );
      expect(written).toBe(fileContent);

      // Assert listFiles was called exactly once (not once-per-file).
      const fetchCalls = (global.fetch as ReturnType<typeof vi.fn>).mock
        .calls as Array<[string, unknown]>;
      const listFilesCalls = fetchCalls.filter(([url]) =>
        /\/v1\/skills\/[^/]+\/files$/.test(url),
      );
      expect(listFilesCalls).toHaveLength(1);

      // Assert the content was fetched via the per-segment-encoded file path
      // (no extra JSON listFiles alongside it).
      const contentGetCalls = fetchCalls.filter(([url]) =>
        url.includes("/v1/skills/skl_abc123/files/scripts/extract.py"),
      );
      expect(contentGetCalls).toHaveLength(1);

      // Total fetch count: list + getManifest + listFiles + content GET = 4.
      expect(fetchCalls).toHaveLength(4);
    });

    // No-files variant of the full materialize sequence:
    //   list(json) -> getManifest(text) -> listFiles(json, empty)
    function mockMaterializeNoFiles(): void {
      (global.fetch as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(mockOkJson({ skills: [mockSkill] })) // list
        .mockResolvedValueOnce(
          mockOkText("---\nname: pdf-extraction\n---\nbody"),
        ) // getManifest
        .mockResolvedValueOnce(mockOkJson({ files: [] })); // listFiles
    }

    it("reconcile=true removes a stale skill dir but keeps desired skills and non-skill files", async () => {
      // Pre-seed a stale materialized skill (has SKILL.md, not in server set),
      // a desired skill dir that will be rewritten, and an unrelated file.
      await fs.mkdir(nodePath.join(tmpDir, "old-skill"), { recursive: true });
      await fs.writeFile(
        nodePath.join(tmpDir, "old-skill", "SKILL.md"),
        "stale",
        "utf8",
      );
      await fs.writeFile(nodePath.join(tmpDir, "README.txt"), "keep me", "utf8");

      mockMaterializeNoFiles();
      await client.skills.materialize(tmpDir, {
        workspaceId: "ws-1",
        enabled: true,
        reconcile: true,
      });

      // Stale skill removed.
      await expect(
        fs.access(nodePath.join(tmpDir, "old-skill")),
      ).rejects.toThrow();
      // Desired skill present.
      await expect(
        fs.access(nodePath.join(tmpDir, "pdf-extraction", "SKILL.md")),
      ).resolves.toBeUndefined();
      // Non-skill file left untouched.
      expect(
        await fs.readFile(nodePath.join(tmpDir, "README.txt"), "utf8"),
      ).toBe("keep me");
    });

    it("reconcile defaults to false: stale skill dir is left in place (additive)", async () => {
      await fs.mkdir(nodePath.join(tmpDir, "old-skill"), { recursive: true });
      await fs.writeFile(
        nodePath.join(tmpDir, "old-skill", "SKILL.md"),
        "stale",
        "utf8",
      );

      mockMaterializeNoFiles();
      await client.skills.materialize(tmpDir, {
        workspaceId: "ws-1",
        enabled: true,
      });

      // Stale skill still present — additive behavior unchanged.
      await expect(
        fs.access(nodePath.join(tmpDir, "old-skill", "SKILL.md")),
      ).resolves.toBeUndefined();
    });

    it("reconcile does not prune when the list() call fails", async () => {
      await fs.mkdir(nodePath.join(tmpDir, "old-skill"), { recursive: true });
      await fs.writeFile(
        nodePath.join(tmpDir, "old-skill", "SKILL.md"),
        "stale",
        "utf8",
      );

      // list() rejects → materialize must reject before any prune runs.
      (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("server unreachable"),
      );

      await expect(
        client.skills.materialize(tmpDir, {
          workspaceId: "ws-1",
          enabled: true,
          reconcile: true,
        }),
      ).rejects.toThrow();

      // Nothing pruned despite reconcile=true.
      await expect(
        fs.access(nodePath.join(tmpDir, "old-skill", "SKILL.md")),
      ).resolves.toBeUndefined();
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
    // save() = resolve -> create -> re-fetch; mock the full sequence. The first
    // call (resolve) must carry the OBO headers from the scoped namespace.
    (global.fetch as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce(mockOkJson({ skill: null })) // resolve: not found
      .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })) // POST create
      .mockResolvedValueOnce(mockOkJson({ skill: mockSkill })); // get re-fetch
    const proxy = client.actingFor({ grantId: "g_bob", subject: { type: "user", id: "bob" } });
    await proxy.skills.save({ name: "test", description: "desc", body: "body" });
    const [, init] = (global.fetch as ReturnType<typeof vi.fn>).mock.calls[0] as [string, RequestInit & { headers: Record<string, string> }];
    expect(init.headers["X-Aether-Subject-ID"]).toBe("bob");
  });
});


describe("browser skill helpers", () => {
  it("rejects filesystem operations clearly without a Node runtime", async () => {
    const { parseSkillFolder } = await import("../src/skills.js");
    const client = new MemoryLayerClient({ baseUrl: "http://localhost:61001" });
    vi.stubGlobal("process", undefined);
    try {
      await expect(parseSkillFolder("/skills")).rejects.toThrow("require Node.js");
      await expect(client.skills.materialize("/skills")).rejects.toThrow("require Node.js");
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
