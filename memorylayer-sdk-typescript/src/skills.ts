import type { MemoryLayerClient } from "./client.js";
import type { AuthorityContext } from "./types.js";
import { MemoryLayerError } from "./errors.js";

/**
 * UTF-8 encode a string and base64 it for the skill-file upsert endpoint.
 * Uses Node's Buffer when present (the SDK's skill helpers are Node-oriented:
 * parseSkillFolder/materialize use fs), falling back to btoa in the browser.
 */
function _toBase64(content: string): string {
  const g = globalThis as unknown as {
    Buffer?: { from(s: string, enc: string): { toString(enc: string): string } };
    btoa?: (s: string) => string;
    TextEncoder?: new () => { encode(s: string): Uint8Array };
  };
  if (g.Buffer) {
    return g.Buffer.from(content, "utf-8").toString("base64");
  }
  if (g.btoa && g.TextEncoder) {
    const bytes = new g.TextEncoder().encode(content);
    let binary = "";
    for (const b of bytes) binary += String.fromCharCode(b);
    return g.btoa(binary);
  }
  throw new MemoryLayerError(
    "No base64 encoder available (need Node Buffer or browser btoa/TextEncoder)",
  );
}

// ------------------------------------------------------------------ //
// Skill types
// ------------------------------------------------------------------ //

export interface Skill {
  id: string;
  tenant_id: string;
  workspace_id: string;
  user_id?: string;
  name: string;
  description: string;
  version: string;
  license?: string;
  compatibility?: string;
  allowed_tools?: string;
  body: string;
  metadata: Record<string, unknown>;
  source_mode: "server" | "filesystem" | "mirrored";
  manifest_hash: string;
  bundle_hash: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface SkillFile {
  id: string;
  skill_id: string;
  path: string;
  kind: "script" | "reference" | "asset" | "other";
  content: string;
  content_hash: string;
  size_bytes: number;
  mime_type?: string;
  created_at: string;
  updated_at: string;
}

export interface SkillManifest {
  name: string;
  description: string;
  version?: string;
  license?: string;
  compatibility?: string;
  allowed_tools?: string;
  body: string;
  metadata?: Record<string, unknown>;
  files?: Array<{ path: string; content: string }>;
  source_mode?: "server" | "filesystem" | "mirrored";
}

export interface SkillListOptions {
  workspaceId?: string;
  scope?: string;
  name?: string;
  tags?: string[];
  enabled?: boolean;
  includeShadowed?: boolean;
  authority?: AuthorityContext;
}

export interface SkillResolveOptions {
  name?: string;
  query?: string;
  scopeHint?: string;
  authority?: AuthorityContext;
}

export interface SkillSyncResult {
  action: "push" | "pull" | "conflict" | "in_sync";
  manifest_hash?: string;
  bundle_hash?: string;
}

export interface ParsedSkillFolder {
  manifest: SkillManifest;
  files: Array<{ path: string; content: string }>;
}

// ------------------------------------------------------------------ //
// SkillsNamespace — attached as client.skills
// ------------------------------------------------------------------ //

export class SkillsNamespace {
  private _client: MemoryLayerClient;
  private _authority?: AuthorityContext;
  private _workspaceId?: string;

  constructor(
    client: MemoryLayerClient,
    authority?: AuthorityContext,
    workspaceId?: string,
  ) {
    this._client = client;
    this._authority = authority;
    this._workspaceId = workspaceId;
  }

  /** Return a scoped proxy that injects OBO headers and workspace on every call. */
  withAuthority(
    authority: AuthorityContext,
    workspaceId?: string,
  ): SkillsNamespace {
    return new SkillsNamespace(
      this._client,
      authority,
      workspaceId ?? this._workspaceId,
    );
  }

  private _req<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
    responseMode: "json" | "text" | "arraybuffer" = "json",
  ): Promise<T> {
    return (this._client as unknown as ClientInternal)._skillsRequest<T>(
      method,
      path,
      body,
      authority ?? this._authority,
      responseMode,
    );
  }

  async list(options: SkillListOptions = {}): Promise<Skill[]> {
    const params = new URLSearchParams();
    const wsId = options.workspaceId ?? this._workspaceId;
    if (wsId) params.set("workspace_id", wsId);
    if (options.scope) params.set("scope", options.scope);
    if (options.name) params.set("name", options.name);
    if (options.enabled !== undefined)
      params.set("enabled", String(options.enabled));
    if (options.includeShadowed)
      params.set("include_shadowed", "true");
    if (options.tags?.length)
      params.set("tags", options.tags.join(","));
    const query = params.toString();
    const response = await this._req<{ skills: Skill[] }>(
      "GET",
      `/v1/skills${query ? `?${query}` : ""}`,
      undefined,
      options.authority,
    );
    return response.skills;
  }

  async get(skillId: string): Promise<Skill> {
    const response = await this._req<{ skill: Skill }>(
      "GET",
      `/v1/skills/${skillId}`,
    );
    return response.skill;
  }

  /**
   * Render the full `SKILL.md` for a skill.
   *
   * The server returns raw markdown (`Content-Type: text/markdown`), not JSON,
   * so this reads the body as text rather than parsing it.
   */
  async getManifest(skillId: string): Promise<string> {
    return this._req<string>(
      "GET",
      `/v1/skills/${skillId}/manifest`,
      undefined,
      undefined,
      "text",
    );
  }

  async listFiles(skillId: string): Promise<SkillFile[]> {
    const response = await this._req<{ files: SkillFile[] }>(
      "GET",
      `/v1/skills/${skillId}/files`,
    );
    return response.files;
  }

  /**
   * Fetch only the raw content of a skill file (no metadata lookup).
   *
   * Used internally by `materialize()` to avoid the extra `listFiles()` call
   * that `getFile()` performs. The caller already has the full `SkillFile`
   * metadata from the single `listFiles()` done by `materialize()`, so we only
   * need the raw bytes here.
   */
  private async _fetchFileContent(
    skillId: string,
    filePath: string,
  ): Promise<string> {
    const encoded = filePath
      .split("/")
      .map((seg) => encodeURIComponent(seg))
      .join("/");
    return this._req<string>(
      "GET",
      `/v1/skills/${skillId}/files/${encoded}`,
      undefined,
      undefined,
      "text",
    );
  }

  /**
   * Fetch a single skill file with its content populated.
   *
   * `GET /v1/skills/{id}/files/{path}` streams the raw file bytes (e.g.
   * `text/x-python`), NOT a JSON envelope — so we read the body as text and
   * merge it onto the file's metadata from `listFiles()`. The metadata lookup
   * is best-effort: if the path isn't in the listing we still return the
   * fetched content with the path filled in.
   */
  async getFile(skillId: string, filePath: string): Promise<SkillFile> {
    const encoded = filePath
      .split("/")
      .map((seg) => encodeURIComponent(seg))
      .join("/");
    const [content, files] = await Promise.all([
      this._req<string>(
        "GET",
        `/v1/skills/${skillId}/files/${encoded}`,
        undefined,
        undefined,
        "text",
      ),
      this.listFiles(skillId),
    ]);
    const info = files.find((f) => f.path === filePath);
    return {
      ...(info ?? ({ path: filePath } as Partial<SkillFile>)),
      skill_id: skillId,
      path: filePath,
      content,
    } as SkillFile;
  }

  /**
   * Delete a single file from a skill bundle.
   *
   * Calls `DELETE /v1/skills/{id}/files/{path}` (per-segment encodeURIComponent
   * so embedded `/` separators are preserved, matching `getFile`). The server
   * responds 204 with an empty body and is idempotent — deleting an
   * already-absent path still succeeds — so the request helper's 204 handling
   * means no JSON parsing happens here.
   */
  async deleteFile(skillId: string, filePath: string): Promise<void> {
    const encoded = filePath
      .split("/")
      .map((seg) => encodeURIComponent(seg))
      .join("/");
    const wsQuery = this._workspaceId
      ? `?workspace_id=${encodeURIComponent(this._workspaceId)}`
      : "";
    await this._req<void>(
      "DELETE",
      `/v1/skills/${skillId}/files/${encoded}${wsQuery}`,
    );
  }

  /**
   * Create or update a skill AND persist its bundle files.
   *
   * The server's create/update inputs (`POST /v1/skills`, `PUT /v1/skills/{id}`)
   * carry NO `files` field — files are persisted out-of-band via
   * `PUT /v1/skills/{id}/files/{path}` with base64 `content_b64`. A previous
   * version sent `manifest.files` on the create body, where the server silently
   * ignored it, so a skill saved with files lost them. This method now:
   *
   *  1. Detects create-vs-update by resolving the skill by name (the unique key
   *     within a scope). Found → UPDATE (`PUT /v1/skills/{id}`); else CREATE
   *     (`POST /v1/skills`). Neither create nor update bodies include `files`.
   *  2. Upserts every manifest file via the dedicated file endpoint.
   *  3. On UPDATE, performs a TRUE file reconcile: it lists the skill's current
   *     server files and DELETEs (`DELETE /v1/skills/{id}/files/{path}`) any whose
   *     path is no longer in the manifest, so the saved skill's bundle is an exact
   *     mirror of the manifest. (CREATE has no pre-existing files to prune.)
   *  4. If the skill create/update succeeds but ANY file PUT fails, this RAISES a
   *     clear error rather than resolving — never silently dropping files. File
   *     PUTs are idempotent upserts, so the failed save is safe to retry.
   *
   * Returns the fully-persisted skill (re-fetched for fresh manifest/bundle
   * hashes). The public signature and return type are unchanged.
   */
  async save(manifest: SkillManifest): Promise<Skill> {
    const files = manifest.files ?? [];

    // Detect an existing skill by name (unique within scope). resolve() honors
    // precedence/shadowing, matching how the rest of the SDK identifies skills.
    let existing: Skill | null = null;
    try {
      existing = await this.resolve({
        name: manifest.name,
        authority: this._authority,
      });
    } catch {
      // resolve failures (e.g. 404-style empties) are treated as "not found";
      // a hard create attempt below surfaces any real error.
      existing = null;
    }

    let skillId: string;
    if (existing) {
      // ── UPDATE path: PUT manifest fields (no files), then reconcile files ──
      const updateBody: Record<string, unknown> = {
        description: manifest.description,
        body: manifest.body,
        metadata: manifest.metadata ?? {},
      };
      if (manifest.version !== undefined) updateBody.version = manifest.version;
      if (manifest.license !== undefined) updateBody.license = manifest.license;
      if (manifest.compatibility !== undefined)
        updateBody.compatibility = manifest.compatibility;
      if (manifest.allowed_tools !== undefined)
        updateBody.allowed_tools = manifest.allowed_tools;
      if (manifest.source_mode !== undefined)
        updateBody.source_mode = manifest.source_mode;
      const wsQuery = this._workspaceId
        ? `?workspace_id=${encodeURIComponent(this._workspaceId)}`
        : "";
      await this._req<{ skill: Skill }>(
        "PUT",
        `/v1/skills/${existing.id}${wsQuery}`,
        updateBody,
      );
      skillId = existing.id;

      // Reconcile deletions: list the skill's current server files and DELETE
      // any whose path is no longer in the manifest, so the saved bundle is an
      // exact mirror of the manifest. A delete failure must surface (don't
      // silently diverge), so this is NOT wrapped in a swallow-all catch.
      const serverFiles = await this.listFiles(skillId);
      const manifestPaths = new Set(files.map((f) => f.path));
      const stale = serverFiles
        .map((f) => f.path)
        .filter((p) => !manifestPaths.has(p));
      for (const path of stale) {
        await this.deleteFile(skillId, path);
      }
    } else {
      // ── CREATE path: POST manifest MINUS files, capture new id ──
      const createBody: Record<string, unknown> = {
        name: manifest.name,
        description: manifest.description,
        version: manifest.version ?? "0.1.0",
        body: manifest.body,
        metadata: manifest.metadata ?? {},
        source_mode: manifest.source_mode ?? "server",
      };
      if (manifest.license !== undefined) createBody.license = manifest.license;
      if (manifest.compatibility !== undefined)
        createBody.compatibility = manifest.compatibility;
      if (manifest.allowed_tools !== undefined)
        createBody.allowed_tools = manifest.allowed_tools;
      if (this._workspaceId) createBody.workspace_id = this._workspaceId;
      const created = await this._req<{ skill: Skill }>(
        "POST",
        "/v1/skills",
        createBody,
      );
      skillId = created.skill.id;
    }

    // ── Persist each manifest file via the dedicated upsert endpoint ──
    // A failure here, AFTER the skill create/update succeeded, must RAISE (never
    // resolve) so files are never silently dropped. PUTs are idempotent upserts,
    // so a failed save is retry-safe; we do NOT auto-delete the skill.
    const wsQuery = this._workspaceId
      ? `?workspace_id=${encodeURIComponent(this._workspaceId)}`
      : "";
    for (const file of files) {
      const encoded = file.path
        .split("/")
        .map((seg) => encodeURIComponent(seg))
        .join("/");
      try {
        await this._req<unknown>(
          "PUT",
          `/v1/skills/${skillId}/files/${encoded}${wsQuery}`,
          { content_b64: _toBase64(file.content) },
        );
      } catch (err) {
        throw new MemoryLayerError(
          `skill ${skillId} (${manifest.name}) saved but file ${file.path} ` +
            `failed to persist: ${err instanceof Error ? err.message : String(err)}. ` +
            `File upserts are idempotent — retry save() to complete persistence.`,
        );
      }
    }

    // Re-fetch so callers get fresh manifest/bundle hashes reflecting files.
    return this.get(skillId);
  }

  async delete(skillId: string): Promise<void> {
    await this._req<void>("DELETE", `/v1/skills/${skillId}`);
  }

  async resolve(options: SkillResolveOptions = {}): Promise<Skill | null> {
    const body: Record<string, unknown> = {};
    if (options.name) body.name = options.name;
    if (options.query) body.query = options.query;
    if (options.scopeHint) body.scope_hint = options.scopeHint;
    const response = await this._req<{ skill: Skill | null }>(
      "POST",
      "/v1/skills/resolve",
      body,
      options.authority,
    );
    return response.skill;
  }

  async pull(
    skillId: string,
  ): Promise<{ skill: Skill; files: SkillFile[] }> {
    const [skill, files] = await Promise.all([
      this.get(skillId),
      this.listFiles(skillId),
    ]);
    return { skill, files };
  }

  async push(manifest: SkillManifest): Promise<Skill> {
    return this.save(manifest);
  }

  /**
   * Write every matching skill's `SKILL.md` + files into `targetDir`.
   *
   * The set of skills written is whatever the internal `list(...)` returns for
   * the given `workspaceId`/`scope`. By default `enabled` is left unset, so
   * both enabled and disabled server-visible skills are materialized (backwards
   * compatible). Pass `enabled: true` to restrict the on-disk cache to the
   * enabled subset (e.g. the OpenClaw skills-sync hook does this so disabled
   * skills are never written into the agent's watched skills dir).
   *
   * By default materialize is purely additive: it writes/overwrites the desired
   * skills but never removes anything already on disk. Pass `reconcile: true` to
   * turn the sync into a true reconcile — after successfully listing AND writing
   * the desired skills, any top-level entry in `targetDir` whose name is NOT in
   * the desired skill-name set is removed. This is how a skill that was disabled,
   * deleted, or renamed in MemoryLayer gets its stale on-disk copy pruned.
   *
   * CONTRACT (reconcile only): `targetDir` MUST be an exclusively
   * MemoryLayer-managed directory (e.g. the OpenClaw plugin's dedicated
   * `extraDir`). Reconcile will delete subdirectories it did not write. As a
   * defensive guard it only removes a directory that *looks* like a materialized
   * skill (i.e. contains a `SKILL.md`); unrelated files and non-skill directories
   * are left untouched. Pruning runs ONLY after the list+materialize succeeded,
   * so a transient server/list failure can never wipe the directory.
   */
  async materialize(
    targetDir: string,
    options: {
      workspaceId?: string;
      scope?: string;
      enabled?: boolean;
      reconcile?: boolean;
    } = {},
  ): Promise<void> {
    if (typeof process === "undefined" || !process.versions?.node) {
      throw new MemoryLayerError("Skill folder operations require Node.js");
    }
    const fs = await import("fs/promises");
    const path = await import("path");
    // If list() throws, this rejects before any prune — a transient server
    // outage can never trigger a wipe of targetDir.
    const skills = await this.list({
      workspaceId: options.workspaceId ?? this._workspaceId,
      scope: options.scope,
      enabled: options.enabled,
    });
    // Desired set: the exact directory names materialize writes below.
    const desiredNames = new Set(skills.map((skill) => skill.name));
    for (const skill of skills) {
      const skillDir = path.join(targetDir, skill.name);
      await fs.mkdir(skillDir, { recursive: true });
      // SKILL.md comes back as raw markdown text (not JSON) from getManifest.
      const manifest = await this.getManifest(skill.id);
      await fs.writeFile(path.join(skillDir, "SKILL.md"), manifest, "utf8");
      // listFiles() returns metadata for all files in one call. We then fetch
      // each file's raw bytes via _fetchFileContent() — which does a single
      // content GET with no extra listFiles() — instead of calling getFile(),
      // which would re-issue listFiles() per file. Net: 1 listFiles + N content
      // GETs rather than 1 + N listFiles calls.
      const files = await this.listFiles(skill.id);
      for (const f of files) {
        const filePath = path.join(skillDir, f.path);
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        const content = await this._fetchFileContent(skill.id, f.path);
        await fs.writeFile(filePath, content, "utf8");
      }
    }

    // Reconcile/prune: only after list+materialize succeeded above.
    if (options.reconcile) {
      let entries: import("fs").Dirent[];
      try {
        entries = await fs.readdir(targetDir, { withFileTypes: true });
      } catch {
        // targetDir doesn't exist (no skills were written, e.g. empty set);
        // nothing to prune.
        return;
      }
      for (const entry of entries) {
        if (!entry.isDirectory()) continue; // never touch loose files
        if (desiredNames.has(entry.name)) continue; // still wanted
        const dir = path.join(targetDir, entry.name);
        // Defensive: only remove dirs that look like a materialized skill.
        try {
          await fs.access(path.join(dir, "SKILL.md"));
        } catch {
          continue; // no SKILL.md → not ours, leave it alone
        }
        await fs.rm(dir, { recursive: true, force: true });
      }
    }
  }
}

// Internal interface to access the private request method from MemoryLayerClient.
// This avoids casting through `any` everywhere.
interface ClientInternal {
  _skillsRequest<T>(
    method: string,
    path: string,
    body?: unknown,
    authority?: AuthorityContext,
    responseMode?: "json" | "text" | "arraybuffer",
  ): Promise<T>;
}

// ------------------------------------------------------------------ //
// parseSkillFolder — Node.js helper
// ------------------------------------------------------------------ //

/**
 * Parse a skill folder from the filesystem.
 * Reads SKILL.md (frontmatter + body) and walks scripts/, references/, assets/.
 * Requires Node.js (uses fs/promises). Not available in browser environments.
 */
export async function parseSkillFolder(
  dir: string,
): Promise<ParsedSkillFolder> {
  if (typeof process === "undefined" || !process.versions?.node) {
    throw new MemoryLayerError("Skill folder operations require Node.js");
  }
  const fs = await import("fs/promises");
  const path = await import("path");

  const manifestPath = path.join(dir, "SKILL.md");
  const raw = await fs.readFile(manifestPath, "utf8");
  const { frontmatter, body } = _parseMarkdownFrontmatter(raw);

  const manifest: SkillManifest = {
    name: String(frontmatter.name ?? ""),
    description: String(frontmatter.description ?? ""),
    version: frontmatter.version ? String(frontmatter.version) : "0.1.0",
    body,
  };
  if (frontmatter.license) manifest.license = String(frontmatter.license);
  if (frontmatter.compatibility)
    manifest.compatibility = String(frontmatter.compatibility);
  if (frontmatter.allowed_tools)
    manifest.allowed_tools = String(frontmatter.allowed_tools);

  const extraKeys = new Set([
    "name",
    "description",
    "version",
    "license",
    "compatibility",
    "allowed_tools",
  ]);
  const metadata: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(frontmatter)) {
    if (!extraKeys.has(k)) metadata[k] = v;
  }
  manifest.metadata = metadata;

  const files: Array<{ path: string; content: string }> = [];
  const knownDirs = ["scripts", "references", "assets"];
  for (const subdir of knownDirs) {
    const subdirPath = path.join(dir, subdir);
    let entries: import("fs").Dirent[];
    try {
      entries = await fs.readdir(subdirPath, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      const filePath = path.join(subdirPath, entry.name);
      const content = await fs.readFile(filePath, "utf8");
      files.push({ path: `${subdir}/${entry.name}`, content });
    }
  }

  return { manifest: { ...manifest, files }, files };
}

/** Minimal YAML-ish frontmatter parser — handles string/number/bool values only. */
function _parseMarkdownFrontmatter(raw: string): {
  frontmatter: Record<string, unknown>;
  body: string;
} {
  const trimmed = raw.trimStart();
  if (!trimmed.startsWith("---")) {
    return { frontmatter: {}, body: raw };
  }
  const end = trimmed.indexOf("\n---", 3);
  if (end === -1) {
    return { frontmatter: {}, body: raw };
  }
  const yamlBlock = trimmed.slice(4, end);
  const body = trimmed.slice(end + 4).trimStart();

  const frontmatter: Record<string, unknown> = {};
  for (const line of yamlBlock.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx === -1) continue;
    const key = line.slice(0, colonIdx).trim();
    const val = line.slice(colonIdx + 1).trim();
    if (!key) continue;
    if (val === "true") frontmatter[key] = true;
    else if (val === "false") frontmatter[key] = false;
    else if (val !== "" && !isNaN(Number(val))) frontmatter[key] = Number(val);
    else frontmatter[key] = val.replace(/^["']|["']$/g, "");
  }
  return { frontmatter, body };
}
