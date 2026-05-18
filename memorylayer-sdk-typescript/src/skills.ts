import type { MemoryLayerClient } from "./client.js";
import type { AuthorityContext } from "./types.js";

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
  ): Promise<T> {
    return (this._client as unknown as ClientInternal)._skillsRequest<T>(
      method,
      path,
      body,
      authority ?? this._authority,
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

  async getManifest(skillId: string): Promise<string> {
    return this._req<string>("GET", `/v1/skills/${skillId}/manifest`);
  }

  async listFiles(skillId: string): Promise<SkillFile[]> {
    const response = await this._req<{ files: SkillFile[] }>(
      "GET",
      `/v1/skills/${skillId}/files`,
    );
    return response.files;
  }

  async getFile(skillId: string, filePath: string): Promise<SkillFile> {
    const encoded = encodeURIComponent(filePath);
    const response = await this._req<{ file: SkillFile }>(
      "GET",
      `/v1/skills/${skillId}/files/${encoded}`,
    );
    return response.file;
  }

  async save(manifest: SkillManifest): Promise<Skill> {
    const body: Record<string, unknown> = {
      name: manifest.name,
      description: manifest.description,
      version: manifest.version ?? "0.1.0",
      body: manifest.body,
      metadata: manifest.metadata ?? {},
      source_mode: manifest.source_mode ?? "server",
    };
    if (manifest.license !== undefined) body.license = manifest.license;
    if (manifest.compatibility !== undefined)
      body.compatibility = manifest.compatibility;
    if (manifest.allowed_tools !== undefined)
      body.allowed_tools = manifest.allowed_tools;
    if (manifest.files) body.files = manifest.files;
    if (this._workspaceId) body.workspace_id = this._workspaceId;
    const response = await this._req<{ skill: Skill }>("POST", "/v1/skills", body);
    return response.skill;
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

  async materialize(
    targetDir: string,
    options: { workspaceId?: string; scope?: string } = {},
  ): Promise<void> {
    const fs = await import("fs/promises");
    const path = await import("path");
    const skills = await this.list({
      workspaceId: options.workspaceId ?? this._workspaceId,
      scope: options.scope,
    });
    for (const skill of skills) {
      const skillDir = path.join(targetDir, skill.name);
      await fs.mkdir(skillDir, { recursive: true });
      const manifest = await this.getManifest(skill.id);
      await fs.writeFile(path.join(skillDir, "SKILL.md"), manifest, "utf8");
      const files = await this.listFiles(skill.id);
      for (const f of files) {
        const filePath = path.join(skillDir, f.path);
        await fs.mkdir(path.dirname(filePath), { recursive: true });
        await fs.writeFile(filePath, f.content, "utf8");
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
