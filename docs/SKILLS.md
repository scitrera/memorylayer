# Skills

Skills are agent harness knowledge units: a `SKILL.md` manifest plus an optional bundle of scripts, references, and assets stored in MemoryLayer and made available to any agent that shares the workspace.

## Overview

A skill is the MemoryLayer implementation of the [agentskills.io](https://agentskills.io) spec. Each skill has:

- A `SKILL.md` file with YAML frontmatter (name, description, version, license, etc.) and a markdown body containing instructions, examples, and usage guidance.
- An optional file bundle organized under `scripts/`, `references/`, and `assets/` subdirectories.
- A memory mirror: when a skill is created or updated, the server writes a procedural memory (`subtype=skill`) so skills are discoverable via vector recall alongside regular memories.

**OSS vs Enterprise:** The full CRUD and file bundle API exists in OSS. Tenant-level RBAC on skill visibility and filesystem-sync daemons are Enterprise concerns. In OSS, scoping is workspace + user-private; tenant scope maps to the single `default_tenant`.

---

## Concepts

### Skill

The top-level record. Identified by `id` (`skl_<12hex>`). Scoped to a workspace and optionally to a specific user (user-private skills).

### SkillFile

A single file within a skill bundle. Files are stored inline in OSS/SQLite. The `kind` is derived from the top-level directory: `scripts/` → `script`, `references/` → `reference`, `assets/` → `asset`, anything else → `other`.

### Scoping

Skills have a 3-tier scope within a workspace:

| Scope | Storage encoding | Visible to |
|-------|-----------------|------------|
| user | `user_id` set + `workspace_id` = caller's workspace | Owning user only |
| workspace | `user_id` null + `workspace_id` = caller's workspace | All users in workspace |
| global | `user_id` null + `workspace_id` = `_global` | All workspaces in tenant |

### Precedence

When multiple skills share the same name, the resolution service returns the precedence winner:

```
user > workspace > tenant/global
```

Within the same scope, `source_mode` breaks ties: `server > mirrored > filesystem`, then most-recently updated.

The `GET /v1/skills` endpoint applies shadowing by default (`include_shadowed=false`), returning only the winner per name. Pass `include_shadowed=true` to see all copies.

### Source Modes

| Mode | Meaning |
|------|---------|
| `server` | MemoryLayer is canonical; filesystem is a checkout |
| `filesystem` | Local directory is canonical; server is a mirror |
| `mirrored` | Bidirectional sync; conflict detection via hash comparison |

### Memory Mirror

On create/update, the server writes a procedural memory with `subtype="skill"` and `metadata.skill_id` pointing to the skill record. This makes skills searchable via `POST /v1/skills/resolve` with a `query` (vector recall) in addition to exact-name lookup.

---

## Data Model

### Skill fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Primary key (`skl_<12hex>`) |
| `tenant_id` | `str` | Tenant scope (empty string in OSS) |
| `workspace_id` | `str` | Workspace scope; `_global` for tenant-wide skills |
| `user_id` | `str \| null` | Set for user-private skills |
| `name` | `str` | 1–64 chars, `[a-z0-9-]`, no leading/trailing/consecutive hyphens |
| `description` | `str` | 1–1024 chars |
| `version` | `str` | Semver-ish string (default `0.1.0`) |
| `license` | `str \| null` | License identifier (e.g. `MIT`) |
| `compatibility` | `str \| null` | Compatibility notes (≤500 chars) |
| `allowed_tools` | `str \| null` | Space-separated tool allowlist (experimental spec field) |
| `body` | `str` | `SKILL.md` body (post-frontmatter markdown) |
| `metadata` | `dict` | Arbitrary spec-allowed extras |
| `source_mode` | `"server" \| "filesystem" \| "mirrored"` | Canonical storage location |
| `manifest_hash` | `str` | SHA-256 of canonical `SKILL.md` |
| `bundle_hash` | `str` | SHA-256 over sorted (path, hash) pairs of all skill files |
| `enabled` | `bool` | Whether the skill is active (default `true`) |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

### SkillFile fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Primary key (`sklf_<12hex>`) |
| `skill_id` | `str` | Parent skill ID |
| `path` | `str` | Relative path within skill root (e.g. `scripts/extract.py`) |
| `kind` | `"script" \| "reference" \| "asset" \| "other"` | Derived from top-level directory |
| `content` | `bytes` | Raw file content |
| `content_hash` | `str` | SHA-256 of content |
| `size_bytes` | `int` | File size in bytes |
| `mime_type` | `str \| null` | MIME type (sniffed if not provided) |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

---

## API Reference

All endpoints are under `/v1/skills`. Workspace scope is read from the `X-Workspace-ID` header or the `workspace_id` query/body parameter.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/skills` | Create a skill |
| `GET` | `/v1/skills` | List skills (shadowed by default) |
| `GET` | `/v1/skills/{id}` | Get a skill by ID |
| `GET` | `/v1/skills/{id}/manifest` | Render full `SKILL.md` text (`text/markdown`) |
| `GET` | `/v1/skills/{id}/files` | List bundle file metadata |
| `GET` | `/v1/skills/{id}/files/{path}` | Stream a single file from the bundle |
| `GET` | `/v1/skills/{id}/bundle` | Stream full bundle as NDJSON or `tar.gz` |
| `PUT` | `/v1/skills/{id}` | Update manifest fields |
| `PUT` | `/v1/skills/{id}/files/{path}` | Upsert a file in the bundle |
| `DELETE` | `/v1/skills/{id}` | Delete skill and all its files |
| `POST` | `/v1/skills/resolve` | Resolve by name (precedence) or query (vector recall) |
| `POST` | `/v1/skills/{id}/sync` | Compare client/server hashes; returns `push \| pull \| conflict \| in_sync` |

### List query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_id` | `str` | from header | Workspace to query |
| `name` | `str` | — | Filter by exact name |
| `enabled` | `bool` | — | Filter by enabled status |
| `include_shadowed` | `bool` | `false` | Include all copies; default returns only precedence winner per name |
| `limit` | `int` | `100` | Max results (1–500) |
| `offset` | `int` | `0` | Pagination offset |

### Bundle format

`GET /v1/skills/{id}/bundle?format=ndjson` (default) streams NDJSON lines:

```json
{"type": "header", "skill_id": "...", "skill_name": "...", "version": "0.1.0", "file_count": 2}
{"type": "file", "path": "scripts/run.py", "kind": "script", "content_b64": "...", "content_hash": "...", "size_bytes": 512, "mime_type": "text/x-python"}
{"type": "footer", "file_count": 2, "bundle_hash": "..."}
```

Pass `format=tar.gz` to receive a gzip-compressed tar archive.

---

## SDK Usage

### Python (async)

```python
import asyncio
from pathlib import Path
from memorylayer import MemoryLayerClient

async def main():
    async with MemoryLayerClient(base_url="http://localhost:61001", workspace_id="my-project") as client:

        # Create a skill
        skill = await client.skills.save(
            name="pdf-extraction",
            description="Extract tables and text from PDF files",
            body="## Usage\n\nCall `scripts/extract.py` with the PDF path as the first argument.",
            version="1.0.0",
            license="MIT",
        )
        print(skill.id)  # skl_abc123...

        # List visible skills (shadowed filtered by default)
        skills = await client.skills.list()

        # Get a skill by ID
        skill = await client.skills.get(skill.id)

        # Get rendered SKILL.md text
        manifest_text = await client.skills.get_manifest(skill.id)

        # Resolve by name — returns precedence winner
        winner = await client.skills.resolve(name="pdf-extraction")

        # Resolve by intent — vector recall against skill memory mirror
        candidates = await client.skills.resolve(query="how do I extract tables from PDFs?")

        # Pull a named skill to disk
        skill_dir = await client.skills.pull("pdf-extraction", out_dir=Path("/tmp/skills"))
        # writes /tmp/skills/pdf-extraction/SKILL.md + bundle files

        # Push a skill directory to server
        skill = await client.skills.push(Path("/tmp/skills/pdf-extraction"))

        # Bulk pull all visible skills to a directory (idempotent via bundle_hash)
        paths = await client.skills.materialize(Path("~/.claude/skills").expanduser())

asyncio.run(main())
```

### Python (sync)

```python
from pathlib import Path
from memorylayer import SyncMemoryLayerClient

with SyncMemoryLayerClient(base_url="http://localhost:61001", workspace_id="my-project") as client:
    skills = client.skills.list(enabled=True)
    skill = client.skills.resolve(name="pdf-extraction")
    client.skills.pull("pdf-extraction", out_dir=Path("~/.claude/skills").expanduser())
```

### TypeScript

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", workspaceId: "my-project" });

// List skills
const skills = await client.skills.list({ enabled: true });

// Save a skill
const skill = await client.skills.save({
  name: "pdf-extraction",
  description: "Extract tables and text from PDF files",
  body: "## Usage\n\nSee scripts/extract.py",
  version: "1.0.0",
  license: "MIT",
});

// Resolve by name
const winner = await client.skills.resolve({ name: "pdf-extraction" });

// Materialize all skills to disk (Node.js only)
await client.skills.materialize("/home/user/.claude/skills");
```

---

## MCP Tools

Skills tools are available in the `full` tool profile. The `cc` (Claude Code default) profile includes read-only tools (`skills_list`, `skills_get`, `skills_get_file`, `skills_search`); `skills_save` requires the `full` profile.

| Tool | Required params | Description |
|------|----------------|-------------|
| `skills_list` | — | List skills; returns name, description, version per skill |
| `skills_get` | `skill_id` or `name` | Fetch full `SKILL.md` body for a skill |
| `skills_get_file` | `skill_id`, `path` | Fetch a single bundle file by relative path |
| `skills_search` | `query` | Vector recall by intent/description |
| `skills_save` | `name`, `description` | Create or update a skill (full profile only) |

`skills_save` optional params: `body`, `version`, `license`, `compatibility`, `allowed_tools`, `source_mode`, `metadata`, `files` (array of `{path, content_b64, mime_type}`).

---

## CLI

The `memorylayer skills` command group requires `memorylayer-server` installed and a running server at `http://localhost:61001` (override with `--server-url`).

```bash
# List visible skills in a workspace
memorylayer skills list --workspace my-project

# Show all copies including shadowed duplicates
memorylayer skills list --show-shadowed

# Push a skill directory to the server
memorylayer skills push ./my-skill-dir --workspace my-project --scope workspace

# Pull a named skill to disk
memorylayer skills pull pdf-extraction --output /tmp/skills --workspace my-project

# Bulk pull all visible skills to ~/.claude/skills
memorylayer skills materialize --workspace my-project

# Sync a skill directory (compare hashes, pull or push as needed)
memorylayer skills sync ./my-skill-dir --workspace my-project

# Watch a directory and push on file change (mirrored mode)
memorylayer skills watch ./my-skill-dir --workspace my-project

# Import all skills from ~/.claude/skills into MemoryLayer
memorylayer skills migrate-from-local --scope user
```

All commands accept `--server-url URL` and `--api-key KEY` for non-default endpoints.

---

## SKILL.md Format

`SKILL.md` uses YAML frontmatter between `---` fences followed by a markdown body. The parser uses stdlib only (no PyYAML required). Fields are rendered in canonical order.

```markdown
---
name: pdf-extraction
description: "Extract tables and text from PDF files"
version: 1.0.0
license: MIT
compatibility: "Requires Python 3.11+"
allowed-tools: computer bash
metadata: {"vendor": "acme"}
---

## Overview

This skill extracts structured data from PDF files using `pdfplumber`.

## Usage

```bash
python scripts/extract.py path/to/file.pdf
```

## Examples

...
```

### Frontmatter field rules

| Field | Required | Constraints |
|-------|----------|-------------|
| `name` | yes | 1–64 chars, `[a-z0-9-]`, no leading/trailing/consecutive hyphens |
| `description` | yes | 1–1024 chars |
| `version` | no | Semver-ish (default `0.1.0`) |
| `license` | no | SPDX identifier recommended (e.g. `MIT`, `Apache-2.0`) |
| `compatibility` | no | Free text, ≤500 chars |
| `allowed-tools` | no | Space-separated tool names (experimental; not enforced server-side) |
| `metadata` | no | Arbitrary key-value pairs; stored in the `metadata` dict field |

Extra frontmatter keys not in the canonical list are collected into the `metadata` dict on ingest.

### File bundle layout

```
my-skill/
  SKILL.md           # required — manifest + instructions
  scripts/           # executable scripts → kind="script"
    extract.py
  references/        # data files, configs → kind="reference"
    schema.json
  assets/            # static assets → kind="asset"
    logo.png
```

Files outside `scripts/`, `references/`, `assets/` at the skill root are also collected with `kind="other"`.

---

## Local File Integration

Skills round-trip with the local filesystem via `pull`, `push`, `materialize`, and `sync`.

**Default local paths:**

- `~/.claude/skills/` — user-scope skills shared across projects
- `.claude/skills/` — project-scope skills (not read by `memorylayer skills materialize` by default)

`memorylayer skills materialize` writes all visible workspace skills to `~/.claude/skills/<skill-name>/` and tracks freshness via a `.bundle_hash` file inside each skill directory. Skills whose `bundle_hash` matches the stored value are skipped (idempotent).

`memorylayer skills migrate-from-local` reads the directory specified by `--dir` (default `~/.claude/skills` or `MEMORYLAYER_SKILLS_LOCAL_PATHS` env var) and pushes all valid skill directories to MemoryLayer.

---

## Verification Recipes

### Scope precedence test

Create the same skill name at two scopes, verify the higher-precedence one wins:

```python
import asyncio
from memorylayer import MemoryLayerClient

async def test_precedence():
    async with MemoryLayerClient(base_url="http://localhost:61001", workspace_id="test") as client:
        # workspace-scoped skill
        await client.skills.save(name="my-skill", description="Workspace copy", body="workspace")

        # user-scoped skill (user_id must be set on create via SkillCreateInput)
        # use the raw API to set user_id
        import httpx
        async with httpx.AsyncClient() as http:
            r = await http.post("http://localhost:61001/v1/skills", json={
                "name": "my-skill",
                "description": "User copy",
                "body": "user",
                "workspace_id": "test",
                "user_id": "alice",
            })
            r.raise_for_status()

        winner = await client.skills.resolve(name="my-skill")
        # winner.body == "user" if user_id header is alice
        print(winner.body)

asyncio.run(test_precedence())
```

### Memory-mirror discovery

```python
async with MemoryLayerClient(base_url="http://localhost:61001", workspace_id="test") as client:
    # Push a skill
    await client.skills.save(
        name="table-extraction",
        description="Extract tables from documents using regex",
        body="...",
    )

    # Discover by intent (routes through procedural memory recall)
    candidates = await client.skills.resolve(query="how do I extract tables?")
    print([s.name for s in candidates])  # ["table-extraction"]
```

### Hybrid-mode sync

```bash
# 1. Push local directory to server (sets manifest_hash)
memorylayer skills push ./my-skill --mode mirrored

# 2. Check sync status (returns push | pull | conflict | in_sync)
curl -X POST http://localhost:61001/v1/skills/skl_abc123/sync \
  -H 'Content-Type: application/json' \
  -d '{"manifest_hash": "<local-sha256>", "bundle_hash": "<local-bundle-sha256>"}'
# {"action": "in_sync", ...}

# 3. After local changes, sync pulls winner automatically
memorylayer skills sync ./my-skill --auto-push
```
