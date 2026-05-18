# MCP Server Registry

MemoryLayer provides a registry for MCP (Model Context Protocol) server configurations. Agents can store, retrieve, and sync MCP server records so that tool configurations are available across sessions and machines without manually editing JSON files.

## Overview

Claude Code manages MCP servers across four scopes (`local`, `project`, `user`, `global`) spread across `.mcp.json`, `~/.claude.json`, and `claude_desktop_config.json`. MemoryLayer mirrors this 4-tier model as a persistent registry with:

- One row per server. A `.mcp.json` with N entries explodes to N records on push and collapses back to one file on export/materialize.
- Secret masking by default on all read endpoints. Literal `${VAR}` placeholders are passed through verbatim.
- A memory mirror: each server record is backed by a procedural memory (`subtype=mcp_server`) for intent-based discovery via vector recall.

**OSS vs Enterprise:** Full CRUD, scoping, and secret masking exist in OSS. At-rest encryption of `env` and `headers` values and RBAC-controlled visibility are Enterprise concerns. In OSS, secret values are stored as plaintext in SQLite; the masking layer hides them from API responses but does not encrypt them.

For Skills (agent knowledge units), see [SKILLS.md](SKILLS.md) — they follow the same scoping and source-mode patterns.

---

## Concepts

### McpServer

One record per MCP server. Identified by `id` (`mcp_<12hex>`). Transport determines which fields are required:

- `stdio` — requires `command`; `args` and `env` are optional.
- `http`, `sse`, `streamable-http` — requires `url`; `headers` is optional.

### Transports

| Value | Protocol |
|-------|---------|
| `stdio` | Local process via stdin/stdout |
| `http` | HTTP-based MCP |
| `sse` | Server-Sent Events |
| `streamable-http` | Streamable HTTP (Claude Code native) |

Transport is inferred during `.mcp.json` import: presence of `command` → `stdio`; presence of `url` → use the `type` field value or default to `http`.

### 4-Tier Scoping

MemoryLayer mirrors Claude Code's MCP scope model exactly:

| Tier | Rank | Storage encoding | Claude Code equivalent |
|------|------|-----------------|------------------------|
| LOCAL | 0 | `user_id` set + `workspace_id` = caller's workspace | `local` scope in `~/.claude.json` under the project path |
| PROJECT | 1 | `user_id` null + `workspace_id` = caller's workspace | `.mcp.json` in project root |
| USER | 2 | `user_id` set + `workspace_id` = `_global_user` | `user` scope in `~/.claude.json` top-level |
| GLOBAL | 3 | `user_id` null + `workspace_id` = `_global` | Global/plugin scope |

**Precedence:** LOCAL > PROJECT > USER > GLOBAL.

Within the same scope, `source_mode` breaks ties: `server > mirrored > filesystem`, then most-recently updated wins.

The `GET /v1/mcp-servers` endpoint applies shadowing by default (`include_shadowed=false`), returning only the precedence winner per name. Pass `include_shadowed=true` to see all copies.

### Secret Masking

`env` and `headers` values are replaced with `***` in all API responses unless `reveal_secrets=true` is passed as a query parameter. Values that are exactly `${VAR_NAME}` (shell-style variable interpolation placeholders) are preserved verbatim and never masked.

```json
{
  "env": {
    "API_KEY": "***",
    "TOOL_PATH": "${TOOL_PATH}"
  }
}
```

### Memory Mirror

On create/update, the server writes a procedural memory with `subtype="mcp_server"` and `metadata.mcp_server_id` pointing back to the record. This makes servers discoverable via `POST /v1/mcp-servers/resolve` with a `query` (vector recall) in addition to exact-name lookup.

---

## Data Model

### McpServer fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Primary key (`mcp_<12hex>`) |
| `tenant_id` | `str` | Tenant scope (default `_default`) |
| `workspace_id` | `str` | Workspace; `_global` for tenant/global, `_global_user` for cross-workspace user scope |
| `user_id` | `str \| null` | Set for LOCAL and USER scopes |
| `name` | `str` | 1–64 chars, `[a-z0-9-]`, no leading/trailing/consecutive hyphens |
| `description` | `str \| null` | Optional description (≤1024 chars) |
| `transport` | `"stdio" \| "http" \| "sse" \| "streamable-http"` | Transport protocol |
| `command` | `str \| null` | Executable command (stdio only) |
| `args` | `list[str]` | Command arguments (stdio only) |
| `env` | `dict[str, str]` | Environment variables (stdio only); may contain `${VAR}` placeholders |
| `url` | `str \| null` | Server URL (http/sse/streamable-http only) |
| `headers` | `dict[str, str]` | HTTP headers (http/sse only); may contain secrets |
| `metadata` | `dict` | Arbitrary metadata (tags, vendor, etc.) |
| `source_mode` | `"server" \| "filesystem" \| "mirrored"` | Canonical storage location |
| `manifest_hash` | `str` | SHA-256 of canonical JSON serialization (sorted keys, no whitespace) |
| `enabled` | `bool` | Whether this server is active |
| `created_at` | `datetime` | Creation timestamp |
| `updated_at` | `datetime` | Last update timestamp |

Transport validation: `stdio` requires `command`; `http`/`sse`/`streamable-http` require `url`. The API returns HTTP 400 if the constraint is violated.

---

## API Reference

All endpoints are under `/v1/mcp-servers`. Workspace scope is read from the `X-Workspace-ID` header or the `workspace_id` query/body parameter. All read endpoints accept `reveal_secrets=true` to return unmasked values.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/mcp-servers` | Create a server |
| `GET` | `/v1/mcp-servers` | List servers (secrets masked by default) |
| `GET` | `/v1/mcp-servers/{id}` | Get a server by ID (masked) |
| `PUT` | `/v1/mcp-servers/{id}` | Update a server |
| `DELETE` | `/v1/mcp-servers/{id}` | Delete a server |
| `POST` | `/v1/mcp-servers/import` | Bulk import from `.mcp.json`-shaped JSON; upserts by name |
| `GET` | `/v1/mcp-servers/export` | Export enabled servers as `.mcp.json`-shaped JSON |
| `POST` | `/v1/mcp-servers/resolve` | Resolve by name (precedence winner) or query (vector recall) |
| `POST` | `/v1/mcp-servers/{id}/sync` | Compare local `manifest_hash` vs stored; returns `push \| pull \| conflict \| in_sync` |

### List query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workspace_id` | `str` | from header | Workspace to query |
| `name` | `str` | — | Filter by exact name |
| `transport` | `str` | — | Filter by transport type |
| `enabled` | `bool` | — | Filter by enabled status |
| `include_shadowed` | `bool` | `false` | Include all copies; default returns only precedence winner per name |
| `reveal_secrets` | `bool` | `false` | Return unmasked env/headers values |
| `limit` | `int` | `100` | Max results (1–500) |
| `offset` | `int` | `0` | Pagination offset |

### Import request body

```json
{
  "mcpServers": {
    "server-name": { "command": "npx", "args": ["-y", "@some/mcp-server"] }
  },
  "workspace_id": "my-project",
  "user_id": null,
  "source_mode": "server"
}
```

Import response: `{"imported": N, "updated": N, "skipped": 0, "errors": [...]}`.

---

## SDK Usage

### Python (async)

```python
import asyncio
from pathlib import Path
from memorylayer import MemoryLayerClient

async def main():
    async with MemoryLayerClient(base_url="http://localhost:61001", workspace_id="my-project") as client:

        # Create a stdio server
        server = await client.mcp_servers.create(
            name="github-mcp",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
        )
        print(server.id)  # mcp_abc123...

        # Create an HTTP server
        server = await client.mcp_servers.create(
            name="my-api",
            transport="http",
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer ${API_KEY}"},
        )

        # List servers (env/headers are masked by default)
        servers = await client.mcp_servers.list()

        # Resolve by name using 4-tier precedence
        winner = await client.mcp_servers.resolve(name="github-mcp")

        # Push a .mcp.json file — creates/updates one record per entry
        results = await client.mcp_servers.push_json(Path(".mcp.json"))

        # Pull all servers back to a .mcp.json file
        out = await client.mcp_servers.pull_json(Path("/tmp/resolved.mcp.json"))

asyncio.run(main())
```

### Python (sync)

```python
from pathlib import Path
from memorylayer import SyncMemoryLayerClient

with SyncMemoryLayerClient(base_url="http://localhost:61001", workspace_id="my-project") as client:
    servers = client.mcp_servers.list(enabled=True, transport="stdio")
    winner = client.mcp_servers.resolve(name="github-mcp")
    client.mcp_servers.pull_json(Path(".mcp.json"))
```

### TypeScript

```typescript
import { MemoryLayerClient } from "@scitrera/memorylayer-sdk";

const client = new MemoryLayerClient({ baseUrl: "http://localhost:61001", workspaceId: "my-project" });

// Create a server
const server = await client.mcpServers.create({
  name: "github-mcp",
  transport: "stdio",
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-github"],
  env: { GITHUB_TOKEN: "${GITHUB_TOKEN}" },
});

// List servers
const servers = await client.mcpServers.list({ transport: "stdio" });

// Resolve by name
const winner = await client.mcpServers.resolve({ name: "github-mcp" });

// Update
await client.mcpServers.update(server.id, { enabled: false });

// Delete
await client.mcpServers.delete(server.id);
```

---

## MCP Tools

MCP server tools are available in the `full` tool profile only (not in the default `cc` profile).

| Tool | Required params | Description |
|------|----------------|-------------|
| `mcp_servers_list` | — | List registered servers; filters: `transport`, `enabled`, `name`, `limit`, `offset` |
| `mcp_servers_get` | `server_id` or `name` | Fetch a single server by ID or name |
| `mcp_servers_save` | `name`, `transport` | Create or update a server (upsert by name) |
| `mcp_servers_delete` | `server_id` | Delete a server by ID |
| `mcp_servers_import` | `mcpServers` | Bulk import from a `mcpServers` JSON object |

`mcp_servers_save` optional params: `command`, `args`, `env`, `url`, `headers`, `description`, `enabled`, `metadata`.

---

## CLI

The `memorylayer mcp` command group requires `memorylayer-server` installed and a running server at `http://localhost:61001` (override with `--server-url`).

```bash
# List registered MCP servers
memorylayer mcp list --workspace my-project

# List only stdio servers in JSON format
memorylayer mcp list --transport stdio --format json

# Push a .mcp.json file to MemoryLayer
memorylayer mcp push .mcp.json --workspace my-project

# Export all registered servers to a .mcp.json file
memorylayer mcp pull --output .mcp.json --workspace my-project

# Two-way sync: push local file, then pull server state back to file
memorylayer mcp sync .mcp.json --workspace my-project

# Write all enabled servers to .mcp.json (idempotent via manifest_hash)
memorylayer mcp materialize --workspace my-project

# Also write to ~/.claude.json user scope
memorylayer mcp materialize --workspace my-project --write-claude-json

# Watch a .mcp.json and push on every save (mirrored mode)
memorylayer mcp watch .mcp.json --workspace my-project

# Import from ~/.claude.json user scope
memorylayer mcp migrate-from-local --scope user

# Import from ~/.claude.json local scope for a specific project
memorylayer mcp migrate-from-local --scope local --project-path /home/user/my-project
```

---

## `.mcp.json` Format

Standard Claude Code `.mcp.json` shape. MemoryLayer reads and writes this format exactly.

### stdio server

```json
{
  "mcpServers": {
    "github-mcp": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

### HTTP server

```json
{
  "mcpServers": {
    "my-api": {
      "type": "http",
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}
```

The `type` field (Claude Code convention) is read during import for transport inference. MemoryLayer stores the normalized `transport` value (`stdio`, `http`, `sse`, or `streamable-http`). On export, the `transport` key is written rather than `type`.

---

## Local File Integration

MemoryLayer integrates with three Claude Code config files:

| File | Scope | Notes |
|------|-------|-------|
| `.mcp.json` | project | Per-project servers; maps to PROJECT tier |
| `~/.claude.json` | user + local | `mcpServers` key = USER tier; `projects[path].mcpServers` = LOCAL tier |
| `claude_desktop_config.json` | global | Not written by MemoryLayer CLI |

### Surgical writer for `~/.claude.json`

`memorylayer mcp materialize --write-claude-json` (or `--write-claude-json` on `mcp pull`) calls `write_claude_json_servers`. This function:

1. Reads the entire existing `~/.claude.json` preserving all other keys.
2. Overwrites only the target scope key (`mcpServers` for `user`, `projects[path].mcpServers` for `local`).
3. Writes back atomically via a temp file + `os.replace`.

Other keys in `~/.claude.json` (theme, model preferences, etc.) are never touched.

**`--write-claude-json` is opt-in.** It is not applied automatically by `memorylayer mcp list`, `mcp push`, or any read commands.

---

## Secrets

### Masking behavior

All `env` and `headers` values are masked to `***` in API responses by default. The mask is applied per-value immediately before serialization; the stored value is unaffected.

To receive unmasked values, pass `?reveal_secrets=true` on any read endpoint:

```bash
curl "http://localhost:61001/v1/mcp-servers/mcp_abc123?reveal_secrets=true" \
  -H "Authorization: Bearer $API_KEY"
```

### `${VAR}` passthrough

Values that match exactly `${VAR_NAME}` (opening `${`, closing `}`, no other content) bypass masking and are returned verbatim. This allows storing shell variable references that the agent or harness expands at runtime without exposing actual secrets.

```json
{ "env": { "GITHUB_TOKEN": "${GITHUB_TOKEN}", "REAL_SECRET": "***" } }
```

### Encryption protocol (OSS passthrough)

The `McpServer` model comments on `env` and `headers` note that values "may contain secrets — encrypted at rest in Enterprise". In OSS, values are stored as plaintext in SQLite. The API-level masking layer provides display protection but not storage encryption. Enterprise deployments replace the storage layer with an encrypted backend; the API contract (masking, `${VAR}` passthrough) is identical in both tiers.

---

## Verification Recipes

### Scope precedence test

```python
import asyncio
import httpx

async def test_precedence():
    base = "http://localhost:61001"
    ws = "test-workspace"

    async with httpx.AsyncClient() as http:
        # Create PROJECT-scoped server (no user_id)
        await http.post(f"{base}/v1/mcp-servers", json={
            "name": "my-server",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@example/server"],
            "workspace_id": ws,
        })

        # Create LOCAL-scoped server (user_id set, same workspace)
        await http.post(f"{base}/v1/mcp-servers", json={
            "name": "my-server",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@example/server-local"],
            "workspace_id": ws,
            "user_id": "alice",
        })

        # Resolve — LOCAL should win (rank 0 < rank 1)
        r = await http.post(f"{base}/v1/mcp-servers/resolve", json={
            "name": "my-server",
        }, headers={"X-Workspace-ID": ws, "X-User-ID": "alice"})
        winner = r.json()["mcp_server"]
        assert winner["args"] == ["-y", "@example/server-local"]

asyncio.run(test_precedence())
```

### Secret masking

```bash
# Create a server with a real secret
curl -X POST http://localhost:61001/v1/mcp-servers \
  -H 'Content-Type: application/json' \
  -d '{"name":"secret-test","transport":"stdio","command":"npx","env":{"API_KEY":"s3cr3t","PATH":"${PATH}"}}'

# Default read — secret is masked, ${PATH} is preserved
curl "http://localhost:61001/v1/mcp-servers?name=secret-test"
# "env": {"API_KEY": "***", "PATH": "${PATH}"}

# Reveal — returns actual value
curl "http://localhost:61001/v1/mcp-servers?name=secret-test&reveal_secrets=true"
# "env": {"API_KEY": "s3cr3t", "PATH": "${PATH}"}
```

### `${VAR}` preservation through export

```bash
# Push a .mcp.json with ${VAR} references
cat > /tmp/test.mcp.json <<'EOF'
{
  "mcpServers": {
    "github-mcp": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
    }
  }
}
EOF

memorylayer mcp push /tmp/test.mcp.json --workspace test-workspace

# Export — ${GITHUB_TOKEN} should survive masking unchanged
memorylayer mcp pull --output /tmp/exported.mcp.json --workspace test-workspace
grep GITHUB_TOKEN /tmp/exported.mcp.json
# "GITHUB_TOKEN": "${GITHUB_TOKEN}"
```
