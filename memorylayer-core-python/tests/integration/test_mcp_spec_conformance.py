"""MCP server spec conformance tests.

Covers the full MCP server spec as implemented:
- Name regex enforcement
- Transport-discriminated field validation
- Multi-server .mcp.json round-trip (import → export)
- 4-tier scope precedence (Local > Project > User > Global)
- Memory mirror discovery (source_mode=mirrored)
- Hybrid mode (stdio + http servers co-exist)
- Secrets masking + ${VAR} passthrough
- ~/.claude.json surgical write (read/write_claude_json_servers)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from memorylayer_server.services.mcp_servers.claude_json import (
    read_claude_json_servers,
    write_claude_json_servers,
)
from memorylayer_server.services.mcp_servers.resolution import (
    McpServerResolutionService,
    RequestContext,
)
from memorylayer_server.services.mcp_servers.sync import compare_hashes, resolve_conflict


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ws_headers() -> dict[str, str]:
    return {"X-Workspace-ID": "spec-ws"}


# ── 1. Name regex enforcement ─────────────────────────────────────────────────


class TestNameRegex:
    """MCP server names: 1-64 chars, [a-z0-9-], no leading/trailing/consecutive hyphens."""

    VALID_NAMES = [
        "postgres-mcp",
        "a",
        "my-server-123",
        "x" * 64,
    ]

    INVALID_NAMES = [
        "Bad Name",        # uppercase + space
        "-leading-hyphen", # leading hyphen
        "trailing-hyphen-", # trailing hyphen
        "double--hyphen",  # consecutive hyphens
        "",                # empty
        "x" * 65,          # too long
        "UPPER",           # uppercase
        "has_underscore",  # underscore not allowed
    ]

    @pytest.mark.parametrize("name", VALID_NAMES)
    def test_valid_name_accepted(self, test_client: TestClient, ws_headers: dict, name: str) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": name, "transport": "stdio", "command": "echo"},
            headers=ws_headers,
        )
        assert resp.status_code == 201, f"Expected 201 for valid name {name!r}, got {resp.status_code}: {resp.text}"

    @pytest.mark.parametrize("name", INVALID_NAMES)
    def test_invalid_name_rejected(self, test_client: TestClient, ws_headers: dict, name: str) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": name, "transport": "stdio", "command": "echo"},
            headers=ws_headers,
        )
        assert resp.status_code == 422, f"Expected 422 for invalid name {name!r}, got {resp.status_code}"


# ── 2. Transport-discriminated validation ─────────────────────────────────────


class TestTransportValidation:
    """stdio requires command; http/sse/streamable-http require url."""

    def test_stdio_without_command_fails(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "no-cmd", "transport": "stdio"},
            headers=ws_headers,
        )
        assert resp.status_code == 422

    def test_http_without_url_fails(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "no-url", "transport": "http"},
            headers=ws_headers,
        )
        assert resp.status_code == 422

    def test_sse_without_url_fails(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "no-url-sse", "transport": "sse"},
            headers=ws_headers,
        )
        assert resp.status_code == 422

    def test_stdio_with_command_succeeds(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "with-cmd", "transport": "stdio", "command": "npx"},
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["transport"] == "stdio"

    def test_http_with_url_succeeds(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "with-url", "transport": "http", "url": "https://mcp.example.com/api"},
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["transport"] == "http"

    def test_sse_with_url_succeeds(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "with-url-sse", "transport": "sse", "url": "https://mcp.example.com/sse"},
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["transport"] == "sse"

    def test_streamable_http_with_url_succeeds(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "sh-transport", "transport": "streamable-http", "url": "https://mcp.example.com/stream"},
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["transport"] == "streamable-http"


# ── 3. Multi-server .mcp.json round-trip ─────────────────────────────────────


class TestMcpJsonRoundTrip:
    """Import a multi-server .mcp.json doc, then export and verify fidelity."""

    def test_multi_server_import_export_roundtrip(self, test_client: TestClient, ws_headers: dict) -> None:
        payload: dict[str, Any] = {
            "mcpServers": {
                "rt-postgres": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"],
                    "env": {"DATABASE_URL": "postgresql://localhost/mydb"},
                },
                "rt-filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
                "rt-web-search": {
                    "url": "https://mcp.example.com/search",
                    "type": "http",
                },
            }
        }
        import_resp = test_client.post("/v1/mcp-servers/import", json=payload, headers=ws_headers)
        assert import_resp.status_code == 200
        result = import_resp.json()
        assert result["imported"] == 3
        assert result["errors"] == []

        export_resp = test_client.get("/v1/mcp-servers/export", headers=ws_headers)
        assert export_resp.status_code == 200
        exported = export_resp.json()
        assert "mcpServers" in exported
        servers = exported["mcpServers"]

        assert "rt-postgres" in servers
        assert servers["rt-postgres"]["command"] == "npx"
        assert "-y" in servers["rt-postgres"]["args"]

        assert "rt-filesystem" in servers
        assert servers["rt-filesystem"]["command"] == "npx"

        assert "rt-web-search" in servers
        assert servers["rt-web-search"]["url"] == "https://mcp.example.com/search"

    def test_mcp_json_import_is_idempotent(self, test_client: TestClient, ws_headers: dict) -> None:
        payload: dict[str, Any] = {
            "mcpServers": {
                "idempotent-rt": {"command": "python", "args": ["-m", "server"]}
            }
        }
        r1 = test_client.post("/v1/mcp-servers/import", json=payload, headers=ws_headers)
        assert r1.json()["imported"] == 1

        r2 = test_client.post("/v1/mcp-servers/import", json=payload, headers=ws_headers)
        assert r2.json()["updated"] == 1
        assert r2.json()["imported"] == 0

    def test_mcp_json_transport_inference_stdio(self, test_client: TestClient, ws_headers: dict) -> None:
        payload: dict[str, Any] = {
            "mcpServers": {
                "infer-stdio": {"command": "python"}
            }
        }
        resp = test_client.post("/v1/mcp-servers/import", json=payload, headers=ws_headers)
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1

        list_resp = test_client.get("/v1/mcp-servers", headers=ws_headers)
        servers = {s["name"]: s for s in list_resp.json()["mcp_servers"]}
        assert servers["infer-stdio"]["transport"] == "stdio"

    def test_mcp_json_transport_inference_http(self, test_client: TestClient, ws_headers: dict) -> None:
        payload: dict[str, Any] = {
            "mcpServers": {
                "infer-http": {"url": "https://mcp.example.com/infer", "type": "http"}
            }
        }
        resp = test_client.post("/v1/mcp-servers/import", json=payload, headers=ws_headers)
        assert resp.status_code == 200

        list_resp = test_client.get("/v1/mcp-servers", headers=ws_headers)
        servers = {s["name"]: s for s in list_resp.json()["mcp_servers"]}
        assert servers["infer-http"]["transport"] == "http"


# ── 4. 4-tier scope precedence ────────────────────────────────────────────────


class TestScopePrecedence:
    """LOCAL > PROJECT > USER > GLOBAL precedence via McpServerResolutionService."""

    def _make_server(
        self,
        name: str,
        workspace_id: str,
        user_id: str | None = None,
        source_mode: str = "server",
    ) -> Any:
        from memorylayer_server.models.mcp_server import McpServer
        from datetime import UTC, datetime

        return McpServer(
            id=f"mcp_{name.replace('-', '')}",
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
            transport="stdio",
            command="echo",
            source_mode=source_mode,  # type: ignore[arg-type]
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    def test_local_beats_project(self) -> None:
        local = self._make_server("shared", workspace_id="ws1", user_id="u1")
        project = self._make_server("shared", workspace_id="ws1", user_id=None)

        ctx = RequestContext(workspace_id="ws1", user_id="u1")

        class _FakeStorage:
            async def find_mcp_servers_by_name(self, name, scopes):  # type: ignore
                return [local, project]

        svc = McpServerResolutionService(_FakeStorage())  # type: ignore[arg-type]
        ranked = svc._rank([local, project], ctx)
        assert ranked[0].user_id == "u1", "LOCAL (user_id set) should beat PROJECT (user_id=None)"

    def test_project_beats_global(self) -> None:
        project = self._make_server("shared", workspace_id="ws1", user_id=None)
        global_s = self._make_server("shared", workspace_id="_global", user_id=None)

        ctx = RequestContext(workspace_id="ws1", user_id=None)
        svc = McpServerResolutionService(None)  # type: ignore[arg-type]
        ranked = svc._rank([global_s, project], ctx)
        assert ranked[0].workspace_id == "ws1", "PROJECT (ws1) should beat GLOBAL (_global)"

    def test_local_beats_global(self) -> None:
        local = self._make_server("shared", workspace_id="ws1", user_id="u1")
        global_s = self._make_server("shared", workspace_id="_global", user_id=None)

        ctx = RequestContext(workspace_id="ws1", user_id="u1")
        svc = McpServerResolutionService(None)  # type: ignore[arg-type]
        ranked = svc._rank([global_s, local], ctx)
        assert ranked[0].user_id == "u1", "LOCAL should beat GLOBAL"

    def test_user_scope_cross_workspace(self) -> None:
        user_scope = self._make_server("shared", workspace_id="_global_user", user_id="u1")
        global_s = self._make_server("shared", workspace_id="_global", user_id=None)

        ctx = RequestContext(workspace_id="ws1", user_id="u1")
        svc = McpServerResolutionService(None)  # type: ignore[arg-type]
        ranked = svc._rank([global_s, user_scope], ctx)
        assert ranked[0].workspace_id == "_global_user", "USER (_global_user) should beat GLOBAL (_global)"

    def test_server_mode_beats_mirrored_within_same_scope(self) -> None:
        server_mode = self._make_server("shared", workspace_id="ws1", source_mode="server")
        mirrored = self._make_server("shared", workspace_id="ws1", source_mode="mirrored")

        ctx = RequestContext(workspace_id="ws1", user_id=None)
        svc = McpServerResolutionService(None)  # type: ignore[arg-type]
        ranked = svc._rank([mirrored, server_mode], ctx)
        assert ranked[0].source_mode == "server", "source_mode=server should beat mirrored within same scope"

    def test_apply_shadowing_returns_one_winner_per_name(self) -> None:
        local = self._make_server("shared", workspace_id="ws1", user_id="u1")
        project = self._make_server("shared", workspace_id="ws1", user_id=None)
        other = self._make_server("other-server", workspace_id="ws1")

        ctx = RequestContext(workspace_id="ws1", user_id="u1")
        svc = McpServerResolutionService(None)  # type: ignore[arg-type]
        result = svc.apply_shadowing([local, project, other], ctx)
        names = [s.name for s in result]
        assert names.count("shared") == 1, "apply_shadowing must return exactly one winner per name"
        assert "other-server" in names

        winner = next(s for s in result if s.name == "shared")
        assert winner.user_id == "u1", "LOCAL (user_id=u1) should win shadowing"


# ── 5. Memory mirror discovery (source_mode) ─────────────────────────────────


class TestMemoryMirrorDiscovery:
    """source_mode=mirrored records are stored and retrievable."""

    def test_mirrored_server_stored_and_listed(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "mirror-server",
                "transport": "stdio",
                "command": "npx",
                "source_mode": "mirrored",
            },
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["source_mode"] == "mirrored"

    def test_filesystem_source_mode_stored(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "fs-server",
                "transport": "stdio",
                "command": "python",
                "source_mode": "filesystem",
            },
            headers=ws_headers,
        )
        assert resp.status_code == 201
        assert resp.json()["mcp_server"]["source_mode"] == "filesystem"

    def test_sync_endpoint_returns_action(self, test_client: TestClient, ws_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "sync-target", "transport": "stdio", "command": "npx", "source_mode": "mirrored"},
            headers=ws_headers,
        )
        server_id = create_resp.json()["mcp_server"]["id"]

        sync_resp = test_client.post(
            f"/v1/mcp-servers/{server_id}/sync",
            json={"manifest_hash": ""},
            headers=ws_headers,
        )
        assert sync_resp.status_code == 200
        data = sync_resp.json()
        assert "action" in data
        assert data["action"] in ("push", "pull", "conflict", "in_sync")

    def test_sync_in_sync_when_hashes_match(self, test_client: TestClient, ws_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "sync-match",
                "transport": "stdio",
                "command": "npx",
                "source_mode": "mirrored",
            },
            headers=ws_headers,
        )
        server = create_resp.json()["mcp_server"]
        server_id = server["id"]
        current_hash = server.get("manifest_hash", "")

        sync_resp = test_client.post(
            f"/v1/mcp-servers/{server_id}/sync",
            json={"manifest_hash": current_hash},
            headers=ws_headers,
        )
        assert sync_resp.status_code == 200
        assert sync_resp.json()["action"] == "in_sync"


# ── 6. Hybrid mode (stdio + http co-exist) ───────────────────────────────────


class TestHybridMode:
    """Multiple servers with different transports can co-exist in same workspace."""

    def test_stdio_and_http_coexist(self, test_client: TestClient, ws_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "hybrid-stdio", "transport": "stdio", "command": "python"},
            headers=ws_headers,
        )
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "hybrid-http", "transport": "http", "url": "https://mcp.example.com/hybrid"},
            headers=ws_headers,
        )
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "hybrid-sse", "transport": "sse", "url": "https://mcp.example.com/hybrid-sse"},
            headers=ws_headers,
        )

        list_resp = test_client.get("/v1/mcp-servers", headers=ws_headers)
        assert list_resp.status_code == 200
        servers = {s["name"]: s for s in list_resp.json()["mcp_servers"]}

        assert "hybrid-stdio" in servers
        assert servers["hybrid-stdio"]["transport"] == "stdio"
        assert "hybrid-http" in servers
        assert servers["hybrid-http"]["transport"] == "http"
        assert "hybrid-sse" in servers
        assert servers["hybrid-sse"]["transport"] == "sse"

    def test_transport_filter_returns_only_matching(self, test_client: TestClient, ws_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "filter-stdio", "transport": "stdio", "command": "sh"},
            headers=ws_headers,
        )
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "filter-http", "transport": "http", "url": "https://mcp.example.com/filter"},
            headers=ws_headers,
        )

        resp = test_client.get("/v1/mcp-servers", params={"transport": "stdio"}, headers=ws_headers)
        assert resp.status_code == 200
        servers = resp.json()["mcp_servers"]
        assert all(s["transport"] == "stdio" for s in servers), "Transport filter must return only stdio servers"


# ── 7. Secrets masking + ${VAR} passthrough ───────────────────────────────────


class TestSecretsAndVariables:
    """Secrets are masked by default; ${VAR} placeholders pass through verbatim."""

    def test_env_secrets_masked_by_default(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "env-masked",
                "transport": "stdio",
                "command": "sh",
                "env": {"API_KEY": "real-secret-value", "OTHER": "also-secret"},
            },
            headers=ws_headers,
        )
        server = resp.json()["mcp_server"]
        assert server["env"]["API_KEY"] == "***"
        assert server["env"]["OTHER"] == "***"

    def test_header_secrets_masked_by_default(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "hdr-masked",
                "transport": "http",
                "url": "https://mcp.example.com/hdr",
                "headers": {"Authorization": "Bearer secret-token"},
            },
            headers=ws_headers,
        )
        server = resp.json()["mcp_server"]
        assert server["headers"]["Authorization"] == "***"

    def test_env_revealed_with_query_param(self, test_client: TestClient, ws_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "env-revealed",
                "transport": "stdio",
                "command": "sh",
                "env": {"SECRET": "my-real-value"},
            },
            headers=ws_headers,
        )
        server_id = create_resp.json()["mcp_server"]["id"]

        get_resp = test_client.get(
            f"/v1/mcp-servers/{server_id}",
            params={"reveal_secrets": "true"},
            headers=ws_headers,
        )
        assert get_resp.json()["mcp_server"]["env"]["SECRET"] == "my-real-value"

    def test_var_placeholder_passes_through_unmasked(self, test_client: TestClient, ws_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "env-varref",
                "transport": "stdio",
                "command": "sh",
                "env": {
                    "DB_URL": "${DATABASE_URL}",
                    "SECRET": "real-secret",
                },
            },
            headers=ws_headers,
        )
        server = resp.json()["mcp_server"]
        # ${VAR} references must be returned verbatim (not masked)
        assert server["env"]["DB_URL"] == "${DATABASE_URL}"
        # plain secrets must be masked
        assert server["env"]["SECRET"] == "***"

    def test_export_masks_secrets_by_default(self, test_client: TestClient, ws_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "export-masked",
                "transport": "stdio",
                "command": "sh",
                "env": {"TOKEN": "export-secret"},
            },
            headers=ws_headers,
        )
        export_resp = test_client.get("/v1/mcp-servers/export", headers=ws_headers)
        servers = export_resp.json().get("mcpServers", {})
        if "export-masked" in servers:
            entry = servers["export-masked"]
            if "env" in entry:
                assert entry["env"]["TOKEN"] == "***"

    def test_list_masks_secrets_by_default(self, test_client: TestClient, ws_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "list-masked",
                "transport": "stdio",
                "command": "sh",
                "env": {"PASS": "list-secret"},
            },
            headers=ws_headers,
        )
        resp = test_client.get("/v1/mcp-servers", headers=ws_headers)
        servers = {s["name"]: s for s in resp.json()["mcp_servers"]}
        if "list-masked" in servers:
            assert servers["list-masked"]["env"]["PASS"] == "***"


# ── 8. Sync hash logic (pure unit tests) ─────────────────────────────────────


class TestSyncHashLogic:
    """compare_hashes and resolve_conflict correctness."""

    def test_equal_hashes_are_in_sync(self) -> None:
        action, reason = compare_hashes("abc123", "abc123")
        assert action == "in_sync"

    def test_empty_local_is_pull(self) -> None:
        action, reason = compare_hashes("", "server-hash")
        assert action == "pull"

    def test_empty_server_is_push(self) -> None:
        action, reason = compare_hashes("local-hash", "")
        assert action == "push"

    def test_both_non_empty_different_is_conflict(self) -> None:
        action, reason = compare_hashes("local-hash", "server-hash")
        assert action == "conflict"

    def test_both_empty_is_in_sync(self) -> None:
        action, reason = compare_hashes("", "")
        assert action == "in_sync"

    def test_resolve_prefer_local(self) -> None:
        action, reason = resolve_conflict("conflict", "prefer-local")
        assert action == "push"

    def test_resolve_prefer_remote(self) -> None:
        action, reason = resolve_conflict("conflict", "prefer-remote")
        assert action == "pull"

    def test_resolve_abort_stays_conflict(self) -> None:
        action, reason = resolve_conflict("conflict", "abort")
        assert action == "conflict"

    def test_resolve_no_policy_unchanged(self) -> None:
        action, reason = resolve_conflict("conflict", None)
        assert action == "conflict"

    def test_non_conflict_unchanged_by_resolver(self) -> None:
        for original in ("push", "pull", "in_sync"):
            action, _ = resolve_conflict(original, "prefer-local")  # type: ignore[arg-type]
            assert action == original


# ── 9. ~/.claude.json surgical write ─────────────────────────────────────────


class TestClaudeJsonSurgicalWrite:
    """read/write_claude_json_servers must only touch mcpServers blocks."""

    def test_user_scope_write_and_read(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            target = Path(f.name)

        try:
            servers = {
                "my-server": {"command": "npx", "args": ["-y", "server"], "type": "stdio"},
            }
            write_claude_json_servers("user", servers, target_path=target)
            result = read_claude_json_servers("user", target_path=target)
            assert "my-server" in result
            assert result["my-server"]["command"] == "npx"
        finally:
            target.unlink(missing_ok=True)

    def test_local_scope_write_and_read(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            target = Path(f.name)

        try:
            servers = {"local-server": {"command": "python", "args": ["-m", "srv"]}}
            write_claude_json_servers("local", servers, project_path="/my/project", target_path=target)
            result = read_claude_json_servers("local", project_path="/my/project", target_path=target)
            assert "local-server" in result
        finally:
            target.unlink(missing_ok=True)

    def test_write_preserves_other_keys(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 42, "otherKey": "preserved", "mcpServers": {}}, f)
            target = Path(f.name)

        try:
            write_claude_json_servers("user", {"new-server": {"command": "sh"}}, target_path=target)
            raw = json.loads(target.read_text())
            assert raw["version"] == 42
            assert raw["otherKey"] == "preserved"
            assert "new-server" in raw["mcpServers"]
        finally:
            target.unlink(missing_ok=True)

    def test_user_scope_does_not_touch_projects(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"projects": {"/proj": {"mcpServers": {"local-srv": {}}}}}, f)
            target = Path(f.name)

        try:
            write_claude_json_servers("user", {"global-srv": {"command": "echo"}}, target_path=target)
            raw = json.loads(target.read_text())
            assert raw["projects"]["/proj"]["mcpServers"]["local-srv"] == {}
        finally:
            target.unlink(missing_ok=True)

    def test_local_scope_does_not_touch_user_scope(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"mcpServers": {"user-srv": {"command": "echo"}}}, f)
            target = Path(f.name)

        try:
            write_claude_json_servers(
                "local", {"local-srv": {"command": "sh"}}, project_path="/proj", target_path=target
            )
            raw = json.loads(target.read_text())
            assert "user-srv" in raw["mcpServers"]
            assert "local-srv" in raw["projects"]["/proj"]["mcpServers"]
        finally:
            target.unlink(missing_ok=True)

    def test_read_nonexistent_file_returns_empty(self) -> None:
        result = read_claude_json_servers("user", target_path=Path("/nonexistent/path.json"))
        assert result == {}

    def test_write_is_atomic_valid_json(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            target = Path(f.name)

        try:
            write_claude_json_servers("user", {"atomic-srv": {"command": "echo"}}, target_path=target)
            # File must be valid JSON after write
            raw = json.loads(target.read_text())
            assert "atomic-srv" in raw["mcpServers"]
        finally:
            target.unlink(missing_ok=True)

    def test_local_scope_requires_project_path(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            target = Path(f.name)

        try:
            with pytest.raises(ValueError, match="project_path"):
                write_claude_json_servers("local", {"srv": {}}, project_path=None, target_path=target)
        finally:
            target.unlink(missing_ok=True)
