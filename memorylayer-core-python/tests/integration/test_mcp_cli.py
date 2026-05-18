"""Integration tests for `memorylayer mcp` CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from memorylayer_server.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def server_url(test_client):
    """Return the base URL used by the test ASGI app via the httpx transport."""
    return "http://testserver"


@pytest.fixture
def mcp_json_file(tmp_path) -> Path:
    doc = {
        "mcpServers": {
            "cli-postgres": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "env": {"DATABASE_URL": "postgresql://localhost/test"},
            },
            "cli-search": {
                "url": "https://search.example.com/mcp",
                "type": "http",
            },
        }
    }
    p = tmp_path / ".mcp.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


class TestMcpList:
    def test_list_empty(self, runner, test_client, monkeypatch):
        """List when workspace has no servers returns 'No MCP servers found.'"""
        import httpx

        def fake_transport(request):
            # Forward to ASGI test client
            method = request.method
            url = str(request.url)
            path = url.replace("http://testserver", "")
            resp = test_client.request(method, path, headers=dict(request.headers))
            return httpx.Response(resp.status_code, json=resp.json())

        monkeypatch.setattr(
            "httpx.Client",
            lambda **kwargs: _MockHttpxClient(test_client),
        )

        result = runner.invoke(cli, ["mcp", "list", "--workspace", "cli-test-ws"])
        assert result.exit_code == 0
        assert "No MCP servers found" in result.output or "server(s)" in result.output

    def test_list_json_format(self, runner, monkeypatch, test_client):
        # Create a server first via API
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "list-json-srv", "transport": "stdio", "command": "echo"},
            headers={"X-Workspace-ID": "cli-json-ws"},
        )

        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="cli-json-ws"))

        result = runner.invoke(cli, ["mcp", "list", "--workspace", "cli-json-ws", "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


class TestMcpPush:
    def test_push_mcp_json(self, runner, monkeypatch, test_client, mcp_json_file):
        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="push-ws"))

        result = runner.invoke(
            cli,
            ["mcp", "push", str(mcp_json_file), "--workspace", "push-ws"],
        )
        assert result.exit_code == 0
        assert "Imported:" in result.output

    def test_push_empty_file(self, runner, monkeypatch, test_client, tmp_path):
        empty = tmp_path / "empty.json"
        empty.write_text('{"mcpServers": {}}', encoding="utf-8")

        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client))

        result = runner.invoke(cli, ["mcp", "push", str(empty)])
        assert result.exit_code == 0
        assert "No mcpServers" in result.output


class TestMcpPull:
    def test_pull_creates_file(self, runner, monkeypatch, test_client, tmp_path):
        # Pre-create a server in the workspace
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "pull-srv", "transport": "stdio", "command": "sh"},
            headers={"X-Workspace-ID": "pull-ws"},
        )

        out_file = tmp_path / "out.mcp.json"
        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="pull-ws"))

        result = runner.invoke(
            cli,
            ["mcp", "pull", "--output", str(out_file), "--workspace", "pull-ws"],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "mcpServers" in data

    def test_pull_server_count_in_output(self, runner, monkeypatch, test_client, tmp_path):
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "pull-count-srv", "transport": "stdio", "command": "node"},
            headers={"X-Workspace-ID": "pull-count-ws"},
        )

        out_file = tmp_path / "count.mcp.json"
        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="pull-count-ws"))

        result = runner.invoke(
            cli,
            ["mcp", "pull", "--output", str(out_file), "--workspace", "pull-count-ws"],
        )
        assert result.exit_code == 0
        assert "server(s)" in result.output


class TestMcpSync:
    def test_sync_push_then_pull(self, runner, monkeypatch, test_client, mcp_json_file, tmp_path):
        """Sync uploads local file then overwrites it with server export."""
        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="sync-ws"))

        sync_file = tmp_path / "sync.mcp.json"
        sync_file.write_text(mcp_json_file.read_text(), encoding="utf-8")

        result = runner.invoke(
            cli,
            ["mcp", "sync", str(sync_file), "--workspace", "sync-ws"],
        )
        assert result.exit_code == 0
        assert "Pushed:" in result.output
        assert "Synced" in result.output

        # File should be valid JSON with mcpServers key
        data = json.loads(sync_file.read_text())
        assert "mcpServers" in data

    def test_sync_nonexistent_file_still_pulls(self, runner, monkeypatch, test_client, tmp_path):
        """Sync with a non-existent local file just pulls from server."""
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "sync-only-srv", "transport": "stdio", "command": "python"},
            headers={"X-Workspace-ID": "sync-only-ws"},
        )

        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="sync-only-ws"))

        new_file = tmp_path / "new.mcp.json"
        result = runner.invoke(
            cli,
            ["mcp", "sync", str(new_file), "--workspace", "sync-only-ws"],
        )
        assert result.exit_code == 0
        assert "Synced" in result.output
        assert new_file.exists()


class _MockHttpxClient:
    """Thin httpx.Client substitute that forwards calls to the FastAPI test client."""

    def __init__(self, test_client, workspace: str | None = None):
        self._tc = test_client
        self._workspace = workspace

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _headers(self, extra: dict | None = None) -> dict:
        h = {}
        if self._workspace:
            h["X-Workspace-ID"] = self._workspace
        if extra:
            for k, v in extra.items():
                if k.lower() not in ("content-type", "x-workspace-id"):
                    h[k] = v
                elif k.lower() == "x-workspace-id":
                    h[k] = v
        return h

    def get(self, url: str, params=None, headers=None):
        path = url.replace("http://localhost:61001", "").replace("http://testserver", "")
        merged = self._headers(headers)
        resp = self._tc.get(path, params=params, headers=merged)
        return _MockResponse(resp)

    def post(self, url: str, content=None, json=None, headers=None):
        path = url.replace("http://localhost:61001", "").replace("http://testserver", "")
        merged = self._headers(headers)
        if content is not None:
            resp = self._tc.post(path, content=content, headers=merged)
        else:
            resp = self._tc.post(path, json=json, headers=merged)
        return _MockResponse(resp)


class _MockResponse:
    def __init__(self, resp):
        self._resp = resp
        self.status_code = resp.status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=None,  # type: ignore[arg-type]
                response=None,  # type: ignore[arg-type]
            )

    def json(self):
        return self._resp.json()


# ── Sync hash comparison logic ────────────────────────────────────────────────


class TestSyncHashLogic:
    """Pure unit tests for sync.compare_hashes and sync.resolve_conflict."""

    def test_equal_hashes_in_sync(self):
        from memorylayer_server.services.mcp_servers.sync import compare_hashes
        action, _ = compare_hashes("abc123", "abc123")
        assert action == "in_sync"

    def test_empty_local_pull(self):
        from memorylayer_server.services.mcp_servers.sync import compare_hashes
        action, _ = compare_hashes("", "server_hash")
        assert action == "pull"

    def test_empty_server_push(self):
        from memorylayer_server.services.mcp_servers.sync import compare_hashes
        action, _ = compare_hashes("local_hash", "")
        assert action == "push"

    def test_both_different_conflict(self):
        from memorylayer_server.services.mcp_servers.sync import compare_hashes
        action, reason = compare_hashes("hash_a", "hash_b")
        assert action == "conflict"

    def test_resolve_conflict_prefer_local_becomes_push(self):
        from memorylayer_server.services.mcp_servers.sync import resolve_conflict
        action, reason = resolve_conflict("conflict", "prefer-local")
        assert action == "push"
        assert "prefer-local" in reason

    def test_resolve_conflict_prefer_remote_becomes_pull(self):
        from memorylayer_server.services.mcp_servers.sync import resolve_conflict
        action, _ = resolve_conflict("conflict", "prefer-remote")
        assert action == "pull"

    def test_resolve_conflict_abort_stays_conflict(self):
        from memorylayer_server.services.mcp_servers.sync import resolve_conflict
        action, _ = resolve_conflict("conflict", "abort")
        assert action == "conflict"

    def test_non_conflict_action_unchanged(self):
        from memorylayer_server.services.mcp_servers.sync import resolve_conflict
        action, _ = resolve_conflict("push", "prefer-local")
        assert action == "push"


# ── Materialize idempotency ───────────────────────────────────────────────────


class TestMaterializeIdempotency:
    def test_second_run_detects_no_change(self, runner, monkeypatch, test_client, tmp_path):
        """Second materialize with identical server state produces '0 changes' message."""
        # Seed a server
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "idem-srv", "transport": "stdio", "command": "echo"},
            headers={"X-Workspace-ID": "idem-ws"},
        )

        out_file = tmp_path / "idem.mcp.json"
        monkeypatch.setattr("httpx.Client", lambda **kwargs: _MockHttpxClient(test_client, workspace="idem-ws"))

        # First run — writes file
        result1 = runner.invoke(
            cli,
            ["mcp", "materialize", "--target", str(out_file), "--workspace", "idem-ws"],
        )
        assert result1.exit_code == 0
        assert out_file.exists()

        mtime1 = out_file.stat().st_mtime

        # Second run — should detect no change
        result2 = runner.invoke(
            cli,
            ["mcp", "materialize", "--target", str(out_file), "--workspace", "idem-ws"],
        )
        assert result2.exit_code == 0
        assert "0 changes" in result2.output or "up-to-date" in result2.output

        # File should not have been rewritten (mtime unchanged)
        mtime2 = out_file.stat().st_mtime
        assert mtime2 == mtime1
