"""Integration tests for /v1/mcp-servers API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    return {"X-Workspace-ID": "test_workspace"}


class TestMcpServerCreate:
    def test_create_stdio_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "postgres-mcp",
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres"],
                "env": {"DATABASE_URL": "postgresql://localhost/mydb"},
                "description": "Query PostgreSQL databases via MCP",
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "mcp_server" in data
        server = data["mcp_server"]
        assert server["name"] == "postgres-mcp"
        assert server["transport"] == "stdio"
        assert server["command"] == "npx"
        assert server["id"].startswith("mcp_")
        assert server["enabled"] is True
        # secrets masked by default
        assert server["env"]["DATABASE_URL"] == "***"

    def test_create_http_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/mcp-servers",
            json={
                "name": "web-search-mcp",
                "transport": "http",
                "url": "https://mcp.example.com/search",
                "headers": {"Authorization": "Bearer sk-test"},
            },
            headers=workspace_headers,
        )
        assert response.status_code == 201
        server = response.json()["mcp_server"]
        assert server["transport"] == "http"
        assert server["url"] == "https://mcp.example.com/search"
        # headers masked
        assert server["headers"]["Authorization"] == "***"

    def test_create_server_invalid_name(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/mcp-servers",
            json={"name": "Bad Name", "transport": "stdio", "command": "npx"},
            headers=workspace_headers,
        )
        assert response.status_code == 422

    def test_create_stdio_missing_command(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/mcp-servers",
            json={"name": "no-cmd", "transport": "stdio"},
            headers=workspace_headers,
        )
        assert response.status_code == 422

    def test_create_http_missing_url(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.post(
            "/v1/mcp-servers",
            json={"name": "no-url", "transport": "http"},
            headers=workspace_headers,
        )
        assert response.status_code == 422


class TestMcpServerGet:
    def test_get_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "get-test-server", "transport": "stdio", "command": "python", "env": {"KEY": "secret"}},
            headers=workspace_headers,
        )
        assert create_resp.status_code == 201
        server_id = create_resp.json()["mcp_server"]["id"]

        get_resp = test_client.get(f"/v1/mcp-servers/{server_id}", headers=workspace_headers)
        assert get_resp.status_code == 200
        server = get_resp.json()["mcp_server"]
        assert server["id"] == server_id
        # masked by default
        assert server["env"]["KEY"] == "***"

    def test_get_server_reveal_secrets(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "reveal-test-server", "transport": "stdio", "command": "python", "env": {"API_KEY": "real-key"}},
            headers=workspace_headers,
        )
        server_id = create_resp.json()["mcp_server"]["id"]

        get_resp = test_client.get(
            f"/v1/mcp-servers/{server_id}",
            params={"reveal_secrets": "true"},
            headers=workspace_headers,
        )
        assert get_resp.status_code == 200
        server = get_resp.json()["mcp_server"]
        assert server["env"]["API_KEY"] == "real-key"

    def test_get_server_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        response = test_client.get("/v1/mcp-servers/nonexistent_id", headers=workspace_headers)
        assert response.status_code == 404


class TestMcpServerList:
    def test_list_servers(self, test_client: TestClient, workspace_headers: dict) -> None:
        for name in ["list-server-a", "list-server-b"]:
            test_client.post(
                "/v1/mcp-servers",
                json={"name": name, "transport": "stdio", "command": "echo"},
                headers=workspace_headers,
            )

        resp = test_client.get("/v1/mcp-servers", headers=workspace_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "mcp_servers" in data
        names = {s["name"] for s in data["mcp_servers"]}
        assert "list-server-a" in names
        assert "list-server-b" in names

    def test_list_servers_masked(self, test_client: TestClient, workspace_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "masked-list-server", "transport": "stdio", "command": "sh", "env": {"TOKEN": "abc123"}},
            headers=workspace_headers,
        )
        resp = test_client.get("/v1/mcp-servers", headers=workspace_headers)
        servers = resp.json()["mcp_servers"]
        masked = next((s for s in servers if s["name"] == "masked-list-server"), None)
        assert masked is not None
        assert masked["env"]["TOKEN"] == "***"


class TestMcpServerUpdate:
    def test_update_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "update-test-server", "transport": "stdio", "command": "python"},
            headers=workspace_headers,
        )
        server_id = create_resp.json()["mcp_server"]["id"]

        update_resp = test_client.put(
            f"/v1/mcp-servers/{server_id}",
            json={"description": "Updated description"},
            headers=workspace_headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["mcp_server"]["description"] == "Updated description"

    def test_update_server_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        resp = test_client.put(
            "/v1/mcp-servers/nonexistent",
            json={"description": "x"},
            headers=workspace_headers,
        )
        assert resp.status_code == 404


class TestMcpServerDelete:
    def test_delete_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        create_resp = test_client.post(
            "/v1/mcp-servers",
            json={"name": "delete-test-server", "transport": "stdio", "command": "sh"},
            headers=workspace_headers,
        )
        server_id = create_resp.json()["mcp_server"]["id"]

        del_resp = test_client.delete(f"/v1/mcp-servers/{server_id}", headers=workspace_headers)
        assert del_resp.status_code == 204

        get_resp = test_client.get(f"/v1/mcp-servers/{server_id}", headers=workspace_headers)
        assert get_resp.status_code == 404

    def test_delete_server_not_found(self, test_client: TestClient, workspace_headers: dict) -> None:
        resp = test_client.delete("/v1/mcp-servers/nonexistent", headers=workspace_headers)
        assert resp.status_code == 404


class TestMcpServerImport:
    def test_import_multi_server_json(self, test_client: TestClient, workspace_headers: dict) -> None:
        payload = {
            "mcpServers": {
                "import-postgres": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-postgres"],
                    "env": {"DATABASE_URL": "postgresql://localhost/mydb"},
                },
                "import-filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
                "import-web-search": {
                    "url": "https://mcp.example.com/search",
                    "type": "http",
                },
            }
        }
        resp = test_client.post("/v1/mcp-servers/import", json=payload, headers=workspace_headers)
        assert resp.status_code == 200
        result = resp.json()
        assert result["imported"] == 3
        assert result["updated"] == 0
        assert result["errors"] == []

    def test_import_idempotent_update(self, test_client: TestClient, workspace_headers: dict) -> None:
        payload = {
            "mcpServers": {
                "idempotent-server": {
                    "command": "python",
                    "args": ["-m", "myserver"],
                }
            }
        }
        resp1 = test_client.post("/v1/mcp-servers/import", json=payload, headers=workspace_headers)
        assert resp1.json()["imported"] == 1

        resp2 = test_client.post("/v1/mcp-servers/import", json=payload, headers=workspace_headers)
        assert resp2.json()["updated"] == 1
        assert resp2.json()["imported"] == 0


class TestMcpServerExport:
    def test_export_roundtrip(self, test_client: TestClient, workspace_headers: dict) -> None:
        servers = [
            {"name": "export-stdio", "transport": "stdio", "command": "npx", "args": ["-y", "server"]},
            {"name": "export-http", "transport": "http", "url": "https://api.example.com/mcp"},
        ]
        for s in servers:
            test_client.post("/v1/mcp-servers", json=s, headers=workspace_headers)

        resp = test_client.get("/v1/mcp-servers/export", headers=workspace_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "mcpServers" in data
        mcp_servers = data["mcpServers"]
        assert "export-stdio" in mcp_servers
        assert "export-http" in mcp_servers
        assert mcp_servers["export-stdio"]["command"] == "npx"
        assert mcp_servers["export-http"]["url"] == "https://api.example.com/mcp"

    def test_export_secrets_masked(self, test_client: TestClient, workspace_headers: dict) -> None:
        test_client.post(
            "/v1/mcp-servers",
            json={"name": "export-secret-server", "transport": "stdio", "command": "sh", "env": {"TOKEN": "real-token"}},
            headers=workspace_headers,
        )
        resp = test_client.get("/v1/mcp-servers/export", headers=workspace_headers)
        data = resp.json()
        if "export-secret-server" in data.get("mcpServers", {}):
            entry = data["mcpServers"]["export-secret-server"]
            if "env" in entry:
                assert entry["env"]["TOKEN"] == "***"


class TestMcpServerStubs:
    def test_resolve_missing_server_returns_empty(self, test_client: TestClient, workspace_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers/resolve",
            json={"name": "nonexistent-server"},
            headers=workspace_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mcp_server"] is None
        assert data["candidates"] == []

    def test_resolve_requires_name_or_query(self, test_client: TestClient, workspace_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers/resolve",
            json={},
            headers=workspace_headers,
        )
        assert resp.status_code == 400

    def test_sync_returns_404_for_unknown_server(self, test_client: TestClient, workspace_headers: dict) -> None:
        resp = test_client.post(
            "/v1/mcp-servers/nonexistent-sync-id/sync",
            json={"manifest_hash": "abc123"},
            headers=workspace_headers,
        )
        assert resp.status_code == 404

    def test_resolve_query_without_memory_service_returns_error(self, test_client: TestClient, workspace_headers: dict) -> None:
        """Query-based resolve must return 501/503, not silently empty results."""
        resp = test_client.post(
            "/v1/mcp-servers/resolve",
            json={"query": "find a postgres tool"},
            headers=workspace_headers,
        )
        # Memory service is not wired in the test deployment; must not return 200 empty
        assert resp.status_code in (501, 503), f"Expected 501 or 503 when memory service unavailable, got {resp.status_code}: {resp.text}"
