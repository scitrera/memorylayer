"""Unit tests for SQLite MCP server storage implementation."""
import pytest

from memorylayer_server.models.mcp_server import McpServer
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.utils import generate_id


def _make_stdio_server(**overrides) -> McpServer:
    defaults = dict(
        id=generate_id("mcp"),
        workspace_id="ws_test",
        name="postgres-mcp",
        description="Query PostgreSQL databases via MCP",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        env={"DATABASE_URL": "postgresql://localhost/mydb"},
        source_mode="server",
    )
    defaults.update(overrides)
    return McpServer(**defaults)


def _make_http_server(**overrides) -> McpServer:
    defaults = dict(
        id=generate_id("mcp"),
        workspace_id="ws_test",
        name="web-search-mcp",
        description="Web search via HTTP MCP",
        transport="http",
        url="https://mcp.example.com/search",
        headers={"Authorization": "Bearer sk-test"},
        source_mode="server",
    )
    defaults.update(overrides)
    return McpServer(**defaults)


@pytest.fixture
async def sqlite_backend(tmp_path):
    db_path = str(tmp_path / "test_mcp_servers.db")
    backend = SQLiteStorageBackend(db_path)
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest.mark.asyncio
class TestMcpServerCRUD:
    async def test_create_and_get_stdio_server(self, sqlite_backend):
        server = _make_stdio_server()
        created = await sqlite_backend.create_mcp_server(server)
        assert created.id == server.id
        assert created.name == "postgres-mcp"
        assert created.transport == "stdio"
        assert created.command == "npx"
        assert created.args == ["-y", "@modelcontextprotocol/server-postgres"]
        assert created.env == {"DATABASE_URL": "postgresql://localhost/mydb"}

        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched is not None
        assert fetched.id == server.id
        assert fetched.description == server.description

    async def test_create_and_get_http_server(self, sqlite_backend):
        server = _make_http_server()
        created = await sqlite_backend.create_mcp_server(server)
        assert created.transport == "http"
        assert created.url == "https://mcp.example.com/search"
        assert created.headers == {"Authorization": "Bearer sk-test"}

        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched is not None
        assert fetched.url == server.url

    async def test_get_server_wrong_workspace(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)
        result = await sqlite_backend.get_mcp_server("other_ws", server.id)
        assert result is None

    async def test_get_server_by_name(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)
        fetched = await sqlite_backend.get_mcp_server_by_name("ws_test", "postgres-mcp")
        assert fetched is not None
        assert fetched.id == server.id

    async def test_get_server_by_name_user_scoped(self, sqlite_backend):
        user_server = _make_stdio_server(id=generate_id("mcp"), user_id="user_a")
        ws_server = _make_stdio_server(id=generate_id("mcp"), user_id=None)
        await sqlite_backend.create_mcp_server(user_server)
        await sqlite_backend.create_mcp_server(ws_server)

        found = await sqlite_backend.get_mcp_server_by_name("ws_test", "postgres-mcp", user_id="user_a")
        assert found is not None
        assert found.user_id == "user_a"

        found_ws = await sqlite_backend.get_mcp_server_by_name("ws_test", "postgres-mcp")
        assert found_ws is not None
        assert found_ws.user_id is None

    async def test_get_server_by_name_not_found(self, sqlite_backend):
        result = await sqlite_backend.get_mcp_server_by_name("ws_test", "nonexistent")
        assert result is None

    async def test_list_servers(self, sqlite_backend):
        s1 = _make_stdio_server(id=generate_id("mcp"), name="server-a", description="A")
        s2 = _make_http_server(id=generate_id("mcp"), name="server-b")
        await sqlite_backend.create_mcp_server(s1)
        await sqlite_backend.create_mcp_server(s2)

        servers = await sqlite_backend.list_mcp_servers("ws_test")
        names = {s.name for s in servers}
        assert "server-a" in names
        assert "server-b" in names

    async def test_list_servers_filter_transport(self, sqlite_backend):
        stdio = _make_stdio_server(id=generate_id("mcp"), name="stdio-server")
        http = _make_http_server(id=generate_id("mcp"), name="http-server")
        await sqlite_backend.create_mcp_server(stdio)
        await sqlite_backend.create_mcp_server(http)

        stdio_only = await sqlite_backend.list_mcp_servers("ws_test", transport="stdio")
        names = {s.name for s in stdio_only}
        assert "stdio-server" in names
        assert "http-server" not in names

    async def test_list_servers_filter_enabled(self, sqlite_backend):
        active = _make_stdio_server(id=generate_id("mcp"), name="active-server")
        disabled = _make_stdio_server(id=generate_id("mcp"), name="disabled-server", enabled=False)
        await sqlite_backend.create_mcp_server(active)
        await sqlite_backend.create_mcp_server(disabled)

        active_only = await sqlite_backend.list_mcp_servers("ws_test", enabled=True)
        names = {s.name for s in active_only}
        assert "active-server" in names
        assert "disabled-server" not in names

    async def test_list_servers_filter_name(self, sqlite_backend):
        s = _make_stdio_server()
        await sqlite_backend.create_mcp_server(s)

        found = await sqlite_backend.list_mcp_servers("ws_test", name="postgres-mcp")
        assert len(found) == 1
        assert found[0].id == s.id

    async def test_list_servers_empty_workspace(self, sqlite_backend):
        results = await sqlite_backend.list_mcp_servers("empty_ws")
        assert results == []

    async def test_update_server(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)

        updated = await sqlite_backend.update_mcp_server(
            "ws_test", server.id, {"description": "Updated description"}
        )
        assert updated is not None
        assert updated.description == "Updated description"

    async def test_update_server_env(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)

        new_env = {"DATABASE_URL": "postgresql://prod/db", "POOL_SIZE": "10"}
        updated = await sqlite_backend.update_mcp_server("ws_test", server.id, {"env": new_env})
        assert updated is not None
        assert updated.env == new_env

    async def test_update_server_not_found(self, sqlite_backend):
        result = await sqlite_backend.update_mcp_server("ws_test", "nonexistent_id", {"description": "x"})
        assert result is None

    async def test_delete_server(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)

        deleted = await sqlite_backend.delete_mcp_server("ws_test", server.id)
        assert deleted is True

        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched is None

    async def test_delete_server_not_found(self, sqlite_backend):
        result = await sqlite_backend.delete_mcp_server("ws_test", "nonexistent_id")
        assert result is False


@pytest.mark.asyncio
class TestFindMcpServersByName:
    async def test_find_across_scopes(self, sqlite_backend):
        user_server = _make_stdio_server(id=generate_id("mcp"), workspace_id="ws1", user_id="user_a")
        ws_server = _make_stdio_server(id=generate_id("mcp"), workspace_id="ws1", user_id=None)
        await sqlite_backend.create_mcp_server(user_server)
        await sqlite_backend.create_mcp_server(ws_server)

        results = await sqlite_backend.find_mcp_servers_by_name(
            "postgres-mcp",
            [
                {"workspace_id": "ws1", "user_id": "user_a"},
                {"workspace_id": "ws1"},
            ],
        )
        ids = {r.id for r in results}
        assert user_server.id in ids
        assert ws_server.id in ids

    async def test_find_empty_scope_filters(self, sqlite_backend):
        server = _make_stdio_server()
        await sqlite_backend.create_mcp_server(server)
        results = await sqlite_backend.find_mcp_servers_by_name("postgres-mcp", [])
        assert results == []

    async def test_find_no_match(self, sqlite_backend):
        results = await sqlite_backend.find_mcp_servers_by_name(
            "nonexistent-server", [{"workspace_id": "ws_test"}]
        )
        assert results == []

    async def test_find_cross_workspace_scopes(self, sqlite_backend):
        global_server = _make_stdio_server(
            id=generate_id("mcp"), workspace_id="_global", user_id=None
        )
        user_server = _make_stdio_server(
            id=generate_id("mcp"), workspace_id="_global_user", user_id="user_b"
        )
        await sqlite_backend.create_mcp_server(global_server)
        await sqlite_backend.create_mcp_server(user_server)

        results = await sqlite_backend.find_mcp_servers_by_name(
            "postgres-mcp",
            [
                {"workspace_id": "_global_user", "user_id": "user_b"},
                {"workspace_id": "_global"},
            ],
        )
        ids = {r.id for r in results}
        assert global_server.id in ids
        assert user_server.id in ids


@pytest.mark.asyncio
class TestMcpServerJsonFields:
    async def test_empty_json_fields_roundtrip(self, sqlite_backend):
        server = _make_stdio_server(args=[], env={})
        await sqlite_backend.create_mcp_server(server)
        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched.args == []
        assert fetched.env == {}

    async def test_metadata_roundtrip(self, sqlite_backend):
        meta = {"vendor": "acme", "version": "2.0", "tags": ["db", "postgres"]}
        server = _make_stdio_server(metadata=meta)
        await sqlite_backend.create_mcp_server(server)
        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched.metadata == meta

    async def test_manifest_hash_stored(self, sqlite_backend):
        server = _make_stdio_server(manifest_hash="abc123def456")
        await sqlite_backend.create_mcp_server(server)
        fetched = await sqlite_backend.get_mcp_server("ws_test", server.id)
        assert fetched.manifest_hash == "abc123def456"
