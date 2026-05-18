"""
Unit tests for in-memory MCP server storage backend.

Tests MCP server CRUD operations against MemoryStorageBackend directly,
without requiring the full service stack.
"""

import pytest

from memorylayer_server.models.mcp_server import McpServer
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend

WORKSPACE_ID = "ws_mcp_test"
USER_ID = "user_abc"


def _make_stdio_server(
    name: str = "postgres-mcp",
    workspace_id: str = WORKSPACE_ID,
    user_id: str | None = None,
    enabled: bool = True,
) -> McpServer:
    return McpServer(
        id=f"mcp_{name.replace('-', '')}001",
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-postgres"],
        env={"DATABASE_URL": "postgres://localhost/mydb"},
        enabled=enabled,
    )


def _make_http_server(
    name: str = "github-mcp",
    workspace_id: str = WORKSPACE_ID,
    user_id: str | None = None,
    enabled: bool = True,
) -> McpServer:
    return McpServer(
        id=f"mcp_{name.replace('-', '')}001",
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        transport="http",
        url="https://mcp.example.com/github",
        headers={"Authorization": "Bearer token123"},
        enabled=enabled,
    )


@pytest.fixture
def backend():
    return MemoryStorageBackend()


@pytest.mark.asyncio
class TestMcpServerCRUD:
    """Test create / get / update / delete for MCP servers."""

    async def test_create_and_get_mcp_server(self, backend):
        server = _make_stdio_server()
        created = await backend.create_mcp_server(server)
        assert created.id == server.id
        assert created.name == "postgres-mcp"

        fetched = await backend.get_mcp_server(WORKSPACE_ID, server.id)
        assert fetched is not None
        assert fetched.id == server.id

    async def test_get_mcp_server_returns_none_for_missing(self, backend):
        result = await backend.get_mcp_server(WORKSPACE_ID, "mcp_nonexistent")
        assert result is None

    async def test_get_mcp_server_wrong_workspace(self, backend):
        server = _make_stdio_server()
        await backend.create_mcp_server(server)
        result = await backend.get_mcp_server("other_ws", server.id)
        assert result is None

    async def test_get_mcp_server_by_name(self, backend):
        server = _make_http_server(name="table-parser")
        await backend.create_mcp_server(server)

        found = await backend.get_mcp_server_by_name(WORKSPACE_ID, "table-parser")
        assert found is not None
        assert found.id == server.id

    async def test_get_mcp_server_by_name_with_user_filter(self, backend):
        server = _make_stdio_server(name="my-server", user_id=USER_ID)
        await backend.create_mcp_server(server)

        found = await backend.get_mcp_server_by_name(WORKSPACE_ID, "my-server", user_id=USER_ID)
        assert found is not None

        not_found = await backend.get_mcp_server_by_name(WORKSPACE_ID, "my-server", user_id="other_user")
        assert not_found is None

    async def test_get_mcp_server_by_name_returns_none_for_missing(self, backend):
        result = await backend.get_mcp_server_by_name(WORKSPACE_ID, "nonexistent-server")
        assert result is None

    async def test_update_mcp_server_fields(self, backend):
        server = _make_stdio_server(name="update-me")
        await backend.create_mcp_server(server)

        updated = await backend.update_mcp_server(WORKSPACE_ID, server.id, {"description": "Updated desc", "enabled": False})
        assert updated is not None
        assert updated.description == "Updated desc"
        assert updated.enabled is False

    async def test_update_mcp_server_returns_none_for_missing(self, backend):
        result = await backend.update_mcp_server(WORKSPACE_ID, "mcp_missing", {"enabled": False})
        assert result is None

    async def test_update_mcp_server_sets_updated_at(self, backend):
        server = _make_stdio_server(name="timestamp-test")
        original_time = server.updated_at
        await backend.create_mcp_server(server)

        updated = await backend.update_mcp_server(WORKSPACE_ID, server.id, {"description": "New desc"})
        assert updated.updated_at >= original_time

    async def test_delete_mcp_server(self, backend):
        server = _make_stdio_server(name="delete-me")
        await backend.create_mcp_server(server)

        result = await backend.delete_mcp_server(WORKSPACE_ID, server.id)
        assert result is True

        fetched = await backend.get_mcp_server(WORKSPACE_ID, server.id)
        assert fetched is None

    async def test_delete_mcp_server_returns_false_for_missing(self, backend):
        result = await backend.delete_mcp_server(WORKSPACE_ID, "mcp_gone")
        assert result is False

    async def test_stdio_and_http_servers_coexist(self, backend):
        stdio = _make_stdio_server(name="stdio-server")
        http = _make_http_server(name="http-server")
        await backend.create_mcp_server(stdio)
        await backend.create_mcp_server(http)

        fetched_stdio = await backend.get_mcp_server(WORKSPACE_ID, stdio.id)
        fetched_http = await backend.get_mcp_server(WORKSPACE_ID, http.id)
        assert fetched_stdio.transport == "stdio"
        assert fetched_http.transport == "http"


@pytest.mark.asyncio
class TestListMcpServers:
    """Test list_mcp_servers filtering and pagination."""

    async def test_list_mcp_servers_basic(self, backend):
        ws = "ws_list_mcp"
        for name in ["alpha-mcp", "beta-mcp", "gamma-mcp"]:
            await backend.create_mcp_server(_make_stdio_server(name=name, workspace_id=ws))

        results = await backend.list_mcp_servers(ws)
        assert len(results) == 3

    async def test_list_mcp_servers_filter_by_transport(self, backend):
        ws = "ws_list_transport"
        await backend.create_mcp_server(_make_stdio_server(name="stdio-one", workspace_id=ws))
        await backend.create_mcp_server(_make_http_server(name="http-one", workspace_id=ws))

        stdio_results = await backend.list_mcp_servers(ws, transport="stdio")
        assert len(stdio_results) == 1
        assert stdio_results[0].name == "stdio-one"

        http_results = await backend.list_mcp_servers(ws, transport="http")
        assert len(http_results) == 1
        assert http_results[0].name == "http-one"

    async def test_list_mcp_servers_filter_by_enabled(self, backend):
        ws = "ws_list_enabled_mcp"
        await backend.create_mcp_server(_make_stdio_server(name="active-mcp", workspace_id=ws, enabled=True))
        await backend.create_mcp_server(_make_stdio_server(name="inactive-mcp", workspace_id=ws, enabled=False))

        active = await backend.list_mcp_servers(ws, enabled=True)
        assert len(active) == 1
        assert active[0].name == "active-mcp"

        inactive = await backend.list_mcp_servers(ws, enabled=False)
        assert len(inactive) == 1
        assert inactive[0].name == "inactive-mcp"

    async def test_list_mcp_servers_filter_by_user_id(self, backend):
        ws = "ws_list_user_mcp"
        await backend.create_mcp_server(_make_stdio_server(name="user-mcp", workspace_id=ws, user_id=USER_ID))
        await backend.create_mcp_server(_make_stdio_server(name="ws-mcp", workspace_id=ws, user_id=None))

        user_servers = await backend.list_mcp_servers(ws, user_id=USER_ID)
        assert len(user_servers) == 1
        assert user_servers[0].name == "user-mcp"

    async def test_list_mcp_servers_pagination(self, backend):
        ws = "ws_list_page_mcp"
        for i in range(5):
            await backend.create_mcp_server(_make_stdio_server(name=f"mcp-{i:02d}", workspace_id=ws))

        page1 = await backend.list_mcp_servers(ws, limit=3, offset=0)
        page2 = await backend.list_mcp_servers(ws, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2

    async def test_list_mcp_servers_empty_workspace(self, backend):
        results = await backend.list_mcp_servers("ws_empty_mcp")
        assert results == []


@pytest.mark.asyncio
class TestFindMcpServersByName:
    """Test find_mcp_servers_by_name for precedence resolution."""

    async def test_find_mcp_servers_across_scopes(self, backend):
        ws_global = "_global"
        ws_project = "ws_proj_mcp"

        global_server = McpServer(
            id="mcp_global001xxx",
            workspace_id=ws_global,
            name="pdf-reader",
            transport="stdio",
            command="npx",
            args=["-y", "pdf-mcp"],
        )
        project_server = McpServer(
            id="mcp_proj001xxxxx",
            workspace_id=ws_project,
            name="pdf-reader",
            transport="stdio",
            command="npx",
            args=["-y", "pdf-mcp-v2"],
        )
        await backend.create_mcp_server(global_server)
        await backend.create_mcp_server(project_server)

        scope_filters = [
            {"workspace_id": ws_project},
            {"workspace_id": ws_global},
        ]
        results = await backend.find_mcp_servers_by_name("pdf-reader", scope_filters)
        assert len(results) == 2
        ids = {s.id for s in results}
        assert "mcp_global001xxx" in ids
        assert "mcp_proj001xxxxx" in ids

    async def test_find_mcp_servers_by_name_no_match(self, backend):
        results = await backend.find_mcp_servers_by_name("nonexistent", [{"workspace_id": WORKSPACE_ID}])
        assert results == []

    async def test_find_mcp_servers_by_name_user_scope_filter(self, backend):
        ws = "ws_scope_mcp"
        user_server = McpServer(
            id="mcp_user001xxxxx",
            workspace_id=ws,
            user_id=USER_ID,
            name="scoped-mcp",
            transport="http",
            url="https://mcp.example.com/user",
        )
        ws_server = McpServer(
            id="mcp_ws001xxxxxxx",
            workspace_id=ws,
            user_id=None,
            name="scoped-mcp",
            transport="http",
            url="https://mcp.example.com/ws",
        )
        await backend.create_mcp_server(user_server)
        await backend.create_mcp_server(ws_server)

        results = await backend.find_mcp_servers_by_name("scoped-mcp", [{"workspace_id": ws, "user_id": USER_ID}])
        assert len(results) == 1
        assert results[0].id == "mcp_user001xxxxx"
