"""Unit tests for McpServerService with a mock storage backend."""

from __future__ import annotations

import pytest

from memorylayer_server.models.mcp_server import McpServer, McpServerCreateInput, McpServerUpdateInput
from memorylayer_server.services.mcp_servers.base import McpServerService, _compute_manifest_hash


class MockStorage:
    """Minimal in-memory stub implementing only the mcp_server ABC methods."""

    def __init__(self):
        self._servers: dict[str, McpServer] = {}

    async def create_mcp_server(self, server: McpServer) -> McpServer:
        self._servers[server.id] = server
        return server

    async def get_mcp_server(self, workspace_id: str, server_id: str):
        s = self._servers.get(server_id)
        if s and s.workspace_id == workspace_id:
            return s
        return None

    async def get_mcp_server_by_name(self, workspace_id: str, name: str, user_id=None):
        for s in self._servers.values():
            if s.workspace_id == workspace_id and s.name == name:
                if user_id is None or s.user_id == user_id:
                    return s
        return None

    async def list_mcp_servers(self, workspace_id, user_id=None, name=None, transport=None, enabled=None, limit=100, offset=0):
        results = [s for s in self._servers.values() if s.workspace_id == workspace_id]
        if user_id is not None:
            results = [s for s in results if s.user_id == user_id]
        if name is not None:
            results = [s for s in results if s.name == name]
        if transport is not None:
            results = [s for s in results if s.transport == transport]
        if enabled is not None:
            results = [s for s in results if s.enabled == enabled]
        return results[offset : offset + limit]

    async def update_mcp_server(self, workspace_id, server_id, updates):
        s = self._servers.get(server_id)
        if not s or s.workspace_id != workspace_id:
            return None
        updated = s.model_copy(update=updates)
        self._servers[server_id] = updated
        return updated

    async def delete_mcp_server(self, workspace_id, server_id):
        s = self._servers.get(server_id)
        if s and s.workspace_id == workspace_id:
            del self._servers[server_id]
            return True
        return False

    async def find_mcp_servers_by_name(self, name, scope_filters):
        results = []
        for s in self._servers.values():
            for f in scope_filters:
                if s.workspace_id == f["workspace_id"]:
                    uid = f.get("user_id")
                    if uid is None or s.user_id == uid:
                        if s.name == name:
                            results.append(s)
        return results


@pytest.fixture
def storage():
    return MockStorage()


@pytest.fixture
def service(storage):
    return McpServerService(storage=storage)


STDIO_INPUT = McpServerCreateInput(
    name="postgres",
    transport="stdio",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres"],
    env={"DB_URL": "postgresql://localhost/mydb"},
)

HTTP_INPUT = McpServerCreateInput(
    name="my-api",
    transport="http",
    url="https://example.com/mcp",
)


class TestMcpServerServiceCRUD:
    @pytest.mark.asyncio
    async def test_create_stdio_server(self, service):
        server = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        assert server.id.startswith("mcp_")
        assert server.name == "postgres"
        assert server.transport == "stdio"
        assert server.command == "npx"
        assert server.workspace_id == "ws1"

    @pytest.mark.asyncio
    async def test_create_sets_manifest_hash(self, service):
        server = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        assert server.manifest_hash != ""
        assert len(server.manifest_hash) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_create_http_server(self, service):
        server = await service.create_mcp_server(HTTP_INPUT, workspace_id="ws1")
        assert server.name == "my-api"
        assert server.transport == "http"
        assert server.url == "https://example.com/mcp"

    @pytest.mark.asyncio
    async def test_get_server(self, service):
        created = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        fetched = await service.get_mcp_server("ws1", created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_server_wrong_workspace_returns_none(self, service):
        created = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        result = await service.get_mcp_server("other-ws", created.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_server_by_name(self, service):
        await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        fetched = await service.get_mcp_server_by_name("ws1", "postgres")
        assert fetched is not None
        assert fetched.name == "postgres"

    @pytest.mark.asyncio
    async def test_list_servers(self, service):
        await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        await service.create_mcp_server(HTTP_INPUT, workspace_id="ws1")
        servers = await service.list_mcp_servers("ws1")
        assert len(servers) == 2

    @pytest.mark.asyncio
    async def test_list_filter_by_transport(self, service):
        await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        await service.create_mcp_server(HTTP_INPUT, workspace_id="ws1")
        servers = await service.list_mcp_servers("ws1", transport="stdio")
        assert len(servers) == 1
        assert servers[0].name == "postgres"

    @pytest.mark.asyncio
    async def test_update_description(self, service):
        created = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        upd = McpServerUpdateInput(description="Updated desc")
        result = await service.update_mcp_server("ws1", created.id, upd)
        assert result is not None
        assert result.description == "Updated desc"

    @pytest.mark.asyncio
    async def test_update_recomputes_hash(self, service):
        created = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        original_hash = created.manifest_hash
        upd = McpServerUpdateInput(description="Changes hash")
        result = await service.update_mcp_server("ws1", created.id, upd)
        assert result.manifest_hash != original_hash

    @pytest.mark.asyncio
    async def test_update_nonexistent_returns_none(self, service):
        upd = McpServerUpdateInput(description="nope")
        result = await service.update_mcp_server("ws1", "mcp_doesnotexist", upd)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_server(self, service):
        created = await service.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        deleted = await service.delete_mcp_server("ws1", created.id)
        assert deleted is True
        fetched = await service.get_mcp_server("ws1", created.id)
        assert fetched is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, service):
        result = await service.delete_mcp_server("ws1", "mcp_doesnotexist")
        assert result is False

    @pytest.mark.asyncio
    async def test_memory_indexer_hook_called(self, service, storage):
        indexed = []

        async def hook(server):
            indexed.append(server)

        svc = McpServerService(storage=storage, memory_indexer=hook)
        server = await svc.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        assert len(indexed) == 1
        assert indexed[0].id == server.id

    @pytest.mark.asyncio
    async def test_memory_indexer_error_does_not_propagate(self, service, storage):
        def bad_hook(server):
            raise RuntimeError("indexer exploded")

        svc = McpServerService(storage=storage, memory_indexer=bad_hook)
        server = await svc.create_mcp_server(STDIO_INPUT, workspace_id="ws1")
        assert server.id.startswith("mcp_")  # no exception raised


class TestComputeManifestHash:
    def test_hash_is_deterministic(self):
        server = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
            args=["-y", "pg"],
        )
        h1 = _compute_manifest_hash(server)
        h2 = _compute_manifest_hash(server)
        assert h1 == h2

    def test_hash_changes_with_description(self):
        base = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
        )
        modified = base.model_copy(update={"description": "new desc"})
        assert _compute_manifest_hash(base) != _compute_manifest_hash(modified)

    def test_hash_changes_with_env(self):
        base = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
            env={},
        )
        with_env = base.model_copy(update={"env": {"KEY": "value"}})
        assert _compute_manifest_hash(base) != _compute_manifest_hash(with_env)
