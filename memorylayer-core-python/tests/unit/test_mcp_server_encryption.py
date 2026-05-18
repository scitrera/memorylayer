"""Unit tests for MCP server encryption passthrough and masking."""

from __future__ import annotations

import pytest

from memorylayer_server.models.mcp_server import McpServer, McpServerCreateInput, McpServerUpdateInput
from memorylayer_server.services.mcp_servers import McpServerService
from memorylayer_server.services.mcp_servers.encryption import (
    decrypt_secrets,
    encrypt_secrets,
    register_encrypter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_encrypter():
    """Return a simple reversible encrypter pair for testing."""

    def encrypt(data: dict) -> dict:
        return {k: f"enc:{v}" for k, v in data.items()}

    def decrypt(data: dict) -> dict:
        return {k: v[4:] if v.startswith("enc:") else v for k, v in data.items()}

    return encrypt, decrypt


class MockStorage:
    def __init__(self):
        self._servers: dict[str, McpServer] = {}
        self.stored_env: dict[str, dict] = {}  # raw as stored

    async def create_mcp_server(self, server: McpServer) -> McpServer:
        self._servers[server.id] = server
        self.stored_env[server.id] = dict(server.env or {})
        return server

    async def get_mcp_server(self, workspace_id, server_id):
        s = self._servers.get(server_id)
        return s if s and s.workspace_id == workspace_id else None

    async def get_mcp_server_by_name(self, workspace_id, name, user_id=None):
        for s in self._servers.values():
            if s.workspace_id == workspace_id and s.name == name:
                return s
        return None

    async def list_mcp_servers(self, workspace_id, user_id=None, name=None, transport=None, enabled=None, limit=100, offset=0):
        return [s for s in self._servers.values() if s.workspace_id == workspace_id]

    async def update_mcp_server(self, workspace_id, server_id, updates):
        s = self._servers.get(server_id)
        if not s or s.workspace_id != workspace_id:
            return None
        updated = s.model_copy(update=updates)
        self._servers[server_id] = updated
        if "env" in updates:
            self.stored_env[server_id] = dict(updates["env"] or {})
        return updated

    async def delete_mcp_server(self, workspace_id, server_id):
        s = self._servers.get(server_id)
        if s and s.workspace_id == workspace_id:
            del self._servers[server_id]
            return True
        return False

    async def find_mcp_servers_by_name(self, name, scope_filters):
        return []


# ── Tests: passthrough (no encrypter registered) ──────────────────────────────


class TestEncryptionPassthrough:
    def setup_method(self):
        # Reset module-level state to passthrough
        register_encrypter.__module__
        import memorylayer_server.services.mcp_servers.encryption as enc_mod

        enc_mod._encrypt_fn = None
        enc_mod._decrypt_fn = None

    def test_encrypt_passthrough_returns_same_dict(self):
        data = {"KEY": "value"}
        assert encrypt_secrets(data) == data

    def test_decrypt_passthrough_returns_same_dict(self):
        data = {"KEY": "value"}
        assert decrypt_secrets(data) == data

    def test_encrypt_empty_dict_returns_empty(self):
        assert encrypt_secrets({}) == {}

    def test_decrypt_empty_dict_returns_empty(self):
        assert decrypt_secrets({}) == {}

    def test_encrypt_none_returns_none(self):
        assert encrypt_secrets(None) is None

    def test_decrypt_none_returns_none(self):
        assert decrypt_secrets(None) is None


# ── Tests: with encrypter registered ─────────────────────────────────────────


class TestEncryptionRegistered:
    def setup_method(self):
        import memorylayer_server.services.mcp_servers.encryption as enc_mod

        enc, dec = _make_encrypter()
        enc_mod._encrypt_fn = enc
        enc_mod._decrypt_fn = dec

    def teardown_method(self):
        import memorylayer_server.services.mcp_servers.encryption as enc_mod

        enc_mod._encrypt_fn = None
        enc_mod._decrypt_fn = None

    def test_encrypt_transforms_values(self):
        result = encrypt_secrets({"KEY": "secret"})
        assert result == {"KEY": "enc:secret"}

    def test_decrypt_transforms_values(self):
        result = decrypt_secrets({"KEY": "enc:secret"})
        assert result == {"KEY": "secret"}

    def test_round_trip(self):
        original = {"DB_URL": "postgresql://user:pass@host/db", "API_KEY": "abc123"}
        assert decrypt_secrets(encrypt_secrets(original)) == original

    def test_encrypt_empty_passthrough(self):
        assert encrypt_secrets({}) == {}

    def test_decrypt_empty_passthrough(self):
        assert decrypt_secrets({}) == {}


# ── Tests: service encrypt/decrypt round-trip ─────────────────────────────────


class TestServiceEncryptionRoundTrip:
    def setup_method(self):
        import memorylayer_server.services.mcp_servers.encryption as enc_mod

        enc, dec = _make_encrypter()
        enc_mod._encrypt_fn = enc
        enc_mod._decrypt_fn = dec
        self.storage = MockStorage()
        self.service = McpServerService(storage=self.storage)

    def teardown_method(self):
        import memorylayer_server.services.mcp_servers.encryption as enc_mod

        enc_mod._encrypt_fn = None
        enc_mod._decrypt_fn = None

    @pytest.mark.asyncio
    async def test_create_returns_plaintext_env(self):
        inp = McpServerCreateInput(
            name="postgres",
            transport="stdio",
            command="npx",
            env={"DB_URL": "postgresql://localhost/mydb"},
        )
        server = await self.service.create_mcp_server(inp, workspace_id="ws1")
        assert server.env == {"DB_URL": "postgresql://localhost/mydb"}

    @pytest.mark.asyncio
    async def test_create_stores_encrypted_env(self):
        inp = McpServerCreateInput(
            name="postgres",
            transport="stdio",
            command="npx",
            env={"DB_URL": "postgresql://localhost/mydb"},
        )
        server = await self.service.create_mcp_server(inp, workspace_id="ws1")
        stored = self.storage.stored_env[server.id]
        assert stored == {"DB_URL": "enc:postgresql://localhost/mydb"}

    @pytest.mark.asyncio
    async def test_update_stores_encrypted_env(self):
        inp = McpServerCreateInput(
            name="postgres",
            transport="stdio",
            command="npx",
            env={"DB_URL": "old"},
        )
        server = await self.service.create_mcp_server(inp, workspace_id="ws1")
        upd = McpServerUpdateInput(env={"DB_URL": "new"})
        result = await self.service.update_mcp_server("ws1", server.id, upd)
        assert result.env == {"DB_URL": "new"}
        stored = self.storage.stored_env[server.id]
        assert stored == {"DB_URL": "enc:new"}

    @pytest.mark.asyncio
    async def test_headers_encrypted_on_create(self):
        inp = McpServerCreateInput(
            name="my-api",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer token123"},
        )
        server = await self.service.create_mcp_server(inp, workspace_id="ws1")
        assert server.headers == {"Authorization": "Bearer token123"}
        # Peek into stored data
        stored_server = self.storage._servers[server.id]
        assert stored_server.headers == {"Authorization": "enc:Bearer token123"}


# ── Tests: _mask_secrets ${VAR} preservation ──────────────────────────────────


class TestMaskSecretsVarPreservation:
    def test_var_placeholder_preserved(self):
        from memorylayer_server.api.v1.mcp_servers import _mask_secrets

        server = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
            env={"DB_URL": "${DATABASE_URL}", "API_KEY": "realvalue"},
        )
        masked = _mask_secrets(server, reveal=False)
        assert masked.env["DB_URL"] == "${DATABASE_URL}"
        assert masked.env["API_KEY"] == "***"

    def test_reveal_returns_plaintext(self):
        from memorylayer_server.api.v1.mcp_servers import _mask_secrets

        server = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
            env={"DB_URL": "postgresql://localhost/mydb"},
        )
        masked = _mask_secrets(server, reveal=True)
        assert masked.env["DB_URL"] == "postgresql://localhost/mydb"

    def test_headers_var_placeholder_preserved(self):
        from memorylayer_server.api.v1.mcp_servers import _mask_secrets

        server = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="my-api",
            transport="http",
            url="https://example.com",
            headers={"Authorization": "${AUTH_TOKEN}", "X-Custom": "literal"},
        )
        masked = _mask_secrets(server, reveal=False)
        assert masked.headers["Authorization"] == "${AUTH_TOKEN}"
        assert masked.headers["X-Custom"] == "***"

    def test_empty_env_returns_empty(self):
        from memorylayer_server.api.v1.mcp_servers import _mask_secrets

        server = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
        )
        masked = _mask_secrets(server, reveal=False)
        assert masked.env == {}
        assert masked.headers == {}
