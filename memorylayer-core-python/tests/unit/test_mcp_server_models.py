"""Unit tests for MCP server domain models."""

import pytest
from pydantic import ValidationError

from memorylayer_server.models.mcp_server import (
    McpJsonDocument,
    McpServer,
    McpServerCreateInput,
    McpServerEntry,
    McpServerUpdateInput,
    validate_mcp_server_name,
)

# --- Name validation ---


class TestValidateMcpServerName:
    def test_valid_simple(self):
        assert validate_mcp_server_name("postgres") == "postgres"

    def test_valid_with_hyphens(self):
        assert validate_mcp_server_name("my-mcp-server") == "my-mcp-server"

    def test_valid_alphanumeric(self):
        assert validate_mcp_server_name("server123") == "server123"

    def test_valid_single_char(self):
        assert validate_mcp_server_name("a") == "a"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_mcp_server_name("")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="64 chars"):
            validate_mcp_server_name("a" * 65)

    def test_exactly_64_chars_ok(self):
        name = "a" * 64
        assert validate_mcp_server_name(name) == name

    def test_leading_hyphen_raises(self):
        with pytest.raises(ValueError, match="must not start or end with a hyphen"):
            validate_mcp_server_name("-server")

    def test_trailing_hyphen_raises(self):
        with pytest.raises(ValueError, match="must not start or end with a hyphen"):
            validate_mcp_server_name("server-")

    def test_consecutive_hyphens_raises(self):
        with pytest.raises(ValueError, match="consecutive hyphens"):
            validate_mcp_server_name("my--server")

    def test_uppercase_raises(self):
        with pytest.raises(ValueError, match="lowercase"):
            validate_mcp_server_name("MyServer")

    def test_underscore_raises(self):
        with pytest.raises(ValueError, match="lowercase"):
            validate_mcp_server_name("my_server")


# --- McpServer model ---


class TestMcpServer:
    def test_stdio_server_valid(self):
        s = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DB_URL": "postgresql://localhost/mydb"},
        )
        assert s.transport == "stdio"
        assert s.command == "npx"
        assert s.enabled is True

    def test_http_server_valid(self):
        s = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="my-api",
            transport="http",
            url="https://example.com/mcp",
        )
        assert s.url == "https://example.com/mcp"

    def test_stdio_without_command_raises(self):
        with pytest.raises(ValidationError, match="command is required"):
            McpServer(
                id="mcp_abc123def456",
                workspace_id="ws1",
                name="postgres",
                transport="stdio",
            )

    def test_http_without_url_raises(self):
        with pytest.raises(ValidationError, match="url is required"):
            McpServer(
                id="mcp_abc123def456",
                workspace_id="ws1",
                name="my-api",
                transport="http",
            )

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError, match="1024 chars"):
            McpServer(
                id="mcp_abc123def456",
                workspace_id="ws1",
                name="postgres",
                transport="stdio",
                command="npx",
                description="x" * 1025,
            )

    def test_default_tenant_id(self):
        s = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="postgres",
            transport="stdio",
            command="npx",
        )
        assert s.tenant_id == "_default"

    def test_sse_transport(self):
        s = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="my-sse",
            transport="sse",
            url="https://example.com/sse",
        )
        assert s.transport == "sse"

    def test_streamable_http_transport(self):
        s = McpServer(
            id="mcp_abc123def456",
            workspace_id="ws1",
            name="my-stream",
            transport="streamable-http",
            url="https://example.com/stream",
        )
        assert s.transport == "streamable-http"


# --- McpServerCreateInput ---


class TestMcpServerCreateInput:
    def test_valid_stdio(self):
        inp = McpServerCreateInput(
            name="postgres",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
        )
        assert inp.name == "postgres"

    def test_valid_http(self):
        inp = McpServerCreateInput(
            name="my-api",
            transport="http",
            url="https://example.com/mcp",
        )
        assert inp.url == "https://example.com/mcp"

    def test_stdio_without_command_raises(self):
        with pytest.raises(ValidationError, match="command is required"):
            McpServerCreateInput(name="bad", transport="stdio")

    def test_http_without_url_raises(self):
        with pytest.raises(ValidationError, match="url is required"):
            McpServerCreateInput(name="bad", transport="http")

    def test_invalid_name_raises(self):
        with pytest.raises(ValidationError):
            McpServerCreateInput(name="Bad-Name", transport="stdio", command="npx")


# --- McpServerUpdateInput ---


class TestMcpServerUpdateInput:
    def test_all_optional(self):
        upd = McpServerUpdateInput()
        assert upd.description is None
        assert upd.enabled is None

    def test_partial_update(self):
        upd = McpServerUpdateInput(enabled=False, description="updated desc")
        assert upd.enabled is False
        assert upd.description == "updated desc"

    def test_description_too_long_raises(self):
        with pytest.raises(ValidationError, match="1024 chars"):
            McpServerUpdateInput(description="x" * 1025)


# --- McpJsonDocument ---

SAMPLE_MCP_JSON = {
    "mcpServers": {
        "postgres": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"],
            "env": {"DB_URL": "postgresql://localhost/mydb"},
        },
        "my-api": {
            "url": "https://example.com/mcp",
            "type": "http",
        },
    }
}


class TestMcpJsonDocument:
    def test_parse_multi_server(self):
        doc = McpJsonDocument(**SAMPLE_MCP_JSON)
        assert len(doc.mcpServers) == 2
        assert "postgres" in doc.mcpServers
        assert "my-api" in doc.mcpServers

    def test_stdio_entry_inferred_transport(self):
        doc = McpJsonDocument(**SAMPLE_MCP_JSON)
        assert doc.mcpServers["postgres"].transport == "stdio"
        assert doc.mcpServers["postgres"].command == "npx"

    def test_http_entry_inferred_transport(self):
        doc = McpJsonDocument(**SAMPLE_MCP_JSON)
        assert doc.mcpServers["my-api"].transport == "http"
        assert doc.mcpServers["my-api"].url == "https://example.com/mcp"

    def test_to_server_create_inputs(self):
        doc = McpJsonDocument(**SAMPLE_MCP_JSON)
        inputs = doc.to_server_create_inputs(workspace_id="ws1")
        assert len(inputs) == 2
        names = {i.name for i in inputs}
        assert names == {"postgres", "my-api"}

    def test_to_server_create_inputs_with_user_id(self):
        doc = McpJsonDocument(**SAMPLE_MCP_JSON)
        inputs = doc.to_server_create_inputs(workspace_id="ws1", user_id="user1")
        for inp in inputs:
            assert inp.user_id == "user1"
            assert inp.workspace_id == "ws1"

    def test_empty_document(self):
        doc = McpJsonDocument()
        assert doc.mcpServers == {}
        inputs = doc.to_server_create_inputs()
        assert inputs == []

    def test_explicit_transport_respected(self):
        doc = McpJsonDocument(mcpServers={"my-sse": McpServerEntry(transport="sse", url="https://example.com/sse")})
        assert doc.mcpServers["my-sse"].transport == "sse"
