"""Unit tests for MCP server JSON parsing and rendering."""

import json

import pytest

from memorylayer_server.models.mcp_server import McpServer
from memorylayer_server.services.mcp_servers.parser import parse_mcp_json, render_mcp_json

SAMPLE_3_SERVER_MCP_JSON = {
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
        "my-sse": {
            "url": "https://example.com/sse",
            "type": "sse",
        },
    }
}

SAMPLE_TEXT = json.dumps(SAMPLE_3_SERVER_MCP_JSON)


class TestParseMcpJson:
    def test_parses_3_servers(self):
        inputs = parse_mcp_json(SAMPLE_TEXT)
        assert len(inputs) == 3

    def test_stdio_server_fields(self):
        inputs = parse_mcp_json(SAMPLE_TEXT)
        pg = next(i for i in inputs if i.name == "postgres")
        assert pg.transport == "stdio"
        assert pg.command == "npx"
        assert pg.args == ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
        assert pg.env == {"DB_URL": "postgresql://localhost/mydb"}

    def test_http_server_fields(self):
        inputs = parse_mcp_json(SAMPLE_TEXT)
        api = next(i for i in inputs if i.name == "my-api")
        assert api.transport == "http"
        assert api.url == "https://example.com/mcp"

    def test_sse_server_fields(self):
        inputs = parse_mcp_json(SAMPLE_TEXT)
        sse = next(i for i in inputs if i.name == "my-sse")
        assert sse.transport == "sse"
        assert sse.url == "https://example.com/sse"

    def test_missing_mcp_servers_key_raises(self):
        bad = json.dumps({"servers": {}})
        with pytest.raises(ValueError, match="mcpServers"):
            parse_mcp_json(bad)

    def test_empty_servers(self):
        text = json.dumps({"mcpServers": {}})
        inputs = parse_mcp_json(text)
        assert inputs == []

    def test_var_placeholder_preserved(self):
        text = json.dumps(
            {
                "mcpServers": {
                    "my-server": {
                        "command": "npx",
                        "env": {"API_KEY": "${MY_API_KEY}"},
                    }
                }
            }
        )
        inputs = parse_mcp_json(text)
        assert inputs[0].env["API_KEY"] == "${MY_API_KEY}"


def _make_server(name: str, transport: str, **kwargs) -> McpServer:
    """Helper to create a minimal McpServer for render tests."""
    base = dict(
        id=f"mcp_{name[:12].ljust(12, '0')}",
        workspace_id="ws1",
        name=name,
        transport=transport,
    )
    base.update(kwargs)
    return McpServer(**base)


class TestRenderMcpJson:
    def test_render_stdio_server(self):
        server = _make_server(
            "postgres",
            "stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={"DB_URL": "postgresql://localhost/mydb"},
        )
        result = render_mcp_json([server])
        entry = result["mcpServers"]["postgres"]
        assert entry["command"] == "npx"
        assert entry["args"] == ["-y", "@modelcontextprotocol/server-postgres"]
        assert entry["env"]["DB_URL"] == "postgresql://localhost/mydb"

    def test_render_http_server(self):
        server = _make_server("my-api", "http", url="https://example.com/mcp")
        result = render_mcp_json([server])
        entry = result["mcpServers"]["my-api"]
        assert entry["url"] == "https://example.com/mcp"
        assert entry["type"] == "http"

    def test_render_sse_server(self):
        server = _make_server("my-sse", "sse", url="https://example.com/sse")
        result = render_mcp_json([server])
        assert result["mcpServers"]["my-sse"]["type"] == "sse"

    def test_strips_server_managed_fields(self):
        server = _make_server("postgres", "stdio", command="npx")
        result = render_mcp_json([server])
        entry = result["mcpServers"]["postgres"]
        for field in (
            "id",
            "tenant_id",
            "workspace_id",
            "user_id",
            "source_mode",
            "manifest_hash",
            "enabled",
            "created_at",
            "updated_at",
            "metadata",
        ):
            assert field not in entry

    def test_render_multiple_servers(self):
        servers = [
            _make_server("postgres", "stdio", command="npx", args=["-y", "@modelcontextprotocol/server-postgres"]),
            _make_server("my-api0", "http", url="https://example.com/mcp"),
            _make_server("my-sse00", "sse", url="https://example.com/sse"),
        ]
        result = render_mcp_json(servers)
        assert set(result["mcpServers"].keys()) == {"postgres", "my-api0", "my-sse00"}

    def test_round_trip_preserves_var_placeholders(self):
        text = json.dumps(
            {
                "mcpServers": {
                    "my-server": {
                        "command": "npx",
                        "env": {"API_KEY": "${MY_API_KEY}"},
                    }
                }
            }
        )
        inputs = parse_mcp_json(text)
        server = _make_server(
            inputs[0].name,
            inputs[0].transport,
            command=inputs[0].command,
            env=inputs[0].env,
        )
        rendered = render_mcp_json([server])
        assert rendered["mcpServers"]["my-server"]["env"]["API_KEY"] == "${MY_API_KEY}"

    def test_empty_list(self):
        result = render_mcp_json([])
        assert result == {"mcpServers": {}}
