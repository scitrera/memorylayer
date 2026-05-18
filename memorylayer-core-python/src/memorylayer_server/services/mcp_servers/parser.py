"""MCP server JSON parsing and rendering utilities.

Handles .mcp.json file format (Claude Code's multi-server config shape):
  {"mcpServers": {"server-name": {...}, ...}}

${VAR} placeholders are passed through verbatim — never expanded.
"""

from __future__ import annotations

import json

from ...models.mcp_server import McpJsonDocument, McpServer, McpServerCreateInput


def parse_mcp_json(text: str) -> list[McpServerCreateInput]:
    """Parse a .mcp.json text blob into a list of McpServerCreateInput objects.

    Accepts both single-server JSON (a bare entry dict) and the standard
    multi-server {"mcpServers": {...}} envelope.
    """
    data = json.loads(text)

    if "mcpServers" not in data:
        raise ValueError("Invalid .mcp.json: expected top-level 'mcpServers' key. Got keys: " + ", ".join(data.keys()))

    doc = McpJsonDocument(**data)
    return doc.to_server_create_inputs()


def render_mcp_json(servers: list[McpServer]) -> dict:
    """Render a list of McpServer records into a .mcp.json-shaped dict.

    Output format: {"mcpServers": {"name": {<runtime fields only>}, ...}}

    Strips server-managed fields: id, tenant_id, workspace_id, user_id,
    source_mode, manifest_hash, enabled, created_at, updated_at, metadata.
    ${VAR} placeholders are preserved verbatim.
    """
    mcp_servers: dict[str, dict] = {}
    for server in servers:
        entry: dict = {}
        if server.transport == "stdio":
            if server.command:
                entry["command"] = server.command
            if server.args:
                entry["args"] = server.args
            if server.env:
                entry["env"] = server.env
        else:
            if server.url:
                entry["url"] = server.url
            if server.headers:
                entry["headers"] = server.headers
            entry["type"] = server.transport
        if server.description:
            entry["description"] = server.description
        mcp_servers[server.name] = entry

    return {"mcpServers": mcp_servers}
