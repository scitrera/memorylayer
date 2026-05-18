"""MCP servers namespace for MemoryLayer.ai Python SDK.

Provides CRUD, resolve, and push/pull helpers for MCP server records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .client import MemoryLayerClient
    from .models import AuthorityContext
    from .sync_client import SyncMemoryLayerClient


class McpServerModel(BaseModel):
    """Pydantic model mirroring the server McpServer."""

    model_config = ConfigDict(extra="allow")

    id: str
    workspace_id: str
    tenant_id: str = "_default"
    user_id: str | None = None
    name: str
    description: str | None = None
    transport: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_mode: str = "server"
    manifest_hash: str = ""
    enabled: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class McpServersAPI:
    """MCP servers namespace — access via client.mcp_servers.<method>."""

    def __init__(self, client: MemoryLayerClient) -> None:
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list(
        self,
        transport: str | None = None,
        enabled: bool | None = None,
        name: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        workspace_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> list[McpServerModel]:
        """List MCP servers with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if transport is not None:
            params["transport"] = transport
        if enabled is not None:
            params["enabled"] = str(enabled).lower()
        if name is not None:
            params["name"] = name
        if user_id is not None:
            params["user_id"] = user_id

        data = await self._client._request("GET", "/mcp-servers", params=params, authority=authority)
        return [McpServerModel(**s) for s in data.get("mcp_servers", [])]

    async def get(self, server_id: str, authority: AuthorityContext | None = None) -> McpServerModel:
        """Get an MCP server by ID."""
        data = await self._client._request("GET", f"/mcp-servers/{server_id}", authority=authority)
        return McpServerModel(**data["mcp_server"])

    async def get_by_name(
        self,
        name: str,
        user_id: str | None = None,
        workspace_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> McpServerModel | None:
        """Get an MCP server by name; returns None if not found."""
        params: dict[str, Any] = {"name": name}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if user_id is not None:
            params["user_id"] = user_id

        try:
            data = await self._client._request("GET", "/mcp-servers/by-name", params=params, authority=authority)
            return McpServerModel(**data["mcp_server"])
        except Exception:
            return None

    async def create(
        self,
        name: str,
        transport: Literal["stdio", "http", "sse", "streamable-http"],
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_mode: str = "server",
        enabled: bool = True,
        workspace_id: str | None = None,
        user_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> McpServerModel:
        """Create a new MCP server record."""
        payload: dict[str, Any] = {
            "name": name,
            "transport": transport,
            "source_mode": source_mode,
            "enabled": enabled,
        }
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if user_id is not None:
            payload["user_id"] = user_id
        if description is not None:
            payload["description"] = description
        if command is not None:
            payload["command"] = command
        if args is not None:
            payload["args"] = args
        if env is not None:
            payload["env"] = env
        if url is not None:
            payload["url"] = url
        if headers is not None:
            payload["headers"] = headers
        if metadata is not None:
            payload["metadata"] = metadata

        data = await self._client._request("POST", "/mcp-servers", json=payload, authority=authority)
        return McpServerModel(**data["mcp_server"])

    async def update(
        self,
        server_id: str,
        description: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        source_mode: str | None = None,
        enabled: bool | None = None,
        authority: AuthorityContext | None = None,
    ) -> McpServerModel:
        """Partially update an MCP server record."""
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if command is not None:
            payload["command"] = command
        if args is not None:
            payload["args"] = args
        if env is not None:
            payload["env"] = env
        if url is not None:
            payload["url"] = url
        if headers is not None:
            payload["headers"] = headers
        if metadata is not None:
            payload["metadata"] = metadata
        if source_mode is not None:
            payload["source_mode"] = source_mode
        if enabled is not None:
            payload["enabled"] = enabled

        data = await self._client._request("PUT", f"/mcp-servers/{server_id}", json=payload, authority=authority)
        return McpServerModel(**data["mcp_server"])

    async def delete(self, server_id: str, authority: AuthorityContext | None = None) -> None:
        """Delete an MCP server record."""
        await self._client._request("DELETE", f"/mcp-servers/{server_id}", authority=authority)

    async def resolve(
        self,
        name: str,
        workspace_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> McpServerModel | None:
        """Resolve an MCP server by name using 4-tier precedence; returns None if not found."""
        payload: dict[str, Any] = {"name": name}
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id

        data = await self._client._request("POST", "/mcp-servers/resolve", json=payload, authority=authority)
        server = data.get("mcp_server")
        if server is not None:
            return McpServerModel(**server)
        return None

    # ------------------------------------------------------------------
    # .mcp.json helpers
    # ------------------------------------------------------------------

    async def push_json(
        self,
        json_path: Path,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
        user_id: str | None = None,
        authority: AuthorityContext | None = None,
    ) -> list[McpServerModel]:
        """Push a .mcp.json file to MemoryLayer, creating/updating one record per server entry."""
        json_path = Path(json_path)
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        mcp_servers = doc.get("mcpServers", {})

        results: list[McpServerModel] = []
        ws_id = self._ws(workspace_id)
        for name, entry in mcp_servers.items():
            transport = entry.get("transport")
            if transport is None:
                transport = "stdio" if entry.get("command") else "http"

            server = await self.create(
                name=name,
                transport=transport,
                command=entry.get("command"),
                args=entry.get("args"),
                env=entry.get("env"),
                url=entry.get("url"),
                headers=entry.get("headers"),
                source_mode=source_mode,
                workspace_id=ws_id,
                user_id=user_id,
                authority=authority,
            )
            results.append(server)
        return results

    async def pull_json(
        self,
        out_path: Path,
        workspace_id: str | None = None,
        user_id: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
        authority: AuthorityContext | None = None,
    ) -> Path:
        """Materialize MCP servers to a .mcp.json file at out_path."""
        servers = await self.list(
            transport=transport,
            enabled=enabled,
            user_id=user_id,
            workspace_id=workspace_id,
            authority=authority,
        )

        mcp_servers: dict[str, Any] = {}
        for server in servers:
            entry: dict[str, Any] = {"transport": server.transport}
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
            mcp_servers[server.name] = entry

        out_path = Path(out_path)
        out_path.write_text(
            json.dumps({"mcpServers": mcp_servers}, indent=2),
            encoding="utf-8",
        )
        return out_path


class SyncMcpServersAPI:
    """Synchronous MCP servers namespace — access via sync_client.mcp_servers.<method>."""

    def __init__(self, client: SyncMemoryLayerClient) -> None:
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    def list(
        self,
        transport: str | None = None,
        enabled: bool | None = None,
        name: str | None = None,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        workspace_id: str | None = None,
    ) -> list[McpServerModel]:
        """List MCP servers with optional filters."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if transport is not None:
            params["transport"] = transport
        if enabled is not None:
            params["enabled"] = str(enabled).lower()
        if name is not None:
            params["name"] = name
        if user_id is not None:
            params["user_id"] = user_id

        data = self._client._request("GET", "/mcp-servers", params=params)
        return [McpServerModel(**s) for s in data.get("mcp_servers", [])]

    def get(self, server_id: str) -> McpServerModel:
        """Get an MCP server by ID."""
        data = self._client._request("GET", f"/mcp-servers/{server_id}")
        return McpServerModel(**data["mcp_server"])

    def get_by_name(
        self,
        name: str,
        user_id: str | None = None,
        workspace_id: str | None = None,
    ) -> McpServerModel | None:
        """Get an MCP server by name; returns None if not found."""
        params: dict[str, Any] = {"name": name}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if user_id is not None:
            params["user_id"] = user_id

        try:
            data = self._client._request("GET", "/mcp-servers/by-name", params=params)
            return McpServerModel(**data["mcp_server"])
        except Exception:
            return None

    def create(
        self,
        name: str,
        transport: Literal["stdio", "http", "sse", "streamable-http"],
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_mode: str = "server",
        enabled: bool = True,
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> McpServerModel:
        """Create a new MCP server record."""
        payload: dict[str, Any] = {
            "name": name,
            "transport": transport,
            "source_mode": source_mode,
            "enabled": enabled,
        }
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if user_id is not None:
            payload["user_id"] = user_id
        if description is not None:
            payload["description"] = description
        if command is not None:
            payload["command"] = command
        if args is not None:
            payload["args"] = args
        if env is not None:
            payload["env"] = env
        if url is not None:
            payload["url"] = url
        if headers is not None:
            payload["headers"] = headers
        if metadata is not None:
            payload["metadata"] = metadata

        data = self._client._request("POST", "/mcp-servers", json=payload)
        return McpServerModel(**data["mcp_server"])

    def update(
        self,
        server_id: str,
        description: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        headers: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        source_mode: str | None = None,
        enabled: bool | None = None,
    ) -> McpServerModel:
        """Partially update an MCP server record."""
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if command is not None:
            payload["command"] = command
        if args is not None:
            payload["args"] = args
        if env is not None:
            payload["env"] = env
        if url is not None:
            payload["url"] = url
        if headers is not None:
            payload["headers"] = headers
        if metadata is not None:
            payload["metadata"] = metadata
        if source_mode is not None:
            payload["source_mode"] = source_mode
        if enabled is not None:
            payload["enabled"] = enabled

        data = self._client._request("PUT", f"/mcp-servers/{server_id}", json=payload)
        return McpServerModel(**data["mcp_server"])

    def delete(self, server_id: str) -> None:
        """Delete an MCP server record."""
        self._client._request("DELETE", f"/mcp-servers/{server_id}")

    def resolve(
        self,
        name: str,
        workspace_id: str | None = None,
    ) -> McpServerModel | None:
        """Resolve an MCP server by name using 4-tier precedence; returns None if not found."""
        payload: dict[str, Any] = {"name": name}
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id

        data = self._client._request("POST", "/mcp-servers/resolve", json=payload)
        server = data.get("mcp_server")
        if server is not None:
            return McpServerModel(**server)
        return None

    def push_json(
        self,
        json_path: Path,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
        user_id: str | None = None,
    ) -> list[McpServerModel]:
        """Push a .mcp.json file to MemoryLayer, creating/updating one record per server entry."""
        json_path = Path(json_path)
        doc = json.loads(json_path.read_text(encoding="utf-8"))
        mcp_servers = doc.get("mcpServers", {})

        results: list[McpServerModel] = []
        ws_id = self._ws(workspace_id)
        for name, entry in mcp_servers.items():
            transport = entry.get("transport")
            if transport is None:
                transport = "stdio" if entry.get("command") else "http"

            server = self.create(
                name=name,
                transport=transport,
                command=entry.get("command"),
                args=entry.get("args"),
                env=entry.get("env"),
                url=entry.get("url"),
                headers=entry.get("headers"),
                source_mode=source_mode,
                workspace_id=ws_id,
                user_id=user_id,
            )
            results.append(server)
        return results

    def pull_json(
        self,
        out_path: Path,
        workspace_id: str | None = None,
        user_id: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
    ) -> Path:
        """Materialize MCP servers to a .mcp.json file at out_path."""
        servers = self.list(
            transport=transport,
            enabled=enabled,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        mcp_servers: dict[str, Any] = {}
        for server in servers:
            entry: dict[str, Any] = {"transport": server.transport}
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
            mcp_servers[server.name] = entry

        out_path = Path(out_path)
        out_path.write_text(
            json.dumps({"mcpServers": mcp_servers}, indent=2),
            encoding="utf-8",
        )
        return out_path
