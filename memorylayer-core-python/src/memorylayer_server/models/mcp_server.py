"""MCP server domain models for MemoryLayer OSS.

Defines the McpServer model plus input/update types and the McpJsonDocument
helper for parsing/rendering multi-server .mcp.json files.
"""

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Same name rules as skills: 1-64 chars, [a-z0-9-], no leading/trailing/consecutive hyphens
_MCP_SERVER_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


def validate_mcp_server_name(name: str) -> str:
    """Validate MCP server name against spec rules.

    Rules: 1-64 chars, lowercase letters/digits/hyphens only,
    no leading, trailing, or consecutive hyphens.
    """
    if not name:
        raise ValueError("MCP server name cannot be empty")
    if len(name) > 64:
        raise ValueError(f"MCP server name must be 64 chars or fewer, got {len(name)}")
    if "--" in name:
        raise ValueError("MCP server name cannot contain consecutive hyphens")
    if not _MCP_SERVER_NAME_RE.match(name):
        raise ValueError(
            "MCP server name must contain only lowercase letters, digits, and hyphens, and must not start or end with a hyphen"
        )
    return name


class McpServer(BaseModel):
    """MCP server record stored in MemoryLayer.

    One row per server. A .mcp.json file with N entries explodes to N rows on push
    and merges back to one file on materialize. Scope is encoded via workspace_id
    and user_id following the 4-tier Claude Code precedence model.
    """

    model_config = {"from_attributes": True}

    id: str = Field(..., description="MCP server ID (mcp_<12hex>)")
    tenant_id: str = Field("_default", description="Tenant scope")
    workspace_id: str = Field(
        ...,
        description="Workspace scope; _global for tenant/global, _global_user for cross-workspace user scope",
    )
    user_id: str | None = Field(None, description="User scope (set for user-private: Claude's 'local' or 'user' scopes)")

    name: str = Field(..., description="Server name (1-64 chars, [a-z0-9-])")
    description: str | None = Field(None, description="Optional description (≤1024 chars)")

    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(..., description="Transport protocol")

    # stdio fields
    command: str | None = Field(None, description="Executable command (stdio only)")
    args: list[str] = Field(default_factory=list, description="Command arguments (stdio only)")
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables (stdio only); may contain ${VAR} placeholders or secrets",
    )

    # http/sse fields
    url: str | None = Field(None, description="Server URL (http/sse/streamable-http only)")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers (http/sse only); may contain secrets — encrypted at rest in Enterprise",
    )

    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata (tags, vendor, etc.)")

    source_mode: Literal["server", "filesystem", "mirrored"] = Field("server", description="Canonical storage location")
    manifest_hash: str = Field("", description="SHA-256 of canonical JSON serialization (sorted keys, no whitespace)")

    enabled: bool = Field(True, description="Whether this server is active")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_mcp_server_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 1024:
            raise ValueError(f"MCP server description must be 1024 chars or fewer, got {len(v)}")
        return v

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "McpServer":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("command is required for stdio transport")
        else:
            if not self.url:
                raise ValueError(f"url is required for {self.transport} transport")
        return self


class McpServerCreateInput(BaseModel):
    """Request model for creating a new MCP server record."""

    name: str = Field(..., description="Server name (1-64 chars, [a-z0-9-])")
    description: str | None = Field(None, description="Optional description (≤1024 chars)")

    transport: Literal["stdio", "http", "sse", "streamable-http"] = Field(..., description="Transport protocol")

    command: str | None = Field(None, description="Executable command (stdio only)")
    args: list[str] = Field(default_factory=list, description="Command arguments (stdio only)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables (stdio only)")

    url: str | None = Field(None, description="Server URL (http/sse/streamable-http only)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers (http/sse only)")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    source_mode: Literal["server", "filesystem", "mirrored"] = Field("server")
    enabled: bool = Field(True)

    workspace_id: str | None = Field(None, description="Target workspace (overrides header)")
    user_id: str | None = Field(None, description="User scope override")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return validate_mcp_server_name(v)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 1024:
            raise ValueError(f"MCP server description must be 1024 chars or fewer, got {len(v)}")
        return v

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "McpServerCreateInput":
        if self.transport == "stdio":
            if not self.command:
                raise ValueError("command is required for stdio transport")
        else:
            if not self.url:
                raise ValueError(f"url is required for {self.transport} transport")
        return self


class McpServerUpdateInput(BaseModel):
    """Request model for updating an existing MCP server (all fields optional)."""

    description: str | None = Field(None, description="New description")
    command: str | None = Field(None, description="New command (stdio only)")
    args: list[str] | None = Field(None, description="New args (stdio only)")
    env: dict[str, str] | None = Field(None, description="New env vars (stdio only)")
    url: str | None = Field(None, description="New URL (http/sse only)")
    headers: dict[str, str] | None = Field(None, description="New headers (http/sse only)")
    metadata: dict[str, Any] | None = Field(None, description="Metadata to merge")
    source_mode: Literal["server", "filesystem", "mirrored"] | None = Field(None)
    manifest_hash: str | None = Field(None, description="Updated manifest hash")
    enabled: bool | None = Field(None, description="Enable/disable the server")

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str | None) -> str | None:
        if v is not None and len(v) > 1024:
            raise ValueError(f"MCP server description must be 1024 chars or fewer, got {len(v)}")
        return v


class McpServerEntry(BaseModel):
    """Per-server shape within a .mcp.json document (no id/tenant/workspace/user fields)."""

    transport: Literal["stdio", "http", "sse", "streamable-http"] | None = Field(
        None, description="Transport protocol (inferred from presence of command/url if omitted)"
    )

    # stdio
    command: str | None = Field(None)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # http/sse
    url: str | None = Field(None)
    headers: dict[str, str] = Field(default_factory=dict)
    type: str | None = Field(None, description="Claude Code 'type' field (e.g. 'sse', 'http')")

    @model_validator(mode="after")
    def infer_transport(self) -> "McpServerEntry":
        if self.transport is None:
            if self.command:
                object.__setattr__(self, "transport", "stdio")
            elif self.url:
                t = self.type or "http"
                if t in ("sse", "http", "streamable-http"):
                    object.__setattr__(self, "transport", t)
                else:
                    object.__setattr__(self, "transport", "http")
        return self


class McpJsonDocument(BaseModel):
    """Represents a .mcp.json file with N server entries.

    Claude Code format: {"mcpServers": {"server-name": {...}, ...}}
    """

    mcpServers: dict[str, McpServerEntry] = Field(default_factory=dict, description="Map of server name -> server config")  # noqa: N815

    def to_server_create_inputs(
        self,
        workspace_id: str | None = None,
        user_id: str | None = None,
        source_mode: Literal["server", "filesystem", "mirrored"] = "server",
    ) -> list[McpServerCreateInput]:
        """Explode this document into a list of McpServerCreateInput objects."""
        result = []
        for name, entry in self.mcpServers.items():
            transport = entry.transport or ("stdio" if entry.command else "http")
            result.append(
                McpServerCreateInput(
                    name=validate_mcp_server_name(name),
                    transport=transport,
                    command=entry.command,
                    args=entry.args,
                    env=entry.env,
                    url=entry.url,
                    headers=entry.headers,
                    source_mode=source_mode,
                    workspace_id=workspace_id,
                    user_id=user_id,
                )
            )
        return result
