"""McpServerService: CRUD business logic for MCP server records."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from ...models.memory import MemoryType, RememberInput
from ...models.mcp_server import McpServer, McpServerCreateInput, McpServerUpdateInput
from ...utils import generate_id
from ..storage import StorageBackend
from .encryption import decrypt_secrets, encrypt_secrets

if TYPE_CHECKING:
    from ..memory import MemoryService

logger = logging.getLogger(__name__)


def _parse_capabilities(server: McpServer) -> str:
    """Extract a short capability hint string from server config for memory indexing."""
    parts = []
    if server.transport == "stdio" and server.args:
        parts.append(server.args[0])
    elif server.url:
        parts.append(server.url)
    return ", ".join(parts) if parts else server.transport


def _compute_manifest_hash(server: McpServer) -> str:
    """SHA-256 of canonical JSON (sorted keys, no whitespace) over runtime config fields."""
    config: dict[str, Any] = {
        "name": server.name,
        "transport": server.transport,
    }
    if server.description is not None:
        config["description"] = server.description
    if server.transport == "stdio":
        if server.command:
            config["command"] = server.command
        if server.args:
            config["args"] = server.args
        if server.env:
            config["env"] = server.env
    else:
        if server.url:
            config["url"] = server.url
        if server.headers:
            config["headers"] = server.headers

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class McpServerService:
    """Service for managing MCP server records in MemoryLayer.

    Wraps StorageBackend with ID generation, hash computation, and an
    optional memory_indexer callable hook for the memory mirror (wired
    in Phase 2 by McpServerResolutionService).
    """

    def __init__(
        self,
        storage: StorageBackend,
        memory_service: "Optional[MemoryService]" = None,
        memory_indexer: Optional[Callable[[McpServer], Any]] = None,
    ) -> None:
        self._storage = storage
        self._memory_service = memory_service
        self._memory_indexer = memory_indexer

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def create_mcp_server(
        self,
        input: McpServerCreateInput,
        workspace_id: str,
        tenant_id: str = "_default",
        user_id: Optional[str] = None,
    ) -> McpServer:
        """Create a new MCP server record, computing manifest hash."""
        now = datetime.now(UTC)
        resolved_user_id = user_id if user_id is not None else input.user_id
        resolved_workspace_id = input.workspace_id or workspace_id

        server = McpServer(
            id=generate_id("mcp"),
            tenant_id=tenant_id,
            workspace_id=resolved_workspace_id,
            user_id=resolved_user_id,
            name=input.name,
            description=input.description,
            transport=input.transport,
            command=input.command,
            args=input.args,
            env=input.env,
            url=input.url,
            headers=input.headers,
            metadata=input.metadata,
            source_mode=input.source_mode,
            manifest_hash="",
            enabled=input.enabled,
            created_at=now,
            updated_at=now,
        )
        server = server.model_copy(update={"manifest_hash": _compute_manifest_hash(server)})
        # Encrypt secrets before persisting
        server = server.model_copy(update={
            "env": encrypt_secrets(server.env),
            "headers": encrypt_secrets(server.headers),
        })
        result = await self._storage.create_mcp_server(server)
        # Decrypt secrets before returning to caller
        result = result.model_copy(update={
            "env": decrypt_secrets(result.env),
            "headers": decrypt_secrets(result.headers),
        })
        await self._maybe_index(result)
        await self._upsert_mirror_memory(result)
        return result

    async def get_mcp_server(
        self,
        workspace_id: str,
        server_id: str,
    ) -> Optional[McpServer]:
        """Get an MCP server by ID."""
        return await self._storage.get_mcp_server(workspace_id, server_id)

    async def get_mcp_server_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: Optional[str] = None,
    ) -> Optional[McpServer]:
        """Get an MCP server by name within a workspace, optionally filtering by user scope."""
        return await self._storage.get_mcp_server_by_name(workspace_id, name, user_id)

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        transport: Optional[str] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[McpServer]:
        """List MCP servers with optional filters."""
        return await self._storage.list_mcp_servers(
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
            transport=transport,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )

    async def update_mcp_server(
        self,
        workspace_id: str,
        server_id: str,
        input: McpServerUpdateInput,
    ) -> Optional[McpServer]:
        """Apply partial updates to an MCP server, recomputing manifest_hash if config changed."""
        updates: dict[str, Any] = {
            k: v for k, v in input.model_dump(exclude_none=True).items()
        }
        updates["updated_at"] = datetime.now(UTC)
        # Encrypt secrets before persisting
        if "env" in updates:
            updates["env"] = encrypt_secrets(updates["env"])
        if "headers" in updates:
            updates["headers"] = encrypt_secrets(updates["headers"])

        result = await self._storage.update_mcp_server(workspace_id, server_id, updates)
        if result is None:
            return None

        new_hash = _compute_manifest_hash(result)
        if new_hash != result.manifest_hash:
            result = await self._storage.update_mcp_server(
                workspace_id, server_id, {"manifest_hash": new_hash}
            )

        # Decrypt secrets before returning to caller
        result = result.model_copy(update={
            "env": decrypt_secrets(result.env),
            "headers": decrypt_secrets(result.headers),
        })
        await self._maybe_index(result)
        await self._upsert_mirror_memory(result)
        return result

    async def delete_mcp_server(self, workspace_id: str, server_id: str) -> bool:
        """Delete an MCP server record and its memory mirror."""
        result = await self._storage.delete_mcp_server(workspace_id, server_id)
        if result and self._memory_service:
            await self._delete_mirror_memory(workspace_id, server_id)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_index(self, server: Optional[McpServer]) -> None:
        """Call the memory_indexer hook after create/update if configured."""
        if not server or not self._memory_indexer:
            return
        try:
            result = self._memory_indexer(server)
            if hasattr(result, "__await__"):
                await result
        except Exception:
            pass  # indexer errors must not break server operations

    async def _upsert_mirror_memory(self, server: Optional[McpServer]) -> None:
        """Create or replace the procedural memory mirror for an MCP server."""
        if not server or not self._memory_service:
            return
        try:
            await self._delete_mirror_memory(server.workspace_id, server.id)

            capabilities = _parse_capabilities(server)
            content = (
                f"{server.name}\n"
                f"{server.description or ''}\n"
                f"Transport: {server.transport}\n"
                f"Capabilities: {capabilities}"
            )
            tags = [
                "mcp_server",
                f"mcp_server:{server.name}",
                f"mcp_transport:{server.transport}",
            ]
            await self._memory_service.remember(
                workspace_id=server.workspace_id,
                input=RememberInput(
                    content=content,
                    type=MemoryType.PROCEDURAL,
                    subtype="mcp_server",
                    tags=tags,
                    metadata={
                        "mcp_server_id": server.id,
                        "mcp_server_name": server.name,
                    },
                    user_id=server.user_id,
                ),
                user_id=server.user_id,
                inline=True,
            )
        except Exception:
            logger.debug("Failed to upsert memory mirror for mcp_server %s", server.id, exc_info=True)

    async def _delete_mirror_memory(self, workspace_id: str, server_id: str) -> None:
        """Delete any existing memory mirror for the given server_id."""
        try:
            existing = await self._storage.search_memories_by_filter(
                workspace_id=workspace_id,
                subtypes=["mcp_server"],
                metadata_filter={"mcp_server_id": server_id},
                limit=10,
            )
            for mem in existing:
                await self._storage.delete_memory(workspace_id, mem.id, hard=True)
        except Exception:
            logger.debug("Failed to delete mirror memory for mcp_server %s", server_id, exc_info=True)
