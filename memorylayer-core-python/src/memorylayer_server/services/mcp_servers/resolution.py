"""McpServerResolutionService: 4-tier precedence-based MCP server lookup.

Implements LOCAL > PROJECT > USER > GLOBAL scope ordering with
source_mode tie-breaking (server > mirrored > filesystem).

Scope encoding:
  LOCAL   (0): user_id set + workspace_id == ctx.workspace_id  (per-project private)
  PROJECT (1): user_id None + workspace_id == ctx.workspace_id (shared .mcp.json)
  USER    (2): user_id set + workspace_id == "_global_user"     (cross-project private)
  GLOBAL  (3): user_id None + workspace_id == "_global"         (tenant/plugin)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models.mcp_server import McpServer
    from ..storage import StorageBackend

_GLOBAL_WORKSPACE_ID = "_global"
_GLOBAL_USER_WORKSPACE_ID = "_global_user"

_MODE_RANK: dict[str, int] = {
    "server": 0,
    "mirrored": 1,
    "filesystem": 2,
}


def _scope_rank(server: "McpServer", ctx_workspace_id: str, ctx_user_id: Optional[str]) -> int:
    """Return the 4-tier scope rank for an MCP server record given the request context."""
    if server.user_id and server.user_id == ctx_user_id:
        if server.workspace_id == ctx_workspace_id:
            return 0  # LOCAL: user-private + current workspace
        if server.workspace_id == _GLOBAL_USER_WORKSPACE_ID:
            return 2  # USER: user-private + cross-workspace
    if server.user_id is None:
        if server.workspace_id == ctx_workspace_id:
            return 1  # PROJECT: shared + current workspace
        if server.workspace_id == _GLOBAL_WORKSPACE_ID:
            return 3  # GLOBAL: tenant/plugin scope
    # Fallback: treat as lower priority than all named tiers
    return 99


def _mode_rank(server: "McpServer") -> int:
    return _MODE_RANK.get(server.source_mode, 99)


class RequestContext:
    """Lightweight context object carrying workspace/user identity for MCP server resolution."""

    __slots__ = ("workspace_id", "user_id", "tenant_id")

    def __init__(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        tenant_id: str = "_default",
    ) -> None:
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.tenant_id = tenant_id


class McpServerResolutionService:
    """Resolves MCP servers by name with deterministic 4-tier scope precedence.

    Precedence: LOCAL > PROJECT > USER > GLOBAL.
    Within scope: server > mirrored > filesystem, then most-recent updated_at.

    Enterprise can subclass and override ``visible_scopes_for`` to inject
    RBAC-filtered visibility without changing resolution logic.
    """

    def __init__(self, storage: "StorageBackend") -> None:
        self._storage = storage

    def visible_scopes_for(self, ctx: RequestContext) -> list[dict]:
        """Return the ordered list of scope filter dicts to search for a given context.

        Each dict contains ``workspace_id`` and optional ``user_id``.
        """
        scopes: list[dict] = []
        if ctx.user_id:
            # LOCAL: user-private within current workspace
            scopes.append({"workspace_id": ctx.workspace_id, "user_id": ctx.user_id})
        # PROJECT: shared within current workspace
        scopes.append({"workspace_id": ctx.workspace_id})
        if ctx.user_id:
            # USER: user-private cross-workspace
            scopes.append({"workspace_id": _GLOBAL_USER_WORKSPACE_ID, "user_id": ctx.user_id})
        # GLOBAL: tenant/plugin scope
        scopes.append({"workspace_id": _GLOBAL_WORKSPACE_ID})
        return scopes

    async def resolve(self, name: str, ctx: RequestContext) -> "Optional[McpServer]":
        """Return the precedence-winning MCP server for the given name + context."""
        scopes = self.visible_scopes_for(ctx)
        candidates = await self._storage.find_mcp_servers_by_name(name, scopes)
        if not candidates:
            return None
        return self._rank(candidates, ctx)[0]

    def apply_shadowing(
        self,
        servers: "list[McpServer]",
        ctx: RequestContext,
    ) -> "list[McpServer]":
        """Given a list of servers, return only the precedence winner per name.

        Used by GET /v1/mcp-servers when ``include_shadowed=false`` (default).
        """
        by_name: dict[str, list["McpServer"]] = {}
        for s in servers:
            by_name.setdefault(s.name, []).append(s)

        result = []
        for name_servers in by_name.values():
            result.append(self._rank(name_servers, ctx)[0])
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rank(
        self,
        candidates: "list[McpServer]",
        ctx: RequestContext,
    ) -> "list[McpServer]":
        """Sort candidates by (scope_rank, mode_rank, -updated_at) ascending."""
        return sorted(
            candidates,
            key=lambda s: (
                _scope_rank(s, ctx.workspace_id, ctx.user_id),
                _mode_rank(s),
                -s.updated_at.timestamp(),
            ),
        )
