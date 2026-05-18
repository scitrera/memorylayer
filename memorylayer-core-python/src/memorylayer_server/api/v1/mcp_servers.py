"""
MCP Servers API endpoints.

Endpoints:
- POST   /v1/mcp-servers              - Create one
- GET    /v1/mcp-servers              - List (masked by default)
- GET    /v1/mcp-servers/{id}         - Get one (masked)
- PUT    /v1/mcp-servers/{id}         - Update
- DELETE /v1/mcp-servers/{id}         - Delete
- POST   /v1/mcp-servers/import       - Import a .mcp.json document (N servers)
- GET    /v1/mcp-servers/export       - Export visible servers as .mcp.json shape
- POST   /v1/mcp-servers/resolve      - Resolve server by name (precedence) or query (vector search)
- POST   /v1/mcp-servers/{id}/sync    - Reconcile mirrored server via hash comparison
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.lifecycle.fastapi import get_logger

from ...models.mcp_server import McpServer, McpServerCreateInput, McpServerUpdateInput
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.mcp_servers import McpServerService
from ...services.mcp_servers.resolution import McpServerResolutionService, RequestContext
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service, get_mcp_servers_resolution_service, get_mcp_servers_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/mcp-servers", tags=["mcp-servers"])

# ── Secret masking ────────────────────────────────────────────────────────────

_MASK = "***"


def _mask_value(v: str) -> str:
    """Mask a secret value, preserving ${VAR} interpolation placeholders verbatim."""
    if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
        return v
    return _MASK


def _mask_secrets(server: McpServer, reveal: bool) -> McpServer:
    """Replace env/headers values with '***' unless reveal=True or value is a ${VAR} placeholder."""
    if reveal:
        return server
    masked_env = {k: _mask_value(v) for k, v in server.env.items()} if server.env else {}
    masked_headers = {k: _mask_value(v) for k, v in server.headers.items()} if server.headers else {}
    return server.model_copy(update={"env": masked_env, "headers": masked_headers})


# ── Request / Response schemas ────────────────────────────────────────────────


class McpServerResponse(BaseModel):
    mcp_server: McpServer


class McpServerListResponse(BaseModel):
    mcp_servers: list[McpServer]
    total_count: int


class McpServerImportRequest(BaseModel):
    mcpServers: dict[str, Any] = Field(..., description="Map of server name -> server config (standard .mcp.json shape)")  # noqa: N815
    workspace_id: str | None = Field(None, description="Target workspace (overrides header)")
    user_id: str | None = Field(None, description="User scope override")
    source_mode: str = Field("server", description="server | filesystem | mirrored")


class McpServerImportResponse(BaseModel):
    imported: int
    updated: int
    skipped: int
    errors: list[str]


class McpServerResolveRequest(BaseModel):
    name: str | None = Field(None, description="Exact server name — returns precedence winner")
    query: str | None = Field(None, description="Intent query — runs vector recall against server memories")
    transport: str | None = Field(None, description="Filter by transport")
    workspace_id: str | None = Field(
        None, description="Workspace to operate against; defaults to the authenticated context's workspace."
    )


class McpServerResolveResponse(BaseModel):
    mcp_server: McpServer | None = None
    candidates: list[McpServer] = Field(default_factory=list)


class McpServerSyncRequest(BaseModel):
    manifest_hash: str = Field("", description="SHA-256 of local config")
    workspace_id: str | None = None


class McpServerSyncResponse(BaseModel):
    action: str = Field(description="push | pull | conflict | in_sync")
    reason: str
    server_manifest_hash: str


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=McpServerResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_mcp_server(
    http_request: Request,
    request: McpServerCreateInput,
    reveal_secrets: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerResponse:
    """Create a new MCP server record."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "write", workspace_id=workspace_id)

        server = await mcp_service.create_mcp_server(
            input=request,
            workspace_id=workspace_id,
            tenant_id=getattr(ctx, "tenant_id", "_default"),
            user_id=request.user_id or getattr(ctx, "user_id", None),
        )
        return McpServerResponse(mcp_server=_mask_secrets(server, reveal_secrets))

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to create MCP server: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create MCP server")


@router.get(
    "",
    response_model=McpServerListResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def list_mcp_servers(
    http_request: Request,
    workspace_id: str | None = Query(None),
    name: str | None = Query(None),
    transport: str | None = Query(None),
    enabled: bool | None = Query(None),
    include_shadowed: bool = Query(False, description="Return all servers including shadowed duplicates"),
    reveal_secrets: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    resolution_service: McpServerResolutionService = Depends(get_mcp_servers_resolution_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerListResponse:
    """List MCP servers for a workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "read", workspace_id=workspace_id)

        servers = await mcp_service.list_mcp_servers(
            workspace_id=workspace_id,
            name=name,
            transport=transport,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )

        if not include_shadowed:
            res_ctx = RequestContext(
                workspace_id=workspace_id,
                user_id=getattr(ctx, "user_id", None),
                tenant_id=getattr(ctx, "tenant_id", "_default"),
            )
            servers = resolution_service.apply_shadowing(servers, res_ctx)

        masked = [_mask_secrets(s, reveal_secrets) for s in servers]
        return McpServerListResponse(mcp_servers=masked, total_count=len(masked))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list MCP servers: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list MCP servers")


@router.post(
    "/resolve",
    response_model=McpServerResolveResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def resolve_mcp_server(
    http_request: Request,
    request: McpServerResolveRequest,
    reveal_secrets: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    resolution_service: McpServerResolutionService = Depends(get_mcp_servers_resolution_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerResolveResponse:
    """Resolve a server by name (precedence winner) or query (vector recall against memory mirror)."""
    try:
        # Pass ``request`` so build_context picks up workspace_id from the
        # body — mirrors the skills.resolve handler.  Without this the
        # body's workspace_id is dropped and resolve_workspace falls back
        # to ``DEFAULT_WORKSPACE_ID="_default"``, raising 403 under any
        # OBO grant scoped to a real workspace.
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "read", workspace_id=workspace_id)

        res_ctx = RequestContext(
            workspace_id=workspace_id,
            user_id=getattr(ctx, "user_id", None),
            tenant_id=getattr(ctx, "tenant_id", "_default"),
        )

        if not request.name and not request.query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'name' or 'query' must be provided",
            )

        if request.name:
            winner = await resolution_service.resolve(request.name, res_ctx)
            if not winner:
                return McpServerResolveResponse(mcp_server=None, candidates=[])
            return McpServerResolveResponse(
                mcp_server=_mask_secrets(winner, reveal_secrets),
                candidates=[],
            )

        # query-based: recall memories with subtype=mcp_server, then resolve by server ID
        # We fetch the memory service directly from the extension registry rather than
        # via DI because this endpoint accepts both name and query modes; memory is only
        # required for the query path and is an optional deployment dependency.
        from ...services.memory import EXT_MEMORY_SERVICE, MemoryService

        memory_service: MemoryService | None = None
        memory_service_error: Exception | None = None
        try:
            from scitrera_app_framework import get_extension

            memory_service = get_extension(EXT_MEMORY_SERVICE)
        except Exception as e:
            memory_service_error = e

        if not memory_service:
            detail = "Vector search requires memory service (not available in this deployment)"
            if memory_service_error:
                logger.error("Memory service unavailable for MCP server query resolve: %s", memory_service_error)
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
            raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=detail)

        from ...models.memory import MemoryType, RecallInput

        recall_result = await memory_service.recall(
            workspace_id=workspace_id,
            input=RecallInput(
                query=request.query,
                types=[MemoryType.PROCEDURAL],
                subtypes=["mcp_server"],
                limit=10,
            ),
            user_id=getattr(ctx, "user_id", None),
        )

        candidates: list = []
        seen_ids: set[str] = set()
        for mem in recall_result.memories:
            server_id = mem.metadata.get("mcp_server_id")
            if not server_id or server_id in seen_ids:
                continue
            seen_ids.add(server_id)
            server = await mcp_service.get_mcp_server(workspace_id, server_id)
            if server:
                candidates.append(_mask_secrets(server, reveal_secrets))

        # Apply transport filter if requested
        if request.transport:
            candidates = [s for s in candidates if s.transport == request.transport]

        return McpServerResolveResponse(
            mcp_server=candidates[0] if candidates else None,
            candidates=candidates,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to resolve MCP server: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to resolve MCP server")


@router.post(
    "/import",
    response_model=McpServerImportResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def import_mcp_servers(
    http_request: Request,
    request: McpServerImportRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerImportResponse:
    """Import a .mcp.json document — explodes N entries into N records."""
    from ...models.mcp_server import McpJsonDocument

    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "write", workspace_id=workspace_id)

        doc = McpJsonDocument(mcpServers=request.mcpServers)
        inputs = doc.to_server_create_inputs(
            workspace_id=workspace_id,
            user_id=request.user_id,
            source_mode=request.source_mode,
        )

        imported = updated = skipped = 0
        errors: list[str] = []

        for inp in inputs:
            try:
                existing = await mcp_service.get_mcp_server_by_name(workspace_id, inp.name)
                if existing:
                    from ...models.mcp_server import McpServerUpdateInput

                    update = McpServerUpdateInput(
                        description=inp.description,
                        command=inp.command,
                        args=inp.args,
                        env=inp.env,
                        url=inp.url,
                        headers=inp.headers,
                        source_mode=inp.source_mode,
                    )
                    await mcp_service.update_mcp_server(workspace_id, existing.id, update)
                    updated += 1
                else:
                    await mcp_service.create_mcp_server(
                        input=inp,
                        workspace_id=workspace_id,
                        tenant_id=getattr(ctx, "tenant_id", "_default"),
                        user_id=inp.user_id or getattr(ctx, "user_id", None),
                    )
                    imported += 1
            except Exception as e:
                errors.append(f"{inp.name}: {e}")

        return McpServerImportResponse(imported=imported, updated=updated, skipped=skipped, errors=errors)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to import MCP servers: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to import MCP servers")


@router.get(
    "/export",
    responses={
        200: {"content": {"application/json": {}}},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def export_mcp_servers(
    http_request: Request,
    workspace_id: str | None = Query(None),
    reveal_secrets: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> dict:
    """Export visible servers as a .mcp.json-shaped JSON object."""
    from ...services.mcp_servers.parser import render_mcp_json

    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "read", workspace_id=workspace_id)

        servers = await mcp_service.list_mcp_servers(workspace_id=workspace_id, enabled=True)
        masked = [_mask_secrets(s, reveal_secrets) for s in servers]
        return render_mcp_json(masked)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export MCP servers: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to export MCP servers")


@router.get(
    "/{server_id}",
    response_model=McpServerResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_mcp_server(
    http_request: Request,
    server_id: str,
    workspace_id: str | None = Query(None),
    reveal_secrets: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerResponse:
    """Get an MCP server by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "read", workspace_id=workspace_id)

        server = await mcp_service.get_mcp_server(workspace_id, server_id)
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP server {server_id} not found")
        return McpServerResponse(mcp_server=_mask_secrets(server, reveal_secrets))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get MCP server %s: %s", server_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get MCP server")


@router.put(
    "/{server_id}",
    response_model=McpServerResponse,
    responses={
        404: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def update_mcp_server(
    http_request: Request,
    server_id: str,
    request: McpServerUpdateInput,
    workspace_id: str | None = Query(None),
    reveal_secrets: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerResponse:
    """Update an MCP server record."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "write", workspace_id=workspace_id)

        server = await mcp_service.update_mcp_server(workspace_id, server_id, request)
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP server {server_id} not found")
        return McpServerResponse(mcp_server=_mask_secrets(server, reveal_secrets))

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to update MCP server %s: %s", server_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update MCP server")


@router.delete(
    "/{server_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def delete_mcp_server(
    http_request: Request,
    server_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """Delete an MCP server record."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "write", workspace_id=workspace_id)

        deleted = await mcp_service.delete_mcp_server(workspace_id, server_id)
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP server {server_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete MCP server %s: %s", server_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete MCP server")


@router.post(
    "/{server_id}/sync",
    response_model=McpServerSyncResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def sync_mcp_server(
    http_request: Request,
    server_id: str,
    request: McpServerSyncRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    mcp_service: McpServerService = Depends(get_mcp_servers_service),
    logger: logging.Logger = Depends(get_logger),
) -> McpServerSyncResponse:
    """Reconcile a mirrored server — compare local manifest_hash vs stored."""
    from ...services.mcp_servers.sync import compare_hashes

    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "mcp_servers", "read", workspace_id=workspace_id)

        server = await mcp_service.get_mcp_server(workspace_id, server_id)
        if not server:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"MCP server {server_id} not found")

        action, reason = compare_hashes(request.manifest_hash, server.manifest_hash)
        return McpServerSyncResponse(
            action=action,
            reason=reason,
            server_manifest_hash=server.manifest_hash,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to sync MCP server %s: %s", server_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to sync MCP server")


# ── Plugin ────────────────────────────────────────────────────────────────────


class McpServersAPIPlugin(Plugin):
    """Plugin to register MCP servers API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
