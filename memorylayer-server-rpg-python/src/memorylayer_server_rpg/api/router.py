# SPDX-License-Identifier: Apache-2.0
"""
RPG (Repository Planning Graph) API endpoints.

Endpoints:
- POST /v1/rpg/sync - Bulk upsert RPG graph (full or incremental)
- GET /v1/rpg/subgraph - Query subgraph from a root path/node
- GET /v1/rpg/search - Search RPG nodes by text
- GET /v1/rpg/status - Get RPG encoding status for workspace
"""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service
from memorylayer_server.api.v1.schemas import ErrorResponse
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services.authentication import AuthenticationError, AuthenticationService
from memorylayer_server.services.authorization import AuthorizationService
from memorylayer_server.services.storage.base import EXT_STORAGE_BACKEND, StorageBackend
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server_rpg.services import RpgService

from .schemas import (
    RpgConflictResponse,
    RpgDeleteEdgesRequest,
    RpgDeleteEdgesResponse,
    RpgDeleteNodesRequest,
    RpgDeleteNodesResponse,
    RpgEnrichRequest,
    RpgEnrichResponse,
    RpgListNodesResponse,
    RpgMaintenanceResponse,
    RpgMergedSubgraphResponse,
    RpgOverlayDeleteResponse,
    RpgOverlayListResponse,
    RpgSearchResponse,
    RpgStatusResponse,
    RpgSubgraphResponse,
    RpgSymbolConflictResponse,
    RpgSyncRequest,
    RpgSyncResponse,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _sanitize_description(text: str | None) -> str | None:
    """Strip HTML tags from description text to prevent XSS."""
    if not text:
        return text
    return _HTML_TAG_RE.sub("", text)


router = APIRouter(prefix="/v1/rpg", tags=["rpg"])


def get_rpg_service(v: Variables = Depends(get_variables_dep)) -> RpgService:
    """FastAPI dependency for RPG service.

    Uses the storage backend extension to construct the service. The storage
    backend itself is typically a singleton managed by the plugin system, so
    this creates lightweight service wrappers on each request.
    """
    storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
    return RpgService(storage)


def get_conflict_service(
    rpg_service: RpgService = Depends(get_rpg_service),
    v: Variables = Depends(get_variables_dep),
):
    """FastAPI dependency for RPG conflict service."""
    from memorylayer_server_rpg.services.conflicts import RpgConflictService

    storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
    return RpgConflictService(storage=storage, rpg_service=rpg_service)


@router.post(
    "/sync",
    response_model=RpgSyncResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def sync_rpg(
    http_request: Request,
    request: RpgSyncRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> RpgSyncResponse:
    """Sync an RPG graph snapshot into MemoryLayer.

    Accepts a full or incremental RPG graph and upserts nodes/edges.
    Nodes are stored as Memory entities, edges as Associations.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        logger.info(
            "RPG sync: workspace=%s, nodes=%d, edges=%d, full_sync=%s",
            ctx.workspace_id,
            len(request.nodes),
            len(request.edges),
            request.full_sync,
        )

        # Sanitize node descriptions before storage to prevent XSS
        sanitized_nodes = []
        for n in request.nodes:
            node_dict = n.model_dump()
            node_dict["description"] = _sanitize_description(node_dict.get("description"))
            sanitized_nodes.append(node_dict)

        result = await rpg_service.sync(
            workspace_id=ctx.workspace_id,
            nodes=sanitized_nodes,
            edges=[e.model_dump() for e in request.edges],
            full_sync=request.full_sync,
            source_commit=request.source_commit,
            context_id=request.context_id,
            task_id=request.task_id,
        )

        sync_response = RpgSyncResponse(
            nodes_created=result["nodes_created"],
            nodes_updated=result["nodes_updated"],
            nodes_deleted=result["nodes_deleted"],
            edges_created=result["edges_created"],
            edges_updated=result["edges_updated"],
            edges_deleted=result["edges_deleted"],
            sync_time_ms=result["sync_time_ms"],
            source_commit=result.get("source_commit"),
        )

        # Schedule enrichment for canonical full syncs
        if request.full_sync and (request.context_id is None or request.context_id == "rpg"):
            try:
                from memorylayer_server.services.tasks import EXT_TASK_SERVICE

                task_service = get_extension(EXT_TASK_SERVICE, v)
                await task_service.schedule_task(
                    "rpg_enrichment",
                    {
                        "workspace_id": ctx.workspace_id,
                        "context_id": request.context_id or "rpg",
                        "source": "post_sync",
                    },
                    priority=7,
                )
                logger.info("Scheduled RPG enrichment for workspace %s", ctx.workspace_id)
            except Exception as sched_exc:
                logger.warning(
                    "Failed to schedule RPG enrichment for workspace %s: %s",
                    ctx.workspace_id,
                    sched_exc,
                )

        return sync_response

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG sync failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG sync failed")


@router.get(
    "/subgraph",
    response_model=RpgSubgraphResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_subgraph(
    http_request: Request,
    path: str | None = Query(None, description="File path to find root node"),
    node_id: str | None = Query(None, description="Direct RPG node ID"),
    depth: int = Query(3, ge=1, le=10, description="Maximum traversal depth"),
    node_types: str | None = Query(None, description="Comma-separated node types to filter"),
    relationship_types: str | None = Query(None, description="Comma-separated edge types to filter"),
    context_id: str = Query("rpg", description="Context partition (default 'rpg' for canonical graph)"),
    max_nodes: int = Query(2000, ge=1, le=10000, description="Maximum number of nodes to return"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgSubgraphResponse:
    """Extract an RPG subgraph rooted at a path or node ID.

    Returns nodes and edges reachable within the specified depth.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        if not path and not node_id:
            raise ValueError("Either 'path' or 'node_id' query parameter is required")

        parsed_node_types = [t.strip() for t in node_types.split(",")] if node_types else None
        parsed_rel_types = [t.strip() for t in relationship_types.split(",")] if relationship_types else None

        result = await rpg_service.get_subgraph(
            workspace_id=ctx.workspace_id,
            root_path=path,
            root_node_id=node_id,
            depth=depth,
            node_types=parsed_node_types,
            relationship_types=parsed_rel_types,
            context_id=context_id,
            max_nodes=max_nodes,
        )

        return RpgSubgraphResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG subgraph query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG subgraph query failed")


@router.get(
    "/search",
    response_model=RpgSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def search_rpg_nodes(
    http_request: Request,
    query: str = Query(..., min_length=1, description="Search text"),
    node_types: str | None = Query(None, description="Comma-separated node types to filter"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    context_id: str = Query("rpg", description="Context partition (default 'rpg' for canonical graph)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgSearchResponse:
    """Search RPG nodes by name, description, or path."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        parsed_node_types = [t.strip() for t in node_types.split(",")] if node_types else None

        result = await rpg_service.search_nodes(
            workspace_id=ctx.workspace_id,
            query=query,
            node_types=parsed_node_types,
            limit=limit,
            context_id=context_id,
        )

        return RpgSearchResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG search failed")


@router.get(
    "/status",
    response_model=RpgStatusResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_rpg_status(
    http_request: Request,
    context_id: str | None = Query(None, description="Filter to specific context (None = all contexts)"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgStatusResponse:
    """Get RPG encoding status for the current workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        result = await rpg_service.get_status(workspace_id=ctx.workspace_id, context_id=context_id)

        return RpgStatusResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("RPG status query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG status query failed")


@router.get(
    "/overlays",
    response_model=RpgOverlayListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_rpg_overlays(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgOverlayListResponse:
    """List all active RPG overlay contexts (rpg-task-*, rpg-intent-*)."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        overlays = await rpg_service.list_overlays(workspace_id=ctx.workspace_id)

        return RpgOverlayListResponse(overlays=overlays, total=len(overlays))

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("RPG list overlays failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG list overlays failed")


@router.delete(
    "/overlays/{context_id}",
    response_model=RpgOverlayDeleteResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_rpg_overlay(
    http_request: Request,
    context_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgOverlayDeleteResponse:
    """Delete all RPG nodes in a specific overlay context."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        if not context_id.startswith("rpg-"):
            raise ValueError("Can only delete overlay contexts (rpg-task-*, rpg-intent-*). Cannot delete base 'rpg' context.")

        result = await rpg_service.delete_overlay(workspace_id=ctx.workspace_id, context_id=context_id)

        return RpgOverlayDeleteResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG delete overlay failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG delete overlay failed")


@router.delete(
    "/nodes",
    response_model=RpgDeleteNodesResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_rpg_nodes(
    http_request: Request,
    request: RpgDeleteNodesRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgDeleteNodesResponse:
    """Delete specific RPG nodes by node ID.

    Accepts a list of RPG node IDs and a context_id, and hard-deletes the
    matching Memory entities. Only nodes in overlay contexts (rpg-*) or the
    canonical 'rpg' context may be deleted.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        context_id = request.context_id
        if context_id != "rpg" and not context_id.startswith("rpg-"):
            raise ValueError(f"context_id must be 'rpg' or start with 'rpg-'. Got: {context_id}")

        logger.info(
            "RPG delete nodes: workspace=%s, context=%s, node_count=%d",
            ctx.workspace_id,
            context_id,
            len(request.node_ids),
        )

        result = await rpg_service.delete_nodes(
            workspace_id=ctx.workspace_id,
            node_ids=request.node_ids,
            context_id=context_id,
        )

        return RpgDeleteNodesResponse(deleted=result["deleted"], not_found=result["not_found"])

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG delete nodes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG delete nodes failed")


@router.delete(
    "/edges",
    response_model=RpgDeleteEdgesResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_rpg_edges_bulk(
    http_request: Request,
    request: RpgDeleteEdgesRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgDeleteEdgesResponse:
    """Bulk-delete RPG edges (associations) by their IDs.

    Accepts a list of association UUIDs and deletes them in a single request,
    replacing the previous pattern of one HTTP call per edge.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        if not request.association_ids:
            return RpgDeleteEdgesResponse(deleted=0, not_found=0)

        logger.info(
            "RPG bulk delete edges: workspace=%s, edge_count=%d",
            ctx.workspace_id,
            len(request.association_ids),
        )

        result = await rpg_service.delete_edges_bulk(
            workspace_id=ctx.workspace_id,
            association_ids=request.association_ids,
        )

        return RpgDeleteEdgesResponse(deleted=result["deleted"], not_found=result["not_found"])

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG bulk delete edges failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG bulk delete edges failed")


@router.get(
    "/nodes",
    response_model=RpgListNodesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_rpg_nodes(
    http_request: Request,
    node_type: str | None = Query(None, description="Filter by RPG node type (e.g. rpg_file, rpg_class)"),
    context_id: str = Query("rpg", description="Context partition to query (default 'rpg')"),
    limit: int = Query(5000, ge=1, le=10000, description="Maximum number of nodes to return"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgListNodesResponse:
    """List RPG nodes, optionally filtered by node type.

    Provides a reliable way to enumerate all nodes of a given type (e.g. all rpg_file
    nodes) without relying on text search heuristics.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        result = await rpg_service.list_nodes(
            workspace_id=ctx.workspace_id,
            node_type=node_type,
            context_id=context_id,
            limit=limit,
        )

        return RpgListNodesResponse(
            nodes=result["nodes"],
            total_count=result["total_count"],
        )

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("RPG list nodes failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG list nodes failed")


@router.get(
    "/merged-subgraph",
    response_model=RpgMergedSubgraphResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_merged_subgraph(
    http_request: Request,
    base_context: str = Query("rpg", description="Base context (default 'rpg')"),
    overlay_context: str = Query("", description="Overlay context to merge on top of base"),
    path: str | None = Query(None, description="Root file path"),
    depth: int = Query(3, ge=1, le=10, description="Maximum traversal depth"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    rpg_service: RpgService = Depends(get_rpg_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgMergedSubgraphResponse:
    """Get a merged RPG subgraph combining canonical base with an overlay context."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        result = await rpg_service.get_merged_subgraph(
            workspace_id=ctx.workspace_id,
            base_context=base_context,
            overlay_context=overlay_context,
            root_path=path,
            depth=depth,
        )

        return RpgMergedSubgraphResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("RPG merged subgraph query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG merged subgraph query failed")


@router.get(
    "/conflicts",
    response_model=RpgConflictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_rpg_conflicts(
    http_request: Request,
    task_id: str = Query(..., min_length=1, description="Task ID to check for conflicts"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    conflict_service=Depends(get_conflict_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgConflictResponse:
    """Detect file-level conflicts between a task's intent and other active intents."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        result = await conflict_service.detect_file_conflicts(
            workspace_id=ctx.workspace_id,
            task_id=task_id,
        )

        return RpgConflictResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG conflict detection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG conflict detection failed")


@router.get(
    "/conflicts/symbols",
    response_model=RpgSymbolConflictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_rpg_symbol_conflicts(
    http_request: Request,
    task_id_a: str = Query(..., min_length=1, description="First task ID to compare"),
    task_id_b: str = Query(..., min_length=1, description="Second task ID to compare"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    conflict_service=Depends(get_conflict_service),
    logger: logging.Logger = Depends(get_logger),
) -> RpgSymbolConflictResponse:
    """Detect symbol-level conflicts between two task overlays.

    Compares the rpg-task-* overlays for both tasks and returns symbols
    modified by both, with severity (high = exact match, medium = same file).
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "read", workspace_id=ctx.workspace_id)

        result = await conflict_service.detect_symbol_conflicts(
            workspace_id=ctx.workspace_id,
            task_id_a=task_id_a,
            task_id_b=task_id_b,
        )

        return RpgSymbolConflictResponse(**result)

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG symbol conflict detection failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG symbol conflict detection failed")


_VALID_MAINTENANCE_OPERATIONS = frozenset(
    {
        "validate",
        "cleanup",
        "statistics",
        "cleanup_intents",
        "recompute_counts",
    }
)


@router.post(
    "/maintenance/{operation}",
    response_model=RpgMaintenanceResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid operation"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def trigger_maintenance(
    http_request: Request,
    operation: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> RpgMaintenanceResponse:
    """Trigger a maintenance operation on the RPG graph for the current workspace.

    Operations:
    - **validate**: Check graph health (dangling edges, orphan nodes).
    - **cleanup**: Remove RPG nodes that have no edges (orphans).
    - **statistics**: Compute node/edge counts by type.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        if operation not in _VALID_MAINTENANCE_OPERATIONS:
            raise ValueError(
                "Invalid operation '{}'. Must be one of: {}".format(operation, ", ".join(sorted(_VALID_MAINTENANCE_OPERATIONS)))
            )

        from memorylayer_server_rpg.services.maintenance import RpgMaintenanceService

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        maintenance = RpgMaintenanceService(storage)

        logger.info("RPG maintenance operation '%s' for workspace %s", operation, ctx.workspace_id)

        if operation == "validate":
            result = await maintenance.validate_graph(ctx.workspace_id)
        elif operation == "cleanup":
            result = await maintenance.cleanup_stale_nodes(ctx.workspace_id)
        elif operation == "cleanup_intents":
            result = await maintenance.cleanup_stale_intents(ctx.workspace_id)
        elif operation == "recompute_counts":
            rpg_service = RpgService(storage)
            result = await rpg_service.recompute_counts(ctx.workspace_id)
        else:  # statistics
            result = await maintenance.compute_statistics(ctx.workspace_id)

        return RpgMaintenanceResponse(
            operation=operation,
            workspace_id=ctx.workspace_id,
            result=result,
        )

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG maintenance operation '%s' failed: %s", operation, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG maintenance failed")


@router.post(
    "/enrich",
    response_model=RpgEnrichResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def trigger_rpg_enrichment(
    http_request: Request,
    request: RpgEnrichRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> RpgEnrichResponse:
    """Manually trigger LLM enrichment for an RPG graph.

    Enrichment adds semantic structure on top of the structural graph:
    - **features**: Identifies logical feature components.
    - **descriptions**: Generates one-line descriptions for classes/interfaces.
    - **data_flows**: Detects data flow relationships between files.

    Set ``async`` to ``false`` to run synchronously and receive stats in the response.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "memories", "write", workspace_id=ctx.workspace_id)

        context_id = request.context_id or "rpg"

        logger.info(
            "RPG enrichment requested: workspace=%s, context=%s, phases=%s, async=%s",
            ctx.workspace_id,
            context_id,
            request.phases,
            request.async_mode,
        )

        if request.async_mode:
            # Schedule via task service
            from memorylayer_server.services.tasks import EXT_TASK_SERVICE

            task_service = get_extension(EXT_TASK_SERVICE, v)
            task_id = await task_service.schedule_task(
                "rpg_enrichment",
                {
                    "workspace_id": ctx.workspace_id,
                    "context_id": context_id,
                    "phases": request.phases,
                    "source": "api",
                },
                priority=5,
            )
            return RpgEnrichResponse(
                workspace_id=ctx.workspace_id,
                context_id=context_id,
                async_mode=True,
                task_id=str(task_id) if task_id is not None else None,
                stats=None,
            )
        else:
            # Run synchronously
            from memorylayer_server.services.llm import EXT_LLM_SERVICE

            from memorylayer_server_rpg.services.enrichment import RpgEnrichmentService

            storage = get_extension(EXT_STORAGE_BACKEND, v)
            llm_service = get_extension(EXT_LLM_SERVICE, v)
            enrichment = RpgEnrichmentService(storage=storage, llm_service=llm_service)

            stats = await enrichment.enrich(
                workspace_id=ctx.workspace_id,
                context_id=context_id,
                phases=request.phases,
            )
            return RpgEnrichResponse(
                workspace_id=ctx.workspace_id,
                context_id=context_id,
                async_mode=False,
                task_id=None,
                stats=stats,
            )

    except HTTPException:
        raise
    except AuthenticationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("RPG enrichment trigger failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="RPG enrichment failed")


class RpgAPIPlugin(Plugin):
    """Plugin to register RPG API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND,)
