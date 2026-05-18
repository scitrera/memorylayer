"""
Graph Analysis API endpoints.

Endpoints:
- GET  /v1/graph/snapshot      - Build graph and return snapshot metadata
- GET  /v1/graph/communities   - Detect communities in the workspace graph
- GET  /v1/graph/centrality    - Compute node centrality scores
- GET  /v1/graph/bridges       - Find cross-community bridge edges
- GET  /v1/graph/stats         - Get aggregate graph statistics
- POST /v1/graph/analyze       - Run full graph analysis
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from ...models.graph_analysis import (
    Bridge,
    CentralNode,
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.graph_analysis import EXT_GRAPH_ANALYSIS_SERVICE, GraphAnalysisService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/graph", tags=["graph-analysis"])

# ── Response schemas ─────────────────────────────────────────────────────────


class GraphSnapshotResponse(BaseModel):
    snapshot: GraphSnapshot


class CommunitiesResponse(BaseModel):
    communities: list[Community]
    total_count: int


class CentralityResponse(BaseModel):
    central_nodes: list[CentralNode]
    total_count: int


class BridgesResponse(BaseModel):
    bridges: list[Bridge]
    total_count: int


class GraphStatsResponse(BaseModel):
    stats: GraphStats


class GraphAnalysisRequest(BaseModel):
    workspace_id: Optional[str] = Field(None, description="Workspace ID (overrides auth context)")
    context_id: Optional[str] = Field(None, description="Context filter")
    include_rpg: bool = Field(False, description="Include RPG (code graph) nodes")


# ── Dependency ───────────────────────────────────────────────────────────────


def get_graph_service(v: Variables = Depends(get_variables_dep)) -> GraphAnalysisService:
    return get_extension(EXT_GRAPH_ANALYSIS_SERVICE, v)


# ── Routes ───────────────────────────────────────────────────────────────────


@router.get(
    "/snapshot",
    response_model=GraphSnapshotResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_graph_snapshot(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    context_id: Optional[str] = Query(None, description="Context filter"),
    include_rpg: bool = Query(False, description="Include RPG nodes"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> GraphSnapshotResponse:
    """Build and return graph snapshot metadata for the workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        snapshot = await graph_service.build_workspace_graph(
            workspace_id, context_id=context_id, include_rpg=include_rpg
        )
        return GraphSnapshotResponse(snapshot=snapshot)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to build graph snapshot for workspace: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build graph snapshot",
        )


@router.get(
    "/communities",
    response_model=CommunitiesResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_communities(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    context_id: Optional[str] = Query(None, description="Context filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> CommunitiesResponse:
    """Detect communities (clusters) in the workspace association graph."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        communities = await graph_service.detect_communities(workspace_id, context_id=context_id)
        return CommunitiesResponse(communities=communities, total_count=len(communities))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to detect communities: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to detect communities",
        )


@router.get(
    "/centrality",
    response_model=CentralityResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_centrality(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    context_id: Optional[str] = Query(None, description="Context filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> CentralityResponse:
    """Compute node centrality scores (degree + betweenness) for the workspace graph."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        central_nodes = await graph_service.compute_centrality(workspace_id, context_id=context_id)
        return CentralityResponse(central_nodes=central_nodes, total_count=len(central_nodes))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to compute centrality: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute centrality",
        )


@router.get(
    "/bridges",
    response_model=BridgesResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_bridges(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    context_id: Optional[str] = Query(None, description="Context filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> BridgesResponse:
    """Find edges that bridge different communities in the workspace graph."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        bridges = await graph_service.get_bridges(workspace_id, context_id=context_id)
        return BridgesResponse(bridges=bridges, total_count=len(bridges))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bridges: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get bridges",
        )


@router.get(
    "/stats",
    response_model=GraphStatsResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_graph_stats(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    context_id: Optional[str] = Query(None, description="Context filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> GraphStatsResponse:
    """Get aggregate graph statistics for the workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        stats = await graph_service.get_statistics(workspace_id, context_id=context_id)
        return GraphStatsResponse(stats=stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get graph stats: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get graph statistics",
        )


@router.post(
    "/analyze",
    response_model=GraphAnalysis,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_graph(
    http_request: Request,
    request: GraphAnalysisRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service: GraphAnalysisService = Depends(get_graph_service),
    logger: logging.Logger = Depends(get_logger),
) -> GraphAnalysis:
    """Run full graph analysis (snapshot + communities + centrality + bridges + stats)."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "graph", "read", workspace_id=workspace_id)

        analysis = await graph_service.analyze(
            workspace_id,
            context_id=request.context_id,
            include_rpg=request.include_rpg,
        )
        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run graph analysis: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run graph analysis",
        )


class GraphAnalysisAPIPlugin(Plugin):
    """Plugin to register graph analysis API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
