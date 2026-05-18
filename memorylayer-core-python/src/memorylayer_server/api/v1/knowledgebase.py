"""
Knowledgebase API endpoints.

Endpoints:
- POST   /v1/knowledgebase/generate                  - Trigger KB generation
- GET    /v1/knowledgebase                            - Get KB metadata + index article
- GET    /v1/knowledgebase/articles                   - List articles
- GET    /v1/knowledgebase/articles/{article_id}      - Get single article
- GET    /v1/knowledgebase/export                     - Download Obsidian vault zip
- GET    /v1/knowledgebase/graph                      - Get cached graph analysis
- GET    /v1/knowledgebase/graph/communities/{cid}    - Community detail
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.graph_analysis import Community, GraphAnalysis
from ...services._constants import EXT_GRAPH_ANALYSIS_SERVICE, EXT_KNOWLEDGEBASE_SERVICE
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.knowledgebase.base import Article, KBGenerateOptions, Knowledgebase
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service
from .schemas import ErrorResponse


# ------------------------------------------------------------------ #
# Request / response schemas
# ------------------------------------------------------------------ #

class KBGenerateRequest(BaseModel):
    """Request body for KB generation."""

    workspace_id: str | None = Field(None, description="Target workspace (overrides session/default)")
    context_id: str | None = Field(None, description="Restrict analysis to this context")
    include_rpg: bool = Field(False, description="Include RPG nodes in graph analysis")
    max_communities: int = Field(50, ge=1, le=500, description="Maximum community articles to generate")
    max_god_nodes: int = Field(20, ge=0, le=200, description="Maximum entity articles to generate")
    regenerate: bool = Field(False, description="Force regeneration even if KB exists")


class ArticleListResponse(BaseModel):
    articles: list[Article]
    total: int


class GraphAnalysisResponse(BaseModel):
    analysis: GraphAnalysis | None = None
    cached: bool = False


# ------------------------------------------------------------------ #
# Router
# ------------------------------------------------------------------ #

router = APIRouter(prefix="/v1/knowledgebase", tags=["knowledgebase"])


# ------------------------------------------------------------------ #
# Dependencies
# ------------------------------------------------------------------ #

def get_knowledgebase_service(v: Variables = Depends(get_variables_dep)):
    """FastAPI dependency for knowledgebase service."""
    return get_extension(EXT_KNOWLEDGEBASE_SERVICE, v)


def get_graph_analysis_service(v: Variables = Depends(get_variables_dep)):
    """FastAPI dependency for graph analysis service."""
    return get_extension(EXT_GRAPH_ANALYSIS_SERVICE, v)


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #

@router.post(
    "/generate",
    response_model=Knowledgebase,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Generation failed"},
    },
)
async def generate_knowledgebase(
    http_request: Request,
    request: KBGenerateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> Knowledgebase:
    """
    Trigger knowledgebase generation for a workspace.

    Runs graph analysis, community labeling, entity deep-dives, and
    produces Obsidian-compatible Markdown articles stored in the backend.

    Returns:
        Knowledgebase metadata including article count and graph stats.
    """
    try:
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(ctx, "knowledgebase", "write", workspace_id=ctx.workspace_id)

        logger.info("Generating knowledgebase for workspace=%s", ctx.workspace_id)

        options = KBGenerateOptions(
            include_rpg=request.include_rpg,
            max_communities=request.max_communities,
            max_god_nodes=request.max_god_nodes,
            regenerate=request.regenerate,
        )

        kb = await kb_service.generate(
            workspace_id=ctx.workspace_id,
            context_id=request.context_id or ctx.context_id,
            options=options,
        )

        logger.info("Knowledgebase generated for workspace=%s: %d articles", ctx.workspace_id, kb.article_count)
        return kb

    except Exception as e:
        logger.error("Failed to generate knowledgebase: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate knowledgebase",
        )


@router.get(
    "",
    response_model=Knowledgebase,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "No knowledgebase found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_knowledgebase(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> Knowledgebase:
    """
    Get knowledgebase metadata for the current workspace.

    Returns article count, community count, generation time, and graph stats.
    Does NOT trigger regeneration.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        kb = await kb_service.get_knowledgebase(workspace_id=ctx.workspace_id)
        if kb is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledgebase found for this workspace. Run POST /v1/knowledgebase/generate first.",
            )

        return kb

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get knowledgebase: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve knowledgebase",
        )


@router.get(
    "/articles",
    response_model=ArticleListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_articles(
    http_request: Request,
    article_type: str | None = Query(None, description="Filter by type: index, community, entity"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> ArticleListResponse:
    """
    List knowledgebase articles for the current workspace.

    Optionally filter by article type (index, community, entity).
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        articles = await kb_service.list_articles(
            workspace_id=ctx.workspace_id,
            article_type=article_type,
            limit=limit,
            offset=offset,
        )

        return ArticleListResponse(articles=articles, total=len(articles))

    except Exception as e:
        logger.error("Failed to list KB articles: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list knowledgebase articles",
        )


@router.get(
    "/articles/{article_id:path}",
    response_model=Article,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Article not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_article(
    http_request: Request,
    article_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> Article:
    """
    Get a single knowledgebase article by ID.

    Article IDs: ``index``, ``community-{n}``, ``entity-{slug}``.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        article = await kb_service.get_article(workspace_id=ctx.workspace_id, article_id=article_id)
        if article is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Article not found: {article_id}",
            )

        return article

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get KB article %s: %s", article_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve article",
        )


@router.get(
    "/export",
    responses={
        200: {"content": {"application/zip": {}}, "description": "Obsidian vault zip"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "No articles to export"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def export_vault(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> StreamingResponse:
    """
    Download the knowledgebase as an Obsidian-compatible vault zip.

    The zip contains:
    - ``index.md``
    - ``communities/community-{n}.md``
    - ``entities/{slug}.md``
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        logger.info("Exporting KB vault for workspace=%s", ctx.workspace_id)

        zip_bytes = await kb_service.export_vault(workspace_id=ctx.workspace_id)

        if not zip_bytes:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No knowledgebase articles found. Run POST /v1/knowledgebase/generate first.",
            )

        filename = f"memorylayer-vault-{ctx.workspace_id}.zip"
        return StreamingResponse(
            iter([zip_bytes]),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export KB vault: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export knowledgebase vault",
        )


@router.get(
    "/graph",
    response_model=GraphAnalysisResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_graph_analysis(
    http_request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service=Depends(get_graph_analysis_service),
    logger: logging.Logger = Depends(get_logger),
) -> GraphAnalysisResponse:
    """
    Get the graph analysis for the current workspace.

    Runs a fresh analysis (snapshot + communities + centrality + bridges + stats).
    For a cached result use GET /v1/knowledgebase which reads the last generated KB.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        logger.debug("Running graph analysis for workspace=%s", ctx.workspace_id)
        analysis: GraphAnalysis = await graph_service.analyze(workspace_id=ctx.workspace_id)

        return GraphAnalysisResponse(analysis=analysis, cached=False)

    except Exception as e:
        logger.error("Failed to get graph analysis: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve graph analysis",
        )


@router.get(
    "/graph/communities/{community_id}",
    response_model=Community,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Community not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_community(
    http_request: Request,
    community_id: int,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    graph_service=Depends(get_graph_analysis_service),
    kb_service=Depends(get_knowledgebase_service),
    logger: logging.Logger = Depends(get_logger),
) -> Community:
    """
    Get detailed information about a single community by ID.

    Combines live community detection data with any cached label from
    the last KB generation run.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "knowledgebase", "read", workspace_id=ctx.workspace_id)

        communities = await graph_service.detect_communities(workspace_id=ctx.workspace_id)
        matched = next((c for c in communities if c.id == community_id), None)

        if matched is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Community {community_id} not found",
            )

        # Enrich with cached label if available
        try:
            article = await kb_service.get_article(
                workspace_id=ctx.workspace_id,
                article_id=f"community-{community_id}",
            )
            if article and not matched.label:
                matched.label = article.title
        except Exception:
            pass

        return matched

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get community %d: %s", community_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve community",
        )


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #

class KnowledgebaseAPIPlugin(Plugin):
    """Plugin to register knowledgebase API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
