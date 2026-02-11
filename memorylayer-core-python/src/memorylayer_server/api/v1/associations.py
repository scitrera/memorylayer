"""
Memory graph operations API endpoints.

Endpoints:
- POST /v1/memories/{memory_id}/associate - Link memories (source_id in URL)
- GET /v1/memories/{memory_id}/associations - Get associations
- POST /v1/memories/{memory_id}/traverse - Graph traversal from memory
"""
import logging

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Plugin, Variables, get_extension

from .. import EXT_MULTI_API_ROUTERS

from ...models.association import AssociateInput
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from ...services.association import AssociationService
from ...services.authentication import (
    AuthenticationService,
    AuthenticationError,
    EXT_AUTHENTICATION_SERVICE,
)
from ...services.authorization import AuthorizationService, EXT_AUTHORIZATION_SERVICE
from .schemas import (
    AssociationCreateRequest,
    MemoryTraverseRequest,
    AssociationResponse,
    AssociationListResponse,
    GraphQueryResult,
    ErrorResponse,
)

router = APIRouter(prefix='/v1', tags=["associations"])


# Dependencies for services
async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


async def get_association_service(v: Variables = Depends(get_variables_dep)) -> AssociationService:
    """Get association service instance from dependency injection."""
    from ...services.association import get_association_service as _get_association_service
    return _get_association_service(v)


@router.post(
    "/memories/{memory_id}/associate",
    response_model=AssociationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Source or target memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_association(
        http_request: Request,
        memory_id: str,
        request: AssociationCreateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        association_service: AssociationService = Depends(get_association_service),
        logger: logging.Logger = Depends(get_logger),
) -> AssociationResponse:
    """
    Create a typed relationship between two memories.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Source memory ID
        request: Association creation request
        auth_service: Authentication service
        association_service: Association service instance

    Returns:
        Created association

    Raises:
        HTTPException: If source/target not found or association fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "associations", "create", workspace_id=ctx.workspace_id
        )

        logger.info(
            "Creating association: %s -[%s]-> %s",
            memory_id,
            request.relationship,
            request.target_id
        )

        # Convert request to domain input
        associate_input = AssociateInput(
            source_id=memory_id,
            target_id=request.target_id,
            relationship=request.relationship,
            strength=request.strength,
            metadata=request.metadata,
        )

        # Create association
        association = await association_service.associate(
            workspace_id=ctx.workspace_id,
            input=associate_input,
        )

        logger.info("Created association: %s", association.id)
        return AssociationResponse(association=association)

    except ValueError as e:
        logger.warning("Invalid association request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        logger.error("Failed to create association: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create association"
        )


@router.get(
    "/memories/{memory_id}/associations",
    response_model=AssociationListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_associations(
        http_request: Request,
        memory_id: str,
        relationships: str | None = None,
        direction: str = "both",
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        association_service: AssociationService = Depends(get_association_service),
        logger: logging.Logger = Depends(get_logger),
) -> AssociationListResponse:
    """
    Get all associations for a memory.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        relationships: Comma-separated list of relationship types to filter by
        direction: Association direction (outgoing, incoming, both)
        auth_service: Authentication service
        association_service: Association service instance

    Returns:
        List of associations

    Raises:
        HTTPException: If memory not found or query fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "associations", "read", workspace_id=ctx.workspace_id
        )

        logger.debug(
            "Getting associations for memory: %s, direction: %s",
            memory_id,
            direction
        )

        # Parse relationship types if provided
        relationship_types = None
        if relationships:
            relationship_types = [
                rel.strip().upper()
                for rel in relationships.split(",")
                if rel.strip()
            ]

        # Get associations
        associations = await association_service.get_related(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
            relationships=relationship_types,
            direction=direction,
        )

        logger.debug("Found %d associations for memory: %s", len(associations), memory_id)
        return AssociationListResponse(
            associations=associations,
            total_count=len(associations)
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid association query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Failed to get associations for memory %s: %s",
            memory_id,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve associations"
        )


@router.post(
    "/memories/{memory_id}/traverse",
    response_model=GraphQueryResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Start memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def traverse_from_memory(
        http_request: Request,
        memory_id: str,
        request: MemoryTraverseRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        association_service: AssociationService = Depends(get_association_service),
        logger: logging.Logger = Depends(get_logger),
) -> GraphQueryResult:
    """
    Traverse memory graph starting from a specific memory.

    Example use cases:
    - Find causal chains: What led to this memory?
    - Find solutions: What addresses this problem?
    - Find related concepts: What's connected to this idea?

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Starting memory for traversal
        request: Traverse request with filters and options
        auth_service: Authentication service
        association_service: Association service instance

    Returns:
        Graph query result with paths and nodes

    Raises:
        HTTPException: If start memory not found or traversal fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "associations", "read", workspace_id=ctx.workspace_id
        )

        logger.info(
            "Traversing graph from memory: %s, max_depth: %d, direction: %s",
            memory_id,
            request.max_depth,
            request.direction
        )

        # Perform traversal via storage backend
        result = await association_service.storage.traverse_graph(
            workspace_id=ctx.workspace_id,
            start_id=memory_id,
            max_depth=request.max_depth,
            relationships=request.relationship_types or None,
            direction=request.direction,
        )

        logger.info(
            "Graph traversal found %d paths, %d unique nodes",
            result.total_paths,
            len(result.unique_nodes)
        )

        return result

    except ValueError as e:
        logger.warning("Invalid graph traversal request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            )
        logger.error("Failed to traverse graph from memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to traverse graph"
        )


class AssociationsAPIPlugin(Plugin):
    """Plugin to register association API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
