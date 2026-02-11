"""
Workspace management API endpoints.

Endpoints:
- POST /v1/workspaces - Create workspace
- GET /v1/workspaces/{workspace_id} - Get workspace
- PUT /v1/workspaces/{workspace_id} - Update workspace
"""
import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Plugin, Variables, get_extension

from .. import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from .schemas import (
    WorkspaceCreateRequest,
    WorkspaceUpdateRequest,
    WorkspaceResponse,
    ErrorResponse,
)
from ...services.workspace import get_workspace_service as _get_workspace_service, WorkspaceService
from ...services.ontology import get_ontology_service as _get_ontology_service
from ...services.authentication import (
    AuthenticationService,
    EXT_AUTHENTICATION_SERVICE,
)
from ...services.authorization import (
    AuthorizationService,
    EXT_AUTHORIZATION_SERVICE,
)

router = APIRouter(prefix="/v1/workspaces", tags=["workspaces"])


async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


def get_workspace_service(v: Variables = Depends(get_variables_dep)) -> WorkspaceService:
    """FastAPI dependency wrapper for workspace service."""
    return _get_workspace_service(v)


@router.post(
    "",
    response_model=WorkspaceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_workspace(
        http_request: Request,
        request: WorkspaceCreateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Create a new workspace.

    Workspaces provide tenant-level memory isolation.

    Args:
        http_request: FastAPI request (for headers)
        request: Workspace creation request
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Created workspace

    Raises:
        HTTPException: If workspace creation fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "workspaces", "create")

        # Generate workspace ID
        workspace_id = f"ws_{uuid4().hex[:16]}"

        logger.info(
            "Creating workspace: %s for tenant: %s, name: %s",
            workspace_id,
            ctx.tenant_id,
            request.name
        )

        # Create workspace
        from ...models.workspace import Workspace
        workspace = Workspace(
            id=workspace_id,
            tenant_id=ctx.tenant_id,
            name=request.name,
            settings=request.settings,
        )

        # Store workspace via workspace service
        workspace = await workspace_service.create_workspace(workspace)

        logger.info("Created workspace: %s", workspace_id)
        return WorkspaceResponse(workspace=workspace)

    except ValueError as e:
        logger.warning("Invalid workspace creation request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create workspace: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workspace"
        )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_workspace(
        http_request: Request,
        workspace_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Retrieve a workspace by ID.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Workspace object

    Raises:
        HTTPException: If workspace not found or access denied
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "read",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.debug("Getting workspace: %s", workspace_id)

        # Get workspace via workspace service
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        return WorkspaceResponse(workspace=workspace)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace"
        )


@router.put(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def update_workspace(
        http_request: Request,
        workspace_id: str,
        request: WorkspaceUpdateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkspaceResponse:
    """
    Update an existing workspace.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        request: Workspace update request
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Updated workspace

    Raises:
        HTTPException: If workspace not found or update fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "write",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.info("Updating workspace: %s", workspace_id)

        # Get existing workspace
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        # Update fields
        if request.name is not None:
            workspace = workspace.model_copy(update={"name": request.name})
        if request.settings is not None:
            workspace = workspace.model_copy(update={"settings": request.settings})
        workspace = await workspace_service.update_workspace(workspace)
        return WorkspaceResponse(workspace=workspace)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid workspace update request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update workspace %s: %s", workspace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update workspace"
        )


@router.get(
    "/{workspace_id}/schema",
    response_model=dict,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Workspace not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_workspace_schema(
        http_request: Request,
        workspace_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Get workspace schema including relationship types and memory subtypes.

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Workspace identifier
        auth_service: Authentication service
        authz_service: Authorization service
        workspace_service: Workspace service instance

    Returns:
        Schema with relationship types, memory subtypes, and customization capability

    Raises:
        HTTPException: If workspace not found or schema retrieval fails
    """
    try:
        # Build auth context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "workspaces", "read",
            resource_id=workspace_id, workspace_id=workspace_id
        )

        logger.debug("Getting schema for workspace: %s", workspace_id)

        # Verify workspace exists
        workspace = await workspace_service.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace not found: {workspace_id}"
            )

        # Get ontology service
        ontology_service = _get_ontology_service()

        # Get relationship types from ontology
        relationship_types = ontology_service.list_relationship_types(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id
        )

        # Get memory subtypes from model
        from ...models.memory import MemorySubtype
        memory_subtypes = [subtype.value for subtype in MemorySubtype]

        return {
            "relationship_types": relationship_types,
            "memory_subtypes": memory_subtypes,
            "can_customize": False,  # OSS: No custom ontologies
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get schema for workspace %s: %s",
            workspace_id,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workspace schema"
        )


class WorkspacesAPIPlugin(Plugin):
    """Plugin to register workspaces API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
