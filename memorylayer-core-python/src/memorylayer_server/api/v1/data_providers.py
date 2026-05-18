"""
Data Provider API endpoints.

Endpoints:
- POST   /v1/data-providers              - Create a data provider
- GET    /v1/data-providers              - List data providers
- GET    /v1/data-providers/{id}         - Get a data provider
- PUT    /v1/data-providers/{id}         - Update a data provider
- DELETE /v1/data-providers/{id}         - Delete a data provider
- POST   /v1/data-providers/{id}/sync    - Trigger sync
"""

import logging
from datetime import UTC, datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from ...models.data_provider import DataProvider
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.data_provider import EXT_DATA_PROVIDER_SERVICE, DataProviderService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/data-providers", tags=["data-providers"])

# ── Request / Response schemas ──────────────────────────────────────────────


class DataProviderCreateRequest(BaseModel):
    workspace_id: Optional[str] = Field(None, description="Workspace ID (overrides auth context)")
    name: str = Field(..., description="Provider name")
    provider_type: str = Field("local", description="Provider type")
    description: Optional[str] = None
    enabled: bool = True
    connection_args: dict[str, Any] = Field(default_factory=dict)
    schedule: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataProviderUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    connection_args: Optional[dict[str, Any]] = None
    schedule: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DataProviderResponse(BaseModel):
    provider: DataProvider


class DataProviderListResponse(BaseModel):
    providers: list[DataProvider]
    total_count: int


class DataProviderSyncResponse(BaseModel):
    provider_id: str
    workspace_id: str
    synced_documents: list


# ── Dependency ───────────────────────────────────────────────────────────────


def get_data_provider_service(v: Variables = Depends(get_variables_dep)) -> DataProviderService:
    return get_extension(EXT_DATA_PROVIDER_SERVICE, v)


# ── Routes ───────────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=DataProviderResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_data_provider(
    http_request: Request,
    request: DataProviderCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Create a new data provider."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=workspace_id)

        now = datetime.now(UTC)
        provider = DataProvider(
            id="",  # will be assigned in create_provider
            workspace_id=workspace_id,
            name=request.name,
            provider_type=request.provider_type,
            description=request.description,
            enabled=request.enabled,
            connection_args=request.connection_args,
            schedule=request.schedule,
            metadata=request.metadata,
            created_at=now,
            updated_at=now,
        )

        result = await dp_service.create_provider(workspace_id, provider)
        return DataProviderResponse(provider=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create data provider: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create data provider",
        )


@router.get(
    "",
    response_model=DataProviderListResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def list_data_providers(
    http_request: Request,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
) -> DataProviderListResponse:
    """List data providers for a workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "read", workspace_id=workspace_id)

        providers, total = await dp_service.list_providers(workspace_id, limit=limit, offset=offset)
        return DataProviderListResponse(providers=providers, total_count=total)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list data providers: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list data providers",
        )


@router.get(
    "/{provider_id}",
    response_model=DataProviderResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def get_data_provider(
    http_request: Request,
    provider_id: str,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Get a data provider by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "read", workspace_id=workspace_id)

        provider = await dp_service.get_provider(workspace_id, provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data provider {provider_id} not found",
            )
        return DataProviderResponse(provider=provider)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get data provider",
        )


@router.put(
    "/{provider_id}",
    response_model=DataProviderResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def update_data_provider(
    http_request: Request,
    provider_id: str,
    request: DataProviderUpdateRequest,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
) -> DataProviderResponse:
    """Update a data provider."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=workspace_id)

        updates = request.model_dump(exclude_none=True)
        updates["updated_at"] = datetime.now(UTC)

        provider = await dp_service.update_provider(workspace_id, provider_id, **updates)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data provider {provider_id} not found",
            )
        return DataProviderResponse(provider=provider)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update data provider",
        )


@router.delete(
    "/{provider_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def delete_data_provider(
    http_request: Request,
    provider_id: str,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
):
    """Delete a data provider."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=workspace_id)

        deleted = await dp_service.delete_provider(workspace_id, provider_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data provider {provider_id} not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete data provider",
        )


@router.post(
    "/{provider_id}/sync",
    response_model=DataProviderSyncResponse,
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
)
async def sync_data_provider(
    http_request: Request,
    provider_id: str,
    workspace_id: Optional[str] = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    dp_service: DataProviderService = Depends(get_data_provider_service),
    logger: logging.Logger = Depends(get_logger),
) -> DataProviderSyncResponse:
    """Trigger a sync for a data provider."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "data_providers", "write", workspace_id=workspace_id)

        # Verify provider exists
        provider = await dp_service.get_provider(workspace_id, provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data provider {provider_id} not found",
            )

        synced = await dp_service.sync(workspace_id, provider_id)
        return DataProviderSyncResponse(
            provider_id=provider_id,
            workspace_id=workspace_id,
            synced_documents=synced,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to sync data provider %s: %s", provider_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to sync data provider",
        )


class DataProviderAPIPlugin(Plugin):
    """Plugin to register data provider API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
