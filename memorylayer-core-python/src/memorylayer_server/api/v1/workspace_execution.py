"""Revisioned workspace-view authority and observer state APIs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables

from ...models.versioned_resource import VersionedResourceNotFoundError, VersionedResourcePreconditionFailedError
from ...models.workspace_execution import (
    WorkspaceView,
    WorkspaceViewCreateInput,
    WorkspaceViewObservation,
    WorkspaceViewObservationInput,
    WorkspaceViewObservationReplaceInput,
    WorkspaceViewReplaceInput,
)
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.versioned_resources import VersionedResourceService
from ...services.workspace import WorkspaceService
from ...services.workspace_execution import WorkspaceExecutionService
from .. import EXT_MULTI_API_ROUTERS
from ._versioned_resource import actor, raise_api_error, required_header
from .deps import get_auth_service, get_authz_service, get_versioned_resource_service, get_workspace_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/views", tags=["workspace-views"])


class WorkspaceViewResponse(BaseModel):
    view: WorkspaceView
    replayed: bool = False


class WorkspaceViewListResponse(BaseModel):
    views: list[WorkspaceView]
    next_page_token: str | None = None


class WorkspaceViewRevision(BaseModel):
    view: WorkspaceView
    action: str
    operation_id: str


class WorkspaceViewRevisionListResponse(BaseModel):
    revisions: list[WorkspaceViewRevision]
    next_page_token: str | None = None


class WorkspaceViewObservationResponse(BaseModel):
    observation: WorkspaceViewObservation
    replayed: bool = False


class WorkspaceViewObservationListResponse(BaseModel):
    observations: list[WorkspaceViewObservation]
    next_page_token: str | None = None


class WorkspaceViewObservationRevision(BaseModel):
    observation: WorkspaceViewObservation
    action: str
    operation_id: str


class WorkspaceViewObservationRevisionListResponse(BaseModel):
    revisions: list[WorkspaceViewObservationRevision]
    next_page_token: str | None = None


async def _authorize_workspace(
    http_request: Request,
    workspace_id: str,
    action: str,
    auth_service: AuthenticationService,
    authz_service: AuthorizationService,
    workspace_service: WorkspaceService,
):
    ctx = await auth_service.build_context(http_request, None)
    await authz_service.require_authorization(
        ctx,
        "workspaces",
        action,
        resource_id=workspace_id,
        workspace_id=workspace_id,
    )
    workspace = await workspace_service.get_workspace(workspace_id)
    if workspace is None or workspace.tenant_id != ctx.tenant_id:
        raise VersionedResourceNotFoundError("workspace not found")
    return ctx


@router.post(
    "",
    response_model=WorkspaceViewResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def create_workspace_view(
    http_request: Request,
    response: Response,
    workspace_id: str,
    request: WorkspaceViewCreateInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-None-Match")
        if expected != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        view, replayed = await WorkspaceExecutionService(resources).create_view(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            request=request,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = view.etag
        return WorkspaceViewResponse(view=view, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "create workspace view", __name__)


@router.get("", response_model=WorkspaceViewListResponse)
async def list_workspace_views(
    http_request: Request,
    workspace_id: str,
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewListResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        views, next_token = await WorkspaceExecutionService(resources).list_views(
            ctx.tenant_id,
            workspace_id,
            limit=limit,
            page_token=page_token,
            include_deleted=include_deleted,
        )
        return WorkspaceViewListResponse(views=views, next_page_token=next_token)
    except Exception as exc:
        raise_api_error(exc, "list workspace views", __name__)


@router.get("/{view_id}/revisions", response_model=WorkspaceViewRevisionListResponse)
async def list_workspace_view_revisions(
    http_request: Request,
    workspace_id: str,
    view_id: str,
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewRevisionListResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        revisions, next_token = await WorkspaceExecutionService(resources).list_view_revisions(
            ctx.tenant_id,
            workspace_id,
            view_id,
            limit=limit,
            page_token=page_token,
        )
        return WorkspaceViewRevisionListResponse(
            revisions=[
                WorkspaceViewRevision(view=view, action=action, operation_id=operation_id) for view, action, operation_id in revisions
            ],
            next_page_token=next_token,
        )
    except Exception as exc:
        raise_api_error(exc, "list workspace view revisions", __name__)


@router.get("/{view_id}", response_model=WorkspaceViewResponse)
async def get_workspace_view(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        view = await WorkspaceExecutionService(resources).get_view(
            ctx.tenant_id,
            workspace_id,
            view_id,
            include_deleted=include_deleted,
        )
        if view is None:
            raise VersionedResourceNotFoundError("workspace view not found")
        response.headers["ETag"] = view.etag
        return WorkspaceViewResponse(view=view)
    except Exception as exc:
        raise_api_error(exc, "get workspace view", __name__)


@router.put(
    "/{view_id}",
    response_model=WorkspaceViewResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def replace_workspace_view(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    request: WorkspaceViewReplaceInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        view, replayed = await WorkspaceExecutionService(resources).replace_view(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            request=request,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = view.etag
        return WorkspaceViewResponse(view=view, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "replace workspace view", __name__)


@router.delete(
    "/{view_id}",
    response_model=WorkspaceViewResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def delete_workspace_view(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        view, replayed = await WorkspaceExecutionService(resources).delete_view(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = view.etag
        return WorkspaceViewResponse(view=view, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "delete workspace view", __name__)


@router.post(
    "/{view_id}/restore",
    response_model=WorkspaceViewResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def restore_workspace_view(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        view, replayed = await WorkspaceExecutionService(resources).restore_view(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = view.etag
        return WorkspaceViewResponse(view=view, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "restore workspace view", __name__)


@router.get("/{view_id}/observations", response_model=WorkspaceViewObservationListResponse)
async def list_workspace_view_observations(
    http_request: Request,
    workspace_id: str,
    view_id: str,
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationListResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        observations, next_token = await WorkspaceExecutionService(resources).list_observations(
            ctx.tenant_id,
            workspace_id,
            view_id,
            limit=limit,
            page_token=page_token,
        )
        return WorkspaceViewObservationListResponse(observations=observations, next_page_token=next_token)
    except Exception as exc:
        raise_api_error(exc, "list workspace view observations", __name__)


@router.get(
    "/{view_id}/observations/{observer_id}/revisions",
    response_model=WorkspaceViewObservationRevisionListResponse,
)
async def list_workspace_view_observation_revisions(
    http_request: Request,
    workspace_id: str,
    view_id: str,
    observer_id: str,
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationRevisionListResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        revisions, next_token = await WorkspaceExecutionService(resources).list_observation_revisions(
            ctx.tenant_id,
            workspace_id,
            view_id,
            observer_id,
            limit=limit,
            page_token=page_token,
        )
        return WorkspaceViewObservationRevisionListResponse(
            revisions=[
                WorkspaceViewObservationRevision(observation=observation, action=action, operation_id=operation_id)
                for observation, action, operation_id in revisions
            ],
            next_page_token=next_token,
        )
    except Exception as exc:
        raise_api_error(exc, "list workspace view observation revisions", __name__)


@router.get("/{view_id}/observations/{observer_id}", response_model=WorkspaceViewObservationResponse)
async def get_workspace_view_observation(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    observer_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationResponse:
    try:
        ctx = await _authorize_workspace(http_request, workspace_id, "read", auth_service, authz_service, workspace_service)
        observation = await WorkspaceExecutionService(resources).get_observation(
            ctx.tenant_id,
            workspace_id,
            view_id,
            observer_id,
        )
        if observation is None:
            raise VersionedResourceNotFoundError("workspace view observation not found")
        response.headers["ETag"] = observation.etag
        return WorkspaceViewObservationResponse(observation=observation)
    except Exception as exc:
        raise_api_error(exc, "get workspace view observation", __name__)


@router.put(
    "/{view_id}/observations/{observer_id}",
    response_model=WorkspaceViewObservationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def put_workspace_view_observation(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    observer_id: str,
    request: WorkspaceViewObservationReplaceInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        if_none_match = http_request.headers.get("If-None-Match", "").strip()
        if_match = http_request.headers.get("If-Match", "").strip()
        if bool(if_none_match) == bool(if_match):
            raise HTTPException(
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                detail="exactly one of If-None-Match or If-Match is required",
            )
        create = bool(if_none_match)
        expected = if_none_match if create else if_match
        if create and expected != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        typed_request: WorkspaceViewObservationInput | WorkspaceViewObservationReplaceInput
        if create:
            typed_request = WorkspaceViewObservationInput(observer_id=observer_id, **request.model_dump())
        else:
            typed_request = request
        observation, replayed = await WorkspaceExecutionService(resources).put_observation(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            request=typed_request,
            observer_id=observer_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
            create=create,
        )
        response.headers["ETag"] = observation.etag
        if create and not replayed:
            response.status_code = status.HTTP_201_CREATED
        return WorkspaceViewObservationResponse(observation=observation, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "publish workspace view observation", __name__)


@router.delete(
    "/{view_id}/observations/{observer_id}",
    response_model=WorkspaceViewObservationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def delete_workspace_view_observation(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    observer_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        observation, replayed = await WorkspaceExecutionService(resources).delete_observation(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            observer_id=observer_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = observation.etag
        return WorkspaceViewObservationResponse(observation=observation, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "delete workspace view observation", __name__)


@router.post(
    "/{view_id}/observations/{observer_id}/restore",
    response_model=WorkspaceViewObservationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def restore_workspace_view_observation(
    http_request: Request,
    response: Response,
    workspace_id: str,
    view_id: str,
    observer_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    workspace_service: WorkspaceService = Depends(get_workspace_service),
    resources: VersionedResourceService = Depends(get_versioned_resource_service),
) -> WorkspaceViewObservationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await _authorize_workspace(http_request, workspace_id, "write", auth_service, authz_service, workspace_service)
        observation, replayed = await WorkspaceExecutionService(resources).restore_observation(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            view_id=view_id,
            observer_id=observer_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = observation.etag
        return WorkspaceViewObservationResponse(observation=observation, replayed=replayed)
    except Exception as exc:
        raise_api_error(exc, "restore workspace view observation", __name__)


class WorkspaceExecutionAPIPlugin(Plugin):
    """Register workspace-view authority routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
