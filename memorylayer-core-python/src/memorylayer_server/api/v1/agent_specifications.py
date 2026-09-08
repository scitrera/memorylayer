"""Typed, workspace-scoped reusable agent-specification API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables

from ...models.agent_specification import AgentSpecification, AgentSpecificationCreateInput, AgentSpecificationReplaceInput
from ...models.versioned_resource import (
    VersionedResource,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.versioned_resources import VersionedResourceService
from .. import EXT_MULTI_API_ROUTERS
from ._versioned_resource import actor, raise_api_error, required_header
from .deps import get_auth_service, get_authz_service, get_versioned_resource_service
from .schemas import ErrorResponse

AGENT_SPECIFICATION_NAMESPACE = "agent.specification.v1"
router = APIRouter(prefix="/v1/agent-specifications", tags=["agent-specifications"])


class AgentSpecificationResponse(BaseModel):
    specification: AgentSpecification
    replayed: bool = False


class AgentSpecificationListResponse(BaseModel):
    specifications: list[AgentSpecification]
    next_page_token: str | None = None


class AgentSpecificationRevision(BaseModel):
    specification: AgentSpecification
    action: str
    operation_id: str


class AgentSpecificationRevisionListResponse(BaseModel):
    revisions: list[AgentSpecificationRevision]
    next_page_token: str | None = None


@router.post(
    "",
    response_model=AgentSpecificationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def create_agent_specification(
    http_request: Request,
    response: Response,
    request: AgentSpecificationCreateInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-None-Match")
        if expected != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "write", workspace_id=workspace_id)
        result = await service.create(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=AGENT_SPECIFICATION_NAMESPACE,
            resource_key=request.key,
            schema_version=request.schema_version,
            content=_content(request),
            metadata=request.metadata,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return AgentSpecificationResponse(specification=_to_specification(result.resource), replayed=result.replayed)
    except Exception as exc:
        raise_api_error(exc, "create agent specification", __name__)


@router.get("", response_model=AgentSpecificationListResponse)
async def list_agent_specifications(
    http_request: Request,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationListResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "read", workspace_id=workspace_id)
        resources, next_token = await service.list_page(
            ctx.tenant_id, workspace_id, AGENT_SPECIFICATION_NAMESPACE, limit=limit, page_token=page_token, include_deleted=include_deleted
        )
        return AgentSpecificationListResponse(specifications=[_to_specification(item) for item in resources], next_page_token=next_token)
    except Exception as exc:
        raise_api_error(exc, "list agent specifications", __name__)


@router.get("/{specification_id}/revisions", response_model=AgentSpecificationRevisionListResponse)
async def list_agent_specification_revisions(
    http_request: Request,
    specification_id: str,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationRevisionListResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "read", workspace_id=workspace_id)
        revisions, next_token = await service.list_revision_page(
            ctx.tenant_id, workspace_id, AGENT_SPECIFICATION_NAMESPACE, specification_id, limit=limit, page_token=page_token
        )
        if (
            not revisions
            and await service.get(ctx.tenant_id, workspace_id, AGENT_SPECIFICATION_NAMESPACE, specification_id, include_deleted=True)
            is None
        ):
            raise VersionedResourceNotFoundError("agent specification not found")
        return AgentSpecificationRevisionListResponse(revisions=[_to_revision(item) for item in revisions], next_page_token=next_token)
    except Exception as exc:
        raise_api_error(exc, "list agent specification revisions", __name__)


@router.get("/{specification_id}", response_model=AgentSpecificationResponse)
async def get_agent_specification(
    http_request: Request,
    response: Response,
    specification_id: str,
    workspace_id: str | None = Query(None),
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "read", workspace_id=workspace_id)
        resource = await service.get(
            ctx.tenant_id, workspace_id, AGENT_SPECIFICATION_NAMESPACE, specification_id, include_deleted=include_deleted
        )
        if resource is None:
            raise VersionedResourceNotFoundError("agent specification not found")
        response.headers["ETag"] = resource.etag
        return AgentSpecificationResponse(specification=_to_specification(resource))
    except Exception as exc:
        raise_api_error(exc, "get agent specification", __name__)


@router.put(
    "/{specification_id}",
    response_model=AgentSpecificationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def replace_agent_specification(
    http_request: Request,
    response: Response,
    specification_id: str,
    request: AgentSpecificationReplaceInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "write", workspace_id=workspace_id)
        result = await service.replace(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=AGENT_SPECIFICATION_NAMESPACE,
            resource_id=specification_id,
            schema_version=request.schema_version,
            content=_content(request),
            metadata=request.metadata,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return AgentSpecificationResponse(specification=_to_specification(result.resource), replayed=result.replayed)
    except Exception as exc:
        raise_api_error(exc, "replace agent specification", __name__)


@router.delete(
    "/{specification_id}",
    response_model=AgentSpecificationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def delete_agent_specification(
    http_request: Request,
    response: Response,
    specification_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "write", workspace_id=workspace_id)
        result = await service.delete(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=AGENT_SPECIFICATION_NAMESPACE,
            resource_id=specification_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return AgentSpecificationResponse(specification=_to_specification(result.resource), replayed=result.replayed)
    except Exception as exc:
        raise_api_error(exc, "delete agent specification", __name__)


@router.post(
    "/{specification_id}/restore",
    response_model=AgentSpecificationResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def restore_agent_specification(
    http_request: Request,
    response: Response,
    specification_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> AgentSpecificationResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "agent_specifications", "write", workspace_id=workspace_id)
        result = await service.restore(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=AGENT_SPECIFICATION_NAMESPACE,
            resource_id=specification_id,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return AgentSpecificationResponse(specification=_to_specification(result.resource), replayed=result.replayed)
    except Exception as exc:
        raise_api_error(exc, "restore agent specification", __name__)


def _content(request: AgentSpecificationCreateInput | AgentSpecificationReplaceInput) -> dict:
    return request.model_dump(exclude={"key", "schema_version", "metadata", "workspace_id"})


def _to_specification(resource: VersionedResource) -> AgentSpecification:
    return AgentSpecification(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        key=resource.resource_key,
        schema_version=resource.schema_version,
        metadata=resource.metadata,
        revision=resource.revision,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
        **resource.content,
    )


def _to_revision(revision: VersionedResourceRevision) -> AgentSpecificationRevision:
    return AgentSpecificationRevision(
        specification=_to_specification(revision.as_resource()), action=revision.action, operation_id=revision.operation_id
    )


class AgentSpecificationsAPIPlugin(Plugin):
    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
