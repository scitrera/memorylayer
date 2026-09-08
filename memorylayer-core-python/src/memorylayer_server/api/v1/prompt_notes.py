"""Typed, workspace-scoped prompt-note resource API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables

from ...models.prompt_note import PromptNote, PromptNoteCreateInput, PromptNoteReplaceInput
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
from ._versioned_resource import actor as _actor
from ._versioned_resource import raise_api_error as _shared_raise_api_error
from ._versioned_resource import required_header as _required_header
from .deps import get_auth_service, get_authz_service, get_versioned_resource_service
from .schemas import ErrorResponse

PROMPT_NOTE_NAMESPACE = "agent.prompt-note.v1"
router = APIRouter(prefix="/v1/prompt-notes", tags=["prompt-notes"])


class PromptNoteResponse(BaseModel):
    note: PromptNote
    replayed: bool = False


class PromptNoteListResponse(BaseModel):
    notes: list[PromptNote]
    next_page_token: str | None = None


class PromptNoteRevision(BaseModel):
    note: PromptNote
    action: str
    operation_id: str


class PromptNoteRevisionListResponse(BaseModel):
    revisions: list[PromptNoteRevision]
    next_page_token: str | None = None


@router.post(
    "",
    response_model=PromptNoteResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def create_prompt_note(
    http_request: Request,
    response: Response,
    request: PromptNoteCreateInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteResponse:
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-None-Match")
        if expected != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "write", workspace_id=workspace_id)
        result = await service.create(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=PROMPT_NOTE_NAMESPACE,
            resource_key=request.key,
            schema_version=request.schema_version,
            content={"title": request.title, "content": request.content, "enabled": request.enabled},
            metadata=request.metadata,
            actor=_actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return PromptNoteResponse(note=_to_note(result.resource), replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "create prompt note")


@router.get("", response_model=PromptNoteListResponse)
async def list_prompt_notes(
    http_request: Request,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteListResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "read", workspace_id=workspace_id)
        resources, next_token = await service.list_page(
            ctx.tenant_id,
            workspace_id,
            PROMPT_NOTE_NAMESPACE,
            limit=limit,
            page_token=page_token,
            include_deleted=include_deleted,
        )
        return PromptNoteListResponse(notes=[_to_note(item) for item in resources], next_page_token=next_token)
    except Exception as exc:
        _raise_api_error(exc, "list prompt notes")


@router.get("/{note_id}/revisions", response_model=PromptNoteRevisionListResponse)
async def list_prompt_note_revisions(
    http_request: Request,
    note_id: str,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteRevisionListResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "read", workspace_id=workspace_id)
        revisions, next_token = await service.list_revision_page(
            ctx.tenant_id,
            workspace_id,
            PROMPT_NOTE_NAMESPACE,
            note_id,
            limit=limit,
            page_token=page_token,
        )
        if (
            not revisions
            and await service.get(
                ctx.tenant_id,
                workspace_id,
                PROMPT_NOTE_NAMESPACE,
                note_id,
                include_deleted=True,
            )
            is None
        ):
            raise VersionedResourceNotFoundError("prompt note not found")
        return PromptNoteRevisionListResponse(
            revisions=[_to_revision(item) for item in revisions],
            next_page_token=next_token,
        )
    except Exception as exc:
        _raise_api_error(exc, "list prompt note revisions")


@router.get("/{note_id}", response_model=PromptNoteResponse)
async def get_prompt_note(
    http_request: Request,
    response: Response,
    note_id: str,
    workspace_id: str | None = Query(None),
    include_deleted: bool = Query(False),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "read", workspace_id=workspace_id)
        resource = await service.get(
            ctx.tenant_id,
            workspace_id,
            PROMPT_NOTE_NAMESPACE,
            note_id,
            include_deleted=include_deleted,
        )
        if resource is None:
            raise VersionedResourceNotFoundError("prompt note not found")
        response.headers["ETag"] = resource.etag
        return PromptNoteResponse(note=_to_note(resource))
    except Exception as exc:
        _raise_api_error(exc, "get prompt note")


@router.put(
    "/{note_id}",
    response_model=PromptNoteResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def replace_prompt_note(
    http_request: Request,
    response: Response,
    note_id: str,
    request: PromptNoteReplaceInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteResponse:
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "write", workspace_id=workspace_id)
        result = await service.replace(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=PROMPT_NOTE_NAMESPACE,
            resource_id=note_id,
            schema_version=request.schema_version,
            content={"title": request.title, "content": request.content, "enabled": request.enabled},
            metadata=request.metadata,
            actor=_actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return PromptNoteResponse(note=_to_note(result.resource), replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "replace prompt note")


@router.delete(
    "/{note_id}",
    response_model=PromptNoteResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def delete_prompt_note(
    http_request: Request,
    response: Response,
    note_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteResponse:
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "write", workspace_id=workspace_id)
        result = await service.delete(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=PROMPT_NOTE_NAMESPACE,
            resource_id=note_id,
            actor=_actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return PromptNoteResponse(note=_to_note(result.resource), replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "delete prompt note")


@router.post(
    "/{note_id}/restore",
    response_model=PromptNoteResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def restore_prompt_note(
    http_request: Request,
    response: Response,
    note_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> PromptNoteResponse:
    try:
        operation_id = _required_header(http_request, "Idempotency-Key")
        expected = _required_header(http_request, "If-Match")
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "prompt_notes", "write", workspace_id=workspace_id)
        result = await service.restore(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=PROMPT_NOTE_NAMESPACE,
            resource_id=note_id,
            actor=_actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return PromptNoteResponse(note=_to_note(result.resource), replayed=result.replayed)
    except Exception as exc:
        _raise_api_error(exc, "restore prompt note")


def _to_note(resource: VersionedResource) -> PromptNote:
    return PromptNote(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        key=resource.resource_key,
        title=str(resource.content["title"]),
        content=str(resource.content["content"]),
        enabled=bool(resource.content["enabled"]),
        schema_version=resource.schema_version,
        metadata=resource.metadata,
        revision=resource.revision,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
    )


def _to_revision(revision: VersionedResourceRevision) -> PromptNoteRevision:
    return PromptNoteRevision(
        note=_to_note(revision.as_resource()),
        action=revision.action,
        operation_id=revision.operation_id,
    )


def _raise_api_error(exc: Exception, operation: str) -> None:
    _shared_raise_api_error(exc, operation, __name__)


class PromptNotesAPIPlugin(Plugin):
    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
