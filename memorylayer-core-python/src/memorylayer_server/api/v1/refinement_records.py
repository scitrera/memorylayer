"""Append-only, workspace-scoped refinement-record API."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, Query, Request, Response, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables

from ...models.refinement_record import RefinementRecord, RefinementRecordCreateInput
from ...models.versioned_resource import VersionedResource, VersionedResourceNotFoundError, VersionedResourcePreconditionFailedError
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.versioned_resources import VersionedResourceService
from .. import EXT_MULTI_API_ROUTERS
from ._versioned_resource import actor, raise_api_error, required_header
from .deps import get_auth_service, get_authz_service, get_versioned_resource_service
from .schemas import ErrorResponse

REFINEMENT_RECORD_NAMESPACE = "agent.refinement-record.v1"
router = APIRouter(prefix="/v1/refinement-records", tags=["refinement-records"])


class RefinementRecordResponse(BaseModel):
    record: RefinementRecord
    replayed: bool = False


class RefinementRecordListResponse(BaseModel):
    records: list[RefinementRecord]
    next_page_token: str | None = None
    scanned_count: int = 0
    scan_truncated: bool = False


@router.post(
    "",
    response_model=RefinementRecordResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}, 412: {"model": ErrorResponse}, 428: {"model": ErrorResponse}},
)
async def create_refinement_record(
    http_request: Request,
    response: Response,
    request: RefinementRecordCreateInput,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> RefinementRecordResponse:
    try:
        operation_id = required_header(http_request, "Idempotency-Key")
        expected = required_header(http_request, "If-None-Match")
        if expected != "*":
            raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "refinement_records", "write", workspace_id=workspace_id)
        result = await service.create(
            tenant_id=ctx.tenant_id,
            workspace_id=workspace_id,
            namespace=REFINEMENT_RECORD_NAMESPACE,
            resource_key=request.key,
            schema_version=request.schema_version,
            content=request.model_dump(mode="json", exclude={"key", "schema_version", "metadata", "workspace_id"}),
            metadata=request.metadata,
            actor=actor(ctx),
            operation_id=operation_id,
            expected_etag=expected,
        )
        response.headers["ETag"] = result.resource.etag
        return RefinementRecordResponse(record=_to_record(result.resource), replayed=result.replayed)
    except Exception as exc:
        raise_api_error(exc, "create refinement record", __name__)


@router.get("", response_model=RefinementRecordListResponse)
async def list_refinement_records(
    http_request: Request,
    workspace_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    page_token: str | None = Query(None),
    phase: list[str] | None = Query(None),
    outcome: list[str] | None = Query(None),
    scope: list[str] | None = Query(None),
    resource_kind: list[str] | None = Query(None),
    refinement_id: str | None = Query(None, max_length=200),
    search: str | None = Query(None, max_length=200),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> RefinementRecordListResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "refinement_records", "read", workspace_id=workspace_id)
        filters = _normalize_filters(
            phase=phase,
            outcome=outcome,
            scope=scope,
            resource_kind=resource_kind,
            refinement_id=refinement_id,
            search=search,
        )
        if _has_filters(filters):
            resources, next_token, scanned_count, scan_truncated = await service.list_filtered_page(
                ctx.tenant_id,
                workspace_id,
                REFINEMENT_RECORD_NAMESPACE,
                limit=limit,
                page_token=page_token,
                predicate=lambda resource: _matches_filters(resource, filters),
                filter_scope=json.dumps(filters, sort_keys=True, separators=(",", ":")),
                scan_limit=max(limit, 1000),
            )
        else:
            resources, next_token = await service.list_page(
                ctx.tenant_id, workspace_id, REFINEMENT_RECORD_NAMESPACE, limit=limit, page_token=page_token
            )
            scanned_count = len(resources)
            scan_truncated = False
        return RefinementRecordListResponse(
            records=[_to_record(item) for item in resources],
            next_page_token=next_token,
            scanned_count=scanned_count,
            scan_truncated=scan_truncated,
        )
    except Exception as exc:
        raise_api_error(exc, "list refinement records", __name__)


@router.get("/{record_id}", response_model=RefinementRecordResponse)
async def get_refinement_record(
    http_request: Request,
    response: Response,
    record_id: str,
    workspace_id: str | None = Query(None),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    service: VersionedResourceService = Depends(get_versioned_resource_service),
) -> RefinementRecordResponse:
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "refinement_records", "read", workspace_id=workspace_id)
        resource = await service.get(ctx.tenant_id, workspace_id, REFINEMENT_RECORD_NAMESPACE, record_id)
        if resource is None:
            raise VersionedResourceNotFoundError("refinement record not found")
        response.headers["ETag"] = resource.etag
        return RefinementRecordResponse(record=_to_record(resource))
    except Exception as exc:
        raise_api_error(exc, "get refinement record", __name__)


def _to_record(resource: VersionedResource) -> RefinementRecord:
    return RefinementRecord(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        key=resource.resource_key,
        schema_version=resource.schema_version,
        metadata=resource.metadata,
        revision=resource.revision,
        etag=resource.etag,
        created_by=resource.created_by,
        created_at=resource.created_at,
        **resource.content,
    )


_FILTER_VALUES = {
    "phase": {"proposal", "decision", "application", "rollback"},
    "outcome": {"proposed", "approved", "rejected", "applied", "partially_applied", "failed", "rolled_back", "no_op"},
    "scope": {"session", "workspace", "user", "tenant", "global"},
    "resource_kind": {"prompt_note", "memory", "skill", "agent_specification"},
}


def _normalize_filter_values(name: str, values: list[str] | None) -> list[str]:
    normalized: set[str] = set()
    for raw in values or []:
        for item in raw.split(","):
            item = item.strip().lower().replace("-", "_")
            if not item or item not in _FILTER_VALUES[name]:
                raise ValueError(f"invalid refinement {name} filter: {item or raw!r}")
            normalized.add(item)
    if len(normalized) > 16:
        raise ValueError(f"too many refinement {name} filters")
    return sorted(normalized)


def _normalize_filters(
    *,
    phase: list[str] | None,
    outcome: list[str] | None,
    scope: list[str] | None,
    resource_kind: list[str] | None,
    refinement_id: str | None,
    search: str | None,
) -> dict[str, object]:
    return {
        "phase": _normalize_filter_values("phase", phase),
        "outcome": _normalize_filter_values("outcome", outcome),
        "scope": _normalize_filter_values("scope", scope),
        "resource_kind": _normalize_filter_values("resource_kind", resource_kind),
        "refinement_id": (refinement_id or "").strip(),
        "search": (search or "").strip().casefold(),
    }


def _has_filters(filters: dict[str, object]) -> bool:
    return any(bool(value) for value in filters.values())


def _matches_filters(resource: VersionedResource, filters: dict[str, object]) -> bool:
    content = resource.content
    for name in ("phase", "outcome", "scope"):
        allowed = filters[name]
        if allowed and content.get(name) not in allowed:
            return False
    refinement_id = filters["refinement_id"]
    if refinement_id and content.get("refinement_id") != refinement_id:
        return False
    resource_kinds = filters["resource_kind"]
    edits = content.get("edits")
    edit_items = edits if isinstance(edits, list) else []
    if resource_kinds and not any(isinstance(edit, dict) and edit.get("resource_kind") in resource_kinds for edit in edit_items):
        return False
    search = filters["search"]
    return not search or _content_contains(resource, str(search))


def _content_contains(resource: VersionedResource, search: str) -> bool:
    content = resource.content
    values: list[object] = [
        resource.id,
        resource.resource_key,
        content.get("refinement_id"),
        content.get("phase"),
        content.get("outcome"),
        content.get("scope"),
        content.get("trigger"),
        content.get("summary"),
        content.get("rationale"),
        content.get("expected_outcome"),
    ]
    for collection in (content.get("evidence"), content.get("edits")):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            values.extend(
                item.get(name)
                for name in (
                    "kind",
                    "reference",
                    "description",
                    "action",
                    "resource_kind",
                    "resource_key",
                    "reason",
                    "error",
                )
            )
    return any(search in value.casefold() for value in values if isinstance(value, str))


class RefinementRecordsAPIPlugin(Plugin):
    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
