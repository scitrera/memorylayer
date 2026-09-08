"""Typed workspace-view authority backed by revisioned resources."""

from __future__ import annotations

import base64
from typing import Any

from ..models.versioned_resource import VersionedResource, VersionedResourceNotFoundError
from ..models.workspace_execution import (
    WorkspaceView,
    WorkspaceViewCreateInput,
    WorkspaceViewObservation,
    WorkspaceViewObservationInput,
    WorkspaceViewObservationReplaceInput,
    WorkspaceViewReplaceInput,
)
from .versioned_resources import VersionedResourceService

WORKSPACE_VIEW_NAMESPACE = "workspace.view.v1"
WORKSPACE_VIEW_OBSERVATION_NAMESPACE_PREFIX = "workspace.view.observation.v1:"


class WorkspaceExecutionService:
    """Domain-specific workspace views over MemoryLayer's CAS/revision store."""

    def __init__(self, resources: VersionedResourceService):
        self._resources = resources

    async def create_view(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        request: WorkspaceViewCreateInput,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceView, bool]:
        result = await self._resources.create(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=WORKSPACE_VIEW_NAMESPACE,
            resource_key=request.view_id,
            schema_version=request.schema_version,
            content=_view_content(request),
            metadata=request.metadata,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_view(result.resource), result.replayed

    async def get_view(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        *,
        include_deleted: bool = False,
    ) -> WorkspaceView | None:
        resource = await self._resources.get_by_key(
            tenant_id,
            workspace_id,
            WORKSPACE_VIEW_NAMESPACE,
            view_id,
            include_deleted=include_deleted,
        )
        return _to_view(resource) if resource else None

    async def list_views(
        self,
        tenant_id: str,
        workspace_id: str,
        *,
        limit: int,
        page_token: str | None,
        include_deleted: bool,
    ) -> tuple[list[WorkspaceView], str | None]:
        resources, next_token = await self._resources.list_page(
            tenant_id,
            workspace_id,
            WORKSPACE_VIEW_NAMESPACE,
            limit=limit,
            page_token=page_token,
            include_deleted=include_deleted,
        )
        return [_to_view(item) for item in resources], next_token

    async def replace_view(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        request: WorkspaceViewReplaceInput,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceView, bool]:
        current = await self._require_view_resource(tenant_id, workspace_id, view_id)
        result = await self._resources.replace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=WORKSPACE_VIEW_NAMESPACE,
            resource_id=current.id,
            schema_version=request.schema_version,
            content=_view_content(request),
            metadata=request.metadata,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_view(result.resource), result.replayed

    async def delete_view(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceView, bool]:
        current = await self._require_view_resource(tenant_id, workspace_id, view_id)
        result = await self._resources.delete(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=WORKSPACE_VIEW_NAMESPACE,
            resource_id=current.id,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_view(result.resource), result.replayed

    async def restore_view(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceView, bool]:
        current = await self._require_view_resource(tenant_id, workspace_id, view_id, include_deleted=True)
        result = await self._resources.restore(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=WORKSPACE_VIEW_NAMESPACE,
            resource_id=current.id,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_view(result.resource), result.replayed

    async def list_view_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        *,
        limit: int,
        page_token: str | None,
    ) -> tuple[list[tuple[WorkspaceView, str, str]], str | None]:
        current = await self._require_view_resource(tenant_id, workspace_id, view_id, include_deleted=True)
        revisions, next_token = await self._resources.list_revision_page(
            tenant_id,
            workspace_id,
            WORKSPACE_VIEW_NAMESPACE,
            current.id,
            limit=limit,
            page_token=page_token,
        )
        return [(_to_view(item.as_resource()), item.action, item.operation_id) for item in revisions], next_token

    async def put_observation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        request: WorkspaceViewObservationInput | WorkspaceViewObservationReplaceInput,
        observer_id: str | None,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
        create: bool,
    ) -> tuple[WorkspaceViewObservation, bool]:
        await self._require_view_resource(tenant_id, workspace_id, view_id)
        resolved_observer_id = request.observer_id if isinstance(request, WorkspaceViewObservationInput) else observer_id
        if not resolved_observer_id:
            raise ValueError("observer_id is required")
        namespace = observation_namespace(view_id)
        content = _observation_content(request)
        if create:
            result = await self._resources.create(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                namespace=namespace,
                resource_key=resolved_observer_id,
                schema_version=request.schema_version,
                content=content,
                metadata=request.metadata,
                actor=actor,
                operation_id=operation_id,
                expected_etag=expected_etag,
            )
        else:
            current = await self._resources.get_by_key(
                tenant_id,
                workspace_id,
                namespace,
                resolved_observer_id,
            )
            if current is None:
                raise VersionedResourceNotFoundError("workspace view observation not found")
            result = await self._resources.replace(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                namespace=namespace,
                resource_id=current.id,
                schema_version=request.schema_version,
                content=content,
                metadata=request.metadata,
                actor=actor,
                operation_id=operation_id,
                expected_etag=expected_etag,
                validate_current=lambda head: _validate_observer_cursor(head, request),
            )
        return _to_observation(result.resource, view_id), result.replayed

    async def get_observation(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        observer_id: str,
    ) -> WorkspaceViewObservation | None:
        await self._require_view_resource(tenant_id, workspace_id, view_id)
        resource = await self._resources.get_by_key(
            tenant_id,
            workspace_id,
            observation_namespace(view_id),
            observer_id,
        )
        return _to_observation(resource, view_id) if resource else None

    async def delete_observation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        observer_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceViewObservation, bool]:
        await self._require_view_resource(tenant_id, workspace_id, view_id)
        namespace = observation_namespace(view_id)
        current = await self._resources.get_by_key(tenant_id, workspace_id, namespace, observer_id)
        if current is None:
            raise VersionedResourceNotFoundError("workspace view observation not found")
        result = await self._resources.delete(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_id=current.id,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_observation(result.resource, view_id), result.replayed

    async def restore_observation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        observer_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> tuple[WorkspaceViewObservation, bool]:
        await self._require_view_resource(tenant_id, workspace_id, view_id)
        namespace = observation_namespace(view_id)
        current = await self._resources.get_by_key(
            tenant_id,
            workspace_id,
            namespace,
            observer_id,
            include_deleted=True,
        )
        if current is None:
            raise VersionedResourceNotFoundError("workspace view observation not found")
        result = await self._resources.restore(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_id=current.id,
            actor=actor,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )
        return _to_observation(result.resource, view_id), result.replayed

    async def list_observations(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        *,
        limit: int,
        page_token: str | None,
    ) -> tuple[list[WorkspaceViewObservation], str | None]:
        await self._require_view_resource(tenant_id, workspace_id, view_id)
        resources, next_token = await self._resources.list_page(
            tenant_id,
            workspace_id,
            observation_namespace(view_id),
            limit=limit,
            page_token=page_token,
        )
        return [_to_observation(item, view_id) for item in resources], next_token

    async def list_observation_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        observer_id: str,
        *,
        limit: int,
        page_token: str | None,
    ) -> tuple[list[tuple[WorkspaceViewObservation, str, str]], str | None]:
        await self._require_view_resource(tenant_id, workspace_id, view_id, include_deleted=True)
        namespace = observation_namespace(view_id)
        current = await self._resources.get_by_key(
            tenant_id,
            workspace_id,
            namespace,
            observer_id,
            include_deleted=True,
        )
        if current is None:
            raise VersionedResourceNotFoundError("workspace view observation not found")
        revisions, next_token = await self._resources.list_revision_page(
            tenant_id,
            workspace_id,
            namespace,
            current.id,
            limit=limit,
            page_token=page_token,
        )
        return [(_to_observation(item.as_resource(), view_id), item.action, item.operation_id) for item in revisions], next_token

    async def _require_view_resource(
        self,
        tenant_id: str,
        workspace_id: str,
        view_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource:
        resource = await self._resources.get_by_key(
            tenant_id,
            workspace_id,
            WORKSPACE_VIEW_NAMESPACE,
            view_id,
            include_deleted=include_deleted,
        )
        if resource is None:
            raise VersionedResourceNotFoundError("workspace view not found")
        return resource


def observation_namespace(view_id: str) -> str:
    encoded = base64.urlsafe_b64encode(view_id.encode()).decode().rstrip("=")
    return WORKSPACE_VIEW_OBSERVATION_NAMESPACE_PREFIX + encoded


def _view_content(request: WorkspaceViewCreateInput | WorkspaceViewReplaceInput) -> dict[str, Any]:
    return request.model_dump(exclude={"view_id", "metadata", "schema_version"}, mode="json")


def _observation_content(request: WorkspaceViewObservationInput | WorkspaceViewObservationReplaceInput) -> dict[str, Any]:
    return request.model_dump(exclude={"observer_id", "metadata", "schema_version"}, mode="json")


def _validate_observer_cursor(
    current: VersionedResource,
    request: WorkspaceViewObservationInput | WorkspaceViewObservationReplaceInput,
) -> None:
    current_generation = str(current.content.get("generation", ""))
    current_sequence = int(current.content.get("sequence", -1))
    if request.generation == current_generation and request.sequence <= current_sequence:
        raise ValueError(
            f"observation sequence must advance within generation {request.generation!r}; "
            f"current={current_sequence}, requested={request.sequence}"
        )


def _to_view(resource: VersionedResource) -> WorkspaceView:
    return WorkspaceView(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        view_id=resource.resource_key,
        schema_version=resource.schema_version,
        metadata=resource.metadata,
        revision=resource.revision,
        sequence=resource.sequence,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
        **resource.content,
    )


def _to_observation(resource: VersionedResource, view_id: str) -> WorkspaceViewObservation:
    return WorkspaceViewObservation(
        id=resource.id,
        tenant_id=resource.tenant_id,
        workspace_id=resource.workspace_id,
        view_id=view_id,
        observer_id=resource.resource_key,
        schema_version=resource.schema_version,
        metadata=resource.metadata,
        resource_revision=resource.revision,
        resource_sequence=resource.sequence,
        etag=resource.etag,
        created_by=resource.created_by,
        updated_by=resource.updated_by,
        created_at=resource.created_at,
        updated_at=resource.updated_at,
        deleted_at=resource.deleted_at,
        **resource.content,
    )
