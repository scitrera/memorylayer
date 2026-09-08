"""Business logic for MemoryLayer's internal versioned-resource primitive."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ...models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourceRevision,
)
from ...utils import generate_id, to_utc_iso
from ..storage import StorageBackend


class VersionedResourceService:
    """Backend-neutral revision, CAS, idempotency, and cursor semantics."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    async def create(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        schema_version: int,
        content: dict[str, Any],
        metadata: dict[str, Any],
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> VersionedResourceMutationResult:
        request_hash = self.request_hash(
            action="create",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=resource_key,
            schema_version=schema_version,
            content=content,
            metadata=metadata,
            expected_etag=expected_etag,
        )
        if replay := await self._replay(
            tenant_id, workspace_id, namespace, operation_id, request_hash
        ):
            return replay
        now = datetime.now(UTC)
        state_hash = self.state_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=resource_key,
            schema_version=schema_version,
            content=content,
            metadata=metadata,
            deleted_at=None,
        )
        resource = VersionedResource(
            id=generate_id("vrs"),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=resource_key,
            schema_version=schema_version,
            content=content,
            metadata=metadata,
            revision=0,
            sequence=0,
            state_hash=state_hash,
            etag="",
            created_by=actor,
            updated_by=actor,
            created_at=now,
            updated_at=now,
        )
        return await self._storage.mutate_versioned_resource(
            VersionedResourceMutation(
                action="create",
                resource=resource,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )

    async def replace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        schema_version: int,
        content: dict[str, Any],
        metadata: dict[str, Any],
        actor: str | None,
        operation_id: str,
        expected_etag: str,
        validate_current: Callable[[VersionedResource], None] | None = None,
    ) -> VersionedResourceMutationResult:
        request_hash = self.request_hash(
            action="replace",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_id=resource_id,
            schema_version=schema_version,
            content=content,
            metadata=metadata,
            expected_etag=expected_etag,
        )
        if replay := await self._replay(
            tenant_id, workspace_id, namespace, operation_id, request_hash
        ):
            return replay
        current = await self._require_active(tenant_id, workspace_id, namespace, resource_id)
        if validate_current is not None:
            validate_current(current)
        now = datetime.now(UTC)
        state_hash = self.state_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=current.resource_key,
            schema_version=schema_version,
            content=content,
            metadata=metadata,
            deleted_at=None,
        )
        desired = current.model_copy(
            update={
                "schema_version": schema_version,
                "content": content,
                "metadata": metadata,
                "state_hash": state_hash,
                "updated_by": actor,
                "updated_at": now,
                "deleted_at": None,
            }
        )
        return await self._storage.mutate_versioned_resource(
            VersionedResourceMutation(
                action="replace",
                resource=desired,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )

    async def delete(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> VersionedResourceMutationResult:
        request_hash = self.request_hash(
            action="delete",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_id=resource_id,
            expected_etag=expected_etag,
        )
        if replay := await self._replay(
            tenant_id, workspace_id, namespace, operation_id, request_hash
        ):
            return replay
        current = await self._require_active(tenant_id, workspace_id, namespace, resource_id)
        now = datetime.now(UTC)
        state_hash = self.state_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=current.resource_key,
            schema_version=current.schema_version,
            content=current.content,
            metadata=current.metadata,
            deleted_at=now,
        )
        desired = current.model_copy(
            update={
                "state_hash": state_hash,
                "updated_by": actor,
                "updated_at": now,
                "deleted_at": now,
            }
        )
        return await self._storage.mutate_versioned_resource(
            VersionedResourceMutation(
                action="delete",
                resource=desired,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )

    async def restore(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        actor: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> VersionedResourceMutationResult:
        """Reactivate a tombstone without changing its stable identity or content."""

        request_hash = self.request_hash(
            action="restore",
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_id=resource_id,
            expected_etag=expected_etag,
        )
        if replay := await self._replay(
            tenant_id, workspace_id, namespace, operation_id, request_hash
        ):
            return replay
        current = await self.get(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            include_deleted=True,
        )
        if current is None:
            raise VersionedResourceNotFoundError("resource not found")
        if not current.is_deleted:
            raise VersionedResourceConflictError("resource is not deleted")
        now = datetime.now(UTC)
        state_hash = self.state_hash(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            namespace=namespace,
            resource_key=current.resource_key,
            schema_version=current.schema_version,
            content=current.content,
            metadata=current.metadata,
            deleted_at=None,
        )
        desired = current.model_copy(
            update={
                "state_hash": state_hash,
                "updated_by": actor,
                "updated_at": now,
                "deleted_at": None,
            }
        )
        return await self._storage.mutate_versioned_resource(
            VersionedResourceMutation(
                action="restore",
                resource=desired,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )

    async def get(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        return await self._storage.get_versioned_resource(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            include_deleted=include_deleted,
        )

    async def get_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        """Resolve a stable domain key without exposing the internal resource ID."""
        return await self._storage.get_versioned_resource_by_key(
            tenant_id,
            workspace_id,
            namespace,
            resource_key,
            include_deleted=include_deleted,
        )

    async def list_page(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        page_token: str | None = None,
        include_deleted: bool = False,
    ) -> tuple[list[VersionedResource], str | None]:
        cursor_scope = self.cursor_scope(
            "heads", tenant_id, workspace_id, namespace, str(include_deleted).lower()
        )
        before = self.decode_cursor(page_token, cursor_scope) if page_token else None
        resources = await self._storage.list_versioned_resources(
            tenant_id,
            workspace_id,
            namespace,
            limit=limit + 1,
            before_sequence=before,
            include_deleted=include_deleted,
        )
        has_more = len(resources) > limit
        resources = resources[:limit]
        next_token = self.encode_cursor(resources[-1].sequence, cursor_scope) if has_more else None
        return resources, next_token

    async def list_filtered_page(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        predicate: Callable[[VersionedResource], bool],
        filter_scope: str,
        page_token: str | None = None,
        scan_limit: int = 1000,
        include_deleted: bool = False,
    ) -> tuple[list[VersionedResource], str | None, int, bool]:
        """Return a bounded filtered page without promoting a second index.

        The opaque cursor is scoped to the complete normalized filter and
        advances by the last resource inspected, not by the last match. Sparse
        searches therefore make deterministic forward progress without
        skipping records. ``scan_truncated`` tells callers that the bounded
        scan stopped before filling the requested result page.
        """
        if limit < 1 or scan_limit < limit:
            raise ValueError("invalid filtered page bounds")
        cursor_scope = self.cursor_scope(
            "filtered-heads",
            tenant_id,
            workspace_id,
            namespace,
            str(include_deleted).lower(),
            filter_scope,
        )
        before = self.decode_cursor(page_token, cursor_scope) if page_token else None
        candidates = await self._storage.list_versioned_resources(
            tenant_id,
            workspace_id,
            namespace,
            limit=scan_limit + 1,
            before_sequence=before,
            include_deleted=include_deleted,
        )
        source_has_more = len(candidates) > scan_limit
        candidates = candidates[:scan_limit]
        matches: list[VersionedResource] = []
        inspected = 0
        for resource in candidates:
            inspected += 1
            if predicate(resource):
                matches.append(resource)
                if len(matches) == limit:
                    break

        stopped_before_candidates_end = inspected < len(candidates)
        has_more = stopped_before_candidates_end or source_has_more
        next_token = (
            self.encode_cursor(candidates[inspected - 1].sequence, cursor_scope)
            if has_more and inspected > 0
            else None
        )
        scan_truncated = has_more and len(matches) < limit and not stopped_before_candidates_end
        return matches, next_token, inspected, scan_truncated

    async def list_revision_page(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        page_token: str | None = None,
    ) -> tuple[list[VersionedResourceRevision], str | None]:
        cursor_scope = self.cursor_scope("revisions", tenant_id, workspace_id, namespace, resource_id)
        before = self.decode_cursor(page_token, cursor_scope) if page_token else None
        revisions = await self._storage.list_versioned_resource_revisions(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            limit=limit + 1,
            before_sequence=before,
        )
        has_more = len(revisions) > limit
        revisions = revisions[:limit]
        next_token = self.encode_cursor(revisions[-1].sequence, cursor_scope) if has_more else None
        return revisions, next_token

    async def _require_active(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
    ) -> VersionedResource:
        current = await self.get(tenant_id, workspace_id, namespace, resource_id)
        if current is None:
            raise VersionedResourceNotFoundError("resource not found")
        return current

    async def _replay(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        return await self._storage.get_versioned_resource_operation(
            tenant_id,
            workspace_id,
            namespace,
            operation_id,
            request_hash,
        )

    @staticmethod
    def state_hash(**state: Any) -> str:
        return _canonical_hash(state)

    @staticmethod
    def request_hash(**request: Any) -> str:
        return _canonical_hash(request)

    @staticmethod
    def cursor_scope(*parts: str) -> str:
        return _canonical_hash(list(parts))

    @staticmethod
    def encode_cursor(sequence: int, cursor_scope: str) -> str:
        payload = json.dumps(
            {"v": 1, "scope": cursor_scope, "before_sequence": sequence},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(payload).decode().rstrip("=")

    @staticmethod
    def decode_cursor(token: str, cursor_scope: str) -> int:
        try:
            padded = token + "=" * (-len(token) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
            if payload.get("v") != 1 or payload.get("scope") != cursor_scope:
                raise ValueError
            sequence = payload["before_sequence"]
            if not isinstance(sequence, int) or sequence < 1:
                raise ValueError
            return sequence
        except Exception as exc:
            raise ValueError("invalid page token") from exc


def _canonical_hash(value: Any) -> str:
    def default(item: Any) -> str:
        if isinstance(item, datetime):
            return to_utc_iso(item)
        raise TypeError(f"unsupported canonical value: {type(item).__name__}")

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=default,
    )
    return hashlib.sha256(payload.encode()).hexdigest()
