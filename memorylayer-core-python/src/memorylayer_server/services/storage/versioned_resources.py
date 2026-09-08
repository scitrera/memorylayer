"""Shared SQLite/libSQL persistence for internal versioned resources."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ...models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from ...utils import parse_datetime_utc, to_utc_iso

_HEAD_COLUMNS = """
    id, tenant_id, workspace_id, namespace, resource_key, schema_version,
    content, metadata, revision, state_hash, etag, created_by, updated_by,
    created_at, updated_at, deleted_at, last_action, last_operation_id,
    last_request_hash
"""


class RelationalVersionedResourceStore:
    """Atomic versioned-resource operations shared by SQLite and Turso.

    Revision and idempotency rows are written by triggers in the same database
    statement as the head mutation.  That avoids a crash window between the
    authoritative head, its immutable history, and its accepted operation.
    """

    def __init__(self, connection: Any):
        self._connection = connection
        self._lock = asyncio.Lock()

    async def create_tables(self) -> None:
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS versioned_resource_heads (
                id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                resource_key TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state_hash TEXT NOT NULL,
                etag TEXT NOT NULL,
                created_by TEXT,
                updated_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                last_action TEXT NOT NULL,
                last_operation_id TEXT NOT NULL,
                last_request_hash TEXT NOT NULL,
                PRIMARY KEY (tenant_id, workspace_id, namespace, id),
                UNIQUE (tenant_id, workspace_id, namespace, resource_key)
            )
        """)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS versioned_resource_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                resource_key TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                revision INTEGER NOT NULL,
                state_hash TEXT NOT NULL,
                etag TEXT NOT NULL,
                created_by TEXT,
                updated_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                action TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                UNIQUE (tenant_id, workspace_id, namespace, id, revision)
            )
        """)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS versioned_resource_operations (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, workspace_id, namespace, operation_id)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_versioned_heads_list "
            "ON versioned_resource_heads(tenant_id, workspace_id, namespace, deleted_at)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_versioned_revisions_resource "
            "ON versioned_resource_revisions(tenant_id, workspace_id, namespace, id, sequence DESC)"
        )

        await self._connection.execute("""
            CREATE TRIGGER IF NOT EXISTS versioned_resource_head_insert
            AFTER INSERT ON versioned_resource_heads
            BEGIN
                INSERT INTO versioned_resource_revisions (
                    id, tenant_id, workspace_id, namespace, resource_key,
                    schema_version, content, metadata, revision, state_hash,
                    etag, created_by, updated_by, created_at, updated_at,
                    deleted_at, action, operation_id, request_hash
                ) VALUES (
                    NEW.id, NEW.tenant_id, NEW.workspace_id, NEW.namespace,
                    NEW.resource_key, NEW.schema_version, NEW.content,
                    NEW.metadata, NEW.revision, NEW.state_hash, NEW.etag,
                    NEW.created_by, NEW.updated_by, NEW.created_at,
                    NEW.updated_at, NEW.deleted_at, NEW.last_action,
                    NEW.last_operation_id, NEW.last_request_hash
                );
                INSERT INTO versioned_resource_operations (
                    tenant_id, workspace_id, namespace, operation_id,
                    request_hash, resource_id, revision
                ) VALUES (
                    NEW.tenant_id, NEW.workspace_id, NEW.namespace,
                    NEW.last_operation_id, NEW.last_request_hash, NEW.id,
                    NEW.revision
                );
            END
        """)
        await self._connection.execute("""
            CREATE TRIGGER IF NOT EXISTS versioned_resource_head_update
            AFTER UPDATE ON versioned_resource_heads
            BEGIN
                INSERT INTO versioned_resource_revisions (
                    id, tenant_id, workspace_id, namespace, resource_key,
                    schema_version, content, metadata, revision, state_hash,
                    etag, created_by, updated_by, created_at, updated_at,
                    deleted_at, action, operation_id, request_hash
                ) VALUES (
                    NEW.id, NEW.tenant_id, NEW.workspace_id, NEW.namespace,
                    NEW.resource_key, NEW.schema_version, NEW.content,
                    NEW.metadata, NEW.revision, NEW.state_hash, NEW.etag,
                    NEW.created_by, NEW.updated_by, NEW.created_at,
                    NEW.updated_at, NEW.deleted_at, NEW.last_action,
                    NEW.last_operation_id, NEW.last_request_hash
                );
                INSERT INTO versioned_resource_operations (
                    tenant_id, workspace_id, namespace, operation_id,
                    request_hash, resource_id, revision
                ) VALUES (
                    NEW.tenant_id, NEW.workspace_id, NEW.namespace,
                    NEW.last_operation_id, NEW.last_request_hash, NEW.id,
                    NEW.revision
                );
            END
        """)
        await self._connection.commit()

    async def mutate(self, mutation: VersionedResourceMutation) -> VersionedResourceMutationResult:
        desired = mutation.resource
        async with self._lock:
            replay = await self._get_operation(
                desired.tenant_id,
                desired.workspace_id,
                desired.namespace,
                mutation.operation_id,
            )
            if replay is not None:
                return self._validate_replay(replay, mutation.request_hash)

            current = await self.get(
                desired.tenant_id,
                desired.workspace_id,
                desired.namespace,
                desired.id,
                include_deleted=True,
            )

            if mutation.action == "create":
                if mutation.expected_etag != "*":
                    raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
                existing_key = await self.get_by_key(
                    desired.tenant_id,
                    desired.workspace_id,
                    desired.namespace,
                    desired.resource_key,
                    include_deleted=True,
                )
                if current is not None or existing_key is not None:
                    raise VersionedResourceConflictError("resource id or key already exists")
                final = desired.model_copy(update={"revision": 1})
                final = final.model_copy(update={"etag": _etag(1, final.state_hash)})
                try:
                    await self._insert_head(final, mutation)
                    await self._connection.commit()
                except Exception:
                    await self._connection.rollback()
                    replay = await self._get_operation(
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_replay(replay, mutation.request_hash)
                    if await self.get_by_key(
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        desired.resource_key,
                        include_deleted=True,
                    ) is not None:
                        raise VersionedResourceConflictError("resource id or key already exists")
                    raise
            else:
                self._validate_current(mutation, current)
                next_revision = current.revision + 1
                final = desired.model_copy(
                    update={
                        "created_at": current.created_at,
                        "created_by": current.created_by,
                        "revision": next_revision,
                        "etag": _etag(next_revision, desired.state_hash),
                    }
                )
                try:
                    cursor = await self._update_head(final, mutation, current.etag)
                    if cursor.rowcount == 0:
                        await self._connection.rollback()
                        return await self._failed_update(mutation)
                    await self._connection.commit()
                except Exception:
                    await self._connection.rollback()
                    replay = await self._get_operation(
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_replay(replay, mutation.request_hash)
                    latest = await self.get(
                        desired.tenant_id,
                        desired.workspace_id,
                        desired.namespace,
                        desired.id,
                        include_deleted=True,
                    )
                    self._validate_current(mutation, latest)
                    raise

            stored = await self.get(
                final.tenant_id,
                final.workspace_id,
                final.namespace,
                final.id,
                include_deleted=True,
            )
            if stored is None:  # pragma: no cover - indicates database corruption
                raise RuntimeError("accepted versioned-resource mutation has no stored head")
            return VersionedResourceMutationResult(resource=stored)

    async def get_operation_result(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        revision = await self._get_operation(tenant_id, workspace_id, namespace, operation_id)
        if revision is None:
            return None
        return self._validate_replay(revision, request_hash)

    async def _failed_update(self, mutation: VersionedResourceMutation) -> VersionedResourceMutationResult:
        desired = mutation.resource
        replay = await self._get_operation(
            desired.tenant_id,
            desired.workspace_id,
            desired.namespace,
            mutation.operation_id,
        )
        if replay is not None:
            return self._validate_replay(replay, mutation.request_hash)
        latest = await self.get(
            desired.tenant_id,
            desired.workspace_id,
            desired.namespace,
            desired.id,
            include_deleted=True,
        )
        self._validate_current(mutation, latest)
        raise VersionedResourcePreconditionFailedError("versioned resource mutation was not admitted")

    @staticmethod
    def _validate_current(
        mutation: VersionedResourceMutation,
        current: VersionedResource | None,
    ) -> None:
        if current is None:
            raise VersionedResourceNotFoundError("resource not found")
        if mutation.expected_etag != current.etag:
            raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
        if mutation.action == "restore":
            if not current.is_deleted:
                raise VersionedResourceConflictError("resource is not deleted")
        elif current.is_deleted:
            raise VersionedResourceNotFoundError("resource not found")
        if mutation.resource.resource_key != current.resource_key:
            raise VersionedResourceConflictError("resource key is immutable")

    async def _insert_head(self, resource: VersionedResource, mutation: VersionedResourceMutation):
        placeholders = ", ".join("?" for _ in range(19))
        return await self._connection.execute(
            f"INSERT INTO versioned_resource_heads ({_HEAD_COLUMNS}) VALUES ({placeholders})",
            self._head_values(resource, mutation),
        )

    async def _update_head(self, resource: VersionedResource, mutation: VersionedResourceMutation, current_etag: str):
        deletion_predicate = "deleted_at IS NOT NULL" if mutation.action == "restore" else "deleted_at IS NULL"
        return await self._connection.execute(
            f"""
            UPDATE versioned_resource_heads SET
                schema_version = ?, content = ?, metadata = ?, revision = ?,
                state_hash = ?, etag = ?, updated_by = ?, updated_at = ?,
                deleted_at = ?, last_action = ?, last_operation_id = ?,
                last_request_hash = ?
            WHERE tenant_id = ? AND workspace_id = ? AND namespace = ?
              AND id = ? AND etag = ? AND {deletion_predicate}
            """,
            (
                resource.schema_version,
                _json(resource.content),
                _json(resource.metadata),
                resource.revision,
                resource.state_hash,
                resource.etag,
                resource.updated_by,
                to_utc_iso(resource.updated_at),
                to_utc_iso(resource.deleted_at),
                mutation.action,
                mutation.operation_id,
                mutation.request_hash,
                resource.tenant_id,
                resource.workspace_id,
                resource.namespace,
                resource.id,
                current_etag,
            ),
        )

    def _head_values(self, resource: VersionedResource, mutation: VersionedResourceMutation) -> tuple:
        return (
            resource.id,
            resource.tenant_id,
            resource.workspace_id,
            resource.namespace,
            resource.resource_key,
            resource.schema_version,
            _json(resource.content),
            _json(resource.metadata),
            resource.revision,
            resource.state_hash,
            resource.etag,
            resource.created_by,
            resource.updated_by,
            to_utc_iso(resource.created_at),
            to_utc_iso(resource.updated_at),
            to_utc_iso(resource.deleted_at),
            mutation.action,
            mutation.operation_id,
            mutation.request_hash,
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
        deleted = "" if include_deleted else " AND h.deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"""
            SELECT h.*, r.sequence FROM versioned_resource_heads h
            JOIN versioned_resource_revisions r
              ON r.tenant_id = h.tenant_id AND r.workspace_id = h.workspace_id
             AND r.namespace = h.namespace AND r.id = h.id
             AND r.revision = h.revision
            WHERE h.tenant_id = ? AND h.workspace_id = ? AND h.namespace = ?
              AND h.id = ?{deleted}
            """,
            (tenant_id, workspace_id, namespace, resource_id),
        )
        row = await cursor.fetchone()
        return _row_to_resource(row) if row else None

    async def get_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        deleted = "" if include_deleted else " AND h.deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"""
            SELECT h.*, r.sequence FROM versioned_resource_heads h
            JOIN versioned_resource_revisions r
              ON r.tenant_id = h.tenant_id AND r.workspace_id = h.workspace_id
             AND r.namespace = h.namespace AND r.id = h.id
             AND r.revision = h.revision
            WHERE h.tenant_id = ? AND h.workspace_id = ? AND h.namespace = ?
              AND h.resource_key = ?{deleted}
            """,
            (tenant_id, workspace_id, namespace, resource_key),
        )
        row = await cursor.fetchone()
        return _row_to_resource(row) if row else None

    async def list(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        before_sequence: int | None = None,
        include_deleted: bool = False,
    ) -> list[VersionedResource]:
        clauses = ["h.tenant_id = ?", "h.workspace_id = ?", "h.namespace = ?"]
        params: list[Any] = [tenant_id, workspace_id, namespace]
        if not include_deleted:
            clauses.append("h.deleted_at IS NULL")
        if before_sequence is not None:
            clauses.append("r.sequence < ?")
            params.append(before_sequence)
        params.append(limit)
        cursor = await self._connection.execute(
            f"""
            SELECT h.*, r.sequence FROM versioned_resource_heads h
            JOIN versioned_resource_revisions r
              ON r.tenant_id = h.tenant_id AND r.workspace_id = h.workspace_id
             AND r.namespace = h.namespace AND r.id = h.id
             AND r.revision = h.revision
            WHERE {' AND '.join(clauses)}
            ORDER BY r.sequence DESC LIMIT ?
            """,
            tuple(params),
        )
        return [_row_to_resource(row) for row in await cursor.fetchall()]

    async def list_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[VersionedResourceRevision]:
        clauses = ["tenant_id = ?", "workspace_id = ?", "namespace = ?", "id = ?"]
        params: list[Any] = [tenant_id, workspace_id, namespace, resource_id]
        if before_sequence is not None:
            clauses.append("sequence < ?")
            params.append(before_sequence)
        params.append(limit)
        cursor = await self._connection.execute(
            f"SELECT * FROM versioned_resource_revisions WHERE {' AND '.join(clauses)} "
            "ORDER BY sequence DESC LIMIT ?",
            tuple(params),
        )
        return [_row_to_revision(row) for row in await cursor.fetchall()]

    async def _get_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
    ) -> VersionedResourceRevision | None:
        cursor = await self._connection.execute(
            """
            SELECT r.* FROM versioned_resource_operations o
            JOIN versioned_resource_revisions r
              ON r.tenant_id = o.tenant_id AND r.workspace_id = o.workspace_id
             AND r.namespace = o.namespace AND r.id = o.resource_id
             AND r.revision = o.revision
            WHERE o.tenant_id = ? AND o.workspace_id = ? AND o.namespace = ?
              AND o.operation_id = ?
            """,
            (tenant_id, workspace_id, namespace, operation_id),
        )
        row = await cursor.fetchone()
        return _row_to_revision(row) if row else None

    @staticmethod
    def _validate_replay(revision: VersionedResourceRevision, request_hash: str) -> VersionedResourceMutationResult:
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError("idempotency key was already used for a different request")
        return VersionedResourceMutationResult(resource=revision.as_resource(), replayed=True)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _etag(revision: int, state_hash: str) -> str:
    return f'"vr-{revision}-{state_hash}"'


def _row_to_resource(row: Any) -> VersionedResource:
    return VersionedResource(
        id=row["id"],
        tenant_id=row["tenant_id"],
        workspace_id=row["workspace_id"],
        namespace=row["namespace"],
        resource_key=row["resource_key"],
        schema_version=row["schema_version"],
        content=json.loads(row["content"]),
        metadata=json.loads(row["metadata"]),
        revision=row["revision"],
        sequence=row["sequence"],
        state_hash=row["state_hash"],
        etag=row["etag"],
        created_by=row["created_by"],
        updated_by=row["updated_by"],
        created_at=parse_datetime_utc(row["created_at"]),
        updated_at=parse_datetime_utc(row["updated_at"]),
        deleted_at=parse_datetime_utc(row["deleted_at"]),
    )


def _row_to_revision(row: Any) -> VersionedResourceRevision:
    resource = _row_to_resource(row)
    return VersionedResourceRevision(
        **resource.model_dump(),
        action=row["action"],
        operation_id=row["operation_id"],
        request_hash=row["request_hash"],
    )
