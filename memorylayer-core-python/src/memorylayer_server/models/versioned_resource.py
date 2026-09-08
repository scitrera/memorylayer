"""Internal primitives for typed, revisioned MemoryLayer resources.

The public API must expose domain-specific models (prompt notes, agent specs,
and so on).  These models are the shared persistence contract underneath those
typed services; they are deliberately not a generic public JSON-store API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

VersionedResourceAction = Literal["create", "replace", "delete", "restore"]


class VersionedResourceError(Exception):
    """Base class for deterministic versioned-resource failures."""


class VersionedResourceConflictError(VersionedResourceError):
    """A logical key, identifier, or idempotency key conflicts."""


class VersionedResourcePreconditionFailedError(VersionedResourceError):
    """A mutation's expected ETag does not match the current head."""


class VersionedResourceNotFoundError(VersionedResourceError):
    """The requested resource head does not exist in the addressed scope."""


class VersionedResource(BaseModel):
    """Current authoritative head of one typed resource."""

    model_config = {"from_attributes": True}

    id: str
    tenant_id: str
    workspace_id: str
    namespace: str
    resource_key: str
    schema_version: int = Field(ge=1)
    content: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    revision: int = Field(ge=0)
    sequence: int = Field(ge=0)
    state_hash: str
    etag: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


class VersionedResourceRevision(VersionedResource):
    """Immutable snapshot produced by one accepted mutation."""

    action: VersionedResourceAction
    operation_id: str
    request_hash: str

    def as_resource(self) -> VersionedResource:
        return VersionedResource.model_validate(
            self.model_dump(exclude={"action", "operation_id", "request_hash"})
        )


class VersionedResourceMutation(BaseModel):
    """Storage-level atomic mutation request."""

    action: VersionedResourceAction
    resource: VersionedResource
    operation_id: str
    request_hash: str
    expected_etag: str


class VersionedResourceMutationResult(BaseModel):
    """Accepted mutation result; replayed is true for an idempotent retry."""

    resource: VersionedResource
    replayed: bool = False
