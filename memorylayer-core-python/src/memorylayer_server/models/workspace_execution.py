"""Durable logical workspace views and observer-published state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WORKSPACE_VIEW_SCHEMA_VERSION = 1
WORKSPACE_VIEW_OBSERVATION_SCHEMA_VERSION = 1

ExecutionSite = Literal["client", "worker", "remote"]
WorkspaceViewKind = Literal["directory", "git_worktree", "checkout", "snapshot", "overlay"]


def normalize_capabilities(values: list[str] | None) -> list[str]:
    """Return a deterministic capability set for hashing and transport."""
    return sorted({value.strip() for value in (values or []) if value.strip()})


def _required_id(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    if "\x00" in value:
        raise ValueError("must not contain NUL")
    return value


class WorkspaceViewCreateInput(BaseModel):
    """Create one durable checkout/worktree/snapshot identity."""

    model_config = ConfigDict(extra="forbid")

    view_id: str = Field(description="Stable logical view identifier within the workspace")
    kind: WorkspaceViewKind
    display_name: str | None = None
    memory_context_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(WORKSPACE_VIEW_SCHEMA_VERSION, ge=1)

    @field_validator("view_id")
    @classmethod
    def _validate_view_id(cls, value: str) -> str:
        return _required_id(value)

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return normalize_capabilities(value)

    @model_validator(mode="after")
    def _schema_supported(self) -> WorkspaceViewCreateInput:
        if self.schema_version != WORKSPACE_VIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported workspace view schema version {self.schema_version}")
        return self


class WorkspaceViewReplaceInput(BaseModel):
    """Replace the mutable description of an existing logical view."""

    model_config = ConfigDict(extra="forbid")

    kind: WorkspaceViewKind
    display_name: str | None = None
    memory_context_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(WORKSPACE_VIEW_SCHEMA_VERSION, ge=1)

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return normalize_capabilities(value)

    @model_validator(mode="after")
    def _schema_supported(self) -> WorkspaceViewReplaceInput:
        if self.schema_version != WORKSPACE_VIEW_SCHEMA_VERSION:
            raise ValueError(f"unsupported workspace view schema version {self.schema_version}")
        return self


class WorkspaceView(BaseModel):
    """Authoritative head for one view within a logical workspace."""

    id: str = Field(description="Stable MemoryLayer resource identifier")
    tenant_id: str
    workspace_id: str
    view_id: str
    kind: WorkspaceViewKind
    display_name: str | None = None
    memory_context_id: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int
    revision: int
    sequence: int
    etag: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class WorkspaceVCSObservation(BaseModel):
    """Portable VCS facts for a view; never contains a host-local path."""

    kind: Literal["git"] = "git"
    repository_id: str | None = Field(None, description="Opaque/fingerprinted repository identity")
    worktree_id: str | None = None
    head_revision: str | None = None
    branch: str | None = None
    dirty: bool | None = None
    detached: bool | None = None


class WorkspaceViewObservationInput(BaseModel):
    """Latest state published by one observer for one workspace view."""

    model_config = ConfigDict(extra="forbid")

    observer_id: str
    generation: str = Field(description="Opaque observer-process generation")
    sequence: int = Field(ge=0, description="Monotonic cursor within the observer generation")
    tool_host_id: str | None = Field(None, description="Last tool host associated with this observation")
    execution_site: ExecutionSite | None = None
    root_ref: str | None = Field(None, description="Opaque root binding interpreted only by the tool host")
    capabilities: list[str] = Field(default_factory=list)
    vcs: WorkspaceVCSObservation | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(WORKSPACE_VIEW_OBSERVATION_SCHEMA_VERSION, ge=1)

    @field_validator("observer_id", "generation")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _required_id(value)

    @field_validator("tool_host_id")
    @classmethod
    def _validate_optional_id(cls, value: str | None) -> str | None:
        return _required_id(value) if value is not None else None

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return normalize_capabilities(value)

    @field_validator("observed_at", "expires_at")
    @classmethod
    def _timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_observation(self) -> WorkspaceViewObservationInput:
        if self.schema_version != WORKSPACE_VIEW_OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"unsupported workspace view observation schema version {self.schema_version}")
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be later than observed_at")
        if (self.tool_host_id is None) != (self.execution_site is None):
            raise ValueError("tool_host_id and execution_site must be provided together")
        return self


class WorkspaceViewObservationReplaceInput(BaseModel):
    """Replacement observation; observer identity remains path-addressed."""

    model_config = ConfigDict(extra="forbid")

    generation: str
    sequence: int = Field(ge=0)
    tool_host_id: str | None = None
    execution_site: ExecutionSite | None = None
    root_ref: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    vcs: WorkspaceVCSObservation | None = None
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = Field(WORKSPACE_VIEW_OBSERVATION_SCHEMA_VERSION, ge=1)

    @field_validator("generation")
    @classmethod
    def _validate_generation(cls, value: str) -> str:
        return _required_id(value)

    @field_validator("tool_host_id")
    @classmethod
    def _validate_optional_id(cls, value: str | None) -> str | None:
        return _required_id(value) if value is not None else None

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return normalize_capabilities(value)

    @field_validator("observed_at", "expires_at")
    @classmethod
    def _timezone_required(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value

    @model_validator(mode="after")
    def _validate_observation(self) -> WorkspaceViewObservationReplaceInput:
        WorkspaceViewObservationInput(
            observer_id="validation",
            **self.model_dump(),
        )
        return self


class WorkspaceViewObservation(BaseModel):
    """Authoritative latest observation from one source, with full revision metadata."""

    id: str
    tenant_id: str
    workspace_id: str
    view_id: str
    observer_id: str
    generation: str
    sequence: int
    tool_host_id: str | None = None
    execution_site: ExecutionSite | None = None
    root_ref: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    vcs: WorkspaceVCSObservation | None = None
    observed_at: datetime
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int
    resource_revision: int
    resource_sequence: int
    etag: str
    created_by: str | None = None
    updated_by: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
