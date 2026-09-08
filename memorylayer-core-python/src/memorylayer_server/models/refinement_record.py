"""Append-only, evidence-backed continual-refinement record models."""

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_REFINEMENT_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")

RefinementPhase = Literal["proposal", "decision", "application", "rollback"]
RefinementScope = Literal["session", "workspace", "user", "tenant", "global"]
RefinementOutcome = Literal["proposed", "approved", "rejected", "applied", "partially_applied", "failed", "rolled_back", "no_op"]
RefinementResourceKind = Literal["prompt_note", "memory", "skill", "agent_specification"]
RefinementAction = Literal["create", "replace", "delete", "restore"]
EvidenceKind = Literal["message", "tool_result", "resource_revision", "evaluation", "user_instruction", "other"]


class RefinementEvidence(BaseModel):
    kind: EvidenceKind
    reference: str = Field(min_length=1, max_length=2048)
    description: str = Field(min_length=1, max_length=4096)
    content_hash: str | None = Field(default=None, max_length=200)


class RefinementResourceSnapshot(BaseModel):
    """Authority response captured before or after one refinement edit."""

    resource_id: str | None = Field(default=None, max_length=200)
    resource_key: str = Field(min_length=1, max_length=200)
    etag: str | None = Field(default=None, max_length=200)
    schema_version: int | None = Field(default=None, ge=1)
    content: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    deleted: bool = False


class RefinementEdit(BaseModel):
    action: RefinementAction
    resource_kind: RefinementResourceKind
    resource_key: str = Field(min_length=1, max_length=200)
    resource_id: str | None = Field(default=None, max_length=200)
    expected_etag: str | None = Field(default=None, max_length=200)
    before_etag: str | None = Field(default=None, max_length=200)
    after_etag: str | None = Field(default=None, max_length=200)
    reason: str = Field(min_length=1, max_length=4096)
    content: dict[str, Any] | None = None
    before: RefinementResourceSnapshot | None = None
    after: RefinementResourceSnapshot | None = None
    applied: bool | None = None
    error: str | None = Field(default=None, max_length=4096)


class RefinementExternalReference(BaseModel):
    system: str = Field(min_length=1, max_length=100)
    id: str = Field(min_length=1, max_length=500)
    attempt_id: str | None = Field(default=None, max_length=500)


class RefinementRecord(BaseModel):
    id: str
    tenant_id: str
    workspace_id: str
    key: str
    refinement_id: str
    phase: RefinementPhase
    trigger: str
    scope: RefinementScope
    summary: str
    rationale: str
    expected_outcome: str
    evidence: list[RefinementEvidence]
    edits: list[RefinementEdit]
    outcome: RefinementOutcome
    parent_record_id: str | None = None
    rollback_of_record_id: str | None = None
    task_ref: RefinementExternalReference | None = None
    approval_ref: RefinementExternalReference | None = None
    schema_version: int
    metadata: dict[str, Any]
    revision: int
    etag: str
    created_by: str | None = None
    created_at: datetime


class RefinementRecordCreateInput(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    refinement_id: str = Field(min_length=1, max_length=200)
    phase: RefinementPhase
    trigger: str = Field(min_length=1, max_length=200)
    scope: RefinementScope
    summary: str = Field(min_length=1, max_length=4096)
    rationale: str = Field(min_length=1, max_length=16384)
    expected_outcome: str = Field(min_length=1, max_length=4096)
    evidence: list[RefinementEvidence] = Field(min_length=1, max_length=256)
    edits: list[RefinementEdit] = Field(default_factory=list, max_length=256)
    outcome: RefinementOutcome
    parent_record_id: str | None = Field(default=None, max_length=200)
    rollback_of_record_id: str | None = Field(default=None, max_length=200)
    task_ref: RefinementExternalReference | None = None
    approval_ref: RefinementExternalReference | None = None
    schema_version: int = Field(default=1, ge=1, le=2)
    metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_id: str | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not _REFINEMENT_KEY.fullmatch(value):
            raise ValueError("key must start with an alphanumeric and contain only alphanumerics, '.', '_', '/', or '-'")
        return value

    @field_validator("refinement_id", "trigger", "summary", "rationale", "expected_outcome")
    @classmethod
    def reject_blank_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_phase_outcome(self) -> "RefinementRecordCreateInput":
        allowed = {
            "proposal": {"proposed", "no_op"},
            "decision": {"approved", "rejected"},
            "application": {"applied", "partially_applied", "failed", "no_op"},
            "rollback": {"rolled_back", "partially_applied", "failed", "no_op"},
        }
        if self.outcome not in allowed[self.phase]:
            raise ValueError(f"outcome {self.outcome!r} is invalid for phase {self.phase!r}")
        if self.outcome != "no_op" and not self.edits:
            raise ValueError("non-no-op refinement records require at least one edit")
        if self.phase == "rollback" and not self.rollback_of_record_id:
            raise ValueError("rollback records require rollback_of_record_id")
        if any(edit.action == "restore" for edit in self.edits) and self.schema_version < 2:
            raise ValueError("restore refinement edits require schema_version 2")
        return self
