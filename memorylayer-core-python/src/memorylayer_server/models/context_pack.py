"""Checkpoint, context-pack, delta, cursor, and token-budget contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .generation import GenerationSummary


class CheckpointCaptureStatus(str, Enum):
    DURABLE = "durable"
    FAILED = "failed"


class CheckpointWorkStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class SessionCheckpointInput(BaseModel):
    transcript_segment: str = Field(..., min_length=1)
    source_kind: str = Field("transcript", min_length=1, max_length=64)
    source_sequence: int | None = Field(None, ge=0)
    source_boundary: int | None = Field(None, ge=0)
    content_hash: str = Field(..., min_length=64, max_length=64)
    idempotency_key: str = Field(..., min_length=1, max_length=255)

    @field_validator("content_hash")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("content_hash must be a hexadecimal SHA-256 digest")
        return normalized


class SessionCheckpoint(BaseModel):
    id: str
    workspace_id: str
    session_id: str
    raw_memory_id: str
    source_kind: str
    source_sequence: int | None = None
    source_boundary: int | None = None
    content_hash: str
    byte_count: int
    capture_status: CheckpointCaptureStatus = CheckpointCaptureStatus.DURABLE
    index_status: CheckpointWorkStatus = CheckpointWorkStatus.PENDING
    enrichment_status: CheckpointWorkStatus = CheckpointWorkStatus.SKIPPED
    idempotency_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ContextEventKind(str, Enum):
    MEMORY_UPSERT = "memory_upsert"
    MEMORY_DELETE = "memory_delete"
    WORKING_UPSERT = "working_upsert"
    WORKING_DELETE = "working_delete"
    CHECKPOINT = "checkpoint"
    COMMIT = "commit"
    SANDBOX_CHECKPOINT = "sandbox_checkpoint"


class SessionContextEvent(BaseModel):
    sequence: int
    workspace_id: str
    session_id: str | None = None
    event_kind: ContextEventKind
    subject_kind: str
    subject_id: str
    event_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPackSections(BaseModel):
    directives: int = Field(8, ge=0, le=100)
    working_memory: int = Field(20, ge=0, le=200)
    recent_activity: int = Field(20, ge=0, le=200)
    contradictions: int = Field(5, ge=0, le=100)
    checkpoints: int = Field(8, ge=0, le=100)


class ContextPackInput(BaseModel):
    topic: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    budget_tokens: int = Field(2048, ge=32, le=200_000)
    section_limits: ContextPackSections = Field(default_factory=ContextPackSections)
    include_directives: bool = True
    include_working_memory: bool = True
    include_recent_activity: bool = True
    include_contradictions: bool = True
    include_sandbox_summary: bool = True
    include_checkpoint_recovery: bool = True


class ContextDeltaInput(BaseModel):
    cursor: str = Field(..., min_length=1)
    budget_tokens: int = Field(2048, ge=32, le=200_000)


class ContextPackItem(BaseModel):
    id: str
    kind: str
    content: str
    importance: float = Field(0.5, ge=0.0, le=1.0)
    event_time: datetime | None = None
    match_signals: list[str] = Field(default_factory=list)
    source_references: list[str] = Field(default_factory=list)
    tombstone: bool = False
    alternate_contents: list[str] = Field(default_factory=list, exclude=True)


class BudgetSummary(BaseModel):
    requested: int | None
    used: int
    estimator: str
    truncated_items: int = 0
    omitted_items: int = 0


class ContextPack(BaseModel):
    rendered: str
    items: list[ContextPackItem] = Field(default_factory=list)
    open_threads: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_contradictions: list[dict[str, Any]] = Field(default_factory=list)
    budget_summary: BudgetSummary
    generation_summary: GenerationSummary
    cursor: str
    degradation_notices: list[str] = Field(default_factory=list)


class ContextDelta(BaseModel):
    rendered: str
    items: list[ContextPackItem] = Field(default_factory=list)
    budget_summary: BudgetSummary
    generation_summary: GenerationSummary
    cursor: str
    has_more: bool = False
    degradation_notices: list[str] = Field(default_factory=list)


class CursorExpiredError(ValueError):
    code = "cursor_expired"

    def __init__(self, message: str = "The cursor is outside the retained event window; request a full context pack"):
        super().__init__(message)


ContextSubjectKind = Literal["memory", "working_memory", "checkpoint", "session", "sandbox"]
