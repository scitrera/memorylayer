"""Typed structural entity-relation and evidence contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RelationEvidenceKind(str, Enum):
    EXPLICIT = "explicit"
    ADAPTER = "adapter"
    PATTERN = "pattern"
    GENERATED = "generated"


class EntityRelationInput(BaseModel):
    source_entity_id: str | None = None
    source_entity_name: str | None = None
    target_entity_id: str | None = None
    target_entity_name: str | None = None
    relationship: str = Field(..., min_length=1, max_length=128)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source_span_start: int | None = Field(None, ge=0)
    source_span_end: int | None = Field(None, ge=0)
    evidence_kind: RelationEvidenceKind = RelationEvidenceKind.EXPLICIT
    extraction_method: str = Field("explicit_api", min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_endpoints_and_span(self) -> EntityRelationInput:
        if not self.source_entity_id and not self.source_entity_name:
            raise ValueError("source_entity_id or source_entity_name is required")
        if not self.target_entity_id and not self.target_entity_name:
            raise ValueError("target_entity_id or target_entity_name is required")
        if self.source_span_end is not None:
            if self.source_span_start is None or self.source_span_end < self.source_span_start:
                raise ValueError("source_span_end requires source_span_start and must not precede it")
        return self


class EntityRelation(BaseModel):
    id: str
    workspace_id: str
    source_entity_id: str
    target_entity_id: str
    relationship: str
    direction: Literal["outgoing"] = "outgoing"
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EntityRelationEvidence(BaseModel):
    id: str
    workspace_id: str
    relation_id: str
    source_memory_id: str
    evidence_kind: RelationEvidenceKind
    source_span_start: int | None = None
    source_span_end: int | None = None
    excerpt_hash: str
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    extraction_method: str
    active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EntityRelationPath(BaseModel):
    seed_entity_id: str
    entity_ids: list[str]
    relations: list[EntityRelation]
    evidence_ids: list[str]
    evidence_memory_ids: list[str]


class EntityRelationWriteResult(BaseModel):
    resolved: int = 0
    unresolved: int = 0
    rejected: int = 0
    duplicate: int = 0
    relation_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class RelationIntent(BaseModel):
    is_relation_query: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    seed_phrases: list[str] = Field(default_factory=list)
    relationship_types: list[str] = Field(default_factory=list)
    direction: Literal["outgoing", "incoming", "both"] = "both"
    max_hops: int = Field(1, ge=1, le=2)
