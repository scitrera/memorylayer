"""Pydantic models for MemoryLayer.ai SDK."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .types import MemorySubtype, MemoryType


class Memory(BaseModel):
    """A memory entry."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    workspace_id: str
    space_id: str | None = None
    user_id: str | None = None
    content: str
    type: MemoryType
    subtype: MemorySubtype | None = None
    importance: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    access_count: int = 0
    last_accessed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class Association(BaseModel):
    """A relationship between two memories."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    workspace_id: str
    source_id: str
    target_id: str
    relationship: str
    strength: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RecallResult(BaseModel):
    """Result from a recall (search) query."""

    memories: list[Memory]
    total_count: int
    query_tokens: int | None = None
    search_latency_ms: int | None = None


class ReflectResult(BaseModel):
    """Result from a reflect (synthesis) query."""

    reflection: str
    source_memories: list[str] = Field(default_factory=list)  # Memory IDs
    confidence: float = Field(ge=0.0, le=1.0)
    tokens_processed: int | None = None


class Session(BaseModel):
    """A working memory session."""

    id: str
    workspace_id: str
    user_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime
    created_at: datetime


class SessionActivity(BaseModel):
    """Recent session activity summary."""

    timestamp: datetime
    summary: str
    memories_created: int
    key_decisions: list[str] = Field(default_factory=list)


class OpenThread(BaseModel):
    """An open topic/thread of work."""

    topic: str
    status: str
    last_activity: datetime
    key_memories: list[str] = Field(default_factory=list)


class ContradictionDetection(BaseModel):
    """Detected contradiction between memories."""

    memory_a: str
    memory_b: str
    relationship: str
    needs_resolution: bool


class SessionBriefing(BaseModel):
    """Session briefing with recent activity and context."""

    workspace_summary: dict[str, Any]
    recent_activity: list[SessionActivity] = Field(default_factory=list)
    open_threads: list[OpenThread] = Field(default_factory=list)
    contradictions_detected: list[ContradictionDetection] = Field(default_factory=list)


class Workspace(BaseModel):
    """A workspace (tenant boundary)."""

    id: str
    tenant_id: str
    name: str
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class MemorySpace(BaseModel):
    """A memory space within a workspace."""

    id: str
    workspace_id: str
    name: str
    description: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
