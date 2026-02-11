"""
Session and working memory models for MemoryLayer.ai.

Sessions provide temporary, TTL-based context storage (working memory tier).
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Session(BaseModel):
    """Working memory session with automatic expiration."""

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Session ID (client-provided or generated)")
    workspace_id: str = Field(..., description="Workspace boundary")
    tenant_id: str = Field(..., description="Tenant this session belongs to")
    context_id: str = Field("_default", description="Context for this session (default: _default)")
    user_id: Optional[str] = Field(None, description="Optional user scope")

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Session metadata (client info, etc.)"
    )

    # v2 additions for session lifecycle
    auto_commit: bool = Field(True, description="Auto-commit working memory on session end")
    committed_at: Optional[datetime] = Field(None, description="When session was committed to long-term memory")

    # Lifecycle
    expires_at: datetime = Field(..., description="Session expiration timestamp")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    @classmethod
    def create_with_ttl(
        cls,
        session_id: str,
        workspace_id: str,
        ttl_seconds: int = 3600,
        **kwargs
    ) -> "Session":
        """Create a session with TTL in seconds."""
        return cls(
            id=session_id,
            workspace_id=workspace_id,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
            **kwargs
        )

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(timezone.utc) > self.expires_at


class WorkingMemory(BaseModel):
    """Key-value working memory within a session."""

    model_config = {"from_attributes": True}

    # Identity
    session_id: str = Field(..., description="Parent session ID")
    key: str = Field(..., description="Context key")

    # Content
    value: Any = Field(..., description="Context value (JSON-serializable)")

    # Lifecycle
    ttl_seconds: Optional[int] = Field(
        None,
        description="Optional TTL override (inherits session TTL if None)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )

    @field_validator("key")
    @classmethod
    def key_not_empty(cls, v: str) -> str:
        """Validate that key is not empty."""
        if not v or not v.strip():
            raise ValueError("Context key cannot be empty")
        return v.strip()




class SessionBriefing(BaseModel):
    """Session briefing summarizing recent activity and context."""

    workspace_summary: dict[str, Any] = Field(
        ...,
        description="Workspace-level summary (total memories, recent activity, etc.)"
    )
    recent_activity: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent sessions/activity summaries"
    )
    open_threads: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ongoing topics/threads"
    )
    contradictions_detected: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Memories with contradictory relationships"
    )
    memories: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant memories for this session (v2 addition)"
    )


class WorkspaceSummary(BaseModel):
    """Workspace-level summary for briefings."""

    total_memories: int = Field(0, description="Total memory count")
    recent_memories: int = Field(0, description="Memories in last 24h")
    active_topics: list[str] = Field(default_factory=list, description="Active topics/tags")
    total_categories: int = Field(0, description="Total categories")
    total_associations: int = Field(0, description="Total associations")


class ActivitySummary(BaseModel):
    """Recent activity summary."""

    timestamp: datetime = Field(..., description="Activity timestamp")
    summary: str = Field(..., description="Activity description")
    memories_created: int = Field(0, description="Memories created")
    key_decisions: list[str] = Field(default_factory=list, description="Key decisions made")


class OpenThread(BaseModel):
    """Ongoing topic/thread."""

    topic: str = Field(..., description="Thread topic")
    status: str = Field(..., description="in_progress, blocked, waiting")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    key_memories: list[str] = Field(default_factory=list, description="Key memory IDs")


class Contradiction(BaseModel):
    """Detected contradiction between memories."""

    memory_a: str = Field(..., description="First memory ID")
    memory_b: str = Field(..., description="Second memory ID")
    relationship: str = Field(..., description="contradicts")
    needs_resolution: bool = Field(True, description="Whether this needs user attention")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp"
    )
