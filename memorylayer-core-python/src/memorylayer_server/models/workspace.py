"""
Workspace and organizational hierarchy models for MemoryLayer.ai v2.

Defines multi-tenant workspace isolation, contexts, and scope boosts.

Hierarchy: Tenant -> Workspace -> Context -> Session -> WorkingMemory
"""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class ScopeBoosts(BaseModel):
    """Locality ranking boosts for memory retrieval."""

    session: float = Field(1.5, ge=0.0, le=10.0, description="Boost for session-local memories")
    context: float = Field(1.3, ge=0.0, le=10.0, description="Boost for context-local memories")
    workspace: float = Field(1.0, ge=0.0, le=10.0, description="Boost for workspace-local memories")
    global_: float = Field(0.8, ge=0.0, le=10.0, description="Boost for global workspace memories")


class WorkspaceSettings(BaseModel):
    """Typed workspace settings schema."""

    # Retention
    default_importance: float = Field(0.5, ge=0.0, le=1.0, description="Default memory importance")
    decay_enabled: bool = Field(True, description="Enable memory decay over time")
    decay_rate: float = Field(0.01, ge=0.0, le=1.0, description="Daily decay rate")

    # Auto-remember
    auto_remember_enabled: bool = Field(False, description="Auto-capture significant interactions")
    auto_remember_min_importance: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Minimum importance for auto-capture"
    )
    auto_remember_exclude_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns to exclude from auto-capture"
    )

    # Embeddings
    embedding_model: str = Field("text-embedding-3-small", description="Embedding model to use")
    embedding_dimensions: int = Field(1536, description="Embedding vector dimensions")

    # Storage tiers
    hot_tier_days: int = Field(7, ge=1, description="Days to keep in hot tier (Redis)")
    warm_tier_days: int = Field(90, ge=1, description="Days to keep in warm tier before archival")
    enable_cold_tier: bool = Field(False, description="Enable LEANN cold tier for old memories")

    # Session behavior (v2)
    session_auto_commit: bool = Field(True, description="Auto-commit sessions on close")

    # Scope boosts for locality ranking (v2)
    scope_boosts: ScopeBoosts = Field(default_factory=ScopeBoosts, description="Locality ranking boosts")


class Workspace(BaseModel):
    """Tenant-level workspace for memory isolation."""

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Unique workspace identifier")
    tenant_id: str = Field("_default", description="Tenant/organization ID")
    name: str = Field(..., description="Human-readable workspace name")

    # Configuration
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace-level settings (retention, auto-remember, etc.)"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Last update timestamp")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Workspace name cannot be empty")
        return v.strip()


class ContextSettings(BaseModel):
    """Typed context settings schema (formerly SpaceSettings)."""

    # Inheritance
    inherit_workspace_settings: bool = Field(True, description="Inherit workspace settings")

    # Overrides (only apply if inherit_workspace_settings=False)
    auto_remember_enabled: Optional[bool] = None
    decay_enabled: Optional[bool] = None

    # Context-specific settings (v2)
    default_importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override default importance")
    session_auto_commit: Optional[bool] = Field(None, description="Override session auto-commit")


class Context(BaseModel):
    """Logical grouping within a workspace (e.g., project, topic, conversation context).

    Formerly known as MemorySpace. Renamed to Context in v2 to better reflect
    its role as a contextual boundary for memories.
    """

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Unique context identifier")
    workspace_id: str = Field(..., description="Parent workspace ID")
    name: str = Field(..., description="Context name (unique within workspace)")
    description: Optional[str] = Field(None, description="Context description")

    # Configuration
    settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Context-level settings (overrides workspace defaults)"
    )

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        """Validate that name is not empty."""
        if not v or not v.strip():
            raise ValueError("Context name cannot be empty")
        return v.strip()


