"""Generation policy, authorization, accounting, and response contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class EnrichmentPolicy(str, Enum):
    """Server or request policy controlling generative model work."""

    DETERMINISTIC = "deterministic"
    ADAPTIVE = "adaptive"
    GENERATIVE = "generative"


class GenerationActivity(str, Enum):
    """Stable, low-cardinality purpose attached to every generative call."""

    FACT_DECOMPOSITION = "fact_decomposition"
    SESSION_EXTRACTION = "session_extraction"
    SEMANTIC_TIERING = "semantic_tiering"
    MEMORY_CLASSIFICATION = "memory_classification"
    RELATIONSHIP_CLASSIFICATION = "relationship_classification"
    CONTRADICTION_DETECTION = "contradiction_detection"
    QUERY_REWRITING = "query_rewriting"
    RERANKING = "reranking"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"


class GenerationAuthorization(BaseModel):
    """Authorization and budget for one logical generative operation."""

    policy: EnrichmentPolicy
    operation_id: str = Field(..., min_length=1)
    workspace_id: str = Field(..., min_length=1)
    activity: GenerationActivity
    max_calls: int = Field(1, ge=0)
    max_input_tokens: int | None = Field(None, ge=0)
    max_output_tokens: int | None = Field(None, ge=0)


class GenerationLedger(BaseModel):
    """Auditable counters for one logical operation."""

    operation_id: str
    workspace_id: str
    activity: GenerationActivity
    policy: EnrichmentPolicy
    attempted: int = 0
    allowed: int = 0
    rejected: int = 0
    completed: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    provider: str | None = None
    profile: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GenerationSummary(BaseModel):
    """Generation cost returned by APIs that perform or trace generation."""

    policy: EnrichmentPolicy
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class GenerationNotAllowedError(PermissionError):
    """Typed error raised before provider selection when policy rejects work."""

    code = "generation_not_allowed"

    def __init__(self, activity: GenerationActivity, policy: EnrichmentPolicy, reason: str):
        self.activity = activity
        self.policy = policy
        self.reason = reason
        super().__init__(f"{self.code}: {activity.value} is not allowed under {policy.value}: {reason}")


class GenerationBudgetExceededError(RuntimeError):
    """Typed error raised before provider selection when a budget is exhausted."""

    code = "generation_budget_exceeded"

    def __init__(self, activity: GenerationActivity, reason: str):
        self.activity = activity
        self.reason = reason
        super().__init__(f"{self.code}: {activity.value}: {reason}")
