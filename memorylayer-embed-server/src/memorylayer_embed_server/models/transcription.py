"""Pydantic models for transcription API."""

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request to transcribe page images to markdown."""

    images: list[str] = Field(..., description="List of base64-encoded page images")
    system_prompt: str | None = Field(None, description="Optional custom system prompt override")
    max_tokens: int | None = Field(None, description="Optional max tokens override")


class ModelAttempt(BaseModel):
    """Record of a single model attempt during cascade."""

    model: str
    provider: str
    success: bool
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "unknown"
    error: str | None = None


class TranscriptionResult(BaseModel):
    """Result for a single page transcription."""

    page_index: int
    content: str
    success: bool = True
    model_used: str | None = None
    provider_used: str | None = None
    attempts: list[ModelAttempt] = Field(default_factory=list)


class TranscriptionStats(BaseModel):
    """Aggregate statistics for a transcription request."""

    total_pages: int
    successful_pages: int
    failed_pages: int
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""

    results: list[TranscriptionResult]
    stats: TranscriptionStats
