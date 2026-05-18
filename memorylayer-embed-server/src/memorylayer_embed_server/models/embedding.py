"""Pydantic models for embedding APIs."""

from pydantic import BaseModel, Field


# ============================================
# OpenAI-Compatible Single-Vector Models
# ============================================

class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    input: str | list[str] = Field(..., description="Text(s) to embed")
    model: str | None = Field(None, description="Model identifier (informational)")
    encoding_format: str = Field("float", description="Encoding format")


class EmbeddingData(BaseModel):
    """Single embedding result."""
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Token usage for embedding request."""
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage = Field(default_factory=EmbeddingUsage)


# ============================================
# Multi-Vector (ColPali) Models
# ============================================

class MultiVectorEmbeddingRequest(BaseModel):
    """Request for multi-vector embeddings."""
    input: str | list[str] = Field(..., description="Text(s) to embed")
    input_type: str = Field("query", description="Input type: 'query' or 'document'")


class MultiVectorData(BaseModel):
    """Single multi-vector embedding result."""
    index: int
    vectors: list[list[float]]
    num_vectors: int


class MultiVectorEmbeddingResponse(BaseModel):
    """Response with multi-vector embeddings."""
    data: list[MultiVectorData]
    model: str
    dimensions: int


# ============================================
# Image Embedding Models
# ============================================

class ImageEmbeddingRequest(BaseModel):
    """Request for image embeddings (single or multi-vector)."""
    images: list[str] = Field(..., description="List of base64-encoded images")
    mode: str = Field("single", description="Embedding mode: 'single' or 'multi'")


# ============================================
# Score Models
# ============================================

class ScoreRequest(BaseModel):
    """Request for MaxSim scoring between query and document vectors."""
    query_vectors: list[list[float]] = Field(..., description="Query multi-vectors")
    document_vectors: list[list[list[float]]] = Field(..., description="List of document multi-vectors")


class ScoreResult(BaseModel):
    """Score result for a single document."""
    index: int
    score: float


class ScoreResponse(BaseModel):
    """Response from scoring endpoint."""
    scores: list[ScoreResult]
