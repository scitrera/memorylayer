"""Document ingestion domain models for MemoryLayer OSS.

Ported from enterprise (memorylayer_saas.models.document), adapted for OSS:
- No blob storage paths (raw content stored in SQLite)
- SQLite-friendly field types
- Full page-level tracking and source attribution
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types for ingestion."""

    MARKDOWN = "markdown"
    TEXT = "text"
    CODE = "code"
    HTML = "html"
    PDF = "pdf"  # optional dep (pymupdf)
    IMAGE = "image"  # optional dep (LLM vision)
    DOCX = "docx"  # optional dep (python-docx/mammoth)
    PPTX = "pptx"  # optional dep (python-pptx)


class DocumentStatus(str, Enum):
    """Document processing lifecycle status."""

    PENDING = "pending"
    PENDING_FETCH = "pending_fetch"  # awaiting byte-fetch from data-connectors
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # some pages succeeded, some failed


class JobStatus(str, Enum):
    """Ingestion job lifecycle status."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DocumentExtractionOptions(BaseModel):
    """Options controlling how a document is parsed and extracted into memories."""

    chunking_strategy: str = Field("page", description="Chunking strategy: page, semantic, fixed")
    chunk_size: int = Field(4096, description="Max chunk size in characters")
    chunk_overlap: int = Field(200, description="Overlap between chunks in characters")
    target_context_id: str = Field("_default", description="Target context for created memories")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Default importance for memories")
    system_prompt: str | None = Field(None, description="Custom LLM transcription/extraction prompt")
    retain_original: bool = Field(True, description="Keep raw content in storage")


class Document(BaseModel):
    """Document domain model.

    Represents an ingested document within a workspace. Each document
    produces DocumentPage records and ultimately Memory records linked
    via source_document_id / source_page_id provenance fields.
    """

    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    filename: str
    document_type: DocumentType
    content_hash: str = Field(..., description="SHA-256 hash for deduplication")
    source_vfs_ref: str | None = Field(None, description="VFS reference from data-connectors (for by-reference ingestion)")
    size_bytes: int
    mime_type: str | None = None
    status: DocumentStatus = DocumentStatus.PENDING
    target_context_id: str = "_default"
    extraction_options: DocumentExtractionOptions = Field(default_factory=DocumentExtractionOptions)
    page_count: int = 0
    chunk_count: int = 0
    memory_ids: list[str] = Field(default_factory=list, description="IDs of memories created from this document")
    deduplicated_count: int = 0
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata discovered during processing")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


class DocumentPage(BaseModel):
    """Page-level representation of an ingested document.

    Provides granular tracking per page/chunk: transcript text, embeddings,
    and transcription model info. Each page links to its parent Document and
    to the Memory records created from it via source_page_id.
    """

    model_config = {"from_attributes": True}

    id: str | None = Field(None, description="Page identifier (generated on persist)")
    document_id: str = Field(..., description="Parent document identifier")
    workspace_id: str = Field(..., description="Workspace identifier")
    page_no: int = Field(..., description="Zero-indexed page number")
    transcript: str | None = Field(None, description="Extracted/transcribed text as markdown")
    embedding: list[float] | None = Field(None, description="Single-vector text embedding", exclude=True)
    multivector: list[list[float]] | None = Field(None, description="ColPali multi-vector embedding (when available)")
    transcript_model: str | None = Field(None, description="Model used for transcription")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary page metadata")
    created_at: datetime | None = Field(None, description="Creation timestamp")


class IngestionJob(BaseModel):
    """Ingestion job tracking model.

    Tracks the progress of document processing, supporting batch ingestion
    of multiple documents in a single job.
    """

    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    document_ids: list[str] = Field(default_factory=list, description="Documents in this batch")
    status: JobStatus = JobStatus.QUEUED
    progress_percent: int = Field(0, ge=0, le=100)
    documents_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list, description="Per-document error details")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DocumentChunk(BaseModel):
    """Intermediate chunk produced during document parsing.

    Used internally by content parsers; not persisted directly.
    """

    content: str
    page_no: int
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
