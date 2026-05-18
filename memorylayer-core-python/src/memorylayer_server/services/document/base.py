"""Document service — base interface and plugin for document ingestion."""

from abc import ABC, abstractmethod
from typing import Optional

from ...config import DEFAULT_MEMORYLAYER_DOCUMENT_PROVIDER, MEMORYLAYER_DOCUMENT_PROVIDER
from ...models.document import Document, DocumentStatus, IngestionJob
from .._constants import EXT_DOCUMENT_SERVICE, EXT_STORAGE_BACKEND, EXT_TASK_SERVICE
from .._plugin_factory import make_service_plugin_base


class DocumentService(ABC):
    """Interface for document ingestion service."""

    @abstractmethod
    async def upload_document(
        self,
        workspace_id: str,
        file_data: bytes,
        filename: str,
        document_type: Optional[str] = None,
        extraction_options: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[Document, IngestionJob]:
        """Upload and queue a document for ingestion.

        Validates file size, detects type, checks for duplicates via content hash,
        stores raw content, creates Document and IngestionJob records, and
        schedules a background processing task.

        Args:
            workspace_id: Target workspace
            file_data: Raw file bytes
            filename: Original filename (used for type detection)
            document_type: Explicit document type override (auto-detected if None)
            extraction_options: Extraction configuration overrides
            metadata: Arbitrary user-supplied metadata

        Returns:
            Tuple of (Document, IngestionJob) for the queued ingestion
        """
        pass

    @abstractmethod
    async def get_document(self, workspace_id: str, doc_id: str) -> Optional[Document]:
        """Get document by ID within a workspace."""
        pass

    @abstractmethod
    async def list_documents(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """List documents in a workspace with optional filters. Returns (documents, total_count)."""
        pass

    @abstractmethod
    async def reprocess_document(
        self,
        workspace_id: str,
        doc_id: str,
        extraction_options: Optional[dict] = None,
    ) -> tuple[Document, IngestionJob]:
        """Re-queue a document for extraction with optional new options.

        Resets document status to pending and creates a new IngestionJob.
        """
        pass

    @abstractmethod
    async def delete_document(
        self,
        workspace_id: str,
        doc_id: str,
        delete_memories: bool = False,
    ) -> bool:
        """Delete a document and optionally cascade to extracted memories."""
        pass

    @abstractmethod
    async def get_document_status(
        self, workspace_id: str, doc_id: str
    ) -> Optional[DocumentStatus]:
        """Get current processing status of a document."""
        pass

    @abstractmethod
    async def get_job(self, workspace_id: str, job_id: str) -> Optional[IngestionJob]:
        """Get ingestion job by ID."""
        pass

    @abstractmethod
    async def list_jobs(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionJob]:
        """List ingestion jobs for a workspace."""
        pass

    @abstractmethod
    async def cancel_job(
        self, workspace_id: str, job_id: str
    ) -> Optional[IngestionJob]:
        """Cancel a queued or running ingestion job."""
        pass


# noinspection PyAbstractClass
DocumentServicePluginBase = make_service_plugin_base(
    ext_name=EXT_DOCUMENT_SERVICE,
    config_key=MEMORYLAYER_DOCUMENT_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_DOCUMENT_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND, EXT_TASK_SERVICE),
)
