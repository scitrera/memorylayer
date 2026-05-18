"""Default document ingestion service implementation.

Pipeline:
1. upload_document(): validate size, detect type, compute SHA-256, check
   duplicates via storage.find_document_by_hash(), create Document +
   IngestionJob records, store raw content, schedule "document_process" task.
2. _handle_document_process (task): parse content via ContentParser ->
   create DocumentPage records -> extract facts via ExtractionService ->
   create memories via MemoryService.remember() with source provenance ->
   update progress -> finalise job.
"""

import hashlib
import logging
from datetime import UTC, datetime

from scitrera_app_framework import Variables, get_extension, get_logger

from ...config import (
    DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
    DEFAULT_MEMORYLAYER_DOCUMENT_PROVIDER,
    MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
)
from ...models.document import (
    Document,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)
from ...models.memory import RememberInput
from ...utils import generate_id
from .._constants import (
    EXT_EXTRACTION_SERVICE,
    EXT_MEMORY_SERVICE,
    EXT_STORAGE_BACKEND,
    EXT_TASK_SERVICE,
)
from ..storage import StorageBackend
from ..tasks import TaskHandlerPlugin, TaskSchedule, TaskService
from .base import DocumentService, DocumentServicePluginBase
from .parsers import ContentChunk, detect_document_type, get_parser

DOCUMENT_PROCESS_TASK = "document_process"


class DefaultDocumentService(DocumentService):
    """Default document ingestion service backed by StorageBackend."""

    def __init__(
        self,
        storage: StorageBackend,
        task_service: TaskService,
        v: Variables,
    ):
        self.storage = storage
        self.task_service = task_service
        self.logger = get_logger(v, name="DocumentService")
        self._v = v

    @property
    def _max_file_size(self) -> int:
        return self._v.get(
            MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
            default=DEFAULT_MEMORYLAYER_DOCUMENT_MAX_FILE_SIZE,
        )

    async def upload_document(
        self,
        workspace_id: str,
        file_data: bytes,
        filename: str,
        document_type: str | None = None,
        extraction_options: dict | None = None,
        metadata: dict | None = None,
    ) -> tuple[Document, IngestionJob]:
        # 1. Validate file size
        max_size = self._max_file_size
        if len(file_data) > max_size:
            raise ValueError("File size %d bytes exceeds maximum allowed size of %d bytes" % (len(file_data), max_size))

        # 2. Detect document type
        if document_type:
            try:
                doc_type = DocumentType(document_type)
            except ValueError:
                raise ValueError("Unsupported document type: %s" % document_type)
        else:
            doc_type = detect_document_type(filename)

        # 3. Compute SHA-256 content hash
        content_hash = hashlib.sha256(file_data).hexdigest()

        # 4. Check for duplicates
        existing = await self.storage.find_document_by_hash(workspace_id, content_hash)
        if existing:
            self.logger.info(
                "Duplicate document detected for workspace %s (hash %s), returning existing doc %s",
                workspace_id,
                content_hash[:16],
                existing.id,
            )
            # Return existing doc with a new job pointing at it
            job_id = generate_id("job")
            now = datetime.now(UTC)
            job = IngestionJob(
                id=job_id,
                workspace_id=workspace_id,
                document_ids=[existing.id],
                status=JobStatus.COMPLETED,
                progress_percent=100,
                documents_processed=1,
                metadata={"deduplicated": True, "original_document_id": existing.id},
                created_at=now,
                started_at=now,
                completed_at=now,
            )
            job = await self.storage.create_job(job)
            return existing, job

        # 5. Resolve extraction options
        opts_dict = extraction_options or {}
        try:
            ext_opts = DocumentExtractionOptions(**opts_dict)
        except Exception as e:
            raise ValueError("Invalid extraction options: %s" % e)

        # 6. Create Document record
        doc_id = generate_id("doc")
        now = datetime.now(UTC)
        doc = Document(
            id=doc_id,
            workspace_id=workspace_id,
            filename=filename,
            document_type=doc_type,
            content_hash=content_hash,
            size_bytes=len(file_data),
            status=DocumentStatus.PENDING,
            target_context_id=ext_opts.target_context_id,
            extraction_options=ext_opts,
            metadata=metadata or {},
            created_at=now,
        )
        doc = await self.storage.create_document(workspace_id, doc)

        # 7. Create IngestionJob record
        job_id = generate_id("job")
        job = IngestionJob(
            id=job_id,
            workspace_id=workspace_id,
            document_ids=[doc_id],
            status=JobStatus.QUEUED,
            metadata={},
            created_at=now,
        )
        job = await self.storage.create_job(job)

        # 8. Store raw content (best-effort; failures logged, not fatal)
        try:
            if ext_opts.retain_original:
                import base64

                await self.storage.update_document(
                    workspace_id,
                    doc_id,
                    metadata={
                        **(metadata or {}),
                        "_raw_content_b64": base64.b64encode(file_data).decode(),
                    },
                )
        except Exception as e:
            self.logger.warning("Failed to store raw content for document %s: %s", doc_id, e)

        # 9. Schedule processing task
        try:
            await self.task_service.schedule_task(
                DOCUMENT_PROCESS_TASK,
                {
                    "workspace_id": workspace_id,
                    "document_id": doc_id,
                    "job_id": job_id,
                    "file_data_b64": __import__("base64").b64encode(file_data).decode(),
                },
            )
            self.logger.info(
                "Scheduled %s task for document %s in workspace %s",
                DOCUMENT_PROCESS_TASK,
                doc_id,
                workspace_id,
            )
        except Exception as e:
            self.logger.error("Failed to schedule processing task for document %s: %s", doc_id, e)
            await self.storage.update_document(workspace_id, doc_id, status=DocumentStatus.FAILED, error_message=str(e))
            await self.storage.update_job(job_id, status=JobStatus.FAILED)

        return doc, job

    async def get_document(self, workspace_id: str, doc_id: str) -> Document | None:
        return await self.storage.get_document(workspace_id, doc_id)

    async def list_documents(
        self,
        workspace_id: str,
        status: str | None = None,
        document_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        return await self.storage.list_documents(
            workspace_id=workspace_id,
            status=status,
            document_type=document_type,
            limit=limit,
            offset=offset,
        )

    async def reprocess_document(
        self,
        workspace_id: str,
        doc_id: str,
        extraction_options: dict | None = None,
    ) -> tuple[Document, IngestionJob]:
        doc = await self.storage.get_document(workspace_id, doc_id)
        if not doc:
            raise ValueError("Document not found: %s" % doc_id)

        # Apply new extraction options if provided
        if extraction_options:
            try:
                ext_opts = DocumentExtractionOptions(**extraction_options)
            except Exception as e:
                raise ValueError("Invalid extraction options: %s" % e)
        else:
            ext_opts = doc.extraction_options

        now = datetime.now(UTC)
        doc = await self.storage.update_document(
            workspace_id,
            doc_id,
            status=DocumentStatus.PENDING,
            extraction_options=ext_opts,
            processing_started_at=None,
            processing_completed_at=None,
            error_message=None,
            page_count=0,
            chunk_count=0,
            memory_ids=[],
        )

        # Create new job
        job_id = generate_id("job")
        job = IngestionJob(
            id=job_id,
            workspace_id=workspace_id,
            document_ids=[doc_id],
            status=JobStatus.QUEUED,
            metadata={"reprocess": True},
            created_at=now,
        )
        job = await self.storage.create_job(job)

        # Re-fetch raw content from stored metadata if available
        raw_b64 = (doc.metadata or {}).get("_raw_content_b64", "")

        try:
            await self.task_service.schedule_task(
                DOCUMENT_PROCESS_TASK,
                {
                    "workspace_id": workspace_id,
                    "document_id": doc_id,
                    "job_id": job_id,
                    "file_data_b64": raw_b64,
                },
            )
            self.logger.info("Scheduled reprocess task for document %s", doc_id)
        except Exception as e:
            self.logger.error("Failed to schedule reprocess task for document %s: %s", doc_id, e)
            await self.storage.update_document(workspace_id, doc_id, status=DocumentStatus.FAILED, error_message=str(e))
            await self.storage.update_job(job_id, status=JobStatus.FAILED)

        return doc, job

    async def delete_document(
        self,
        workspace_id: str,
        doc_id: str,
        delete_memories: bool = False,
    ) -> bool:
        return await self.storage.delete_document(workspace_id, doc_id, delete_memories=delete_memories)

    async def get_document_status(self, workspace_id: str, doc_id: str) -> DocumentStatus | None:
        doc = await self.storage.get_document(workspace_id, doc_id)
        return doc.status if doc else None

    async def get_job(self, workspace_id: str, job_id: str) -> IngestionJob | None:
        job = await self.storage.get_job(job_id, workspace_id=workspace_id)
        if job and job.workspace_id != workspace_id:
            return None
        return job

    async def list_jobs(
        self,
        workspace_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[IngestionJob]:
        return await self.storage.list_jobs(
            workspace_id=workspace_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    async def cancel_job(self, workspace_id: str, job_id: str) -> IngestionJob | None:
        job = await self.get_job(workspace_id, job_id)
        if not job:
            return None

        if job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            self.logger.debug("Job %s is in terminal state %s, cannot cancel", job_id, job.status)
            return job

        updated = await self.storage.update_job(job_id, status=JobStatus.CANCELLED, completed_at=datetime.now(UTC))

        # Best-effort task cancellation (asyncio backend may not support it)
        try:
            await self.task_service.cancel_task(job_id)
        except Exception as e:
            self.logger.debug("Could not cancel background task for job %s: %s", job_id, e)

        self.logger.info("Cancelled ingestion job %s in workspace %s", job_id, workspace_id)
        return updated


class DefaultDocumentServicePlugin(DocumentServicePluginBase):
    """Plugin for the default document ingestion service."""

    PROVIDER_NAME = DEFAULT_MEMORYLAYER_DOCUMENT_PROVIDER

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        task_service: TaskService = get_extension(EXT_TASK_SERVICE, v)
        return DefaultDocumentService(storage=storage, task_service=task_service, v=v)


# ---------------------------------------------------------------------------
# Background task handler
# ---------------------------------------------------------------------------


class DocumentProcessTaskHandler(TaskHandlerPlugin):
    """Background task handler that processes an uploaded document.

    Pipeline per document:
    1. Decode raw file bytes
    2. Parse into ContentChunks via the appropriate ContentParser
    3. Persist each chunk as a DocumentPage
    4. Run ExtractionService to get fact strings from the chunk text
    5. Store each fact as a Memory via MemoryService.remember()
    6. Update Document + IngestionJob with progress and final status
    """

    def get_task_type(self) -> str:
        return DOCUMENT_PROCESS_TASK

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        return None  # On-demand only

    async def handle(self, v: Variables, payload: dict) -> None:

        logger: logging.Logger = get_logger(v, name=DOCUMENT_PROCESS_TASK)

        workspace_id: str = payload.get("workspace_id", "")
        document_id: str = payload.get("document_id", "")
        job_id: str = payload.get("job_id", "")
        file_data_b64: str = payload.get("file_data_b64", "")

        if not workspace_id or not document_id or not job_id:
            logger.warning(
                "Missing required payload fields for %s: workspace_id=%s, document_id=%s, job_id=%s",
                DOCUMENT_PROCESS_TASK,
                workspace_id,
                document_id,
                job_id,
            )
            return

        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)

        # Fetch document record
        doc = await storage.get_document(workspace_id, document_id)
        if not doc:
            logger.warning("Document %s not found in workspace %s", document_id, workspace_id)
            return

        # Check job is not cancelled
        job = await storage.get_job(job_id)
        if not job or job.status == JobStatus.CANCELLED:
            logger.info("Job %s is cancelled or missing, skipping processing", job_id)
            return

        # Mark as running
        now = datetime.now(UTC)
        await storage.update_document(workspace_id, document_id, status=DocumentStatus.PROCESSING, processing_started_at=now)
        await storage.update_job(job_id, status=JobStatus.RUNNING, started_at=now)

        try:
            await self._process_document(v, logger, storage, doc, job, file_data_b64)
        except Exception as e:
            logger.error(
                "Unhandled error processing document %s in workspace %s: %s",
                document_id,
                workspace_id,
                e,
                exc_info=True,
            )
            await storage.update_document(
                workspace_id,
                document_id,
                status=DocumentStatus.FAILED,
                error_message=str(e),
                processing_completed_at=datetime.now(UTC),
            )
            await storage.update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(UTC),
                errors=[{"document_id": document_id, "error": str(e)}],
            )

    async def _process_document(
        self,
        v: Variables,
        logger: logging.Logger,
        storage: StorageBackend,
        doc: Document,
        job: IngestionJob,
        file_data_b64: str,
    ) -> None:
        import base64

        workspace_id = doc.workspace_id
        document_id = doc.id
        job_id = job.id
        ext_opts = doc.extraction_options

        # Decode raw content
        if file_data_b64:
            file_data = base64.b64decode(file_data_b64)
        else:
            # Try to recover from stored metadata
            raw_b64 = (doc.metadata or {}).get("_raw_content_b64", "")
            if raw_b64:
                file_data = base64.b64decode(raw_b64)
            else:
                logger.warning("No file data available for document %s, cannot process", document_id)
                await storage.update_document(
                    workspace_id,
                    document_id,
                    status=DocumentStatus.FAILED,
                    error_message="No file data available for processing",
                    processing_completed_at=datetime.now(UTC),
                )
                await storage.update_job(job_id, status=JobStatus.FAILED, completed_at=datetime.now(UTC))
                return

        # Parse content into chunks
        parser = get_parser(doc.document_type)
        try:
            chunks: list[ContentChunk] = await parser.parse(file_data, doc.filename, ext_opts)
        except Exception as e:
            logger.error("Parser failed for document %s: %s", document_id, e)
            chunks = []

        if not chunks:
            logger.warning("No content extracted from document %s", document_id)
            await storage.update_document(
                workspace_id,
                document_id,
                status=DocumentStatus.FAILED,
                error_message="No content could be extracted from document",
                processing_completed_at=datetime.now(UTC),
            )
            await storage.update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now(UTC),
                errors=[{"document_id": document_id, "error": "No content extracted"}],
            )
            return

        logger.info("Parsed document %s into %d chunks", document_id, len(chunks))

        # Persist chunks as DocumentPage records
        pages: list[DocumentPage] = []
        for chunk in chunks:
            page = DocumentPage(
                document_id=document_id,
                workspace_id=workspace_id,
                page_no=chunk.page_number if chunk.page_number is not None else len(pages),
                transcript=chunk.text,
                metadata=chunk.metadata,
                created_at=datetime.now(UTC),
            )
            try:
                page = await storage.create_page(workspace_id, document_id, page)
                pages.append(page)
            except Exception as e:
                logger.warning("Failed to persist page %d for document %s: %s", len(pages), document_id, e)

        await storage.update_document(workspace_id, document_id, page_count=len(pages))

        # Extract facts and create memories
        memory_ids: list[str] = []
        total_memories = 0
        errors: list[dict] = []

        # Optional services — failures degrade gracefully
        memory_service = None
        extraction_service = None
        try:
            memory_service = self.get_extension(EXT_MEMORY_SERVICE, v)
        except Exception as e:
            logger.warning("MemoryService not available for document processing: %s", e)

        try:
            extraction_service = self.get_extension(EXT_EXTRACTION_SERVICE, v)
        except Exception as e:
            logger.debug("ExtractionService not available: %s", e)

        for page_idx, (chunk, page) in enumerate(zip(chunks, pages)):
            # Update progress
            progress = int((page_idx + 1) / len(chunks) * 90)  # reserve last 10% for finalization
            await storage.update_job(job_id, progress_percent=progress)

            page_id = page.id
            content = chunk.text.strip()
            if not content:
                continue

            if memory_service is not None:
                try:
                    facts = [content]
                    if extraction_service is not None:
                        try:
                            fact_list = await extraction_service.decompose_to_facts(content)
                            if fact_list:
                                facts = [f["content"] for f in fact_list if f.get("content")]
                        except Exception as e:
                            logger.debug("Fact decomposition failed for page %d: %s", page_idx, e)

                    for fact in facts:
                        if not fact.strip():
                            continue
                        try:
                            remember_input = RememberInput(
                                content=fact,
                                importance=ext_opts.importance,
                                context_id=ext_opts.target_context_id,
                                source_document_id=document_id,
                                source_page_id=page_id,
                                metadata={
                                    "source": "document_ingestion",
                                    "document_id": document_id,
                                    "filename": doc.filename,
                                    "document_type": doc.document_type.value,
                                    "page_no": chunk.page_number,
                                },
                            )
                            memory = await memory_service.remember(workspace_id, remember_input)
                            memory_ids.append(memory.id)
                            total_memories += 1
                        except Exception as e:
                            logger.warning(
                                "Failed to store memory for page %d of document %s: %s",
                                page_idx,
                                document_id,
                                e,
                            )
                            errors.append({"page": page_idx, "error": str(e)})
                except Exception as e:
                    logger.error(
                        "Unexpected error processing page %d of document %s: %s",
                        page_idx,
                        document_id,
                        e,
                    )
                    errors.append({"page": page_idx, "error": str(e)})

        # Finalize document + job
        final_status = DocumentStatus.COMPLETED if not errors else DocumentStatus.PARTIAL
        if errors and not memory_ids:
            final_status = DocumentStatus.FAILED

        completed_at = datetime.now(UTC)
        await storage.update_document(
            workspace_id,
            document_id,
            status=final_status,
            page_count=len(pages),
            chunk_count=len(pages),
            memory_ids=memory_ids,
            processing_completed_at=completed_at,
        )
        await storage.update_job(
            job_id,
            status=JobStatus.COMPLETED if final_status != DocumentStatus.FAILED else JobStatus.FAILED,
            progress_percent=100,
            documents_processed=1,
            total_memories_created=total_memories,
            completed_at=completed_at,
            errors=errors,
        )

        logger.info(
            "Finished processing document %s: %d pages, %d memories created, %d errors",
            document_id,
            len(pages),
            total_memories,
            len(errors),
        )
