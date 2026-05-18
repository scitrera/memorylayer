"""Document ingestion API endpoints.

Endpoints:
- POST   /v1/documents                            - Upload document (multipart/form-data)
- GET    /v1/documents                            - List documents (workspace filter, status filter)
- GET    /v1/documents/jobs                       - List ingestion jobs
- GET    /v1/documents/jobs/{job_id}              - Get ingestion job
- POST   /v1/documents/jobs/{job_id}/cancel       - Cancel ingestion job
- POST   /v1/documents/search                     - ColPali MaxSim page search
- GET    /v1/documents/{document_id}              - Get document
- GET    /v1/documents/{document_id}/memories     - Get memories extracted from document
- GET    /v1/documents/{document_id}/pages        - Get document pages
- GET    /v1/documents/{document_id}/pages/{page_id} - Get a single page
- POST   /v1/documents/{document_id}/reprocess    - Re-extract document
- DELETE /v1/documents/{document_id}              - Delete document
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field
from scitrera_app_framework import Plugin, Variables, get_extension

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.document import Document, DocumentPage, IngestionJob
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.document import EXT_DOCUMENT_SERVICE, DocumentService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_auth_service, get_authz_service
from .schemas import ErrorResponse

router = APIRouter(prefix="/v1/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_document_service(v: Variables = Depends(get_variables_dep)) -> DocumentService:
    """FastAPI dependency wrapper for document service."""
    return get_extension(EXT_DOCUMENT_SERVICE, v)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Response model for a single document."""

    id: str
    workspace_id: str
    filename: str
    document_type: str
    content_hash: str
    size_bytes: int
    mime_type: Optional[str] = None
    status: str
    target_context_id: str
    page_count: int
    chunk_count: int
    memory_ids: list[str]
    deduplicated_count: int
    error_message: Optional[str] = None
    metadata: dict[str, Any]
    created_at: Any
    processing_started_at: Optional[Any] = None
    processing_completed_at: Optional[Any] = None


class DocumentListResponse(BaseModel):
    """Response model for a list of documents."""

    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class IngestionJobResponse(BaseModel):
    """Response model for an ingestion job."""

    id: str
    workspace_id: str
    document_ids: list[str]
    status: str
    progress_percent: int
    documents_processed: int
    total_memories_created: int
    errors: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: Any
    started_at: Optional[Any] = None
    completed_at: Optional[Any] = None


class IngestionJobListResponse(BaseModel):
    """Response model for a list of ingestion jobs."""

    jobs: list[IngestionJobResponse]
    total: int
    limit: int
    offset: int


class DocumentPageResponse(BaseModel):
    """Response model for a document page."""

    id: Optional[str] = None
    document_id: str
    workspace_id: str
    page_no: int
    transcript: Optional[str] = None
    transcript_model: Optional[str] = None
    metadata: dict[str, Any]
    created_at: Optional[Any] = None
    relevance_score: Optional[float] = None


class DocumentPageListResponse(BaseModel):
    """Response model for a list of document pages."""

    pages: list[DocumentPageResponse]
    total: int


class PageSearchRequest(BaseModel):
    """Request body for document page search."""

    query: str = Field(..., description="Natural language search query")
    limit: int = Field(10, ge=1, le=100, description="Maximum results to return")
    doc_ids: Optional[list[str]] = Field(
        None, description="Optional list of document IDs to restrict search"
    )


class PageSearchResponse(BaseModel):
    """Response for document page search."""

    pages: list[DocumentPageResponse]
    total_count: int
    query: str


class DocumentMemoriesResponse(BaseModel):
    """Response model for memories extracted from a document."""

    document_id: str
    memories: list[dict[str, Any]]
    total: int


class ReprocessRequest(BaseModel):
    """Request model for reprocessing a document."""

    extraction_options: Optional[dict[str, Any]] = Field(
        None, description="Extraction options overrides"
    )


class DeleteDocumentResponse(BaseModel):
    """Response model for document deletion."""

    deleted: bool
    document_id: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _doc_to_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        workspace_id=doc.workspace_id,
        filename=doc.filename,
        document_type=doc.document_type.value,
        content_hash=doc.content_hash,
        size_bytes=doc.size_bytes,
        mime_type=doc.mime_type,
        status=doc.status.value,
        target_context_id=doc.target_context_id,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        memory_ids=doc.memory_ids,
        deduplicated_count=doc.deduplicated_count,
        error_message=doc.error_message,
        metadata={k: v for k, v in (doc.metadata or {}).items() if not k.startswith("_raw_content")},
        created_at=doc.created_at,
        processing_started_at=doc.processing_started_at,
        processing_completed_at=doc.processing_completed_at,
    )


def _job_to_response(job: IngestionJob) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        document_ids=job.document_ids,
        status=job.status.value,
        progress_percent=job.progress_percent,
        documents_processed=job.documents_processed,
        total_memories_created=job.total_memories_created,
        errors=job.errors,
        metadata=job.metadata,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _page_to_response(page: DocumentPage) -> DocumentPageResponse:
    return DocumentPageResponse(
        id=page.id,
        document_id=page.document_id,
        workspace_id=page.workspace_id,
        page_no=page.page_no,
        transcript=page.transcript,
        transcript_model=page.transcript_model,
        metadata=page.metadata,
        created_at=page.created_at,
    )


# ---------------------------------------------------------------------------
# Routes — fixed paths must come before parameterised paths
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(..., description="Document file to ingest"),
    document_type: Optional[str] = Form(None, description="Override detected document type"),
    target_context_id: Optional[str] = Form(None, description="Target context for extracted memories"),
    chunk_size: Optional[int] = Form(None, description="Max chunk size in characters"),
    importance: Optional[float] = Form(None, description="Default importance for extracted memories"),
    retain_original: Optional[bool] = Form(None, description="Retain raw content in storage"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> dict:
    """Upload a document for background ingestion.

    The document is parsed, chunked, and facts are extracted as memories
    in a background task. Returns a document record and ingestion job ID
    immediately; poll GET /v1/documents/{id} for processing status.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "create", workspace_id=ctx.workspace_id)

        file_data = await file.read()
        filename = file.filename or "upload"
        mime_type = file.content_type

        extraction_options: dict = {}
        if target_context_id:
            extraction_options["target_context_id"] = target_context_id
        if chunk_size is not None:
            extraction_options["chunk_size"] = chunk_size
        if importance is not None:
            extraction_options["importance"] = importance
        if retain_original is not None:
            extraction_options["retain_original"] = retain_original

        # Include MIME type detection hint
        if mime_type and not document_type:
            from ...services.document.parsers import detect_document_type
            detected = detect_document_type(filename, mime_type)
            document_type = detected.value

        logger.info(
            "Uploading document '%s' (%d bytes) to workspace %s",
            filename,
            len(file_data),
            ctx.workspace_id,
        )

        doc, job = await document_service.upload_document(
            workspace_id=ctx.workspace_id,
            file_data=file_data,
            filename=filename,
            document_type=document_type,
            extraction_options=extraction_options if extraction_options else None,
        )

        logger.info("Queued document %s (job %s) in workspace %s", doc.id, job.id, ctx.workspace_id)

        return {
            "document": _doc_to_response(doc).model_dump(),
            "job": _job_to_response(job).model_dump(),
        }

    except ValueError as e:
        logger.warning("Invalid document upload request: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("Failed to upload document: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document")


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_documents(
    http_request: Request,
    doc_status: Optional[str] = Query(None, alias="status", description="Filter by document status"),
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> DocumentListResponse:
    """List documents in the workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        docs, total = await document_service.list_documents(
            workspace_id=ctx.workspace_id,
            status=doc_status,
            document_type=document_type,
            limit=limit,
            offset=offset,
        )

        return DocumentListResponse(
            documents=[_doc_to_response(d) for d in docs],
            total=total,
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error("Failed to list documents: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list documents")


@router.get(
    "/jobs",
    response_model=IngestionJobListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_jobs(
    http_request: Request,
    job_status: Optional[str] = Query(None, alias="status", description="Filter by job status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> IngestionJobListResponse:
    """List ingestion jobs for the workspace."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        jobs = await document_service.list_jobs(
            workspace_id=ctx.workspace_id,
            status=job_status,
            limit=limit,
            offset=offset,
        )

        return IngestionJobListResponse(
            jobs=[_job_to_response(j) for j in jobs],
            total=len(jobs),
            limit=limit,
            offset=offset,
        )

    except Exception as e:
        logger.error("Failed to list ingestion jobs: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list ingestion jobs")


@router.get(
    "/jobs/{job_id}",
    response_model=IngestionJobResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> IngestionJobResponse:
    """Get a specific ingestion job by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        job = await document_service.get_job(ctx.workspace_id, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found: %s" % job_id)

        return _job_to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get ingestion job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get ingestion job")


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=IngestionJobResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def cancel_job(
    http_request: Request,
    job_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> IngestionJobResponse:
    """Cancel a queued or running ingestion job."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ctx.workspace_id)

        job = await document_service.cancel_job(ctx.workspace_id, job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found: %s" % job_id)

        logger.info("Cancelled ingestion job %s in workspace %s", job_id, ctx.workspace_id)
        return _job_to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel ingestion job %s: %s", job_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to cancel ingestion job")


@router.post(
    "/search",
    response_model=PageSearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        501: {"model": ErrorResponse, "description": "Page search not supported by the active storage backend"},
    },
)
async def search_document_pages(
    http_request: Request,
    request: PageSearchRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> PageSearchResponse:
    """Search document pages by ColPali MaxSim visual similarity.

    Embeds the query as a multi-vector via the configured embed-server peer and
    ranks pages whose stored ``multivector`` columns score highest under
    MaxSim late interaction. Requires an embed-server peer that exposes
    ``/v1/embeddings/multi`` (the ColPali endpoint) and a storage backend
    that implements ``search_pages_by_maxsim``.
    """
    from ...services.document import EXT_EMBED_SERVER_CLIENT
    from ...services.storage import EXT_STORAGE_BACKEND, StorageBackend

    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "documents", "read", workspace_id=ctx.workspace_id,
        )

        embed_client = get_extension(EXT_EMBED_SERVER_CLIENT, v)
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)

        try:
            await embed_client.connect()
            mv_results = await embed_client.embed_texts_multivector([request.query])
        finally:
            await embed_client.close()

        if not mv_results or "vectors" not in mv_results[0]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embed server returned no multi-vector embedding for query",
            )
        query_multivector = mv_results[0]["vectors"]

        try:
            results = await storage.search_pages_by_maxsim(
                workspace_id=ctx.workspace_id,
                query_multivector=query_multivector,
                limit=request.limit,
                doc_ids=request.doc_ids,
            )
        except NotImplementedError as nie:
            logger.info(
                "Page search not supported by active storage backend: %s", nie,
            )
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Page search is not supported by the active storage backend",
            )

        pages = [
            DocumentPageResponse(
                id=page.id,
                document_id=page.document_id,
                workspace_id=page.workspace_id,
                page_no=page.page_no,
                transcript=page.transcript,
                transcript_model=page.transcript_model,
                metadata=page.metadata or {},
                created_at=page.created_at,
                relevance_score=score,
            )
            for page, score in results
        ]

        return PageSearchResponse(
            pages=pages,
            total_count=len(pages),
            query=request.query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Page search failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Page search failed",
        )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> DocumentResponse:
    """Get a document by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        doc = await document_service.get_document(ctx.workspace_id, document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found: %s" % document_id)

        return _doc_to_response(doc)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get document")


@router.get(
    "/{document_id}/memories",
    response_model=DocumentMemoriesResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_memories(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> DocumentMemoriesResponse:
    """Get memories extracted from a document."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        doc = await document_service.get_document(ctx.workspace_id, document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found: %s" % document_id)

        # Fetch memories via storage backend directly
        from ...services.storage import EXT_STORAGE_BACKEND, StorageBackend

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        memories = await storage.get_document_memories(ctx.workspace_id, document_id)

        memory_dicts = [
            {
                "id": m.id,
                "content": m.content,
                "type": m.type.value if m.type else None,
                "subtype": m.subtype,
                "importance": m.importance,
                "tags": m.tags,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in memories
        ]

        return DocumentMemoriesResponse(
            document_id=document_id,
            memories=memory_dicts,
            total=len(memory_dicts),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memories for document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get document memories"
        )


@router.get(
    "/{document_id}/pages",
    response_model=DocumentPageListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_pages(
    http_request: Request,
    document_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> DocumentPageListResponse:
    """Get all pages for a document."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        doc = await document_service.get_document(ctx.workspace_id, document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found: %s" % document_id)

        from ...services.storage import EXT_STORAGE_BACKEND, StorageBackend

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        pages = await storage.get_pages(document_id, workspace_id=ctx.workspace_id)

        return DocumentPageListResponse(
            pages=[_page_to_response(p) for p in pages],
            total=len(pages),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get pages for document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get document pages")


@router.get(
    "/{document_id}/pages/{page_id}",
    response_model=DocumentPageResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Page not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_document_page(
    http_request: Request,
    document_id: str,
    page_id: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> DocumentPageResponse:
    """Get a specific page of a document by page ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "read", workspace_id=ctx.workspace_id)

        # Verify parent document exists and belongs to workspace
        doc = await document_service.get_document(ctx.workspace_id, document_id)
        if not doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found: %s" % document_id)

        from ...services.storage import EXT_STORAGE_BACKEND, StorageBackend

        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        page = await storage.get_page(page_id)

        if not page or page.document_id != document_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found: %s" % page_id)

        return _page_to_response(page)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get page %s of document %s: %s", page_id, document_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get document page")


@router.post(
    "/{document_id}/reprocess",
    response_model=dict,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reprocess_document(
    http_request: Request,
    document_id: str,
    request: ReprocessRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> dict:
    """Re-queue a document for extraction with optional new options."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "write", workspace_id=ctx.workspace_id)

        doc, job = await document_service.reprocess_document(
            workspace_id=ctx.workspace_id,
            doc_id=document_id,
            extraction_options=request.extraction_options,
        )

        logger.info("Queued reprocess for document %s (job %s) in workspace %s", document_id, job.id, ctx.workspace_id)

        return {
            "document": _doc_to_response(doc).model_dump(),
            "job": _job_to_response(job).model_dump(),
        }

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reprocess document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to reprocess document")


@router.delete(
    "/{document_id}",
    response_model=DeleteDocumentResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Document not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_document(
    http_request: Request,
    document_id: str,
    delete_memories: bool = Query(False, description="Also delete extracted memories"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    document_service: DocumentService = Depends(get_document_service),
    logger: logging.Logger = Depends(get_logger),
) -> DeleteDocumentResponse:
    """Delete a document and optionally cascade to extracted memories."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(ctx, "documents", "delete", workspace_id=ctx.workspace_id)

        success = await document_service.delete_document(
            workspace_id=ctx.workspace_id,
            doc_id=document_id,
            delete_memories=delete_memories,
        )

        if not success:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found: %s" % document_id)

        logger.info("Deleted document %s from workspace %s (delete_memories=%s)", document_id, ctx.workspace_id, delete_memories)
        return DeleteDocumentResponse(deleted=True, document_id=document_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document %s: %s", document_id, e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete document")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class DocumentsAPIPlugin(Plugin):
    """Plugin to register documents API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
