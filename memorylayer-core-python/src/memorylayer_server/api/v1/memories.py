"""
Memory CRUD and recall operations API endpoints.

Endpoints:
- POST /v1/memories - Store a memory
- GET /v1/memories/{memory_id} - Get single memory
- PUT /v1/memories/{memory_id} - Update memory
- DELETE /v1/memories/{memory_id} - Delete memory (soft delete)
- POST /v1/memories/recall - Query memories with mode (rag/llm/hybrid)
- POST /v1/memories/reflect - Synthesize memories
- POST /v1/memories/{memory_id}/decay - Decay importance
- POST /v1/memories/batch - Batch operations (create, update, delete)
"""
import logging

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Plugin, Variables, get_extension

from .. import EXT_MULTI_API_ROUTERS
from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.memory import RememberInput, RecallInput, ReflectInput
from ...models.auth import RequestContext
from ...services.memory import MemoryService, EXT_MEMORY_SERVICE
from ...services.reflect import ReflectService, EXT_REFLECT_SERVICE
from ...services.authentication import (
    AuthenticationService,
    AuthenticationError,
    get_authentication_service,
    EXT_AUTHENTICATION_SERVICE,
)
from ...services.authorization import AuthorizationService, EXT_AUTHORIZATION_SERVICE

from .schemas import (
    MemoryCreateRequest,
    MemoryUpdateRequest,
    MemoryRecallRequest,
    MemoryReflectRequest,
    MemoryDecayRequest,
    MemoryBatchRequest,
    MemoryResponse,
    RecallResult,
    ReflectResult,
    ErrorResponse,
    BatchOperationResponse,
    BatchOperationResult,
)

router = APIRouter(prefix="/v1/memories", tags=["memories"])


# Dependencies for services
async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


async def get_memory_service(v: Variables = Depends(get_variables_dep)) -> MemoryService:
    """Get memory service instance for FastAPI dependency injection."""
    return get_extension(EXT_MEMORY_SERVICE, v)


async def get_reflect_service(v: Variables = Depends(get_variables_dep)) -> ReflectService:
    """Get reflect service instance for FastAPI dependency injection."""
    return get_extension(EXT_REFLECT_SERVICE, v)


@router.post(
    "",
    response_model=MemoryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_memory(
        http_request: Request,
        request: MemoryCreateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Store a new memory with automatic embedding and classification.

    Authentication:
        - Authorization header: Bearer <api_key> (optional in OSS)
        - X-Session-ID header: Session for workspace context (optional)

    Workspace Resolution:
        1. request.workspace_id (explicit override)
        2. session.workspace_id (from X-Session-ID header)
        3. "_default" (fallback)

    Args:
        http_request: FastAPI request (for headers)
        request: Memory creation request
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Created memory with generated ID and embedding

    Raises:
        HTTPException: If memory creation fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "memories", "create", workspace_id=ctx.workspace_id
        )

        logger.info(
            "Creating memory in workspace: %s, content length: %d",
            ctx.workspace_id,
            len(request.content)
        )

        # Convert request to domain input
        remember_input = RememberInput(
            content=request.content,
            type=request.type,
            subtype=request.subtype,
            importance=request.importance,
            tags=request.tags,
            metadata=request.metadata,
            associations=request.associations,
            context_id=request.context_id or ctx.context_id,
        )

        # Store memory
        memory = await memory_service.remember(
            workspace_id=ctx.workspace_id,
            input=remember_input,
        )

        logger.info("Created memory: %s", memory.id)
        return MemoryResponse(memory=memory)

    except AuthenticationError as e:
        logger.warning("Authentication failed: %s", e)
        raise HTTPException(
            status_code=e.status_code,
            detail=e.message
        )
    except ValueError as e:
        logger.warning("Invalid memory creation request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create memory: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create memory"
        )


@router.get(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_memory(
        http_request: Request,
        memory_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Retrieve a single memory by ID.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Memory object

    Raises:
        HTTPException: If memory not found
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "memories", "read",
            resource_id=memory_id, workspace_id=ctx.workspace_id
        )

        logger.debug("Getting memory: %s", memory_id)

        memory = await memory_service.get(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
        )

        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            )

        return MemoryResponse(memory=memory)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memory"
        )


@router.put(
    "/{memory_id}",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def update_memory(
        http_request: Request,
        memory_id: str,
        request: MemoryUpdateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Update an existing memory.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        request: Memory update request
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Updated memory object

    Raises:
        HTTPException: If memory not found or update fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "memories", "write",
            resource_id=memory_id, workspace_id=ctx.workspace_id
        )

        logger.info("Updating memory: %s", memory_id)

        # Check if memory exists
        existing_memory = await memory_service.get(ctx.workspace_id, memory_id)
        if not existing_memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            )

        # Build update kwargs from non-None fields
        update_kwargs = {}
        if request.content is not None:
            update_kwargs["content"] = request.content
        if request.type is not None:
            update_kwargs["type"] = request.type
        if request.subtype is not None:
            update_kwargs["subtype"] = request.subtype
        if request.importance is not None:
            update_kwargs["importance"] = request.importance
        if request.tags is not None:
            update_kwargs["tags"] = request.tags
        if request.metadata is not None:
            update_kwargs["metadata"] = request.metadata
        if request.pinned is not None:
            update_kwargs["pinned"] = 1 if request.pinned else 0

        # Update memory via storage
        # Note: This assumes memory_service has access to storage.update_memory
        updated_memory = await memory_service.storage.update_memory(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
            **update_kwargs
        )

        if not updated_memory:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update memory"
            )

        logger.info("Updated memory: %s", memory_id)
        return MemoryResponse(memory=updated_memory)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid memory update request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to update memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update memory"
        )


@router.delete(
    "/{memory_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_memory(
        http_request: Request,
        memory_id: str,
        hard: bool = False,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> None:
    """
    Delete a memory (soft delete by default).

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        hard: If True, permanently delete; if False, soft delete
        auth_service: Authentication service
        memory_service: Memory service instance

    Raises:
        HTTPException: If memory not found or deletion fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "memories", "delete",
            resource_id=memory_id, workspace_id=ctx.workspace_id
        )

        logger.info("Deleting memory: %s (hard=%s)", memory_id, hard)

        success = await memory_service.forget(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
            hard=hard,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            )

        logger.info("Deleted memory: %s", memory_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory"
        )


@router.post(
    "/recall",
    response_model=RecallResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def recall_memories(
        http_request: Request,
        request: MemoryRecallRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> RecallResult:
    """
    Query memories using vector similarity and optional filters.

    Supports three retrieval modes:
    - RAG: Fast vector similarity search (~30ms)
    - LLM: Query rewriting + enhanced search (~500ms)
    - HYBRID: RAG first, LLM if insufficient (balanced)

    Args:
        http_request: FastAPI request (for headers)
        request: Memory recall request with query and filters
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Recall result with matched memories and metadata

    Raises:
        HTTPException: If recall fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "memories", "read", workspace_id=ctx.workspace_id
        )

        logger.debug(
            "(API) Recalling memories in workspace: %s, mode: %s, query: %s",
            ctx.workspace_id,
            request.mode,
            request.query[:50]
        )

        # Convert request to domain input
        recall_input = RecallInput(
            query=request.query,
            types=request.types,
            subtypes=request.subtypes,
            tags=request.tags,
            context_id=request.context_id or ctx.context_id,
            mode=request.mode,
            tolerance=request.tolerance,
            limit=request.limit,
            min_relevance=request.min_relevance,
            recency_weight=request.recency_weight,
            include_associations=request.include_associations,
            traverse_depth=request.traverse_depth,
            max_expansion=request.max_expansion,
            created_after=request.created_after,
            created_before=request.created_before,
            context=request.context,
            rag_threshold=request.rag_threshold,
            include_archived=request.include_archived,
        )

        # Perform recall
        result = await memory_service.recall(
            workspace_id=ctx.workspace_id,
            input=recall_input,
        )

        logger.debug(
            "Recalled %d memories in %d ms using %s mode",
            len(result.memories),
            result.search_latency_ms,
            result.mode_used
        )

        return result

    except ValueError as e:
        logger.warning("Invalid recall request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to recall memories: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recall memories"
        )


@router.post(
    "/reflect",
    response_model=ReflectResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def reflect_memories(
        http_request: Request,
        request: MemoryReflectRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        reflection_service: ReflectService = Depends(get_reflect_service),
        logger: logging.Logger = Depends(get_logger),
) -> ReflectResult:
    """
    Synthesize memories into a coherent reflection.

    Args:
        http_request: FastAPI request (for headers)
        request: Memory reflection request
        auth_service: Authentication service
        reflection_service: Reflection service instance

    Returns:
        Reflection result with synthesized content

    Raises:
        HTTPException: If reflection fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "memories", "read", workspace_id=ctx.workspace_id
        )

        logger.info(
            "Reflecting on memories in workspace: %s, query: %s",
            ctx.workspace_id,
            request.query[:50]
        )

        # Convert detail_level string to DetailLevel enum
        from ...models.memory import DetailLevel
        detail_level = DetailLevel.FULL
        if request.detail_level:
            detail_level_map = {
                "abstract": DetailLevel.ABSTRACT,
                "overview": DetailLevel.OVERVIEW,
                "full": DetailLevel.FULL,
            }
            detail_level = detail_level_map.get(request.detail_level.lower(), DetailLevel.FULL)

        # Convert request to domain input
        reflect_input = ReflectInput(
            query=request.query,
            detail_level=detail_level,
            include_sources=request.include_sources,
            depth=request.depth,
            types=request.types,
            subtypes=request.subtypes,
            tags=request.tags,
            context_id=request.context_id or ctx.context_id,
        )

        # Perform reflection
        result = await reflection_service.reflect(
            workspace_id=ctx.workspace_id,
            input=reflect_input,
        )

        logger.info(
            "Reflected on %d source memories, generated %d tokens",
            len(result.source_memories),
            result.tokens_processed
        )

        return result

    except ValueError as e:
        logger.warning("Invalid reflect request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to reflect memories: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reflect memories"
        )


@router.post(
    "/{memory_id}/decay",
    response_model=MemoryResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Memory not found"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def decay_memory(
        http_request: Request,
        memory_id: str,
        request: MemoryDecayRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> MemoryResponse:
    """
    Reduce memory importance by decay rate.

    Args:
        http_request: FastAPI request (for headers)
        memory_id: Memory identifier
        request: Decay request with rate
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Updated memory with decayed importance

    Raises:
        HTTPException: If memory not found or decay fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "memories", "write",
            resource_id=memory_id, workspace_id=ctx.workspace_id
        )

        logger.info("Decaying memory: %s by rate: %f", memory_id, request.decay_rate)

        updated_memory = await memory_service.decay(
            workspace_id=ctx.workspace_id,
            memory_id=memory_id,
            decay_rate=request.decay_rate,
        )

        if not updated_memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found: {memory_id}"
            )

        logger.info("Decayed memory: %s", memory_id)
        return MemoryResponse(memory=updated_memory)

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Invalid decay request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to decay memory %s: %s", memory_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decay memory"
        )


@router.post(
    "/batch",
    response_model=BatchOperationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def batch_operations(
        http_request: Request,
        request: MemoryBatchRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        memory_service: MemoryService = Depends(get_memory_service),
        logger: logging.Logger = Depends(get_logger),
) -> BatchOperationResponse:
    """
    Perform multiple memory operations in a single request.

    Supported operation types:
    - create: Create a new memory
    - update: Update an existing memory
    - delete: Delete a memory

    Args:
        http_request: FastAPI request (for headers)
        request: Batch request with list of operations
        auth_service: Authentication service
        memory_service: Memory service instance

    Returns:
        Results for each operation with success/error status

    Raises:
        HTTPException: If batch request is invalid
    """
    try:
        # Build request context and check authorization (batch requires write access)
        ctx = await auth_service.build_context(http_request, None)
        await authz_service.require_authorization(
            ctx, "memories", "write", workspace_id=ctx.workspace_id
        )

        logger.info(
            "Processing batch operations in workspace: %s, count: %d",
            ctx.workspace_id,
            len(request.operations)
        )

        results = []
        successful = 0
        failed = 0

        for i, operation in enumerate(request.operations):
            op_type = operation.get("type")
            op_data = operation.get("data", {})

            logger.debug("Processing batch operation %d: %s", i, op_type)

            try:
                # CREATE operation
                if op_type == "create":
                    # Convert data to RememberInput
                    remember_input = RememberInput(
                        content=op_data.get("content"),
                        type=op_data.get("type"),
                        subtype=op_data.get("subtype"),
                        importance=op_data.get("importance", 0.5),
                        tags=op_data.get("tags", []),
                        metadata=op_data.get("metadata", {}),
                        associations=op_data.get("associations", []),
                        context_id=op_data.get("context_id"),
                    )

                    # Create memory
                    memory = await memory_service.remember(
                        workspace_id=ctx.workspace_id,
                        input=remember_input,
                    )

                    results.append(BatchOperationResult(
                        index=i,
                        type=op_type,
                        status="success",
                        memory_id=memory.id,
                    ))
                    successful += 1

                # UPDATE operation
                elif op_type == "update":
                    memory_id = op_data.get("memory_id")
                    if not memory_id:
                        raise ValueError("memory_id is required for update operation")

                    # Check if memory exists
                    existing = await memory_service.get(ctx.workspace_id, memory_id)
                    if not existing:
                        raise ValueError(f"Memory not found: {memory_id}")

                    # Build update kwargs
                    update_kwargs = {}
                    if "content" in op_data:
                        update_kwargs["content"] = op_data["content"]
                    if "type" in op_data:
                        update_kwargs["type"] = op_data["type"]
                    if "subtype" in op_data:
                        update_kwargs["subtype"] = op_data["subtype"]
                    if "importance" in op_data:
                        update_kwargs["importance"] = op_data["importance"]
                    if "tags" in op_data:
                        update_kwargs["tags"] = op_data["tags"]
                    if "metadata" in op_data:
                        update_kwargs["metadata"] = op_data["metadata"]

                    # Update memory
                    updated = await memory_service.storage.update_memory(
                        workspace_id=ctx.workspace_id,
                        memory_id=memory_id,
                        **update_kwargs
                    )

                    results.append(BatchOperationResult(
                        index=i,
                        type=op_type,
                        status="success",
                        memory_id=memory_id,
                    ))
                    successful += 1

                # DELETE operation
                elif op_type == "delete":
                    memory_id = op_data.get("memory_id")
                    if not memory_id:
                        raise ValueError("memory_id is required for delete operation")

                    hard = op_data.get("hard", False)

                    # Delete memory
                    success = await memory_service.forget(
                        workspace_id=ctx.workspace_id,
                        memory_id=memory_id,
                        hard=hard,
                    )

                    if not success:
                        raise ValueError(f"Memory not found: {memory_id}")

                    results.append(BatchOperationResult(
                        index=i,
                        type=op_type,
                        status="success",
                        memory_id=memory_id,
                    ))
                    successful += 1

                # Unknown operation type
                else:
                    raise ValueError(f"Unknown operation type: {op_type}")

            except Exception as e:
                logger.warning(
                    "Batch operation %d failed: %s - %s",
                    i,
                    op_type,
                    str(e)
                )
                results.append(BatchOperationResult(
                    index=i,
                    type=op_type or "unknown",
                    status="error",
                    error=str(e),
                ))
                failed += 1

        logger.info(
            "Completed batch operations: %d successful, %d failed",
            successful,
            failed
        )

        return BatchOperationResponse(
            total_operations=len(request.operations),
            successful=successful,
            failed=failed,
            results=results,
        )

    except ValueError as e:
        logger.warning("Invalid batch request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to process batch operations: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch operations"
        )


class MemoriesAPIPlugin(Plugin):
    """Plugin to register memories API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
