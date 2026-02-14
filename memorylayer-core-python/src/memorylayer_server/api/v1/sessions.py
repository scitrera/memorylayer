"""
Working memory/session context API endpoints.

Endpoints:
- POST /v1/sessions - Create session
- GET /v1/sessions/{session_id} - Get session
- DELETE /v1/sessions/{session_id} - Delete session
- POST /v1/sessions/{session_id}/memory - Set working memory key
- GET /v1/sessions/{session_id}/memory - Get working memory
- POST /v1/sessions/{session_id}/commit - Commit session to long-term memory
- POST /v1/sessions/{session_id}/touch - Update session expiration
- GET /v1/sessions/briefing - Session briefing
"""
import logging
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Request, status
from scitrera_app_framework import Plugin, Variables, get_extension

from .. import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep

from .schemas import (
    SessionCreateRequest,
    WorkingMemorySetRequest,
    SessionResponse,
    SessionStartResponse,
    WorkingMemoryResponse,
    SessionBriefingResponse,
    CommitOptions,
    CommitResponse,
    ErrorResponse,
)
from ...services.session import get_session_service as _get_session_service, SessionService
from ...services.workspace import get_workspace_service as _get_workspace_service, WorkspaceService
from ...services.authentication import (
    AuthenticationService,
    EXT_AUTHENTICATION_SERVICE,
)
from ...services.authorization import AuthorizationService, EXT_AUTHORIZATION_SERVICE
from ...config import DEFAULT_TENANT_ID, DEFAULT_CONTEXT_ID

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])


async def get_auth_service(v: Variables = Depends(get_variables_dep)) -> AuthenticationService:
    """Get authentication service instance."""
    return get_extension(EXT_AUTHENTICATION_SERVICE, v)


async def get_authz_service(v: Variables = Depends(get_variables_dep)) -> AuthorizationService:
    """Get authorization service instance."""
    return get_extension(EXT_AUTHORIZATION_SERVICE, v)


def get_session_service(v: Variables = Depends(get_variables_dep)) -> SessionService:
    """FastAPI dependency wrapper for session service."""
    return _get_session_service(v)


def get_workspace_service(v: Variables = Depends(get_variables_dep)) -> WorkspaceService:
    """FastAPI dependency wrapper for workspace service."""
    return _get_workspace_service(v)


@router.post(
    "",
    response_model=SessionStartResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_session(
        http_request: Request,
        request: SessionCreateRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        workspace_service: WorkspaceService = Depends(get_workspace_service),
        logger: logging.Logger = Depends(get_logger),
) -> SessionStartResponse:
    """
    Create a new working memory session.

    Sessions provide TTL-based temporary context storage. Workspaces and contexts
    are auto-created if they don't exist, enabling a "just works" experience.

    Args:
        http_request: FastAPI request (for headers)
        request: Session creation request
        auth_service: Authentication service for identity/workspace resolution
        session_service: Session service instance
        workspace_service: Workspace service for auto-creation

    Returns:
        Created session with optional briefing

    Raises:
        HTTPException: If session creation fails
    """
    try:
        # Build request context and check authorization
        ctx = await auth_service.build_context(http_request, request)
        await authz_service.require_authorization(
            ctx, "sessions", "create", workspace_id=ctx.workspace_id
        )

        # Generate session ID if not provided
        session_id = request.session_id or f"sess_{uuid4().hex[:16]}"

        # Use workspace from resolved context
        workspace_id = ctx.workspace_id

        # Use context_id from request if provided, otherwise use default
        context_id = request.context_id or DEFAULT_CONTEXT_ID

        logger.info(
            "Creating session: %s in workspace: %s, ttl: %d, context: %s",
            session_id,
            workspace_id,
            request.ttl_seconds,
            context_id
        )

        # Auto-create workspace if it doesn't exist (OSS "just works" pattern)
        workspace = await workspace_service.ensure_workspace(
            workspace_id=workspace_id,
            tenant_id=DEFAULT_TENANT_ID,
            auto_create=True,
        )
        if not workspace:
            raise ValueError(f"Failed to ensure workspace: {workspace_id}")

        # Ensure _default context partition exists
        await workspace_service.ensure_default_context(workspace_id)

        # Create session with context_id
        from ...models.session import Session
        session = Session.create_with_ttl(
            session_id=session_id,
            workspace_id=workspace_id,
            ttl_seconds=request.ttl_seconds,
            metadata=request.metadata,
            context_id=context_id,
            tenant_id=DEFAULT_TENANT_ID,  # Will be from auth in multi-tenant
        )

        # Store session via session service
        session = await session_service.create_session(workspace_id, session, context_id=context_id)

        # Set initial working memory if provided
        if request.working_memory:
            logger.info("Setting initial working memory: %d keys", len(request.working_memory))
            for key, value in request.working_memory.items():
                await session_service.set_working_memory(
                    workspace_id=workspace_id,
                    session_id=session_id,
                    key=key,
                    value=value
                )

        # Generate briefing if requested
        briefing = None
        if request.briefing:
            logger.info("Generating briefing for session: %s", session_id)
            briefing = await session_service.get_briefing(
                workspace_id,
                lookback_minutes=60,
                detail_level="abstract",
                limit=10,
                include_memories=True,
                include_contradictions=True,
            )

        logger.info("Created session: %s", session_id)
        return SessionStartResponse(session=session, briefing=briefing)

    except ValueError as e:
        logger.warning("Invalid session creation request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to create session: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session"
        )


# NOTE: /briefing must be defined BEFORE /{session_id} to avoid route collision
@router.get(
    "/briefing",
    response_model=SessionBriefingResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_briefing(
        http_request: Request,
        workspace_id: Optional[str] = None,
        lookback_minutes: int = 60,
        detail_level: str = "abstract",
        limit: int = 10,
        include_memories: bool = True,
        include_contradictions: bool = True,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> SessionBriefingResponse:
    """
    Get a briefing of recent workspace activity and context.

    Provides:
    - Workspace summary (total memories, recent activity)
    - Recent sessions and activity
    - Open threads/topics
    - Detected contradictions

    Args:
        http_request: FastAPI request (for headers)
        workspace_id: Optional explicit workspace ID override
        auth_service: Authentication service for workspace resolution
        session_service: Session service instance

    Returns:
        Session briefing with activity summary

    Raises:
        HTTPException: If briefing generation fails
    """
    try:
        # Build context and check authorization
        ctx = await auth_service.build_context(http_request, None)
        # Use explicit workspace_id if provided, otherwise fall back to context
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(
            ctx, "sessions", "read", workspace_id=workspace_id
        )

        logger.info("Generating briefing for workspace: %s", workspace_id)

        briefing = await session_service.get_briefing(
            workspace_id,
            lookback_minutes=lookback_minutes,
            detail_level=detail_level,
            limit=limit,
            include_memories=include_memories,
            include_contradictions=include_contradictions,
        )
        return SessionBriefingResponse(briefing=briefing)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate briefing: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate briefing"
        )


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found or expired"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_session(
        http_request: Request,
        session_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> SessionResponse:
    """
    Retrieve a session by ID.

    Args:
        session_id: Session identifier
        session_service: Session service instance

    Returns:
        Session object

    Raises:
        HTTPException: If session not found or expired
    """
    try:
        # Build context and check authorization
        ctx = await auth_service.build_context(http_request, None)

        logger.debug("Getting session: %s", session_id)

        # Session service get() doesn't require workspace_id
        session = await session_service.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found or expired: {session_id}"
            )

        # Check authorization for the session's workspace
        await authz_service.require_authorization(
            ctx, "sessions", "read",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        return SessionResponse(session=session)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_session(
        http_request: Request,
        session_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> None:
    """
    Delete a session and all its context data.

    Args:
        session_id: Session identifier
        session_service: Session service instance

    Raises:
        HTTPException: If session not found or deletion fails
    """
    try:
        # Build context
        ctx = await auth_service.build_context(http_request, None)

        logger.info("Deleting session: %s", session_id)

        # Get session to find its workspace
        session = await session_service.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check authorization
        await authz_service.require_authorization(
            ctx, "sessions", "delete",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        success = await session_service.delete_session(session.workspace_id, session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


@router.post(
    "/{session_id}/memory",
    response_model=WorkingMemoryResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found or expired"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def set_working_memory(
        http_request: Request,
        session_id: str,
        request: WorkingMemorySetRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> WorkingMemoryResponse:
    """
    Set a key-value working memory entry in a session.

    Args:
        session_id: Session identifier
        request: Working memory set request
        session_service: Session service instance

    Returns:
        Working memory entry

    Raises:
        HTTPException: If session not found or memory set fails
    """
    try:
        # Build context
        ctx = await auth_service.build_context(http_request, None)

        # Get session to find its workspace
        session = await session_service.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check authorization
        await authz_service.require_authorization(
            ctx, "sessions", "write",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        logger.info(
            "Setting working memory in session: %s, key: %s",
            session_id,
            request.key
        )

        memory = await session_service.set_working_memory(
            workspace_id=session.workspace_id,
            session_id=session_id,
            key=request.key,
            value=request.value,
            ttl_seconds=request.ttl_seconds,
        )
        return WorkingMemoryResponse(
            key=memory.key,
            value=memory.value,
            ttl_seconds=memory.ttl_seconds,
            created_at=memory.created_at,
            updated_at=memory.updated_at
        )

    except ValueError as e:
        logger.warning("Invalid working memory set request: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to set working memory in session %s: %s",
            session_id,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set working memory"
        )


@router.get(
    "/{session_id}/memory",
    response_model=dict,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found or expired"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_working_memory(
        http_request: Request,
        session_id: str,
        key: str | None = None,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Get session working memory data.

    Args:
        session_id: Session identifier
        key: Optional specific key to retrieve (returns all if omitted)
        session_service: Session service instance

    Returns:
        Working memory data (single entry if key specified, all entries otherwise)

    Raises:
        HTTPException: If session not found
    """
    try:
        # Build context
        ctx = await auth_service.build_context(http_request, None)

        # Get session to find its workspace
        session = await session_service.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check authorization
        await authz_service.require_authorization(
            ctx, "sessions", "read",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        logger.debug("Getting working memory from session: %s, key: %s", session_id, key)

        if key:
            memory = await session_service.get_working_memory(session.workspace_id, session_id, key)
            if not memory:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Working memory key not found: {key}"
                )
            return {key: memory.value}
        else:
            memories = await session_service.get_all_working_memory(session.workspace_id, session_id)
            return {mem.key: mem.value for mem in memories}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get working memory from session %s: %s",
            session_id,
            e,
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve working memory"
        )


@router.post(
    "/{session_id}/commit",
    response_model=CommitResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found or expired"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def commit_session(
        http_request: Request,
        session_id: str,
        options: Optional[CommitOptions] = None,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> CommitResponse:
    """
    Commit session working memory to long-term memory.

    Extracts memories from session working memory using the extraction service,
    deduplicates them, and creates new long-term memories.

    Args:
        session_id: Session identifier
        options: Optional commit options (min_importance, deduplicate, categories, max_memories)
        session_service: Session service instance

    Returns:
        CommitResponse with extraction statistics and breakdown by category

    Raises:
        HTTPException: If session not found or commit fails
    """
    try:
        # Build context
        ctx = await auth_service.build_context(http_request, None)

        # Get session to find its workspace
        session = await session_service.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check authorization
        await authz_service.require_authorization(
            ctx, "sessions", "write",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        logger.info("Committing session: %s with options: %s", session_id, options)

        # Convert Pydantic model to service CommitOptions
        from ...services.session.base import CommitOptions as ServiceCommitOptions
        service_options = None
        if options:
            service_options = ServiceCommitOptions(
                include_working_memory=True,
                importance_threshold=options.min_importance,
                delete_after_commit=False,
                tags=[]
            )

        result = await session_service.commit_session(
            session.workspace_id,
            session_id,
            options=service_options
        )

        # Build response from CommitResult fields
        return CommitResponse(
            session_id=session_id,
            memories_extracted=result.memories_extracted,
            memories_deduplicated=result.memories_deduplicated,
            memories_created=result.memories_committed,
            breakdown=result.extraction_summary.get('breakdown', {}),
            extraction_time_ms=result.extraction_summary.get('extraction_time_ms', 0)
        )

    except ValueError as e:
        logger.warning("Session commit failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to commit session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to commit session"
        )


@router.post(
    "/{session_id}/touch",
    status_code=status.HTTP_200_OK,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        404: {"model": ErrorResponse, "description": "Session not found or expired"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def touch_session(
        http_request: Request,
        session_id: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        authz_service: AuthorizationService = Depends(get_authz_service),
        session_service: SessionService = Depends(get_session_service),
        logger: logging.Logger = Depends(get_logger),
) -> dict:
    """
    Update session expiration (extend TTL).

    Args:
        session_id: Session identifier
        session_service: Session service instance

    Returns:
        Updated expiration timestamp

    Raises:
        HTTPException: If session not found or touch fails
    """
    try:
        # Build context
        ctx = await auth_service.build_context(http_request, None)

        # Get session to find its workspace
        session = await session_service.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )

        # Check authorization
        await authz_service.require_authorization(
            ctx, "sessions", "write",
            resource_id=session_id, workspace_id=session.workspace_id
        )

        logger.debug("Touching session: %s", session_id)

        updated_session = await session_service.touch_session(session.workspace_id, session_id)
        if not updated_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        return {"expires_at": updated_session.expires_at.isoformat()}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to touch session %s: %s", session_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to touch session"
        )


class SessionsAPIPlugin(Plugin):
    """Plugin to register sessions API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
