"""
Context environment API endpoints.

Endpoints:
- POST /v1/context/execute - Execute code in sandbox
- POST /v1/context/inspect - Inspect sandbox state
- POST /v1/context/load - Load memories into sandbox
- POST /v1/context/inject - Inject values into sandbox
- POST /v1/context/query - Query LLM with sandbox context
- POST /v1/context/rlm - Run reasoning loop over memories
- GET /v1/context/status - Get sandbox status
- DELETE /v1/context/cleanup - Clean up sandbox
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header, status
from scitrera_app_framework import Plugin, Variables

from .. import EXT_MULTI_API_ROUTERS
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from .schemas import (
    ContextExecuteRequest,
    ContextExecuteResponse,
    ContextInspectResponse,
    ContextLoadRequest,
    ContextLoadResponse,
    ContextInjectRequest,
    ContextInjectResponse,
    ContextQueryRequest,
    ContextQueryResponse,
    ContextRLMRequest,
    ContextRLMResponse,
    ContextStatusResponse,
    ErrorResponse,
)
from ...services.context_environment import (
    get_context_environment_service as _get_ctx_env_service,
    ContextEnvironmentService,
)
from ...services.session import SessionService
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from .deps import get_auth_service, get_authz_service, get_session_service

router = APIRouter(prefix="/v1/context", tags=["context-environment"])


def get_context_env_service(v: Variables = Depends(get_variables_dep)) -> ContextEnvironmentService:
    """FastAPI dependency wrapper for context environment service."""
    return _get_ctx_env_service(v)


async def _resolve_session_id(
    x_session_id: Optional[str],
    session_service: SessionService,
    logger: logging.Logger,
) -> str:
    """Resolve and validate session ID from header."""
    if not x_session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Session-ID header is required",
        )

    session = await session_service.get(x_session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found or expired: {x_session_id}",
        )
    return x_session_id


@router.post(
    "/execute",
    response_model=ContextExecuteResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def execute_code(
    request: ContextExecuteRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextExecuteResponse:
    """Execute Python code in the session's sandbox environment."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.debug("Context execute for session: %s", session_id)

        result = await ctx_env_service.execute(
            session_id=session_id,
            code=request.code,
            result_var=request.result_var,
            return_result=request.return_result,
            max_return_chars=request.max_return_chars,
        )

        return ContextExecuteResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context execute failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute code",
        )


@router.post(
    "/inspect",
    response_model=ContextInspectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def inspect_state(
    variable: Optional[str] = None,
    preview_chars: int = 200,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextInspectResponse:
    """Inspect the session's sandbox state or a specific variable."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.debug("Context inspect for session: %s, variable: %s", session_id, variable)

        result = await ctx_env_service.inspect(
            session_id=session_id,
            variable=variable,
            preview_chars=preview_chars,
        )

        return ContextInspectResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context inspect failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to inspect state",
        )


@router.post(
    "/load",
    response_model=ContextLoadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def load_memories(
    request: ContextLoadRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextLoadResponse:
    """Load memories into the session's sandbox as a variable."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.info("Context load for session: %s, query: %s", session_id, request.query[:50])

        result = await ctx_env_service.load(
            session_id=session_id,
            var=request.var,
            query=request.query,
            limit=request.limit,
            types=request.types,
            tags=request.tags,
            min_relevance=request.min_relevance,
            include_embeddings=request.include_embeddings,
        )

        return ContextLoadResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context load failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load memories",
        )


@router.post(
    "/inject",
    response_model=ContextInjectResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def inject_value(
    request: ContextInjectRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextInjectResponse:
    """Inject a value into the session's sandbox state."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.debug("Context inject for session: %s, key: %s", session_id, request.key)

        result = await ctx_env_service.inject(
            session_id=session_id,
            key=request.key,
            value=request.value,
            parse_json=request.parse_json,
        )

        return ContextInjectResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context inject failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to inject value",
        )


@router.post(
    "/query",
    response_model=ContextQueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def query_llm(
    request: ContextQueryRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextQueryResponse:
    """Send sandbox variables and a prompt to the LLM."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.info("Context query for session: %s", session_id)

        result = await ctx_env_service.query(
            session_id=session_id,
            prompt=request.prompt,
            variables=request.variables,
            max_context_chars=request.max_context_chars,
            result_var=request.result_var,
        )

        return ContextQueryResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to query LLM",
        )


@router.post(
    "/rlm",
    response_model=ContextRLMResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def run_rlm(
    request: ContextRLMRequest,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextRLMResponse:
    """Run a Recursive Language Model (RLM) loop."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.info("Context RLM for session: %s, goal: %s", session_id, request.goal[:50])

        result = await ctx_env_service.rlm(
            session_id=session_id,
            goal=request.goal,
            memory_query=request.memory_query,
            memory_limit=request.memory_limit,
            max_iterations=request.max_iterations,
            variables=request.variables,
            result_var=request.result_var,
            detail_level=request.detail_level,
        )

        return ContextRLMResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context RLM failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run RLM",
        )


@router.get(
    "/status",
    response_model=ContextStatusResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_status(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> ContextStatusResponse:
    """Get the status of a session's sandbox environment."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        result = await ctx_env_service.status(session_id)

        return ContextStatusResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context status failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get status",
        )


@router.post(
    "/checkpoint",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"model": ErrorResponse, "description": "Missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def checkpoint_environment(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """Checkpoint the session's sandbox state for persistence hooks."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)
        logger.info("Context checkpoint for session: %s", session_id)
        await ctx_env_service.checkpoint(session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context checkpoint failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to checkpoint environment",
        )


@router.delete(
    "/cleanup",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        400: {"model": ErrorResponse, "description": "Missing session"},
        404: {"model": ErrorResponse, "description": "Session not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def cleanup_environment(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    ctx_env_service: ContextEnvironmentService = Depends(get_context_env_service),
    session_service: SessionService = Depends(get_session_service),
    logger: logging.Logger = Depends(get_logger),
) -> None:
    """Clean up and remove a session's sandbox environment."""
    try:
        session_id = await _resolve_session_id(x_session_id, session_service, logger)

        logger.info("Context cleanup for session: %s", session_id)

        await ctx_env_service.cleanup_environment(session_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context cleanup failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup environment",
        )


class ContextEnvironmentAPIPlugin(Plugin):
    """Plugin to register context environment API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
