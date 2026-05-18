"""Token management API endpoints via Aether gRPC.

Manages API tokens through the shared ``AsyncServiceClient`` from
:class:`AetherServiceConnection`, replacing the previous HTTP proxy approach.

Endpoints:
- GET    /v1/tokens          -- List API tokens
- POST   /v1/tokens          -- Create API token
- GET    /v1/tokens/{id}     -- Get token details
- DELETE /v1/tokens/{id}     -- Delete token
- POST   /v1/tokens/{id}/revoke -- Revoke token
"""

from datetime import UTC, datetime
from logging import Logger

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from scitrera_app_framework import Plugin, Variables, get_extension

from memorylayer_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_server.api.v1.deps import get_auth_service, get_authz_service
from memorylayer_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
from memorylayer_server.services.authentication import AuthenticationError, AuthenticationService
from memorylayer_server.services.authorization import AuthorizationService

router = APIRouter(prefix="/v1/tokens", tags=["tokens"])


# ------------------------------------------------------------------ #
# Request / Response schemas
# ------------------------------------------------------------------ #


class TokenCreateRequest(BaseModel):
    """Request body for creating an API token."""

    name: str
    principal_type: str = "User"
    workspace_patterns: list[str] = ["*"]
    scopes: list[str] = ["*"]
    expires_in_days: int | None = None


class TokenResponse(BaseModel):
    """API token representation."""

    id: str
    name: str
    principal_type: str = "User"
    workspace_patterns: list[str]
    scopes: list[str]
    created_at: str
    expires_at: str | None = None
    revoked: bool = False


class TokenCreateResponse(TokenResponse):
    """Response for a newly created token (includes plaintext token)."""

    token: str


class TokenListResponse(BaseModel):
    """List of API tokens."""

    tokens: list[TokenResponse]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _get_aether_client(v: Variables):
    """Return the shared Aether client from AetherServiceConnection."""
    agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
    client = agent_service.client
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Aether service not connected",
        )
    return client


def _ts_to_iso(ts: int | str | None) -> str | None:
    """Convert a unix timestamp (int) or string to an ISO 8601 string."""
    if ts is None or ts == 0 or ts == "":
        return None
    if isinstance(ts, int):
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    return str(ts)


def _parse_token_info(token_info) -> TokenResponse:
    """Map an Aether ``TokenInfo`` protobuf to a ``TokenResponse``."""
    return TokenResponse(
        id=str(token_info.id),
        name=token_info.name or "",
        principal_type=token_info.principal_type or "User",
        workspace_patterns=list(token_info.workspace_patterns) or ["*"],
        scopes=list(token_info.scopes) or ["*"],
        created_at=_ts_to_iso(token_info.created_at) or "",
        expires_at=_ts_to_iso(token_info.expires_at),
        revoked=bool(token_info.revoked),
    )


def _handle_grpc_error(exc: Exception, logger: Logger, operation: str) -> None:
    """Convert a gRPC/Aether error into an appropriate HTTPException."""
    error_msg = str(exc)
    logger.warning("Aether token %s error: %s", operation, error_msg)

    if "not found" in error_msg.lower():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
    if "unauthorized" in error_msg.lower() or "permission" in error_msg.lower():
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Aether authentication failed",
        )
    raise HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail="Upstream token service error",
    )


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@router.get(
    "",
    response_model=TokenListResponse,
    responses={
        502: {"model": ErrorResponse, "description": "Upstream service error"},
        503: {"model": ErrorResponse, "description": "Aether service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def list_tokens(
    http_request: Request,
    include_revoked: bool = False,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: Logger = Depends(get_logger),
) -> TokenListResponse:
    """List API tokens via Aether gRPC.

    Args:
        include_revoked: If true, includes revoked tokens in the result. Default false.
    """
    """List all API tokens via Aether gRPC."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        client = _get_aether_client(v)
        logger.debug("Listing tokens via Aether gRPC")

        response = await client.list_tokens(limit=0, offset=0, include_revoked=include_revoked, timeout=10.0)
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Token list request timed out",
            )

        if not response.success:
            logger.error("Aether list_tokens failed: %s", response.error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token service error: {response.error}",
            )

        tokens = [_parse_token_info(t) for t in response.tokens]
        logger.info("Listed %d tokens", len(tokens))
        return TokenListResponse(tokens=tokens)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list tokens: %s", exc, exc_info=True)
        _handle_grpc_error(exc, logger, "list")


@router.post(
    "",
    response_model=TokenCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        502: {"model": ErrorResponse, "description": "Upstream service error"},
        503: {"model": ErrorResponse, "description": "Aether service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_token(
    http_request: Request,
    request: TokenCreateRequest,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: Logger = Depends(get_logger),
) -> TokenCreateResponse:
    """Create a new API token via Aether gRPC."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        client = _get_aether_client(v)
        logger.info("Creating token %r via Aether gRPC", request.name)

        expires_in_hours = 0
        if request.expires_in_days is not None:
            expires_in_hours = request.expires_in_days * 24

        response = await client.create_token(
            name=request.name,
            principal_type=request.principal_type,
            workspace_patterns=request.workspace_patterns,
            scopes=request.scopes,
            expires_in_hours=expires_in_hours,
            timeout=10.0,
        )
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Token create request timed out",
            )

        if not response.success:
            logger.error("Aether create_token failed: %s", response.error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token service error: {response.error}",
            )

        # The created token info is in response.created_token, plaintext in response.plaintext_token
        token_info = response.created_token
        result = TokenCreateResponse(
            id=str(token_info.id),
            name=token_info.name or request.name,
            principal_type=token_info.principal_type or request.principal_type,
            workspace_patterns=list(token_info.workspace_patterns) or request.workspace_patterns,
            scopes=list(token_info.scopes) or request.scopes,
            created_at=_ts_to_iso(token_info.created_at) or "",
            expires_at=_ts_to_iso(token_info.expires_at),
            revoked=bool(token_info.revoked),
            token=response.plaintext_token,
        )

        logger.info("Created token %r with id=%s", result.name, result.id)
        return result

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_grpc_error(exc, logger, "create")


@router.get(
    "/{token_id}",
    response_model=TokenResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Token not found"},
        502: {"model": ErrorResponse, "description": "Upstream service error"},
        503: {"model": ErrorResponse, "description": "Aether service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_token(
    http_request: Request,
    token_id: str,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: Logger = Depends(get_logger),
) -> TokenResponse:
    """Get details for a single API token via Aether gRPC."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "read", workspace_id=ctx.workspace_id)

        client = _get_aether_client(v)
        logger.debug("Fetching token: %s", token_id)

        response = await client.get_token(token_id, timeout=10.0)
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Token get request timed out",
            )

        if not response.success:
            if "not found" in (response.error or "").lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
            logger.error("Aether get_token failed: %s", response.error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token service error: {response.error}",
            )

        return _parse_token_info(response.token)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_grpc_error(exc, logger, "get")


@router.delete(
    "/{token_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Token not found"},
        502: {"model": ErrorResponse, "description": "Upstream service error"},
        503: {"model": ErrorResponse, "description": "Aether service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def delete_token(
    http_request: Request,
    token_id: str,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Delete an API token via Aether gRPC."""
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        client = _get_aether_client(v)
        logger.info("Deleting token: %s", token_id)

        response = await client.delete_token(token_id, timeout=10.0)
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Token delete request timed out",
            )

        if not response.success:
            if "not found" in (response.error or "").lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
            logger.error("Aether delete_token failed: %s", response.error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token service error: {response.error}",
            )

        logger.info("Deleted token: %s", token_id)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_grpc_error(exc, logger, "delete")


@router.post(
    "/{token_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Token not found"},
        502: {"model": ErrorResponse, "description": "Upstream service error"},
        503: {"model": ErrorResponse, "description": "Aether service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def revoke_token(
    http_request: Request,
    token_id: str,
    v: Variables = Depends(get_variables_dep),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    logger: Logger = Depends(get_logger),
) -> None:
    """Revoke an API token via Aether gRPC.

    Revoked tokens are invalidated immediately but remain visible
    in token listings with revoked=True.
    """
    try:
        ctx = await auth_service.build_context(http_request)
        await authz_service.require_authorization(ctx, "admin", "write", workspace_id=ctx.workspace_id)

        client = _get_aether_client(v)
        logger.info("Revoking token: %s", token_id)

        response = await client.revoke_token(token_id, timeout=10.0)
        if response is None:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Token revoke request timed out",
            )

        if not response.success:
            if "not found" in (response.error or "").lower():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Token not found")
            logger.error("Aether revoke_token failed: %s", response.error)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Token service error: {response.error}",
            )

        logger.info("Revoked token: %s", token_id)

    except AuthenticationError as exc:
        logger.warning("Authentication failed: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except HTTPException:
        raise
    except Exception as exc:
        _handle_grpc_error(exc, logger, "revoke")


# ------------------------------------------------------------------ #
# Plugin registration
# ------------------------------------------------------------------ #


class TokensAPIPlugin(Plugin):
    """Plugin to register token management API routes (gRPC-backed)."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        """Return the router. No HTTP client needed — uses shared gRPC client."""
        logger.info("Token API initialized (gRPC-backed via AetherServiceConnection)")
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True
