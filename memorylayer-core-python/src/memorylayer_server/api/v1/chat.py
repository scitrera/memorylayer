"""
Chat history API endpoints.

Endpoints:
- POST   /v1/threads              - Create a new chat thread
- GET    /v1/threads              - List threads (filter by workspace, user)
- GET    /v1/threads/{id}         - Get thread metadata
- PUT    /v1/threads/{id}         - Update thread (e.g. rename)
- GET    /v1/threads/{id}/full    - Get thread with messages inlined
- DELETE /v1/threads/{id}         - Delete thread and messages
- POST   /v1/threads/{id}/messages   - Append messages
- GET    /v1/threads/{id}/messages   - Get messages (paginated)
- POST   /v1/threads/{id}/decompose  - Trigger memory decomposition
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from scitrera_app_framework import Plugin, Variables

from memorylayer_server.lifecycle.fastapi import get_logger

from ...models.chat import (
    AppendMessagesInput,
    ChatMessageContent,
    CreateThreadInput,
    MessageInput,
)
from ...services.audit import AuditEvent, AuditService
from ...services.authentication import AuthenticationService
from ...services.authorization import AuthorizationService
from ...services.chat import ChatService
from .. import EXT_MULTI_API_ROUTERS
from .deps import get_audit_service, get_auth_service, get_authz_service, get_chat_service
from .schemas import (
    ErrorResponse,
    MessageListResponse,
    MessagesAppendRequest,
    MessagesAppendResponse,
    ThreadCreateRequest,
    ThreadDecomposeResponse,
    ThreadListResponse,
    ThreadResponse,
    ThreadUpdateRequest,
    ThreadWithMessagesResponse,
)

router = APIRouter(prefix="/v1/threads", tags=["chat"])


@router.post(
    "",
    response_model=ThreadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def create_thread(
    http_request: Request,
    request: ThreadCreateRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadResponse:
    """Create a new chat thread."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = request.workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        input_data = CreateThreadInput(
            thread_id=request.thread_id,
            user_id=request.user_id,
            context_id=request.context_id or "_default",
            observer_id=request.observer_id,
            subject_id=request.subject_id,
            title=request.title,
            metadata=request.metadata,
            expires_at=request.expires_at,
        )

        thread = await chat_service.create_thread(
            workspace_id=workspace_id,
            tenant_id=ctx.tenant_id,
            input=input_data,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="create",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread.thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread create")
        return ThreadResponse(thread=thread)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create thread: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat thread",
        )


@router.get(
    "",
    response_model=ThreadListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def list_threads(
    http_request: Request,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    user_id: str | None = Query(None, description="User filter"),
    limit: int = Query(50, ge=1, le=200, description="Max threads to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadListResponse:
    """List chat threads, optionally filtered by workspace and user."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=workspace_id)

        threads = await chat_service.list_threads(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread list")
        return ThreadListResponse(threads=threads, total_count=len(threads))

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list threads: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list threads",
        )


@router.get(
    "/{thread_id}",
    response_model=ThreadResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def get_thread(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadResponse:
    """Get thread metadata by ID."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=workspace_id)

        thread = await chat_service.get_thread(workspace_id, thread_id)
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread read")
        return ThreadResponse(thread=thread)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get thread",
        )


@router.put(
    "/{thread_id}",
    response_model=ThreadResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def update_thread(
    http_request: Request,
    thread_id: str,
    request: ThreadUpdateRequest,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadResponse:
    """Update a thread (e.g. rename)."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        updates = request.model_dump(exclude_none=True)
        if not updates:
            # Nothing to update, just return the current thread
            thread = await chat_service.get_thread(workspace_id, thread_id)
            if not thread:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Thread {thread_id} not found",
                )
            return ThreadResponse(thread=thread)

        thread = await chat_service.update_thread(workspace_id, thread_id, **updates)
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="update",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread update")
        return ThreadResponse(thread=thread)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update thread",
        )


@router.get(
    "/{thread_id}/full",
    response_model=ThreadWithMessagesResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def get_thread_full(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    limit: int = Query(100, ge=1, le=1000, description="Max messages to return"),
    offset: int = Query(0, ge=0, description="Message pagination offset"),
    order: str = Query("asc", pattern="^(asc|desc)$", description="Message order"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadWithMessagesResponse:
    """Get thread with all messages inlined (paginated)."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=workspace_id)

        result = await chat_service.get_thread_with_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            order=order,
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread full read")
        return ThreadWithMessagesResponse(
            thread=result.thread,
            messages=result.messages,
            total_messages=result.total_messages,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get full thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get thread with messages",
        )


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def delete_thread(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
):
    """Delete a thread and all its messages."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        deleted = await chat_service.delete_thread(workspace_id, thread_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="delete",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread delete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete thread",
        )


@router.post(
    "/{thread_id}/messages",
    response_model=MessagesAppendResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def append_messages(
    http_request: Request,
    thread_id: str,
    request: MessagesAppendRequest,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> MessagesAppendResponse:
    """Append messages to a chat thread."""
    try:
        ctx = await auth_service.build_context(http_request, request)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        # Convert API schema to domain input
        msg_inputs = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                content = [ChatMessageContent(**block) if isinstance(block, dict) else block for block in content]
            msg_inputs.append(
                MessageInput(
                    role=msg.role,
                    content=content,
                    metadata=msg.metadata,
                )
            )

        input_data = AppendMessagesInput(messages=msg_inputs)

        messages = await chat_service.append_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            input=input_data,
        )

        # Get updated thread for message count
        thread = await chat_service.get_thread(workspace_id, thread_id)
        new_count = thread.message_count if thread else len(messages)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="create",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="message",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for message append")
        return MessagesAppendResponse(
            messages=messages,
            thread_id=thread_id,
            new_message_count=new_count,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to append messages to thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to append messages",
        )


@router.get(
    "/{thread_id}/messages",
    response_model=MessageListResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def get_messages(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    limit: int = Query(100, ge=1, le=1000, description="Max messages to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    after_index: int | None = Query(None, ge=0, description="Get messages after this index"),
    order: str = Query("asc", pattern="^(asc|desc)$", description="Message order"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> MessageListResponse:
    """Get messages from a chat thread with pagination."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=workspace_id)

        # Verify thread exists
        thread = await chat_service.get_thread(workspace_id, thread_id)
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

        messages = await chat_service.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            after_index=after_index,
            order=order,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="message",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for message read")
        return MessageListResponse(
            messages=messages,
            thread_id=thread_id,
            total_count=thread.message_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get messages for thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get messages",
        )


@router.post(
    "/{thread_id}/decompose",
    response_model=ThreadDecomposeResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def decompose_thread(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadDecomposeResponse:
    """Trigger on-demand memory decomposition for unprocessed messages."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        result = await chat_service.trigger_decomposition(workspace_id, thread_id)

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="create",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                    resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread decompose")
        return ThreadDecomposeResponse(
            thread_id=result.thread_id,
            workspace_id=result.workspace_id,
            messages_processed=result.messages_processed,
            memories_created=result.memories_created,
            from_index=result.from_index,
            to_index=result.to_index,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to decompose thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to trigger decomposition",
        )


class ChatAPIPlugin(Plugin):
    """Plugin to register chat API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def is_multi_extension(self, v: Variables) -> bool:
        return True
