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

from ...models.auth import RequestContext
from ...models.chat import (
    USER_CHAT_HOME_WORKSPACE,
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


# ---------------------------------------------------------------------------
# Per-user thread scoping chokepoint (SECURITY).
#
# User-owned chat threads all live in the shared USER_CHAT_HOME_WORKSPACE
# sentinel and the harness sends a shared client thread_id ("_default") for
# every user. Per-user isolation is enforced by SCOPING the storage lookup on
# the OBO HUMAN subject (RequestContext.effective_subject_id — the human, NOT
# the sahara sandbox actor): thread identity is (workspace_id, user_id, id) with
# ``id`` stored VERBATIM. The storage-internal surrogate ``row_id`` never leaves
# storage, so there is no mangled id to present back to the client — responses
# carry the verbatim client ``id``.
#
# Every thread-id-keyed route computes ``user_key = _scope_user_id(ctx, ws)`` and
# threads it into the service/storage as the scoping ``user_id``.
#
# FAIL-CLOSED: sentinel access with no resolvable OBO human user is DENIED
# (401); we never fall back to an unscoped (shared) query.
# ---------------------------------------------------------------------------


def _obo_user_key(ctx: RequestContext) -> str | None:
    """Return the OBO human user key for per-user scoping, or None.

    Uses ``effective_subject_id()`` which yields the OBO grant subject (the
    human) when on-behalf-of is active and falls back to ``user_id`` for direct
    human web-app requests. The sahara sandbox *actor* is never returned here.
    """
    key = ctx.effective_subject_id()
    if key is None:
        return None
    key = key.strip()
    return key or None


def _scope_user_id(ctx: RequestContext, workspace_id: str) -> str | None:
    """Return the owner scoping key for a thread-keyed request (SECURITY chokepoint).

    Threads are keyed ``(workspace_id, user_id, id)`` in BOTH homes — the
    ``_user_chat`` sentinel (user-owned, cross-workspace) and a real workspace
    (workspace-homed). The namespace decides a thread's HOME; it does NOT decide
    its KEYING. Both resolve to the OBO human subject, so a shared client id like
    ``"_default"`` resolves to THIS user's thread in either home.

    The homes differ only in how a MISSING OBO human is treated:

    * sentinel -> HTTP 401 (fail-closed). A user-owned thread with no user is
      meaningless, and an unscoped fallback would enumerate other users' history.
    * real workspace -> ``None``. Service/system writes carry no OBO human and
      legitimately land unattributed rows here; rejecting them would break every
      non-user caller.

    DESIGN: workspace-homed threads are deliberately NOT a collaboration feature.
    They are per-user threads *homed* to a workspace, not shared across its
    members. Populating ``user_id`` is what implements that — and it preserves the
    option of relaxing to shared reads later (a read-time filter choice) without
    having lost authorship, which NULLing the column would make unrecoverable.
    The schema already assumes a populated column: ``uq_chat_threads_ws_user_id``
    keys on ``COALESCE(user_id,'')`` and ``idx_chat_threads_tenant_user_ownership``
    is partial ``WHERE user_id IS NOT NULL``.

    Do NOT reintroduce a rule that infers keying from the mere PRESENCE of an OBO
    user: the human subject rides on essentially every authenticated request,
    including operations on workspace-homed threads. Identity answers "who is
    acting", never "how is this thread keyed". Conflating the two is what split
    every workspace-homed thread into two rows (create stamped the caller's id
    while every thread-id-keyed route scoped to NULL), stranding the auto-title
    task on the invisible half.
    """
    user_key = _obo_user_key(ctx)
    if user_key:
        return user_key
    if workspace_id == USER_CHAT_HOME_WORKSPACE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user-owned chat threads require an authenticated user identity",
        )
    return None


def _present_messages(messages, client_id: str):
    """Set each message's ``thread_id`` to the request's client id.

    Messages reference the parent thread's storage-internal surrogate, which must
    NEVER appear in a response — callers only know the client id (e.g. "_default").
    """
    presented = []
    for message in messages:
        if getattr(message, "thread_id", None) != client_id:
            message = message.model_copy()
            message.thread_id = client_id
        presented.append(message)
    return presented


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

        # SECURITY: for the _user_chat sentinel, stamp the OBO human user as the
        # thread owner (fail-closed if absent). The owner is a SCOPING COLUMN, not
        # part of the id — parent_thread stays the verbatim client id.
        # Owner key comes from the scoping chokepoint, NEVER from request.user_id.
        # That field is caller-supplied and unvalidated, so trusting it let a client
        # stamp ANY id as the thread owner — and, because every other thread-id-keyed
        # route scopes via _scope_user_id, it also produced a row none of them could
        # then find (see _scope_user_id's docstring).
        thread_user_id = _scope_user_id(ctx, workspace_id)
        parent_thread = request.parent_thread

        # thread_id is server-owned (see ThreadCreateRequest docstring); not
        # forwarded from the request even if a client tries to sneak one in.
        input_data = CreateThreadInput(
            user_id=thread_user_id,
            context_id=request.context_id or "_default",
            observer_id=request.observer_id,
            subject_id=request.subject_id,
            title=request.title,
            metadata=request.metadata,
            expires_at=request.expires_at,
            idle_action=request.idle_action,
            scope=request.scope,
            ownership=request.ownership,
            parent_thread=parent_thread,
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
    scope_filter: str | None = Query(None, description="Scope filter: 'web' | 'office' | None (return all)"),
    ownership_filter: str | None = Query(None, description="Ownership filter: 'user' | 'workspace' | None (return all)"),
    include_hidden: bool = Query(False, description="Include archived/hidden threads (for an Archived view)"),
    parent_thread: str | None = Query(None, description="Parent thread id to list its sub-threads; omit for top-level threads only"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadListResponse:
    """List chat threads, optionally filtered by workspace, user, scope, and ownership.

    By default returns only top-level threads; pass ``parent_thread`` to list a
    thread's sub-threads.
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=workspace_id)

        # SECURITY: force-scope to the OBO human whenever there is one, in BOTH
        # homes — threads are per-user in each (see _scope_user_id), so the
        # caller-supplied ``user_id`` query param must never widen the result to
        # another user's threads. The sentinel additionally fails closed inside
        # _scope_user_id when no OBO human resolves.
        #
        # When no OBO human resolves (service/system callers on a real workspace)
        # the explicit param is left as-is rather than cleared: clearing it would
        # drop the storage filter entirely and hand back EVERY user's threads in
        # the workspace. parent_thread stays the verbatim client id (owner scoping
        # is a column, applied via user_id below).
        user_key = _scope_user_id(ctx, workspace_id)
        if user_key:
            user_id = user_key

        threads = await chat_service.list_threads(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
            scope_filter=scope_filter,
            ownership_filter=ownership_filter,
            include_hidden=include_hidden,
            parent_thread=parent_thread,
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
        return ThreadListResponse(
            threads=threads,
            total_count=len(threads),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list threads: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list threads",
        )


@router.get(
    "/user/{user_id}",
    response_model=ThreadListResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def list_user_threads(
    http_request: Request,
    user_id: str,
    ownership: str = Query('user', description="Ownership filter: 'user' | 'workspace'"),
    scope_filter: str | None = Query(None, description="Scope filter: 'web' | 'office' | None (return all)"),
    limit: int = Query(50, ge=1, le=200, description="Max threads to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    include_hidden: bool = Query(False, description="Include archived/hidden threads (for an Archived view)"),
    parent_thread: str | None = Query(None, description="Parent thread id to list its sub-threads; omit for top-level threads only"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadListResponse:
    """List chat threads owned by a user across all workspaces.

    Unlike GET /v1/threads, this route is keyed on (tenant, user, ownership)
    and does NOT require a workspace filter. Used for the user-session-scoped
    right rail in the web app — each returned thread carries its existing
    metadata (where callers may stash a preferredWorkspace for sidebar grouping).
    """
    try:
        ctx = await auth_service.build_context(http_request, None)
        # Authorization scope: read on the user's own tenant. We don't require a
        # workspace_id since this is a cross-workspace read.
        await authz_service.require_authorization(ctx, "threads", "read", workspace_id=ctx.workspace_id)

        # SECURITY: this route lists user-owned threads across workspaces. The
        # ``user_id`` path segment is caller-supplied and MUST NOT be trusted —
        # a spoofed id would enumerate another user's chat history. Force the
        # scope to the OBO human subject (fail-closed for user-owned listings).
        obo_user = _obo_user_key(ctx)
        if ownership == 'user':
            if not obo_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="user-owned chat threads require an authenticated user identity",
                )
            user_id = obo_user
        elif obo_user:
            # workspace-owned listing: still pin to the authenticated user.
            user_id = obo_user

        threads = await chat_service.list_user_threads(
            tenant_id=ctx.tenant_id,
            user_id=user_id,
            ownership=ownership,
            scope_filter=scope_filter,
            limit=limit,
            offset=offset,
            include_hidden=include_hidden,
            parent_thread=parent_thread,
        )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="read",
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    resource_type="thread",
                )
            )
        except Exception:
            logger.debug("Audit record failed for user thread list")
        return ThreadListResponse(
            threads=threads,
            total_count=len(threads),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list user threads: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list user threads",
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        thread = await chat_service.get_thread(workspace_id, thread_id, user_id=user_key)
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        updates = request.model_dump(exclude_none=True)
        if not updates:
            # Nothing to update, just return the current thread
            thread = await chat_service.get_thread(workspace_id, thread_id, user_id=user_key)
            if not thread:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Thread {thread_id} not found",
                )
            return ThreadResponse(thread=thread)

        thread = await chat_service.update_thread(workspace_id, thread_id, user_id=user_key, **updates)
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        result = await chat_service.get_thread_with_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            order=order,
            user_id=user_key,
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
            messages=_present_messages(result.messages, thread_id),
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        deleted = await chat_service.delete_thread(workspace_id, thread_id, user_id=user_key)
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
    "/{thread_id}/archive",
    response_model=ThreadResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def archive_thread(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadResponse:
    """Archive (hide) a thread without deleting it. Hidden threads are excluded
    from default listings but can be restored; a new message also un-archives."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        thread = await chat_service.hide_thread(workspace_id, thread_id, user_id=user_key)
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread {thread_id} not found")

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat", action="update", tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id, user_id=ctx.user_id,
                    resource_type="thread", resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread archive")
        return ThreadResponse(thread=thread)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to archive thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive thread",
        )


@router.post(
    "/{thread_id}/restore",
    response_model=ThreadResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def restore_thread(
    http_request: Request,
    thread_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
) -> ThreadResponse:
    """Restore (un-archive) a previously hidden thread."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        thread = await chat_service.unhide_thread(workspace_id, thread_id, user_id=user_key)
        if not thread:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Thread {thread_id} not found")

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat", action="update", tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id, user_id=ctx.user_id,
                    resource_type="thread", resource_id=thread_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for thread restore")
        return ThreadResponse(thread=thread)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restore thread %s: %s", thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore thread",
        )


@router.delete(
    "/{thread_id}/messages/{message_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse, "description": "Message not found"},
        401: {"model": ErrorResponse, "description": "Authentication failed"},
        403: {"model": ErrorResponse, "description": "Authorization denied"},
    },
)
async def delete_message(
    http_request: Request,
    thread_id: str,
    message_id: str,
    workspace_id: str | None = Query(None, description="Workspace filter"),
    auth_service: AuthenticationService = Depends(get_auth_service),
    authz_service: AuthorizationService = Depends(get_authz_service),
    chat_service: ChatService = Depends(get_chat_service),
    audit_service: AuditService = Depends(get_audit_service),
    logger: logging.Logger = Depends(get_logger),
):
    """Delete a single message from a thread by ID. Returns 204 on success, 404 if not found."""
    try:
        ctx = await auth_service.build_context(http_request, None)
        workspace_id = workspace_id or ctx.workspace_id
        await authz_service.require_authorization(ctx, "threads", "write", workspace_id=workspace_id)

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        deleted = await chat_service.delete_message(workspace_id, thread_id, message_id, user_id=user_key)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Message {message_id} not found in thread {thread_id}",
            )

        try:
            await audit_service.record(
                AuditEvent(
                    event_type="chat",
                    action="delete",
                    tenant_id=ctx.tenant_id,
                    workspace_id=workspace_id,
                    user_id=ctx.user_id,
                    resource_type="message",
                    resource_id=message_id,
                )
            )
        except Exception:
            logger.debug("Audit record failed for message delete")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete message %s from thread %s: %s", message_id, thread_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete message",
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user
        # (fail-closed). The stored id is verbatim; responses round-trip it.
        user_key = _scope_user_id(ctx, workspace_id)

        # Convert API schema to domain input
        msg_inputs = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, list):
                content = [ChatMessageContent(**block) if isinstance(block, dict) else block for block in content]
            msg_inputs.append(
                MessageInput(
                    id=msg.id,
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
            tenant_id=ctx.tenant_id,
            user_id=user_key,
        )

        # Get updated thread for message count (owner-scoped)
        thread = await chat_service.get_thread(workspace_id, thread_id, user_id=user_key)
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
            messages=_present_messages(messages, thread_id),
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        # Verify thread exists (owner-scoped)
        thread = await chat_service.get_thread(workspace_id, thread_id, user_id=user_key)
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
            user_id=user_key,
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
            messages=_present_messages(messages, thread_id),
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

        # SECURITY: owner-scope the _user_chat sentinel to the OBO human user.
        user_key = _scope_user_id(ctx, workspace_id)

        result = await chat_service.trigger_decomposition(workspace_id, thread_id, user_id=user_key)

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
            thread_id=thread_id,
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
