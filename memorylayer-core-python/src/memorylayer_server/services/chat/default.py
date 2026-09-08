"""
Default Chat Service implementation.

Delegates persistence to StorageBackend and schedules decomposition tasks
when the unprocessed message threshold is exceeded.
"""

import logging
from datetime import UTC, datetime

from scitrera_app_framework import Variables, get_extension, get_logger

from ...config import (
    DEFAULT_CONTEXT_ID,
    DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
    DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
    DEFAULT_TENANT_ID,
    MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
    MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
)
from ...models.chat import (
    AppendMessagesInput,
    ChatMessage,
    ChatThread,
    ChatThreadWithMessages,
    CreateThreadInput,
    DecompositionResult,
    ownership_for_home,
)
from ...utils import generate_id
from .._constants import EXT_STORAGE_BACKEND, EXT_TASK_SERVICE
from ..storage import StorageBackend
from ..tasks import TaskService
from .base import ChatService, ChatServicePluginBase

CHAT_DECOMPOSITION_TASK = "chat_decomposition"


class DefaultChatService(ChatService):
    """Default chat service backed by StorageBackend."""

    def __init__(
        self,
        storage: StorageBackend,
        task_service: TaskService,
        v: Variables,
    ):
        self.storage = storage
        self.task_service = task_service
        self.logger = get_logger(v, name="ChatService")
        self._v = v

    @property
    def _auto_decompose_threshold(self) -> int:
        return self._v.get(
            MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
            default=DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
        )

    @property
    def _auto_decompose_interval(self) -> int:
        return self._v.get(
            MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
            default=DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
        )

    async def create_thread(
        self,
        workspace_id: str,
        tenant_id: str,
        input: CreateThreadInput,
    ) -> ChatThread:
        thread_id = input.thread_id or generate_id("thread")
        now = datetime.now(UTC)

        tenant = tenant_id or DEFAULT_TENANT_ID
        ownership = input.ownership
        scope = input.scope
        user_id = input.user_id

        # Sub-thread invariant: a child always shares its parent's workspace +
        # ownership (and tenant/user/scope). Look the parent up in this workspace
        # and inherit; a missing parent here means a cross-workspace/unknown parent
        # was requested, which would violate the shared-context guarantee. The
        # parent is owner-scoped to the same user so a shared client id resolves
        # to THIS user's parent thread.
        if input.parent_thread:
            parent = await self.storage.get_thread(workspace_id, input.parent_thread, user_id=user_id)
            if not parent:
                raise ValueError(
                    f"parent thread '{input.parent_thread}' not found in workspace {workspace_id}"
                )
            ownership = parent.ownership
            scope = parent.scope
            user_id = parent.user_id
            tenant = parent.tenant_id

        thread = ChatThread(
            id=thread_id,
            workspace_id=workspace_id,
            tenant_id=tenant,
            user_id=user_id,
            context_id=input.context_id or DEFAULT_CONTEXT_ID,
            observer_id=input.observer_id,
            subject_id=input.subject_id,
            title=input.title,
            metadata=input.metadata or {},
            message_count=0,
            last_decomposed_at=None,
            last_decomposed_index=0,
            expires_at=input.expires_at,
            idle_action=input.idle_action,
            created_at=now,
            updated_at=now,
            scope=scope,
            ownership=ownership,
            parent_thread=input.parent_thread,
        )

        result = await self.storage.create_thread(thread)
        self.logger.info("Created chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def get_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> ChatThread | None:
        thread = await self.storage.get_thread(workspace_id, thread_id, user_id=user_id)
        if thread and thread.is_expired:
            self.logger.debug("Thread %s is expired, returning None", thread_id)
            return None
        return thread

    async def list_threads(
        self,
        workspace_id: str,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        scope_filter: str | None = None,
        ownership_filter: str | None = None,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list[ChatThread]:
        return await self.storage.list_threads(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
            scope_filter=scope_filter,
            ownership_filter=ownership_filter,
            include_hidden=include_hidden,
            parent_thread=parent_thread,
        )

    async def list_user_threads(
        self,
        tenant_id: str,
        user_id: str,
        *,
        ownership: str = 'user',
        scope_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list[ChatThread]:
        return await self.storage.list_user_threads(
            tenant_id=tenant_id or DEFAULT_TENANT_ID,
            user_id=user_id,
            ownership=ownership,
            scope_filter=scope_filter,
            limit=limit,
            offset=offset,
            include_hidden=include_hidden,
            parent_thread=parent_thread,
        )

    async def hide_thread(
        self, workspace_id: str, thread_id: str, user_id: str | None = None
    ) -> ChatThread | None:
        """Archive (hide) a thread. Flushes any undecomposed tail first (async)."""
        thread = await self.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            return None
        await self._schedule_decompose_if_needed(workspace_id, thread)
        result = await self.storage.hide_thread(workspace_id, thread_id, user_id=user_id)
        if result:
            self.logger.info("Archived chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def unhide_thread(
        self, workspace_id: str, thread_id: str, user_id: str | None = None
    ) -> ChatThread | None:
        """Restore (un-archive) a thread."""
        result = await self.storage.unhide_thread(workspace_id, thread_id, user_id=user_id)
        if result:
            self.logger.info("Restored chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def _schedule_decompose_if_needed(self, workspace_id: str, thread: ChatThread) -> bool:
        """Schedule a chat_decomposition task when the thread has undecomposed
        messages. Returns True if a task was scheduled. Used on hide/delete
        transitions so the tail is extracted before (or in lieu of) removal."""
        if thread.unprocessed_count <= 0:
            return False
        try:
            await self.task_service.schedule_task(
                CHAT_DECOMPOSITION_TASK,
                {"workspace_id": workspace_id, "thread_id": thread.id, "user_id": thread.user_id},
            )
        except Exception as e:
            self.logger.warning("Failed to schedule decomposition for thread %s: %s", thread.id, e)
        return True

    async def update_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
        **updates,
    ) -> ChatThread | None:
        thread = await self.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            return None
        result = await self.storage.update_thread(workspace_id, thread_id, user_id=user_id, **updates)
        if result:
            self.logger.info("Updated chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def delete_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> bool:
        result = await self.storage.delete_thread(workspace_id, thread_id, user_id=user_id)
        if result:
            self.logger.info("Deleted chat thread %s from workspace %s", thread_id, workspace_id)
        return result

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        input: AppendMessagesInput,
        tenant_id: str = "",
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        # Get or auto-create thread on first append (owner-scoped)
        thread = await self.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            self.logger.info("Auto-creating thread %s in workspace %s", thread_id, workspace_id)
            # SECURITY: stamp user_id so an auto-created thread is attributable to
            # the OBO human subject supplied by the API. This applies to BOTH homes
            # (see _scope_user_id in api/v1/chat.py): the sentinel and a real
            # workspace are both keyed (workspace_id, user_id, id).
            #
            # ownership must be stamped explicitly and match the home. Letting it
            # fall through to the CreateThreadInput default ('user') labelled every
            # auto-created row 'user' even when it lived in a real workspace, so the
            # ownership-filtered listings could not see the very rows that held the
            # messages.
            thread = await self.create_thread(
                workspace_id=workspace_id,
                tenant_id=tenant_id,
                input=CreateThreadInput(
                    thread_id=thread_id,
                    title=thread_id,
                    user_id=user_id,
                    ownership=ownership_for_home(workspace_id),
                ),
            )

        result = await self.storage.append_messages(workspace_id, thread_id, input.messages, user_id=user_id)

        self.logger.debug(
            "Appended %d messages to thread %s (new total: %d)",
            len(result),
            thread_id,
            thread.message_count + len(result),
        )

        # Check if we should schedule auto-decomposition
        await self._maybe_schedule_decomposition(workspace_id, thread_id, thread, len(result))

        return result

    async def delete_message(
        self,
        workspace_id: str,
        thread_id: str,
        message_id: str,
        user_id: str | None = None,
    ) -> bool:
        return await self.storage.delete_message(workspace_id, thread_id, message_id, user_id=user_id)

    async def get_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        after_index: int | None = None,
        order: str = "asc",
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        return await self.storage.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            after_index=after_index,
            order=order,
            user_id=user_id,
        )

    async def get_thread_with_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        order: str = "asc",
        user_id: str | None = None,
    ) -> ChatThreadWithMessages | None:
        thread = await self.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            return None

        messages = await self.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            order=order,
            user_id=user_id,
        )

        return ChatThreadWithMessages(
            thread=thread,
            messages=messages,
            total_messages=thread.message_count,
        )

    async def trigger_decomposition(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> DecompositionResult:
        thread = await self.get_thread(workspace_id, thread_id, user_id=user_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found in workspace {workspace_id}")

        if thread.unprocessed_count == 0:
            return DecompositionResult(
                thread_id=thread_id,
                workspace_id=workspace_id,
                messages_processed=0,
                memories_created=0,
                from_index=thread.last_decomposed_index,
                to_index=thread.last_decomposed_index,
            )

        # Schedule decomposition task synchronously (will run in background). The
        # owner (user_id) rides on the payload so the worker can re-resolve the
        # owner-scoped thread from a shared client id like "_default".
        await self.task_service.schedule_task(
            CHAT_DECOMPOSITION_TASK,
            {
                "workspace_id": workspace_id,
                "thread_id": thread_id,
                "user_id": thread.user_id,
                "force": True,
            },
        )

        return DecompositionResult(
            thread_id=thread_id,
            workspace_id=workspace_id,
            messages_processed=thread.unprocessed_count,
            memories_created=0,  # Actual count determined by async task
            from_index=thread.last_decomposed_index,
            to_index=thread.message_count,
        )

    async def _maybe_schedule_decomposition(
        self,
        workspace_id: str,
        thread_id: str,
        thread: ChatThread,
        new_message_count: int,
    ) -> None:
        """Schedule decomposition if threshold conditions are met."""
        new_total = thread.message_count + new_message_count
        unprocessed = new_total - thread.last_decomposed_index

        if unprocessed < self._auto_decompose_threshold:
            return

        # Check time interval since last decomposition
        if thread.last_decomposed_at:
            elapsed = (datetime.now(UTC) - thread.last_decomposed_at).total_seconds()
            if elapsed < self._auto_decompose_interval:
                return

        self.logger.info(
            "Auto-scheduling decomposition for thread %s (%d unprocessed messages)",
            thread_id,
            unprocessed,
        )

        try:
            await self.task_service.schedule_task(
                CHAT_DECOMPOSITION_TASK,
                {
                    "workspace_id": workspace_id,
                    "thread_id": thread_id,
                    "user_id": thread.user_id,
                },
            )
        except Exception as e:
            self.logger.warning("Failed to schedule chat decomposition for thread %s: %s", thread_id, e)


class DefaultChatServicePlugin(ChatServicePluginBase):
    """Plugin for default chat service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        storage: StorageBackend = get_extension(EXT_STORAGE_BACKEND, v)
        task_service: TaskService = get_extension(EXT_TASK_SERVICE, v)
        return DefaultChatService(storage=storage, task_service=task_service, v=v)
