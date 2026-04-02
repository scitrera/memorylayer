"""
Default Chat Service implementation.

Delegates persistence to StorageBackend and schedules decomposition tasks
when the unprocessed message threshold is exceeded.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import get_logger, get_extension, Variables

from .base import ChatService, ChatServicePluginBase
from ..storage import StorageBackend
from ..tasks import TaskService
from .._constants import EXT_STORAGE_BACKEND, EXT_TASK_SERVICE
from ...config import (
    DEFAULT_TENANT_ID,
    DEFAULT_CONTEXT_ID,
    MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
    DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_THRESHOLD,
    MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
    DEFAULT_MEMORYLAYER_CHAT_AUTO_DECOMPOSE_INTERVAL,
)
from ...models.chat import (
    ChatThread,
    ChatMessage,
    ChatThreadWithMessages,
    CreateThreadInput,
    AppendMessagesInput,
    DecompositionResult,
)
from ...utils import generate_id, utc_now_iso

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
        thread_id = input.thread_id or generate_id()
        now = datetime.now(timezone.utc)

        thread = ChatThread(
            id=thread_id,
            workspace_id=workspace_id,
            tenant_id=tenant_id or DEFAULT_TENANT_ID,
            user_id=input.user_id,
            context_id=input.context_id or DEFAULT_CONTEXT_ID,
            observer_id=input.observer_id,
            subject_id=input.subject_id,
            title=input.title,
            metadata=input.metadata or {},
            message_count=0,
            last_decomposed_at=None,
            last_decomposed_index=0,
            expires_at=input.expires_at,
            created_at=now,
            updated_at=now,
        )

        result = await self.storage.create_thread(thread)
        self.logger.info("Created chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def get_thread(
            self,
            workspace_id: str,
            thread_id: str,
    ) -> Optional[ChatThread]:
        thread = await self.storage.get_thread(workspace_id, thread_id)
        if thread and thread.is_expired:
            self.logger.debug("Thread %s is expired, returning None", thread_id)
            return None
        return thread

    async def list_threads(
            self,
            workspace_id: str,
            user_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
    ) -> list[ChatThread]:
        return await self.storage.list_threads(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    async def update_thread(
            self,
            workspace_id: str,
            thread_id: str,
            **updates,
    ) -> Optional[ChatThread]:
        thread = await self.get_thread(workspace_id, thread_id)
        if not thread:
            return None
        result = await self.storage.update_thread(workspace_id, thread_id, **updates)
        if result:
            self.logger.info("Updated chat thread %s in workspace %s", thread_id, workspace_id)
        return result

    async def delete_thread(
            self,
            workspace_id: str,
            thread_id: str,
    ) -> bool:
        result = await self.storage.delete_thread(workspace_id, thread_id)
        if result:
            self.logger.info("Deleted chat thread %s from workspace %s", thread_id, workspace_id)
        return result

    async def append_messages(
            self,
            workspace_id: str,
            thread_id: str,
            input: AppendMessagesInput,
    ) -> list[ChatMessage]:
        # Verify thread exists and is not expired
        thread = await self.get_thread(workspace_id, thread_id)
        if not thread:
            raise ValueError(f"Thread {thread_id} not found in workspace {workspace_id}")

        result = await self.storage.append_messages(workspace_id, thread_id, input.messages)

        self.logger.debug(
            "Appended %d messages to thread %s (new total: %d)",
            len(result), thread_id, thread.message_count + len(result),
        )

        # Check if we should schedule auto-decomposition
        await self._maybe_schedule_decomposition(workspace_id, thread_id, thread, len(result))

        return result

    async def get_messages(
            self,
            workspace_id: str,
            thread_id: str,
            limit: int = 100,
            offset: int = 0,
            after_index: Optional[int] = None,
            order: str = "asc",
    ) -> list[ChatMessage]:
        return await self.storage.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            after_index=after_index,
            order=order,
        )

    async def get_thread_with_messages(
            self,
            workspace_id: str,
            thread_id: str,
            limit: int = 100,
            offset: int = 0,
            order: str = "asc",
    ) -> Optional[ChatThreadWithMessages]:
        thread = await self.get_thread(workspace_id, thread_id)
        if not thread:
            return None

        messages = await self.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            limit=limit,
            offset=offset,
            order=order,
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
    ) -> DecompositionResult:
        thread = await self.get_thread(workspace_id, thread_id)
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

        # Schedule decomposition task synchronously (will run in background)
        await self.task_service.schedule_task(
            CHAT_DECOMPOSITION_TASK,
            {
                "workspace_id": workspace_id,
                "thread_id": thread_id,
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
            elapsed = (datetime.now(timezone.utc) - thread.last_decomposed_at).total_seconds()
            if elapsed < self._auto_decompose_interval:
                return

        self.logger.info(
            "Auto-scheduling decomposition for thread %s (%d unprocessed messages)",
            thread_id, unprocessed,
        )

        try:
            await self.task_service.schedule_task(
                CHAT_DECOMPOSITION_TASK,
                {
                    "workspace_id": workspace_id,
                    "thread_id": thread_id,
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
