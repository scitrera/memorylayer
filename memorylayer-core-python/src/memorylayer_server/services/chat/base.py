"""
Chat Service — Base interface and plugin for chat history management.

Manages conversation threads and their messages, with automatic decomposition
of chat history into long-term memories via background tasks.
"""

import logging
from abc import ABC, abstractmethod

from ...config import DEFAULT_MEMORYLAYER_CHAT_SERVICE, MEMORYLAYER_CHAT_SERVICE
from ...models.chat import (
    AppendMessagesInput,
    ChatMessage,
    ChatThread,
    ChatThreadWithMessages,
    CreateThreadInput,
    DecompositionResult,
)
from .._constants import EXT_CHAT_SERVICE, EXT_STORAGE_BACKEND, EXT_TASK_SERVICE
from .._plugin_factory import make_service_plugin_base


class ChatService(ABC):
    """Interface for chat history service."""

    logger: logging.Logger = None

    @abstractmethod
    async def create_thread(
        self,
        workspace_id: str,
        tenant_id: str,
        input: CreateThreadInput,
    ) -> ChatThread:
        """Create a new chat thread."""
        pass

    @abstractmethod
    async def get_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> ChatThread | None:
        """Get thread metadata by ID, scoped by owner.

        ``user_id`` scopes the lookup to the owner of a user-owned (``_user_chat``)
        thread — required so a shared client id like ``"_default"`` resolves to the
        correct human's thread. ``None`` matches workspace-owned (NULL user) rows.
        """
        pass

    @abstractmethod
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
        """List threads in a workspace, optionally filtered by user, scope, and ownership.

        ``include_hidden=False`` (default) excludes archived/hidden threads.
        ``parent_thread=None`` (default) returns only top-level threads; pass a
        parent id to list that thread's sub-threads.
        """
        pass

    @abstractmethod
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
        """List threads owned by a user across all workspaces.

        Unlike list_threads, this method is keyed on (tenant_id, user_id, ownership)
        and returns threads regardless of workspace. Used for the user-session-scoped
        right rail in the web app. ``include_hidden=False`` excludes archived threads.
        ``parent_thread=None`` returns only top-level threads.
        """
        pass

    @abstractmethod
    async def hide_thread(
        self, workspace_id: str, thread_id: str, user_id: str | None = None
    ) -> ChatThread | None:
        """Archive (hide) a thread without deleting it, scoped by owner."""
        pass

    @abstractmethod
    async def unhide_thread(
        self, workspace_id: str, thread_id: str, user_id: str | None = None
    ) -> ChatThread | None:
        """Restore (un-archive) a previously hidden thread, scoped by owner."""
        pass

    @abstractmethod
    async def update_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
        **updates,
    ) -> ChatThread | None:
        """Update thread fields (e.g. title, metadata), scoped by owner."""
        pass

    @abstractmethod
    async def delete_thread(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a thread and all its messages, scoped by owner."""
        pass

    @abstractmethod
    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        input: AppendMessagesInput,
        tenant_id: str = "",
        user_id: str | None = None,
    ) -> list[ChatMessage]:
        """Append messages to a thread. Auto-creates the thread if it doesn't exist.

        ``user_id``, when provided, is stamped as the owner on an auto-created
        thread (used to attribute user-owned ``_user_chat`` threads to the OBO
        human subject).
        """
        pass

    @abstractmethod
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
        """Get messages from a thread with pagination, scoped by owner."""
        pass

    @abstractmethod
    async def get_thread_with_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        order: str = "asc",
        user_id: str | None = None,
    ) -> ChatThreadWithMessages | None:
        """Get thread metadata with messages inlined, scoped by owner."""
        pass

    @abstractmethod
    async def delete_message(
        self,
        workspace_id: str,
        thread_id: str,
        message_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a single message from a thread, scoped by owner.

        Returns True if found and deleted, False if not found.
        Idempotent: a missing message returns False without raising.
        """
        pass

    @abstractmethod
    async def trigger_decomposition(
        self,
        workspace_id: str,
        thread_id: str,
        user_id: str | None = None,
    ) -> DecompositionResult:
        """Trigger on-demand memory decomposition for unprocessed messages, scoped by owner."""
        pass


# noinspection PyAbstractClass
ChatServicePluginBase = make_service_plugin_base(
    ext_name=EXT_CHAT_SERVICE,
    config_key=MEMORYLAYER_CHAT_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_CHAT_SERVICE,
    dependencies=(EXT_STORAGE_BACKEND, EXT_TASK_SERVICE),
)
