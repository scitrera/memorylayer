"""
Chat Service — Base interface and plugin for chat history management.

Manages conversation threads and their messages, with automatic decomposition
of chat history into long-term memories via background tasks.
"""
import logging
from abc import ABC, abstractmethod
from typing import Optional

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_CHAT_SERVICE, DEFAULT_MEMORYLAYER_CHAT_SERVICE
from ...models.chat import (
    ChatThread,
    ChatMessage,
    ChatThreadWithMessages,
    CreateThreadInput,
    AppendMessagesInput,
    DecompositionResult,
)
from .._constants import EXT_CHAT_SERVICE, EXT_STORAGE_BACKEND, EXT_TASK_SERVICE


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
    ) -> Optional[ChatThread]:
        """Get thread metadata by ID."""
        pass

    @abstractmethod
    async def list_threads(
            self,
            workspace_id: str,
            user_id: Optional[str] = None,
            limit: int = 50,
            offset: int = 0,
    ) -> list[ChatThread]:
        """List threads in a workspace, optionally filtered by user."""
        pass

    @abstractmethod
    async def delete_thread(
            self,
            workspace_id: str,
            thread_id: str,
    ) -> bool:
        """Delete a thread and all its messages."""
        pass

    @abstractmethod
    async def append_messages(
            self,
            workspace_id: str,
            thread_id: str,
            input: AppendMessagesInput,
    ) -> list[ChatMessage]:
        """Append messages to a thread. Returns the created messages with IDs and indexes."""
        pass

    @abstractmethod
    async def get_messages(
            self,
            workspace_id: str,
            thread_id: str,
            limit: int = 100,
            offset: int = 0,
            after_index: Optional[int] = None,
            order: str = "asc",
    ) -> list[ChatMessage]:
        """Get messages from a thread with pagination."""
        pass

    @abstractmethod
    async def get_thread_with_messages(
            self,
            workspace_id: str,
            thread_id: str,
            limit: int = 100,
            offset: int = 0,
            order: str = "asc",
    ) -> Optional[ChatThreadWithMessages]:
        """Get thread metadata with messages inlined."""
        pass

    @abstractmethod
    async def trigger_decomposition(
            self,
            workspace_id: str,
            thread_id: str,
    ) -> DecompositionResult:
        """Trigger on-demand memory decomposition for unprocessed messages."""
        pass


# noinspection PyAbstractClass
class ChatServicePluginBase(Plugin):
    """Base plugin for chat service."""

    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_CHAT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CHAT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_CHAT_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_CHAT_SERVICE, DEFAULT_MEMORYLAYER_CHAT_SERVICE)

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_TASK_SERVICE)
