"""Chat message history implementation using MemoryLayer as the backend."""

import logging
from typing import Any, Sequence

import httpx
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict,
)

from memorylayer import SyncMemoryLayerClient

logger = logging.getLogger(__name__)


class MemoryLayerChatMessageHistory(BaseChatMessageHistory):
    """
    Chat message history that persists to MemoryLayer.

    This class implements LangChain's BaseChatMessageHistory interface,
    storing messages as episodic memories in MemoryLayer. Messages are
    isolated by session_id, allowing multiple conversations to be tracked
    independently.

    Usage:
        from memorylayer_langchain import MemoryLayerChatMessageHistory

        # Create history for a conversation
        history = MemoryLayerChatMessageHistory(
            session_id="user_123_conversation_1",
            base_url="http://localhost:61001",
            api_key="your-api-key",
            workspace_id="ws_123"
        )

        # Add messages
        history.add_user_message("Hello!")
        history.add_ai_message("Hi there! How can I help?")

        # Get all messages
        messages = history.messages

        # Clear history
        history.clear()

    With RunnableWithMessageHistory (LCEL):
        from langchain_core.runnables.history import RunnableWithMessageHistory

        chain_with_history = RunnableWithMessageHistory(
            runnable=your_chain,
            get_session_history=lambda session_id: MemoryLayerChatMessageHistory(
                session_id=session_id,
                base_url="http://localhost:61001",
                api_key="your-api-key",
                workspace_id="ws_123"
            ),
            input_messages_key="input",
            history_messages_key="history",
        )
    """

    def __init__(
        self,
        session_id: str,
        base_url: str = "http://localhost:61001",
        api_key: str | None = None,
        workspace_id: str | None = None,
        timeout: float = 30.0,
        memory_tags: list[str] | None = None,
        max_messages: int = 1000,
    ) -> None:
        """
        Initialize MemoryLayer chat message history.

        Args:
            session_id: Unique identifier for this conversation session.
                Used to isolate messages between different conversations.
            base_url: MemoryLayer API base URL (default: http://localhost:61001)
            api_key: API key for authentication
            workspace_id: Workspace ID for multi-tenant isolation
            timeout: Request timeout in seconds (default: 30.0)
            memory_tags: Additional tags to add to stored memories
            max_messages: Maximum number of messages to retrieve (default: 1000)
        """
        self.session_id = session_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace_id = workspace_id
        self.timeout = timeout
        self.memory_tags = memory_tags or []
        self.max_messages = max_messages

        # Build headers (kept for backward compatibility with tests)
        self._headers: dict[str, str] = {}
        if self.api_key:
            self._headers["Authorization"] = f"Bearer {self.api_key}"
        if self.workspace_id:
            self._headers["X-Workspace-ID"] = self.workspace_id

        # Initialize the synchronous client
        self._client = SyncMemoryLayerClient(
            base_url=self.base_url,
            api_key=self.api_key,
            workspace_id=self.workspace_id,
            timeout=self.timeout,
        )
        self._client.connect()

    def close(self) -> None:
        """Close the underlying MemoryLayer client and release resources."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            # SyncMemoryLayerClient is expected to provide a close() method.
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            # Avoid raising during cleanup; just log the exception.
            logger.exception("Error while closing SyncMemoryLayerClient")

    def __enter__(self) -> "MemoryLayerChatMessageHistory":
        """Allow use as a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Ensure resources are cleaned up when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup if the instance is garbage-collected."""
        try:
            self.close()
        except Exception:
            # Never propagate exceptions from __del__.
            pass

    def _message_to_role(self, message: BaseMessage) -> str:
        """Convert LangChain message to role string."""
        if isinstance(message, HumanMessage):
            return "human"
        elif isinstance(message, AIMessage):
            return "ai"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            return message.type

    def _role_to_message_class(self, role: str) -> type[BaseMessage]:
        """Convert role string to LangChain message class."""
        role_map: dict[str, type[BaseMessage]] = {
            "human": HumanMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }
        return role_map.get(role, HumanMessage)

    @property
    def messages(self) -> list[BaseMessage]:
        """
        Retrieve all messages for this session from MemoryLayer.

        Returns:
            List of LangChain BaseMessage objects, ordered by creation time.
        """
        try:
            # Search for memories with this session's tag
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"chat history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "chat_message"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )

            memories = result.memories

            # Sort by message_index from metadata
            memories.sort(key=lambda m: m.metadata.get("message_index", 0))

            # Convert to LangChain messages
            messages: list[BaseMessage] = []
            for memory in memories:
                role = memory.metadata.get("role", "human")
                content = memory.content

                # Reconstruct the message using stored dict if available
                if "message_data" in memory.metadata:
                    # Full message reconstruction with all metadata
                    restored = messages_from_dict([memory.metadata["message_data"]])
                    if restored:
                        messages.append(restored[0])
                        continue

                # Fallback: create message from role and content
                message_class = self._role_to_message_class(role)
                messages.append(message_class(content=content))

            return messages

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to retrieve messages: {e}")
            return []

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a single message to the history.

        Args:
            message: LangChain message to add
        """
        self.add_messages([message])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """
        Add multiple messages to the history.

        This is the preferred method for adding messages as it can be
        more efficient when adding multiple messages at once.

        Args:
            messages: Sequence of LangChain messages to add
        """
        # Get current message count for indexing
        current_messages = self.messages
        start_index = len(current_messages)

        for i, message in enumerate(messages):
            role = self._message_to_role(message)
            content = message.content if isinstance(message.content, str) else str(message.content)

            # Serialize the full message for perfect reconstruction
            message_data = messages_to_dict([message])[0]

            # Build tags for this message
            tags = [
                f"session:{self.session_id}",
                "chat_message",
                f"role:{role}",
                *self.memory_tags,
            ]

            # Store as episodic memory using the sync client
            try:
                from memorylayer import MemoryType
                self._client.remember(
                    content=content,
                    type=MemoryType.EPISODIC,
                    importance=0.5,
                    tags=tags,
                    metadata={
                        "session_id": self.session_id,
                        "role": role,
                        "message_index": start_index + i,
                        "message_data": message_data,
                    },
                )
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to store message: {e}")
                raise

    def clear(self) -> None:
        """
        Remove all messages for this session.

        This deletes all memories tagged with this session's ID.
        """
        try:
            # First, get all memories for this session
            from memorylayer import RecallMode, SearchTolerance
            result = self._client.recall(
                query=f"chat history for session {self.session_id}",
                tags=[f"session:{self.session_id}", "chat_message"],
                mode=RecallMode.RAG,
                limit=self.max_messages,
                min_relevance=0.0,
                tolerance=SearchTolerance.LOOSE,
            )

            memories = result.memories

            # Delete each memory
            for memory in memories:
                try:
                    self._client.forget(memory.id)
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Failed to delete memory {memory.id}: {e}")

        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to clear messages: {e}")
            raise
