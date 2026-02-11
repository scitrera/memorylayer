"""MemoryLayer ChatStore implementation for LlamaIndex.

This module provides a persistent chat storage backend using MemoryLayer.ai.

Utility Functions:
    chat_message_to_memory_payload: Convert a LlamaIndex ChatMessage to MemoryLayer memory payload
    memory_to_chat_message: Convert a MemoryLayer memory dict to a LlamaIndex ChatMessage
    message_role_to_string: Convert a LlamaIndex MessageRole to a string
    string_to_message_role: Convert a string to a LlamaIndex MessageRole
"""

import asyncio
import logging
import threading
from typing import Any, Optional

from llama_index.core.base.llms.types import TextBlock
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.storage.chat_store.base import BaseChatStore
from memorylayer import MemoryLayerClient, SyncMemoryLayerClient
from memorylayer.models import Memory
from memorylayer.types import MemoryType
from pydantic import Field

logger = logging.getLogger(__name__)


# Tag prefix for identifying chat messages
CHAT_KEY_TAG_PREFIX = "llamaindex_chat_key:"

# Role mapping from LlamaIndex MessageRole to string representation
ROLE_TO_STRING_MAP: dict[MessageRole, str] = {
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "assistant",
    MessageRole.SYSTEM: "system",
    MessageRole.TOOL: "tool",
    MessageRole.CHATBOT: "chatbot",
    MessageRole.MODEL: "model",
    MessageRole.FUNCTION: "function",
}

# Reverse mapping from string to MessageRole (including aliases)
STRING_TO_ROLE_MAP: dict[str, MessageRole] = {
    "user": MessageRole.USER,
    "human": MessageRole.USER,
    "assistant": MessageRole.ASSISTANT,
    "ai": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
    "tool": MessageRole.TOOL,
    "chatbot": MessageRole.CHATBOT,
    "model": MessageRole.MODEL,
    "function": MessageRole.FUNCTION,
}


def message_role_to_string(role: MessageRole) -> str:
    """
    Convert a LlamaIndex MessageRole to a string representation.

    This function maps LlamaIndex message roles to their string equivalents
    for storage in MemoryLayer metadata.

    Args:
        role: A LlamaIndex MessageRole enum value

    Returns:
        String representation of the role (e.g., "user", "assistant", "system", "tool")

    Examples:
        >>> from llama_index.core.llms import MessageRole
        >>> message_role_to_string(MessageRole.USER)
        'user'
        >>> message_role_to_string(MessageRole.ASSISTANT)
        'assistant'
    """
    if role in ROLE_TO_STRING_MAP:
        return ROLE_TO_STRING_MAP[role]
    # Fallback: use the value if it's a string, otherwise convert
    if hasattr(role, "value"):
        return str(role.value)
    return str(role)


def string_to_message_role(role_str: str) -> MessageRole:
    """
    Convert a string to a LlamaIndex MessageRole.

    This function maps string role representations (from MemoryLayer metadata)
    back to LlamaIndex MessageRole enum values.

    Args:
        role_str: String representation of the role (e.g., "user", "assistant")

    Returns:
        Corresponding LlamaIndex MessageRole enum value.
        Defaults to MessageRole.USER if the string is not recognized.

    Examples:
        >>> string_to_message_role("user")
        <MessageRole.USER: 'user'>
        >>> string_to_message_role("assistant")
        <MessageRole.ASSISTANT: 'assistant'>
        >>> string_to_message_role("unknown")  # Falls back to USER
        <MessageRole.USER: 'user'>
    """
    normalized = role_str.lower().strip()
    if normalized in STRING_TO_ROLE_MAP:
        return STRING_TO_ROLE_MAP[normalized]

    # Try to create MessageRole directly from the value
    try:
        return MessageRole(role_str)
    except ValueError:
        pass

    # Fallback to USER for unknown roles
    logger.warning(f"Unknown message role '{role_str}', defaulting to USER")
    return MessageRole.USER


def chat_message_to_memory_payload(
    message: ChatMessage,
    key: str,
    index: int,
    *,
    tag_prefix: str = CHAT_KEY_TAG_PREFIX,
    memory_type: str = "episodic",
    importance: float = 0.5,
    additional_tags: Optional[list[str]] = None,
    additional_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Convert a LlamaIndex ChatMessage to a MemoryLayer memory payload.

    This utility function transforms a ChatMessage into a dictionary suitable
    for creating a memory via the MemoryLayer API. It preserves message content,
    role, ordering, and any additional kwargs from the original message.

    The resulting memory uses:
    - type=episodic (default) for storing conversational context
    - Tags for key-based organization (e.g., by user or session)
    - Metadata for storing role, message index, and block data

    Args:
        message: The LlamaIndex ChatMessage to convert
        key: The chat key (e.g., user ID, session ID) for organizing messages
        index: Message index for maintaining conversation order
        tag_prefix: Prefix for the chat key tag (default: "llamaindex_chat_key:")
        memory_type: Memory type (default: "episodic")
        importance: Memory importance score 0.0-1.0 (default: 0.5)
        additional_tags: Additional tags to add to the memory
        additional_metadata: Additional metadata to merge with standard metadata

    Returns:
        Dictionary payload ready for MemoryLayer API /memories POST request

    Examples:
        >>> from llama_index.core.llms import ChatMessage, MessageRole
        >>> msg = ChatMessage.from_str("Hello!", role=MessageRole.USER)
        >>> payload = chat_message_to_memory_payload(msg, key="user_123", index=0)
        >>> payload["content"]
        'user: Hello!'
        >>> payload["type"]
        'episodic'
        >>> payload["metadata"]["role"]
        'user'
        >>> payload["metadata"]["message_index"]
        0
    """
    # Get content from message - try to extract clean text from blocks
    content = str(message)
    if message.blocks:
        text_parts = []
        for block in message.blocks:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        if text_parts:
            content = "\n".join(text_parts)

    # Build tags list
    tags = [f"{tag_prefix}{key}"]
    if additional_tags:
        tags.extend(additional_tags)

    # Build metadata
    metadata: dict[str, Any] = {
        "chat_key": key,
        "message_index": index,
        "role": message_role_to_string(message.role),
        "additional_kwargs": message.additional_kwargs or {},
        "blocks": [
            block.model_dump() if hasattr(block, "model_dump") else str(block)
            for block in (message.blocks or [])
        ],
    }
    if additional_metadata:
        metadata.update(additional_metadata)

    return {
        "content": content,
        "type": memory_type,
        "importance": importance,
        "tags": tags,
        "metadata": metadata,
    }


def memory_to_chat_message(memory: dict[str, Any] | Memory) -> ChatMessage:
    """
    Convert a MemoryLayer memory dict or Memory object to a LlamaIndex ChatMessage.

    This utility function reconstructs a ChatMessage from a memory dict or Memory object
    retrieved from the MemoryLayer API. It attempts to preserve the original
    message structure including role, content blocks, and additional kwargs.

    The function handles two scenarios:
    1. Full reconstruction: If blocks are stored in metadata, it reconstructs
       the original message structure
    2. Fallback: If no blocks are available, it creates a simple text message
       from the content field

    Args:
        memory: Memory dictionary or Memory object from MemoryLayer API response containing:
            - content: The text content of the message
            - metadata: Dictionary with role, blocks, additional_kwargs, etc.

    Returns:
        Reconstructed LlamaIndex ChatMessage instance

    Examples:
        >>> memory = {
        ...     "id": "mem_123",
        ...     "content": "user: Hello!",
        ...     "metadata": {
        ...         "role": "user",
        ...         "message_index": 0,
        ...         "blocks": [{"block_type": "text", "text": "Hello!"}],
        ...         "additional_kwargs": {}
        ...     }
        ... }
        >>> msg = memory_to_chat_message(memory)
        >>> msg.role
        <MessageRole.USER: 'user'>
    """
    # Handle both Memory objects and dicts
    if isinstance(memory, Memory):
        metadata = memory.metadata
        content = memory.content
    else:
        metadata = memory.get("metadata", {})
        content = memory.get("content", "")

    role_str = metadata.get("role", "user")
    role = string_to_message_role(role_str)

    # Reconstruct blocks from metadata if available
    blocks_data = metadata.get("blocks", [])
    additional_kwargs = metadata.get("additional_kwargs", {})

    if blocks_data:
        # Try to reconstruct from stored blocks
        blocks = []
        for block_data in blocks_data:
            if isinstance(block_data, dict) and block_data.get("block_type") == "text":
                blocks.append(TextBlock(text=block_data.get("text", "")))
            elif isinstance(block_data, str):
                blocks.append(TextBlock(text=block_data))

        if blocks:
            return ChatMessage(
                role=role,
                blocks=blocks,
                additional_kwargs=additional_kwargs,
            )

    # Fallback: use content field
    return ChatMessage.from_str(content=content, role=role, **additional_kwargs)


def get_message_index(memory: dict[str, Any] | Memory) -> int:
    """
    Extract the message index from a MemoryLayer memory.

    Args:
        memory: Memory dictionary or Memory object from MemoryLayer API response

    Returns:
        Message index (default 0 if not found)
    """
    if isinstance(memory, Memory):
        return memory.metadata.get("message_index", 0)
    return memory.get("metadata", {}).get("message_index", 0)


def get_chat_key(memory: dict[str, Any] | Memory) -> Optional[str]:
    """
    Extract the chat key from a MemoryLayer memory.

    Args:
        memory: Memory dictionary or Memory object from MemoryLayer API response

    Returns:
        Chat key string, or None if not found
    """
    if isinstance(memory, Memory):
        return memory.metadata.get("chat_key")
    return memory.get("metadata", {}).get("chat_key")


def string_to_memory_type(type_str: Optional[str]) -> Optional[MemoryType]:
    """
    Convert a string to MemoryType enum.

    Args:
        type_str: String representation of memory type (e.g., "episodic", "semantic")

    Returns:
        MemoryType enum value, or None if type_str is None
    """
    if type_str is None:
        return None
    try:
        return MemoryType(type_str)
    except ValueError:
        logger.warning(f"Unknown memory type '{type_str}', returning None")
        return None


class MemoryLayerChatStore(BaseChatStore):
    """
    A chat store that persists messages using MemoryLayer.ai.

    This chat store saves chat messages as MemoryLayer memories with type=episodic,
    enabling persistent chat history across application restarts.

    Usage:
        from memorylayer_llamaindex import MemoryLayerChatStore
        from llama_index.core.memory import ChatMemoryBuffer

        # Create the chat store
        chat_store = MemoryLayerChatStore(
            base_url="http://localhost:61001",
            api_key="your-api-key",
            workspace_id="ws_123"
        )

        # Use with ChatMemoryBuffer
        memory = ChatMemoryBuffer.from_defaults(
            chat_store=chat_store,
            chat_store_key="user_123"
        )

    Attributes:
        base_url: MemoryLayer API base URL
        api_key: API key for authentication
        workspace_id: Default workspace ID for operations
        timeout: Request timeout in seconds
    """

    base_url: str = Field(default="http://localhost:61001", description="MemoryLayer API base URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    workspace_id: Optional[str] = Field(default=None, description="Workspace ID")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")

    _sync_client: Optional[SyncMemoryLayerClient] = None
    _async_client: Optional[MemoryLayerClient] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_url: str = "http://localhost:61001",
        api_key: Optional[str] = None,
        workspace_id: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize MemoryLayerChatStore.

        Args:
            base_url: MemoryLayer API base URL (default: http://localhost:61001)
            api_key: API key for authentication
            workspace_id: Default workspace ID for operations
            timeout: Request timeout in seconds (default: 30.0)
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            workspace_id=workspace_id,
            timeout=timeout,
            **kwargs,
        )
        self._sync_client = None
        self._async_client = None
        # Threading lock for synchronizing add_message operations per key
        self._sync_message_locks: dict[str, Any] = {}
        # Asyncio locks for synchronizing async_add_message operations per key
        self._async_message_locks: dict[str, asyncio.Lock] = {}

    def _get_sync_client(self) -> SyncMemoryLayerClient:
        """Get or create sync SDK client."""
        if self._sync_client is None:
            self._sync_client = SyncMemoryLayerClient(
                base_url=self.base_url,
                api_key=self.api_key,
                workspace_id=self.workspace_id,
                timeout=self.timeout,
            )
            self._sync_client.connect()
        return self._sync_client

    def _get_async_client(self) -> MemoryLayerClient:
        """Get or create async SDK client."""
        if self._async_client is None:
            self._async_client = MemoryLayerClient(
                base_url=self.base_url,
                api_key=self.api_key,
                workspace_id=self.workspace_id,
                timeout=self.timeout,
            )
        return self._async_client

    def close(self) -> None:
        """Close any underlying synchronous HTTP client resources."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close any underlying asynchronous HTTP client resources."""
        if self._async_client is not None:
            # SDK client uses context manager, close via __aexit__
            if hasattr(self._async_client, '_client') and self._async_client._client:
                await self._async_client._client.aclose()
            self._async_client = None

    def __enter__(self) -> "MemoryLayerChatStore":
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Exit the runtime context and close synchronous resources."""
        self.close()

    async def __aenter__(self) -> "MemoryLayerChatStore":
        """
        Async context manager entry.

        Ensures that the async client is created and ready for use.
        """
        # Ensure the async client is initialized and entered
        client = self._get_async_client()
        await client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """
        Async context manager exit.

        Closes the async HTTP client.
        """
        await self.aclose()

    def _make_chat_key_tag(self, key: str) -> str:
        """Create a tag for identifying chat messages by key."""
        return f"{CHAT_KEY_TAG_PREFIX}{key}"

    def _chat_message_to_memory_payload(
        self, message: ChatMessage, key: str, index: int
    ) -> dict[str, Any]:
        """
        Convert a ChatMessage to a MemoryLayer memory payload.

        This is a convenience method that delegates to the module-level
        chat_message_to_memory_payload function.

        Args:
            message: The ChatMessage to convert
            key: The chat key (e.g., user ID)
            index: Message index for ordering

        Returns:
            Memory payload dict for API request
        """
        return chat_message_to_memory_payload(message, key, index)

    def _memory_to_chat_message(self, memory: dict[str, Any]) -> ChatMessage:
        """
        Convert a MemoryLayer memory to a ChatMessage.

        This is a convenience method that delegates to the module-level
        memory_to_chat_message function.

        Args:
            memory: Memory dict from API response

        Returns:
            ChatMessage instance
        """
        return memory_to_chat_message(memory)


    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MemoryLayerChatStore"

    # ========== Sync Methods ==========

    def set_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """
        Set messages for a key, replacing any existing messages.

        Args:
            key: The chat key (e.g., user ID or session ID)
            messages: List of messages to store
        """
        # First delete existing messages for this key
        self.delete_messages(key)

        # Then add all new messages
        client = self._get_sync_client()
        for idx, message in enumerate(messages):
            payload = self._chat_message_to_memory_payload(message, key, idx)
            client.remember(
                content=payload["content"],
                type=string_to_memory_type(payload.get("type")),
                importance=payload.get("importance", 0.5),
                tags=payload.get("tags"),
                metadata=payload.get("metadata"),
            )

    def _get_memories_for_key(self, key: str) -> list[Memory]:
        """
        Internal helper to fetch raw memory objects for a chat key.

        Args:
            key: The chat key

        Returns:
            List of Memory objects from MemoryLayer API
        """
        client = self._get_sync_client()
        tag = self._make_chat_key_tag(key)

        # Use recall to fetch memories with the chat key tag
        result = client.recall(
            query=f"chat history for {key}",
            tags=[tag],
            limit=1000,  # High limit to get all messages
            min_relevance=0.0,  # Get all matches regardless of relevance
        )

        # Sort by message index
        memories = result.memories
        memories.sort(key=lambda m: m.metadata.get("message_index", 0))

        return memories

    def get_messages(self, key: str) -> list[ChatMessage]:
        """
        Get messages for a key.

        Args:
            key: The chat key

        Returns:
            List of chat messages, ordered by index
        """
        memories = self._get_memories_for_key(key)
        return [self._memory_to_chat_message(mem) for mem in memories]

    def add_message(self, key: str, message: ChatMessage) -> None:
        """
        Add a message for a key.

        Args:
            key: The chat key
            message: Message to add
        """
        # Get or create the lock for this specific chat key (setdefault is atomic in CPython)
        lock = self._sync_message_locks.setdefault(key, threading.Lock())

        # Serialize index computation and write for this key to avoid race conditions
        with lock:
            # Get current message count to determine index
            existing = self.get_messages(key)
            index = len(existing)

            client = self._get_sync_client()
            payload = self._chat_message_to_memory_payload(message, key, index)
            client.remember(
                content=payload["content"],
                type=string_to_memory_type(payload.get("type")),
                importance=payload.get("importance", 0.5),
                tags=payload.get("tags"),
                metadata=payload.get("metadata"),
            )

    def delete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """
        Delete all messages for a key.

        Args:
            key: The chat key

        Returns:
            List of deleted messages, or None if none existed
        """
        client = self._get_sync_client()

        # Fetch memories once and extract both messages and IDs
        memories = self._get_memories_for_key(key)
        if not memories:
            return None

        # Convert to ChatMessage objects for return value
        messages = [self._memory_to_chat_message(mem) for mem in memories]

        # Delete each memory using the IDs we already have
        for memory in memories:
            try:
                client.forget(memory.id)
            except Exception:
                # Ignore errors (e.g., already deleted)
                pass

        return messages

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Delete a specific message by index.

        Args:
            key: The chat key
            idx: Message index to delete

        Returns:
            Deleted message, or None if not found
        """
        client = self._get_sync_client()

        # Fetch memories with this tag
        memories = self._get_memories_for_key(key)

        # Find memory with matching index
        target_memory = None
        for memory in memories:
            if memory.metadata.get("message_index") == idx:
                target_memory = memory
                break

        if target_memory is None:
            return None

        # Convert to ChatMessage before deletion
        deleted_message = self._memory_to_chat_message(target_memory)

        # Delete the memory
        try:
            client.forget(target_memory.id)
        except Exception:
            # Ignore errors (e.g., already deleted)
            pass

        return deleted_message

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Delete the last message for a key.

        Args:
            key: The chat key

        Returns:
            Deleted message, or None if no messages exist
        """
        messages = self.get_messages(key)
        if not messages:
            return None

        # Delete the message with the highest index
        return self.delete_message(key, len(messages) - 1)

    def get_keys(self) -> list[str]:
        """
        Get all chat keys.

        Returns:
            List of unique chat keys
        """
        client = self._get_sync_client()

        # Search for all memories with our tag prefix
        result = client.recall(
            query="chat history",
            limit=1000,
            min_relevance=0.0,
        )

        # Extract unique keys from metadata
        keys: set[str] = set()
        for memory in result.memories:
            chat_key = memory.metadata.get("chat_key")
            if chat_key:
                keys.add(chat_key)

        return list(keys)

    # ========== Async Methods ==========

    async def aset_messages(self, key: str, messages: list[ChatMessage]) -> None:
        """
        Async version of set_messages.

        Args:
            key: The chat key
            messages: List of messages to store
        """
        # First delete existing messages
        await self.adelete_messages(key)

        # Then add all new messages
        client = self._get_async_client()
        async with client:
            for idx, message in enumerate(messages):
                payload = self._chat_message_to_memory_payload(message, key, idx)
                await client.remember(
                    content=payload["content"],
                    type=string_to_memory_type(payload.get("type")),
                    importance=payload.get("importance", 0.5),
                    tags=payload.get("tags"),
                    metadata=payload.get("metadata"),
                )

    async def _aget_memories_for_key(self, key: str) -> list[Memory]:
        """
        Internal async helper to fetch raw memory objects for a chat key.

        Args:
            key: The chat key

        Returns:
            List of Memory objects from MemoryLayer API
        """
        client = self._get_async_client()
        tag = self._make_chat_key_tag(key)

        async with client:
            result = await client.recall(
                query=f"chat history for {key}",
                tags=[tag],
                limit=1000,
                min_relevance=0.0,
            )

        # Sort by message index
        memories = result.memories
        memories.sort(key=lambda m: m.metadata.get("message_index", 0))

        return memories

    async def aget_messages(self, key: str) -> list[ChatMessage]:
        """
        Async version of get_messages.

        Args:
            key: The chat key

        Returns:
            List of chat messages, ordered by index
        """
        memories = await self._aget_memories_for_key(key)
        return [self._memory_to_chat_message(mem) for mem in memories]

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """
        Async version of add_message.

        Args:
            key: The chat key
            message: Message to add
        """
        # Get or create the lock for this specific chat key (setdefault is atomic)
        lock = self._async_message_locks.setdefault(key, asyncio.Lock())

        # Serialize index computation and write for this key to avoid race conditions
        async with lock:
            # Get current message count to determine index
            existing = await self.aget_messages(key)
            index = len(existing)

            client = self._get_async_client()
            payload = self._chat_message_to_memory_payload(message, key, index)
            async with client:
                await client.remember(
                    content=payload["content"],
                    type=string_to_memory_type(payload.get("type")),
                    importance=payload.get("importance", 0.5),
                    tags=payload.get("tags"),
                    metadata=payload.get("metadata"),
                )

    async def adelete_messages(self, key: str) -> Optional[list[ChatMessage]]:
        """
        Async version of delete_messages.

        Args:
            key: The chat key

        Returns:
            List of deleted messages, or None if none existed
        """
        client = self._get_async_client()

        # Fetch memories once and extract both messages and IDs
        memories = await self._aget_memories_for_key(key)
        if not memories:
            return None

        # Convert to ChatMessage objects for return value
        messages = [self._memory_to_chat_message(mem) for mem in memories]

        # Delete each memory using the IDs we already have
        async with client:
            for memory in memories:
                try:
                    await client.forget(memory.id)
                except Exception:
                    # Ignore errors (e.g., already deleted)
                    pass

        return messages

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """
        Async version of delete_message.

        Args:
            key: The chat key
            idx: Message index to delete

        Returns:
            Deleted message, or None if not found
        """
        client = self._get_async_client()

        # Fetch memories with this tag
        memories = await self._aget_memories_for_key(key)

        # Find memory with matching index
        target_memory = None
        for memory in memories:
            if memory.metadata.get("message_index") == idx:
                target_memory = memory
                break

        if target_memory is None:
            return None

        # Convert to ChatMessage before deletion
        deleted_message = self._memory_to_chat_message(target_memory)

        # Delete the memory
        async with client:
            try:
                await client.forget(target_memory.id)
            except Exception:
                # Ignore errors (e.g., already deleted)
                pass

        return deleted_message

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """
        Async version of delete_last_message.

        Args:
            key: The chat key

        Returns:
            Deleted message, or None if no messages exist
        """
        messages = await self.aget_messages(key)
        if not messages:
            return None

        return await self.adelete_message(key, len(messages) - 1)

    async def aget_keys(self) -> list[str]:
        """
        Async version of get_keys.

        Returns:
            List of unique chat keys
        """
        client = self._get_async_client()

        async with client:
            result = await client.recall(
                query="chat history",
                limit=1000,
                min_relevance=0.0,
            )

        keys: set[str] = set()
        for memory in result.memories:
            chat_key = memory.metadata.get("chat_key")
            if chat_key:
                keys.add(chat_key)

        return list(keys)

    def __del__(self) -> None:
        """Clean up SDK clients."""
        if getattr(self, "_sync_client", None) is not None:
            try:
                self._sync_client.close()
            except Exception:
                # Suppress exceptions during destructor cleanup.
                pass

        async_client = getattr(self, "_async_client", None)
        if async_client is not None:
            try:
                # Avoid blocking or creating new event loops in __del__.
                # If there's a running loop, schedule aclose; otherwise rely
                # on explicit or context-managed cleanup.
                if hasattr(async_client, "_client") and async_client._client is not None:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running event loop; best-effort cleanup only.
                        return
                    else:
                        loop.create_task(async_client._client.aclose())
            except Exception:
                # Suppress exceptions during destructor cleanup.
                pass
