"""MemoryLayer.ai LlamaIndex Integration - Persistent memory for LlamaIndex applications."""

from .chat_store import (
    CHAT_KEY_TAG_PREFIX,
    MemoryLayerChatStore,
    chat_message_to_memory_payload,
    get_chat_key,
    get_message_index,
    memory_to_chat_message,
    message_role_to_string,
    string_to_message_role,
)

__version__ = "0.0.4"

__all__ = [
    # Main class
    "MemoryLayerChatStore",
    # Utility functions for ChatMessage <-> Memory conversion
    "chat_message_to_memory_payload",
    "memory_to_chat_message",
    # Role conversion utilities
    "message_role_to_string",
    "string_to_message_role",
    # Helper functions
    "get_message_index",
    "get_chat_key",
    # Constants
    "CHAT_KEY_TAG_PREFIX",
]
