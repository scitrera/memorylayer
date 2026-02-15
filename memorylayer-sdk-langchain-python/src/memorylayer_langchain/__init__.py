"""MemoryLayer LangChain Integration - Persistent memory for LangChain agents."""

from .chat_message_history import MemoryLayerChatMessageHistory
from .memory import MemoryLayerConversationSummaryMemory, MemoryLayerMemory

# Re-export SDK's sync client for backwards compatibility
from memorylayer import SyncMemoryLayerClient, sync_client

__version__ = "0.0.4"

__all__ = [
    # LangChain Chat History (LCEL compatible)
    "MemoryLayerChatMessageHistory",
    # Legacy BaseMemory implementations
    "MemoryLayerMemory",
    "MemoryLayerConversationSummaryMemory",
    # Synchronous client utilities (from SDK)
    "SyncMemoryLayerClient",
    "sync_client",
]
