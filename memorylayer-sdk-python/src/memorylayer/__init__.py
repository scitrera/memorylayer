"""MemoryLayer.ai Python SDK - Memory infrastructure for AI agents."""

from .client import MemoryLayerClient
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    EnterpriseRequiredError,
    MemoryLayerError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    Association,
    ChatMessage,
    ChatMessageContent,
    ChatThread,
    ChatThreadWithMessages,
    DatasetColumn,
    DatasetInfo,
    DatasetJobInfo,
    DatasetSliceResult,
    DecompositionResult,
    DocumentInfo,
    DocumentPage,
    JobInfo,
    Memory,
    PageSearchResult,
    RecallResult,
    ReflectResult,
    Session,
    SessionBriefing,
    Workspace,
)
from .sync_client import SyncMemoryLayerClient, sync_client
from .types import (
    MemorySubtype,
    MemoryType,
    RecallMode,
    RelationshipCategory,
    RelationshipType,
    SearchTolerance,
)

__version__ = "0.0.6"

__all__ = [
    # Main clients
    "MemoryLayerClient",
    "SyncMemoryLayerClient",
    "sync_client",
    # Models
    "Memory",
    "RecallResult",
    "ReflectResult",
    "Association",
    "Session",
    "SessionBriefing",
    "Workspace",
    # Chat history models
    "ChatMessage",
    "ChatMessageContent",
    "ChatThread",
    "ChatThreadWithMessages",
    "DecompositionResult",
    # Document models (Enterprise)
    "DocumentInfo",
    "DocumentPage",
    "JobInfo",
    "PageSearchResult",
    # Dataset models (Enterprise)
    "DatasetColumn",
    "DatasetInfo",
    "DatasetJobInfo",
    "DatasetSliceResult",
    # Types
    "MemoryType",
    "MemorySubtype",
    "RecallMode",
    "SearchTolerance",
    "RelationshipType",
    "RelationshipCategory",
    # Exceptions
    "MemoryLayerError",
    "AuthenticationError",
    "AuthorizationError",
    "EnterpriseRequiredError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
]
