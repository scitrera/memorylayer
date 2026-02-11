"""MemoryLayer.ai Python SDK - Memory infrastructure for AI agents."""

from .client import MemoryLayerClient
from .exceptions import (
    AuthenticationError,
    MemoryLayerError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .models import (
    Association,
    Memory,
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

__version__ = "0.0.3"

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
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
]
