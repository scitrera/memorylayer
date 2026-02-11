"""
Core domain models for MemoryLayer.ai.

Exports all Pydantic models for memory, associations, workspaces, and sessions.
"""
from .association import (
    Association,
    AssociateInput,
    GraphPath,
    GraphQueryInput,
    GraphQueryResult,
    KNOWN_RELATIONSHIP_TYPES,
    RelationshipCategory,
    get_relationship_category,
)
from .memory import (
    DetailLevel,
    Memory,
    MemoryStatus,
    MemorySubtype,
    MemoryType,
    RecallInput,
    RecallMode,
    RecallResult,
    ReflectInput,
    ReflectResult,
    RememberInput,
    SearchTolerance,
)
from .session import (
    ActivitySummary,
    Contradiction,
    OpenThread,
    Session,
    SessionBriefing,
    WorkingMemory,
    WorkspaceSummary,
)
from .auth import AuthIdentity, RequestContext
from .workspace import (
    Context,
    ContextSettings,
    Workspace,
    WorkspaceSettings,
)

__all__ = [
    # Memory models
    "Memory",
    "MemoryStatus",
    "MemoryType",
    "MemorySubtype",
    "RememberInput",
    "RecallInput",
    "RecallResult",
    "RecallMode",
    "SearchTolerance",
    "DetailLevel",
    "ReflectInput",
    "ReflectResult",
    # Association models
    "Association",
    "AssociateInput",
    "KNOWN_RELATIONSHIP_TYPES",
    "RelationshipCategory",
    "get_relationship_category",
    "GraphQueryInput",
    "GraphQueryResult",
    "GraphPath",
    # Tenant models (v2)
    "Tenant",
    "TenantSettings",
    # Workspace models
    "Workspace",
    "WorkspaceSettings",
    "Context",
    "ContextSettings",
    # Session models
    "Session",
    "WorkingMemory",
    "SessionBriefing",
    "WorkspaceSummary",
    "ActivitySummary",
    "OpenThread",
    "Contradiction",
    # Auth models
    "AuthIdentity",
    "RequestContext",
]
