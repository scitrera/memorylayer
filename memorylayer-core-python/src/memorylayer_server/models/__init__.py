"""
Core domain models for MemoryLayer.ai.

Exports all Pydantic models for memory, associations, workspaces, and sessions.
"""

from .association import (
    KNOWN_RELATIONSHIP_TYPES,
    AssociateInput,
    Association,
    GraphPath,
    GraphQueryInput,
    GraphQueryResult,
    RelationshipCategory,
    get_relationship_category,
)
from .auth import AuthIdentity, RequestContext
from .chat import (
    AppendMessagesInput,
    ChatMessage,
    ChatMessageContent,
    ChatThread,
    ChatThreadWithMessages,
    CreateThreadInput,
    DecompositionResult,
    MessageInput,
)
from .data_provider import (
    DataProvider,
    DataProviderType,
)
from .document import (
    Document,
    DocumentChunk,
    DocumentExtractionOptions,
    DocumentPage,
    DocumentStatus,
    DocumentType,
    IngestionJob,
    JobStatus,
)
from .graph_analysis import (
    Bridge,
    CentralNode,
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)
from .generation import (
    EnrichmentPolicy,
    GenerationActivity,
    GenerationAuthorization,
    GenerationBudgetExceededError,
    GenerationLedger,
    GenerationNotAllowedError,
    GenerationSummary,
)
from .memory import (
    DetailLevel,
    Memory,
    MemoryMutation,
    MemoryMutationResult,
    MemoryReplaceInput,
    MemoryRevision,
    MemoryScope,
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
    SessionMemorySections,
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
from .workspace import (
    Context,
    ContextSettings,
    Workspace,
    WorkspaceSettings,
)
from .workspace_execution import (
    ExecutionSite,
    WorkspaceVCSObservation,
    WorkspaceView,
    WorkspaceViewCreateInput,
    WorkspaceViewKind,
    WorkspaceViewObservation,
    WorkspaceViewObservationInput,
    WorkspaceViewObservationReplaceInput,
    WorkspaceViewReplaceInput,
)

__all__ = [
    # Memory models
    "Memory",
    "MemoryMutation",
    "MemoryMutationResult",
    "MemoryReplaceInput",
    "MemoryRevision",
    "MemoryScope",
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
    "SessionMemorySections",
    # Association models
    "Association",
    "AssociateInput",
    "KNOWN_RELATIONSHIP_TYPES",
    "RelationshipCategory",
    "get_relationship_category",
    "GraphQueryInput",
    "GraphQueryResult",
    "GraphPath",
    # Workspace models
    "Workspace",
    "WorkspaceSettings",
    "Context",
    "ContextSettings",
    "ExecutionSite",
    "WorkspaceViewKind",
    "WorkspaceView",
    "WorkspaceViewCreateInput",
    "WorkspaceViewReplaceInput",
    "WorkspaceVCSObservation",
    "WorkspaceViewObservation",
    "WorkspaceViewObservationInput",
    "WorkspaceViewObservationReplaceInput",
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
    # Chat history models
    "ChatMessage",
    "ChatMessageContent",
    "ChatThread",
    "ChatThreadWithMessages",
    "CreateThreadInput",
    "AppendMessagesInput",
    "MessageInput",
    "DecompositionResult",
    # Data provider models
    "DataProvider",
    "DataProviderType",
    # Document models
    "Document",
    "DocumentChunk",
    "DocumentExtractionOptions",
    "DocumentPage",
    "DocumentStatus",
    "DocumentType",
    "IngestionJob",
    "JobStatus",
    # Graph analysis models
    "Bridge",
    "CentralNode",
    "Community",
    "GraphAnalysis",
    "GraphSnapshot",
    "GraphStats",
    # Generation policy models
    "EnrichmentPolicy",
    "GenerationActivity",
    "GenerationAuthorization",
    "GenerationBudgetExceededError",
    "GenerationLedger",
    "GenerationNotAllowedError",
    "GenerationSummary",
]
