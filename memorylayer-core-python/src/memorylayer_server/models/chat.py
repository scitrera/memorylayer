"""
Chat history models for MemoryLayer.ai.

Chat threads provide persistent, append-only conversation storage scoped to
workspace / user / thread. Messages accumulate over time and are periodically
decomposed into long-term memories via background tasks.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


USER_CHAT_HOME_WORKSPACE = "_user_chat"
"""Storage workspace sentinel for user-owned chat threads.

User-scoped chat history is cross-workspace: when a thread has
``ownership='user'`` (the default) it lives under this sentinel
regardless of which app workspace the user was in at write time.
The user's actual workspace context rides on per-message metadata
under :data:`MESSAGE_META_APP_WORKSPACE_KEY`.

MUST match the same constant in ``memorylayer-sdk-python``
(``memorylayer/constants.py``). Keep in sync.
"""

THREAD_OWNERSHIP_USER = "user"
THREAD_OWNERSHIP_WORKSPACE = "workspace"


def ownership_for_home(workspace_id: str | None) -> str:
    """Return the ``ownership`` discriminator implied by a thread's storage home.

    The sentinel and a real workspace are the only two homes, so ownership is a
    denormalized restatement of ``workspace_id == USER_CHAT_HOME_WORKSPACE``.
    Deriving it HERE, in one place, is deliberate: the alternative is scattering
    bare ``== USER_CHAT_HOME_WORKSPACE`` tests at each write site, which is how
    the column drifted out of sync with the home in the first place (auto-created
    rows kept the 'user' default while living in a real workspace).

    NOTE this maps HOME -> ownership. It says nothing about KEYING: threads in
    both homes are keyed ``(workspace_id, user_id, id)``. See ``_scope_user_id``
    in ``api/v1/chat.py``.
    """
    return (
        THREAD_OWNERSHIP_USER
        if workspace_id == USER_CHAT_HOME_WORKSPACE
        else THREAD_OWNERSHIP_WORKSPACE
    )


MESSAGE_META_APP_WORKSPACE_KEY = "app_workspace"
"""Reserved key on :class:`ChatMessage.metadata` for the originating app
workspace at write time (mirrors ``asgi_bridge``'s
``scope["state"]["app_workspace"]`` naming).

When a writer appends to a user-owned thread from a workspace context X
(e.g., the user's current app workspace), the thread row lives in
:data:`USER_CHAT_HOME_WORKSPACE`, but each message preserves X under this
key so per-message indexing / filtering by origin workspace is still
possible.

MUST match the same constant in ``memorylayer-sdk-python``
(``memorylayer/constants.py``). Keep in sync.
"""


# ---------------------------------------------------------------------------
# Owner-scoped user chat threads (the _user_chat sentinel).
#
# SECURITY: user-owned chat threads all live in the shared USER_CHAT_HOME_WORKSPACE
# sentinel and the harness sends a shared client thread_id (e.g. "_default") for
# every user. Per-user isolation is enforced by SCOPING on the OBO human subject:
# thread identity is (workspace_id, user_id, id) with ``id`` stored VERBATIM. An
# opaque surrogate ``row_id`` (storage-internal, never exposed to callers) is the
# physical primary key. Every thread lookup for the sentinel MUST be scoped by the
# effective user; a missing OBO human user fails closed (never an unscoped query).
# ---------------------------------------------------------------------------


class ChatMessageContent(BaseModel):
    """Structured content block within a chat message (tool calls, images, etc.)."""

    type: str = Field(..., description="Content block type: text, tool_call, tool_result, image, etc.")
    text: str | None = Field(None, description="Text content (for type=text)")
    data: dict[str, Any] | None = Field(None, description="Structured data (tool args, image ref, etc.)")


class ChatMessage(BaseModel):
    """A single turn in a chat thread."""

    model_config = {"from_attributes": True}

    id: str = Field(..., description="Message ID (auto-generated)")
    thread_id: str = Field(..., description="Parent thread ID")
    message_index: int = Field(..., description="Sequential index within the thread (0-based)")
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str | list[ChatMessageContent] = Field(..., description="Message content — plain string or structured blocks")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("Role cannot be empty")
        return v


class ChatThread(BaseModel):
    """A conversation thread — long-lived, append-only message log."""

    model_config = {"from_attributes": True}

    # Identity
    id: str = Field(..., description="Thread ID (client-provided or auto-generated)")
    workspace_id: str = Field(..., description="Workspace boundary")
    tenant_id: str = Field("_default", description="Tenant")
    user_id: str | None = Field(None, description="User who owns this conversation")
    context_id: str = Field("_default", description="Context within the workspace")

    # Entity attribution (for persona tracking / inference)
    observer_id: str | None = Field(None, description="Entity doing the observing (typically the AI agent)")
    subject_id: str | None = Field(None, description="Entity being observed (typically the human user)")

    # Display
    title: str | None = Field(None, description="Optional display title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata")

    # Surface scope — separates web-app threads from Office add-in threads.
    # NULL in the database is treated as "web" at read time (no backfill required).
    scope: str | None = Field(None, description="Surface scope: 'web' | 'office' | None (≡ 'web')")

    # Ownership discriminator — separates user-owned threads (the user-session-scoped
    # right rail in the web app) from workspace-shared threads. Defaults to 'user';
    # existing per-workspace `_default` rows backfill to 'user' via server_default.
    ownership: str = Field('user', description="Thread ownership: 'user' | 'workspace'")

    # Sub-thread parent. NULL = top-level thread (the default; listings return only
    # top-level threads unless a parent is requested). A child always shares its
    # parent's workspace + ownership (enforced at create time).
    parent_thread: str | None = Field(None, description="Parent thread id (None = top-level)")

    # Counters and watermarks
    message_count: int = Field(0, description="Total messages in thread")
    last_decomposed_at: datetime | None = Field(None, description="When decomposition last ran")
    last_decomposed_index: int = Field(0, description="Message index watermark for decomposition")

    # Lifecycle
    expires_at: datetime | None = Field(None, description="Optional absolute expiration (None = permanent)")
    # Per-thread idle policy. None = no idle handling (current behavior); 'hide'
    # = archive the thread once idle past the server idle threshold; 'delete' =
    # remove it. Idle is measured against ``updated_at`` (bumped on every append).
    idle_action: str | None = Field(None, description="Idle policy: None | 'hide' | 'delete'")
    # Archive flag. Set when the thread is hidden (manually or by the idle 'hide'
    # policy); cleared on revival (a new append un-hides). NULL = visible.
    hidden_at: datetime | None = Field(None, description="When the thread was archived/hidden (None = visible)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    @field_validator("idle_action")
    @classmethod
    def validate_idle_action(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if v == "":
            return None
        if v not in ("hide", "delete"):
            raise ValueError("idle_action must be None, 'hide', or 'delete'")
        return v

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

    @property
    def is_hidden(self) -> bool:
        return self.hidden_at is not None

    @property
    def unprocessed_count(self) -> int:
        return max(0, self.message_count - self.last_decomposed_index)


class ChatThreadWithMessages(BaseModel):
    """A thread with its messages inlined — used for full retrieval."""

    thread: ChatThread
    messages: list[ChatMessage] = Field(default_factory=list)
    total_messages: int = Field(0, description="Total message count (may exceed len(messages) if paginated)")


# Input models (for service layer — no IDs, no timestamps)


class CreateThreadInput(BaseModel):
    """Input for creating a new chat thread."""

    thread_id: str | None = Field(None, description="Client-provided thread ID (auto-generated if omitted)")
    user_id: str | None = Field(None, description="User scope")
    context_id: str = Field("_default", description="Context within workspace")
    observer_id: str | None = Field(None, description="Observer entity ID")
    subject_id: str | None = Field(None, description="Subject entity ID")
    title: str | None = Field(None, description="Display title")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    expires_at: datetime | None = Field(None, description="Optional expiration")
    idle_action: str | None = Field(None, description="Idle policy: None | 'hide' | 'delete'")
    scope: str | None = Field(None, description="Surface scope: 'web' | 'office' | None (≡ 'web')")
    ownership: str = Field('user', description="Thread ownership: 'user' | 'workspace'")
    parent_thread: str | None = Field(None, description="Parent thread id for a sub-thread (None = top-level)")

    @field_validator("idle_action")
    @classmethod
    def validate_idle_action(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().lower()
        if v == "":
            return None
        if v not in ("hide", "delete"):
            raise ValueError("idle_action must be None, 'hide', or 'delete'")
        return v


class AppendMessagesInput(BaseModel):
    """Input for appending messages to a thread."""

    messages: list["MessageInput"] = Field(..., min_length=1, description="Messages to append")


class MessageInput(BaseModel):
    """A single message to append (index assigned by the service; id assigned by the service unless the caller supplies one)."""

    id: str | None = Field(default=None, description="Optional client-provided id; server generates one when omitted.")
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str | list[ChatMessageContent] = Field(..., description="Message content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("Role cannot be empty")
        return v


class DecompositionResult(BaseModel):
    """Result of decomposing a thread's messages into memories."""

    thread_id: str
    workspace_id: str
    messages_processed: int = 0
    memories_created: int = 0
    from_index: int = 0
    to_index: int = 0
