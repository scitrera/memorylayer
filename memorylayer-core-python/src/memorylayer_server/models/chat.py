"""
Chat history models for MemoryLayer.ai.

Chat threads provide persistent, append-only conversation storage scoped to
workspace / user / thread. Messages accumulate over time and are periodically
decomposed into long-term memories via background tasks.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


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

    # Counters and watermarks
    message_count: int = Field(0, description="Total messages in thread")
    last_decomposed_at: datetime | None = Field(None, description="When decomposition last ran")
    last_decomposed_index: int = Field(0, description="Message index watermark for decomposition")

    # Lifecycle
    expires_at: datetime | None = Field(None, description="Optional expiration (None = permanent)")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update timestamp",
    )

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at

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


class AppendMessagesInput(BaseModel):
    """Input for appending messages to a thread."""

    messages: list["MessageInput"] = Field(..., min_length=1, description="Messages to append")


class MessageInput(BaseModel):
    """A single message to append (no ID or index — assigned by the service)."""

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
