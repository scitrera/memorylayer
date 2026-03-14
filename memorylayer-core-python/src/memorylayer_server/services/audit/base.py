"""Audit Service - Pluggable audit logging interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_AUDIT_SERVICE, DEFAULT_MEMORYLAYER_AUDIT_SERVICE

from .._constants import EXT_AUDIT_SERVICE

# Re-export for backward compatibility
__all__ = [
    "AuditEvent",
    "AuditService",
    "AuditServicePluginBase",
    "EXT_AUDIT_SERVICE",
]


@dataclass
class AuditEvent:
    """Structured audit event."""

    event_type: str
    """Event category: 'auth', 'memory', 'session', 'workspace', 'admin'."""

    action: str
    """Operation performed: 'create', 'read', 'update', 'delete', 'login', 'deny'."""

    tenant_id: str
    """Tenant this event belongs to."""

    workspace_id: Optional[str] = None
    """Workspace scope, if applicable."""

    user_id: Optional[str] = None
    """Acting user or principal, if known."""

    resource_type: Optional[str] = None
    """Type of resource acted upon: 'memory', 'session', 'workspace', etc."""

    resource_id: Optional[str] = None
    """Identifier of the specific resource."""

    metadata: dict = field(default_factory=dict)
    """Additional context (IP address, user-agent, request ID, etc.)."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp when the event occurred."""

    id: str = field(default_factory=lambda: uuid4().hex)
    """Unique event identifier (uuid4 hex)."""


class AuditService(ABC):
    """Abstract audit service interface.

    Provides a structured event recording interface that can be implemented
    by different backends (no-op, PostgreSQL, etc.).
    """

    @abstractmethod
    async def record(self, event: AuditEvent) -> None:
        """Record a single audit event.

        Args:
            event: The audit event to record.
        """
        pass

    @abstractmethod
    async def record_batch(self, events: list[AuditEvent]) -> None:
        """Record multiple audit events in a single operation.

        Args:
            events: List of audit events to record.
        """
        pass

    @abstractmethod
    async def query(
        self,
        tenant_id: str,
        workspace_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events.

        Args:
            tenant_id: Tenant to query events for (required).
            workspace_id: Filter by workspace, or None for all workspaces.
            event_type: Filter by event type, or None for all types.
            since: Return only events at or after this UTC datetime.
            limit: Maximum number of events to return.

        Returns:
            List of matching audit events ordered by timestamp descending.
        """
        pass


# noinspection PyAbstractClass
class AuditServicePluginBase(Plugin):
    """Base plugin for audit service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_AUDIT_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_AUDIT_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_AUDIT_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        # Set a default value for MEMORYLAYER_AUDIT_SERVICE; defaults are lower priority than .set(...) values
        v.set_default_value(MEMORYLAYER_AUDIT_SERVICE, DEFAULT_MEMORYLAYER_AUDIT_SERVICE)
