"""No-op audit service - silently discards all events (OSS default)."""
from datetime import datetime
from logging import Logger
from typing import Optional

from scitrera_app_framework.api import Variables

from .base import AuditEvent, AuditService, AuditServicePluginBase


class NoopAuditService(AuditService):
    """No-op audit service that silently ignores all events."""

    async def record(self, event: AuditEvent) -> None:
        pass

    async def record_batch(self, events: list[AuditEvent]) -> None:
        pass

    async def query(
        self,
        tenant_id: str,
        workspace_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        return []


class NoopAuditServicePlugin(AuditServicePluginBase):
    """Plugin for no-op audit service."""
    PROVIDER_NAME = 'noop'

    def initialize(self, v: Variables, logger: Logger) -> Optional[NoopAuditService]:
        return NoopAuditService()
