"""No-op audit service - silently discards all events (OSS default)."""

from datetime import datetime
from logging import Logger

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
        workspace_id: str | None = None,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        return []


class NoopAuditServicePlugin(AuditServicePluginBase):
    """Plugin for no-op audit service."""

    PROVIDER_NAME = "noop"

    def initialize(self, v: Variables, logger: Logger) -> NoopAuditService | None:
        return NoopAuditService()
