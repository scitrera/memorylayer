"""
Unit tests for the audit service — AuditEvent dataclass and NoopAuditService.
"""

from datetime import UTC, datetime

import pytest

from memorylayer_server.services.audit.base import AuditEvent
from memorylayer_server.services.audit.noop import NoopAuditService

# ============================================================================
# AuditEvent dataclass tests
# ============================================================================


class TestAuditEvent:
    """Test AuditEvent dataclass default values and field types."""

    def test_id_is_32_char_hex_string_by_default(self):
        """id defaults to a 32-character hex string (uuid4.hex)."""
        event = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        assert isinstance(event.id, str)
        assert len(event.id) == 32
        # uuid4().hex contains only hex characters
        int(event.id, 16)  # raises ValueError if not valid hex

    def test_id_is_unique_across_instances(self):
        """Each AuditEvent gets a distinct id by default."""
        e1 = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        e2 = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        assert e1.id != e2.id

    def test_timestamp_defaults_to_utc(self):
        """timestamp defaults to a timezone-aware UTC datetime."""
        before = datetime.now(UTC)
        event = AuditEvent(event_type="memory", action="create", tenant_id="t1")
        after = datetime.now(UTC)

        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == UTC
        assert before <= event.timestamp <= after

    def test_metadata_defaults_to_empty_dict(self):
        """metadata defaults to an empty dict and is not shared between instances."""
        e1 = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        e2 = AuditEvent(event_type="auth", action="login", tenant_id="t1")

        assert e1.metadata == {}
        assert isinstance(e1.metadata, dict)
        # Verify field_factory creates independent dicts (no shared mutable default)
        e1.metadata["key"] = "value"
        assert e2.metadata == {}

    def test_optional_fields_default_to_none(self):
        """workspace_id, user_id, resource_type, and resource_id default to None."""
        event = AuditEvent(event_type="session", action="delete", tenant_id="t1")

        assert event.workspace_id is None
        assert event.user_id is None
        assert event.resource_type is None
        assert event.resource_id is None

    def test_required_fields_are_stored(self):
        """event_type, action, and tenant_id are stored as supplied."""
        event = AuditEvent(event_type="workspace", action="update", tenant_id="acme")

        assert event.event_type == "workspace"
        assert event.action == "update"
        assert event.tenant_id == "acme"

    def test_optional_fields_accept_values(self):
        """Optional fields can be set and are stored correctly."""
        event = AuditEvent(
            event_type="memory",
            action="read",
            tenant_id="t1",
            workspace_id="ws-1",
            user_id="user-42",
            resource_type="memory",
            resource_id="mem-99",
            metadata={"ip": "127.0.0.1"},
        )

        assert event.workspace_id == "ws-1"
        assert event.user_id == "user-42"
        assert event.resource_type == "memory"
        assert event.resource_id == "mem-99"
        assert event.metadata == {"ip": "127.0.0.1"}


# ============================================================================
# NoopAuditService tests
# ============================================================================


class TestNoopAuditService:
    """Test NoopAuditService — all operations complete silently."""

    @pytest.mark.asyncio
    async def test_record_single_event_succeeds_silently(self):
        """record() completes without raising for a valid AuditEvent."""
        service = NoopAuditService()
        event = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        # Must not raise
        result = await service.record(event)
        assert result is None

    @pytest.mark.asyncio
    async def test_record_batch_succeeds_silently(self):
        """record_batch() completes without raising for a list of events."""
        service = NoopAuditService()
        events = [
            AuditEvent(event_type="memory", action="create", tenant_id="t1"),
            AuditEvent(event_type="memory", action="delete", tenant_id="t1"),
        ]
        result = await service.record_batch(events)
        assert result is None

    @pytest.mark.asyncio
    async def test_record_batch_empty_list_succeeds_silently(self):
        """record_batch() accepts an empty list without raising."""
        service = NoopAuditService()
        result = await service.record_batch([])
        assert result is None

    @pytest.mark.asyncio
    async def test_query_returns_empty_list(self):
        """query() always returns an empty list regardless of filters."""
        service = NoopAuditService()
        results = await service.query(tenant_id="t1")
        assert results == []
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_query_with_all_filters_returns_empty_list(self):
        """query() with every optional filter still returns an empty list."""
        service = NoopAuditService()
        results = await service.query(
            tenant_id="t1",
            workspace_id="ws-1",
            event_type="auth",
            since=datetime.now(UTC),
            limit=50,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_query_after_record_still_returns_empty_list(self):
        """NoopAuditService does not persist events — query always returns empty."""
        service = NoopAuditService()
        event = AuditEvent(event_type="auth", action="login", tenant_id="t1")
        await service.record(event)
        results = await service.query(tenant_id="t1")
        assert results == []
