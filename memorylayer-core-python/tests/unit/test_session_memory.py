"""
Unit tests for Phase 4: Session Memory (token budget tracking and structured sections).

Tests:
- Token estimation helper
- Cumulative token tracking in touch_session()
- Extraction trigger thresholds (init and growth)
- SessionMemorySections model (sections, total_tokens, add_entry, budget enforcement)
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.models.memory import (
    DEFAULT_SESSION_SECTION_TOKEN_BUDGET,
    SESSION_MEMORY_SECTION_NAMES,
    SessionMemorySections,
)
from memorylayer_server.services.session.persistent import PersistentSessionService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id="sess-1", workspace_id="ws-1", metadata=None):
    """Build a minimal Session-like mock."""
    from memorylayer_server.models.session import Session

    return Session(
        id=session_id,
        workspace_id=workspace_id,
        tenant_id="tenant-1",
        context_id="_default",
        metadata=metadata or {},
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


def _make_working_memory(key, value):
    """Build a minimal WorkingMemory-like mock."""
    from memorylayer_server.models.session import WorkingMemory

    return WorkingMemory(session_id="sess-1", key=key, value=value)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    """Tests for PersistentSessionService._estimate_tokens()."""

    def test_empty_string_returns_zero(self):
        assert PersistentSessionService._estimate_tokens("") == 0

    def test_four_chars_is_one_token(self):
        assert PersistentSessionService._estimate_tokens("abcd") == 1

    def test_twelve_chars_is_three_tokens(self):
        assert PersistentSessionService._estimate_tokens("a" * 12) == 3

    def test_non_divisible_truncates(self):
        # 10 chars → 10 // 4 = 2
        assert PersistentSessionService._estimate_tokens("a" * 10) == 2

    def test_long_content(self):
        content = "x" * 4000  # 4000 chars → 1000 tokens
        assert PersistentSessionService._estimate_tokens(content) == 1000


# ---------------------------------------------------------------------------
# touch_session token tracking
# ---------------------------------------------------------------------------


def _make_variables(token_trigger_init=10000, token_trigger_growth=5000):
    """Build a mock Variables that returns config values via environ()."""
    v = MagicMock()

    def _environ(key, default=None, type_fn=None):
        from memorylayer_server.config import (
            MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH,
            MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT,
        )

        mapping = {
            MEMORYLAYER_SESSION_TOKEN_TRIGGER_INIT: token_trigger_init,
            MEMORYLAYER_SESSION_TOKEN_TRIGGER_GROWTH: token_trigger_growth,
        }
        val = mapping.get(key, default)
        return type_fn(val) if type_fn and val is not None else val

    v.environ = _environ
    return v


class TestTouchSessionTokenTracking:
    """Tests for token budget tracking inside touch_session()."""

    def _make_service(self, storage, task_service=None, token_trigger_init=10000, token_trigger_growth=5000):
        v = _make_variables(token_trigger_init=token_trigger_init, token_trigger_growth=token_trigger_growth)
        return PersistentSessionService(
            storage=storage,
            v=v,
            task_service=task_service,
            default_touch_ttl=3600,
        )

    @pytest.mark.asyncio
    async def test_cumulative_tokens_stored_in_metadata(self):
        """touch_session() stores cumulative_tokens in session metadata."""
        session = _make_session()
        wm_entries = [_make_working_memory("key1", "a" * 400)]  # 100 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=10000)
        await svc.touch_session("ws-1", "sess-1")

        # update_session should have been called with metadata containing cumulative_tokens
        call_kwargs = storage.update_session.call_args
        metadata = call_kwargs.kwargs.get("metadata") or (call_kwargs.args[3] if len(call_kwargs.args) > 3 else None)
        assert metadata is not None
        assert metadata["cumulative_tokens"] == 100

    @pytest.mark.asyncio
    async def test_extraction_not_triggered_below_init_threshold(self):
        """No extraction task is scheduled when tokens are below init threshold."""
        session = _make_session()
        wm_entries = [_make_working_memory("key1", "a" * 400)]  # 100 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=1000, token_trigger_growth=500)
        await svc.touch_session("ws-1", "sess-1")

        task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_extraction_triggered_at_init_threshold(self):
        """Extraction task is scheduled when cumulative tokens reach init threshold."""
        session = _make_session()
        wm_entries = [_make_working_memory("key1", "a" * 4000)]  # 1000 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=500, token_trigger_growth=500)
        await svc.touch_session("ws-1", "sess-1")

        task_service.schedule_task.assert_called_once()
        call_args = task_service.schedule_task.call_args
        assert call_args.args[0] == "session_extraction"
        payload = call_args.args[1]
        assert payload["workspace_id"] == "ws-1"
        assert payload["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_extraction_triggered_on_growth_threshold(self):
        """Extraction is re-triggered after growth threshold exceeded."""
        # Session has already had one extraction at 1000 tokens
        session = _make_session(
            metadata={
                "cumulative_tokens": 1000,
                "last_extraction_tokens": 1000,
            }
        )
        # New working memory totals 1600 tokens (600 more than last extraction)
        wm_entries = [_make_working_memory("key1", "a" * 6400)]  # 1600 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=1000, token_trigger_growth=500)
        await svc.touch_session("ws-1", "sess-1")

        task_service.schedule_task.assert_called_once()
        call_args = task_service.schedule_task.call_args
        assert call_args.args[0] == "session_extraction"

    @pytest.mark.asyncio
    async def test_extraction_not_retriggered_below_growth_threshold(self):
        """No extraction if growth since last extraction is below growth threshold."""
        session = _make_session(
            metadata={
                "cumulative_tokens": 1000,
                "last_extraction_tokens": 1000,
            }
        )
        # New total is only 1200 (200 more than last extraction, below 500 growth threshold)
        wm_entries = [_make_working_memory("key1", "a" * 4800)]  # 1200 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=1000, token_trigger_growth=500)
        await svc.touch_session("ws-1", "sess-1")

        task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_crash_when_task_service_is_none(self):
        """touch_session() works fine when task_service is not set."""
        session = _make_session()
        wm_entries = [_make_working_memory("key1", "a" * 4000)]  # 1000 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        svc = self._make_service(storage, task_service=None, token_trigger_init=100)
        # Should not raise even though token budget would be exceeded
        result = await svc.touch_session("ws-1", "sess-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_last_extraction_tokens_updated_on_trigger(self):
        """last_extraction_tokens is set in metadata when extraction is triggered."""
        session = _make_session()
        wm_entries = [_make_working_memory("key1", "a" * 4000)]  # 1000 tokens

        storage = AsyncMock()
        storage.get_session = AsyncMock(return_value=session)
        storage.get_all_working_memory = AsyncMock(return_value=wm_entries)
        storage.update_session = AsyncMock(return_value=session)

        task_service = AsyncMock()
        task_service.schedule_task = AsyncMock(return_value="task-id")

        svc = self._make_service(storage, task_service, token_trigger_init=500)
        await svc.touch_session("ws-1", "sess-1")

        call_kwargs = storage.update_session.call_args
        metadata = call_kwargs.kwargs.get("metadata") or (call_kwargs.args[3] if len(call_kwargs.args) > 3 else None)
        assert metadata is not None
        assert metadata.get("last_extraction_tokens") == 1000


# ---------------------------------------------------------------------------
# SessionMemorySections model
# ---------------------------------------------------------------------------


class TestSessionMemorySections:
    """Tests for the SessionMemorySections Pydantic model."""

    def test_default_construction_has_all_sections(self):
        sms = SessionMemorySections()
        for name in SESSION_MEMORY_SECTION_NAMES:
            assert name in sms.sections
            assert sms.sections[name] == []

    def test_total_tokens_zero_when_empty(self):
        sms = SessionMemorySections()
        assert sms.total_tokens == 0

    def test_total_tokens_computed_correctly(self):
        sms = SessionMemorySections(
            sections={
                "context": ["a" * 400],  # 100 tokens
                "decisions": ["b" * 800],  # 200 tokens
                "learnings": [],
                "errors": [],
                "progress": [],
                "open_items": [],
            }
        )
        assert sms.total_tokens == 300

    def test_add_entry_to_valid_section(self):
        sms = SessionMemorySections()
        result = sms.add_entry("context", "Some context entry")
        assert result is True
        assert "Some context entry" in sms.sections["context"]

    def test_add_entry_to_unknown_section_returns_false(self):
        sms = SessionMemorySections()
        result = sms.add_entry("nonexistent_section", "Some entry")
        assert result is False

    def test_add_entry_enforces_section_budget(self):
        # Budget of 128 tokens = 512 chars
        sms = SessionMemorySections(section_token_budget=128)
        # First entry: 100 tokens (400 chars) - fits
        result1 = sms.add_entry("context", "a" * 400)
        assert result1 is True
        # Second entry: 100 tokens (400 chars) - would exceed budget (100+100 > 128)
        result2 = sms.add_entry("context", "b" * 400)
        assert result2 is False
        # Only first entry added
        assert len(sms.sections["context"]) == 1

    def test_add_entry_budget_independent_per_section(self):
        sms = SessionMemorySections(section_token_budget=128)
        # Fill context section (100 tokens)
        sms.add_entry("context", "a" * 400)
        # decisions section should still accept entries independently
        result = sms.add_entry("decisions", "b" * 32)
        assert result is True
        assert len(sms.sections["decisions"]) == 1

    def test_all_canonical_section_names(self):
        assert "context" in SESSION_MEMORY_SECTION_NAMES
        assert "decisions" in SESSION_MEMORY_SECTION_NAMES
        assert "learnings" in SESSION_MEMORY_SECTION_NAMES
        assert "errors" in SESSION_MEMORY_SECTION_NAMES
        assert "progress" in SESSION_MEMORY_SECTION_NAMES
        assert "open_items" in SESSION_MEMORY_SECTION_NAMES

    def test_default_section_token_budget(self):
        sms = SessionMemorySections()
        assert sms.section_token_budget == DEFAULT_SESSION_SECTION_TOKEN_BUDGET

    def test_custom_section_token_budget(self):
        sms = SessionMemorySections(section_token_budget=512)
        assert sms.section_token_budget == 512

    def test_sections_can_be_provided_at_construction(self):
        sms = SessionMemorySections(
            sections={
                "context": ["Entry 1", "Entry 2"],
                "decisions": [],
                "learnings": [],
                "errors": [],
                "progress": [],
                "open_items": [],
            }
        )
        assert sms.sections["context"] == ["Entry 1", "Entry 2"]

    def test_total_tokens_updates_after_add_entry(self):
        sms = SessionMemorySections()
        assert sms.total_tokens == 0
        sms.add_entry("context", "a" * 40)  # 10 tokens
        assert sms.total_tokens == 10
        sms.add_entry("decisions", "b" * 80)  # 20 tokens
        assert sms.total_tokens == 30
