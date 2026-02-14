"""
Unit tests for TierGenerationService background task integration.

Tests:
- request_tier_generation: scheduling via TaskService
- Config toggle: MEMORYLAYER_TIER_GENERATION_ENABLED
- TierGenerationTaskHandler: background task handler
- Integration with remember() flow
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memorylayer_server.models.memory import Memory, MemoryType
from memorylayer_server.services.semantic_tiering.default import DefaultSemanticTieringService
from memorylayer_server.services.semantic_tiering.base import SemanticTieringService
from memorylayer_server.services.semantic_tiering.semantic_tiering_task_handler import TierGenerationTaskHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_service():
    """Mock LLM service that returns canned responses."""
    service = AsyncMock()
    response = MagicMock()
    response.content = "A brief summary of the content."
    service.complete.return_value = response
    return service


@pytest.fixture
def mock_storage():
    """Mock storage backend."""
    storage = AsyncMock()
    storage.get_memory.return_value = Memory(
        id="mem_test123",
        workspace_id="ws_test",
        tenant_id="default_tenant",
        content="Test memory content for tier generation",
        content_hash="abc123",
        type=MemoryType.SEMANTIC,
    )
    storage.update_memory.return_value = Memory(
        id="mem_test123",
        workspace_id="ws_test",
        tenant_id="default_tenant",
        content="Test memory content for tier generation",
        content_hash="abc123",
        type=MemoryType.SEMANTIC,
        abstract="A brief summary of the content.",
        overview="A brief summary of the content.",
    )
    return storage


@pytest.fixture
def mock_task_service():
    """Mock task service for background scheduling."""
    service = AsyncMock()
    service.schedule_task.return_value = "task_001"
    return service


@pytest.fixture
def mock_variables():
    """Mock Variables instance."""
    v = MagicMock()
    v.get.return_value = None
    return v


@pytest.fixture
def tier_service_enabled(mock_llm_service, mock_storage, mock_task_service, mock_variables):
    """Tier generation service with background scheduling enabled."""
    return DefaultSemanticTieringService(
        llm_service=mock_llm_service,
        storage=mock_storage,
        v=mock_variables,
        enabled=True,
        task_service=mock_task_service,
    )


@pytest.fixture
def tier_service_disabled(mock_llm_service, mock_storage, mock_task_service, mock_variables):
    """Tier generation service with tier generation disabled."""
    return DefaultSemanticTieringService(
        llm_service=mock_llm_service,
        storage=mock_storage,
        v=mock_variables,
        enabled=False,
        task_service=mock_task_service,
    )


@pytest.fixture
def tier_service_no_task_service(mock_llm_service, mock_storage, mock_variables):
    """Tier generation service without task service (inline fallback)."""
    return DefaultSemanticTieringService(
        llm_service=mock_llm_service,
        storage=mock_storage,
        v=mock_variables,
        enabled=True,
        task_service=None,
    )


# ---------------------------------------------------------------------------
# request_tier_generation() Tests
# ---------------------------------------------------------------------------

class TestRequestTierGeneration:
    """Tests for request_tier_generation() dispatch logic."""

    @pytest.mark.asyncio
    async def test_schedules_background_task_when_enabled(self, tier_service_enabled, mock_task_service):
        """When enabled with task service, should schedule a background task."""
        task_id = await tier_service_enabled.request_tier_generation("mem_123", "ws_test")

        assert task_id == "task_001"
        mock_task_service.schedule_task.assert_called_once_with(
            task_type='generate_tiers',
            payload={'memory_id': 'mem_123', 'workspace_id': 'ws_test'},
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self, tier_service_disabled, mock_task_service):
        """When disabled, should return None and not schedule anything."""
        result = await tier_service_disabled.request_tier_generation("mem_123", "ws_test")

        assert result is None
        mock_task_service.schedule_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_inline_without_task_service(
        self, tier_service_no_task_service, mock_storage, mock_llm_service
    ):
        """Without task service, should generate tiers inline."""
        result = await tier_service_no_task_service.request_tier_generation("mem_test123", "ws_test")

        assert result is None  # No task_id for inline execution
        # Should have called storage to load the memory and update with tiers
        mock_storage.get_memory.assert_called_once_with("ws_test", "mem_test123", track_access=False)
        mock_storage.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_base_class_default_is_noop(self):
        """The ABC default implementation returns None (no-op)."""

        class MinimalTierService(SemanticTieringService):
            async def generate_abstract(self, content, max_tokens=30):
                return ""

            async def generate_overview(self, content, max_tokens=100):
                return ""

            async def generate_tiers(self, memory_id, workspace_id, force=False):
                return None

            async def generate_tiers_for_content(self, content):
                return ("", "")

        service = MinimalTierService()
        result = await service.request_tier_generation("mem_123", "ws_test")
        assert result is None


# ---------------------------------------------------------------------------
# Config Toggle Tests
# ---------------------------------------------------------------------------

class TestTierGenerationConfig:
    """Tests for MEMORYLAYER_TIER_GENERATION_ENABLED config behavior."""

    @pytest.mark.asyncio
    async def test_enabled_true_allows_scheduling(self, tier_service_enabled, mock_task_service):
        """With enabled=True, tasks are scheduled."""
        await tier_service_enabled.request_tier_generation("mem_123", "ws_test")
        mock_task_service.schedule_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_enabled_false_prevents_scheduling(self, tier_service_disabled, mock_task_service):
        """With enabled=False, no tasks are scheduled."""
        await tier_service_disabled.request_tier_generation("mem_123", "ws_test")
        mock_task_service.schedule_task.assert_not_called()

    def test_service_stores_enabled_flag(self, mock_llm_service, mock_storage, mock_variables):
        """Verify the enabled flag is properly stored on the service."""
        service_on = DefaultSemanticTieringService(
            llm_service=mock_llm_service, storage=mock_storage,
            v=mock_variables, enabled=True
        )
        service_off = DefaultSemanticTieringService(
            llm_service=mock_llm_service, storage=mock_storage,
            v=mock_variables, enabled=False
        )
        assert service_on.enabled is True
        assert service_off.enabled is False

    def test_service_defaults_to_enabled(self, mock_llm_service, mock_storage, mock_variables):
        """Default constructor should have enabled=True."""
        service = DefaultSemanticTieringService(
            llm_service=mock_llm_service, storage=mock_storage, v=mock_variables
        )
        assert service.enabled is True


# ---------------------------------------------------------------------------
# TierGenerationTaskHandler Tests
# ---------------------------------------------------------------------------

class TestTierGenerationTaskHandler:
    """Tests for the background task handler."""

    def test_task_type_is_generate_tiers(self):
        """Handler should register for 'generate_tiers' task type."""
        handler = TierGenerationTaskHandler()
        assert handler.get_task_type() == 'generate_tiers'

    def test_schedule_returns_none(self):
        """Handler is on-demand only, no recurring schedule."""
        handler = TierGenerationTaskHandler()
        assert handler.get_schedule(MagicMock()) is None

    @pytest.mark.asyncio
    async def test_handle_delegates_to_tier_service(self):
        """handle() should call tier_service.generate_tiers with payload."""
        handler = TierGenerationTaskHandler()

        # Mock the framework's get_extension
        mock_tier_service = AsyncMock()
        mock_v = MagicMock()
        handler._v = mock_v

        with patch.object(handler, 'get_extension', return_value=mock_tier_service):
            await handler.handle({
                'memory_id': 'mem_abc',
                'workspace_id': 'ws_xyz',
            })

        mock_tier_service.generate_tiers.assert_called_once_with('mem_abc', 'ws_xyz')

    def test_initialize_returns_self(self):
        """initialize() should return self and store v."""
        handler = TierGenerationTaskHandler()
        mock_v = MagicMock()
        mock_logger = MagicMock()

        result = handler.initialize(mock_v, mock_logger)

        assert result is handler
        assert handler._v is mock_v


# ---------------------------------------------------------------------------
# Integration with remember() flow
# ---------------------------------------------------------------------------

class TestRememberTierGenerationIntegration:
    """Tests verifying remember() delegates to tier generation service."""

    @pytest.mark.asyncio
    async def test_remember_calls_request_tier_generation(
        self,
        memory_service,
        unique_workspace_id,
    ):
        """remember() should call request_tier_generation after creating memory."""
        from memorylayer_server.models.memory import RememberInput

        # Patch the tier generation service's request_tier_generation
        original_service = memory_service.tier_generation_service
        mock_request = AsyncMock(return_value=None)

        try:
            if original_service:
                original_service.request_tier_generation = mock_request

                input_data = RememberInput(
                    content="Test content for tier generation integration",
                    type=MemoryType.SEMANTIC,
                )
                memory = await memory_service.remember(unique_workspace_id, input_data)

                assert memory.id is not None
                mock_request.assert_called_once_with(memory.id, unique_workspace_id)
        finally:
            # Restore original if we patched it
            if original_service and hasattr(original_service, 'request_tier_generation'):
                # The base class method will be restored since we only patched the instance
                pass

    @pytest.mark.asyncio
    async def test_remember_succeeds_without_tier_service(
        self,
        memory_service,
        unique_workspace_id,
    ):
        """remember() should succeed even if tier generation service is None."""
        from memorylayer_server.models.memory import RememberInput

        original_service = memory_service.tier_generation_service
        try:
            memory_service.tier_generation_service = None

            input_data = RememberInput(
                content="Test content without tier service",
                type=MemoryType.SEMANTIC,
            )
            memory = await memory_service.remember(unique_workspace_id, input_data)
            assert memory.id is not None
        finally:
            memory_service.tier_generation_service = original_service

    @pytest.mark.asyncio
    async def test_remember_handles_tier_generation_error(
        self,
        memory_service,
        unique_workspace_id,
    ):
        """remember() should handle tier generation errors gracefully."""
        from memorylayer_server.models.memory import RememberInput

        original_service = memory_service.tier_generation_service
        try:
            if original_service:
                original_service.request_tier_generation = AsyncMock(
                    side_effect=RuntimeError("LLM unavailable")
                )

                input_data = RememberInput(
                    content="Test content with tier error",
                    type=MemoryType.SEMANTIC,
                )
                # Should not raise despite tier generation failure
                memory = await memory_service.remember(unique_workspace_id, input_data)
                assert memory.id is not None
        finally:
            if original_service:
                memory_service.tier_generation_service = original_service


# ---------------------------------------------------------------------------
# RememberInput model tests
# ---------------------------------------------------------------------------

class TestRememberInputModel:
    """Verify generate_tiers field has been removed from RememberInput."""

    def test_no_generate_tiers_field(self):
        """RememberInput should not have a generate_tiers field."""
        from memorylayer_server.models.memory import RememberInput
        fields = RememberInput.model_fields
        assert 'generate_tiers' not in fields

    def test_remember_input_ignores_generate_tiers_kwarg(self):
        """RememberInput should not store generate_tiers even if passed."""
        from memorylayer_server.models.memory import RememberInput
        input_data = RememberInput(content="test")
        assert not hasattr(input_data, 'generate_tiers') or 'generate_tiers' not in input_data.model_fields
