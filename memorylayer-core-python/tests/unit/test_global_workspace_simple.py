"""Simple tests for _global workspace functionality without full framework."""
import pytest
from memorylayer_server.models import RecallInput
from memorylayer_server.config import GLOBAL_WORKSPACE_ID, DEFAULT_TENANT_ID


@pytest.mark.asyncio
async def test_global_workspace_id_constant():
    """Test that GLOBAL_WORKSPACE_ID is correctly defined."""
    assert GLOBAL_WORKSPACE_ID == "_global"


@pytest.mark.asyncio
async def test_default_tenant_id_constant():
    """Test that DEFAULT_TENANT_ID is correctly defined."""
    assert DEFAULT_TENANT_ID == "_default"


@pytest.mark.asyncio
async def test_recall_input_include_global_default():
    """Test that RecallInput.include_global defaults to True."""
    recall_input = RecallInput(query="test query")
    assert recall_input.include_global is True


@pytest.mark.asyncio
async def test_recall_input_include_global_can_be_disabled():
    """Test that RecallInput.include_global can be set to False."""
    recall_input = RecallInput(query="test query", include_global=False)
    assert recall_input.include_global is False


@pytest.mark.asyncio
async def test_recall_input_serialization():
    """Test that RecallInput serializes include_global correctly."""
    recall_input = RecallInput(query="test query", include_global=False)
    data = recall_input.model_dump()
    assert "include_global" in data
    assert data["include_global"] is False

    # Test round-trip
    restored = RecallInput(**data)
    assert restored.include_global is False
