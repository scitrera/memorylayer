"""Pytest configuration and fixtures for memorylayer-langchain tests."""

import pytest


@pytest.fixture
def base_url() -> str:
    """Test base URL."""
    return "http://test.memorylayer.ai"


@pytest.fixture
def api_key() -> str:
    """Test API key."""
    return "test_api_key"


@pytest.fixture
def workspace_id() -> str:
    """Test workspace ID."""
    return "ws_test"


@pytest.fixture
def session_id() -> str:
    """Test session ID."""
    return "sess_test_123"
