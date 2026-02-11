"""Pytest fixtures for MemoryLayer.ai integration tests.

These fixtures extend the base test fixtures from tests/conftest.py.
The `fastapi_app` fixture is inherited from the parent conftest and provides
a properly initialized FastAPI app using the test framework's Variables instance.
"""

from typing import AsyncGenerator, Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


@pytest.fixture(scope="session")
def test_client(fastapi_app: FastAPI) -> Generator[TestClient, None, None]:
    """
    Create TestClient for FastAPI app.

    Uses the fastapi_app fixture from tests/conftest.py which provides
    a properly initialized app with test isolation.
    """
    with TestClient(fastapi_app) as client:
        yield client


@pytest.fixture
async def async_client(fastapi_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Create async client for FastAPI app.

    Uses the fastapi_app fixture from tests/conftest.py which provides
    a properly initialized app with test isolation.
    """
    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    """Default workspace headers for requests."""
    return {"X-Workspace-ID": "test_workspace"}
