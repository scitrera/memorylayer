"""Integration tests for health check endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_health_check(test_client: TestClient) -> None:
    """Test basic health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"


def test_readiness_check(test_client: TestClient) -> None:
    """Test readiness check endpoint."""
    response = test_client.get("/health/ready")

    # May return 200 or 503 depending on service availability
    assert response.status_code in [200, 503]

    data = response.json()
    assert "status" in data
    assert "services" in data


def test_root_endpoint(test_client: TestClient) -> None:
    """Test root endpoint returns API information."""
    response = test_client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "description" in data
    assert "MemoryLayer" in data["name"]
