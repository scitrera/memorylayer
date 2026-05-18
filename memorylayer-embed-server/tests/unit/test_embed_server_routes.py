"""FastAPI route tests for memorylayer-embed-server.

Uses ``fastapi.testclient.TestClient`` against a hand-built FastAPI app
that includes the embed-server routers and a stub ``Variables`` with the
``MockSingleVectorProvider`` / ``MockMultiVectorProvider`` wired into a
real ``DualEmbeddingService``.

These tests do **not** import torch or download any model. They cover
the routing surface (request validation, status codes, response shapes)
and prove that the mock-provider integration test image will work.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memorylayer_embed_server.api.health import router as health_router
from memorylayer_embed_server.api.v1.embeddings import router as embeddings_router
from memorylayer_embed_server.api.v1.embeddings_images import router as embeddings_images_router
from memorylayer_embed_server.api.v1.embeddings_multi import router as embeddings_multi_router
from memorylayer_embed_server.api.v1.score import router as score_router
from memorylayer_embed_server.lifecycle.fastapi import get_logger, get_variables_dep
from memorylayer_embed_server.services.embedding.dual_service import DualEmbeddingService
from memorylayer_embed_server.services.embedding.mock_providers import (
    MockMultiVectorProvider,
    MockSingleVectorProvider,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dual_service() -> DualEmbeddingService:
    """DualEmbeddingService backed by the deterministic mock providers."""
    return DualEmbeddingService(
        v=None,
        single_vector_provider=MockSingleVectorProvider(v=None, dimensions=64),
        multi_vector_provider=MockMultiVectorProvider(
            v=None,
            dimensions=32,
            num_tokens=4,
        ),
    )


@pytest.fixture
def app_factory(dual_service):
    """Build a FastAPI app pre-wired with the routers and a stub Variables."""

    def _make_app(with_dual_service: bool = True) -> FastAPI:
        app = FastAPI()
        app.include_router(embeddings_router)
        app.include_router(embeddings_multi_router)
        app.include_router(embeddings_images_router)
        app.include_router(score_router)
        app.include_router(health_router)

        v = MagicMock(name="Variables")

        def _v_get(key, default=None):
            if key == "dual_embedding_service":
                return dual_service if with_dual_service else None
            if key == "cascade_transcriber":
                return None
            if key == "gpu_monitor":
                return None
            if key == "health_check_callables":
                return []
            return default

        v.get.side_effect = _v_get

        async def _override_v(request=None):
            return v

        async def _override_logger(request=None):
            import logging

            return logging.getLogger("embed-server-tests")

        app.dependency_overrides[get_variables_dep] = _override_v
        app.dependency_overrides[get_logger] = _override_logger
        return app

    return _make_app


@pytest.fixture
def client(app_factory) -> TestClient:
    return TestClient(app_factory(with_dual_service=True))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_basic(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_health_ready_when_embedding_configured(client):
    """With dual_embedding_service wired, readiness should be ready."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["services"]["embedding"]["single_vector"] == "available"
    assert body["services"]["embedding"]["multi_vector"] == "available"


def test_health_ready_not_ready_without_embedding(app_factory):
    """Without dual_embedding_service, readiness flips to 503/not_ready."""
    client = TestClient(app_factory(with_dual_service=False))
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"


# ---------------------------------------------------------------------------
# /v1/embeddings (single-vector)
# ---------------------------------------------------------------------------


def test_embeddings_single_text(client):
    resp = client.post("/v1/embeddings", json={"input": "hello world"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    entry = body["data"][0]
    assert entry["object"] == "embedding"
    assert entry["index"] == 0
    assert len(entry["embedding"]) == 64
    # L2-normalized → unit norm
    sumsq = sum(x * x for x in entry["embedding"])
    assert abs(sumsq - 1.0) < 1e-3


def test_embeddings_batch_preserves_order(client):
    resp = client.post("/v1/embeddings", json={"input": ["alpha", "beta", "gamma"]})
    assert resp.status_code == 200
    body = resp.json()
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    # Determinism: same input → same vectors
    resp2 = client.post("/v1/embeddings", json={"input": "alpha"})
    assert resp2.json()["data"][0]["embedding"] == body["data"][0]["embedding"]


def test_embeddings_rejects_empty_list(client):
    resp = client.post("/v1/embeddings", json={"input": []})
    assert resp.status_code == 400


def test_embeddings_503_without_provider(app_factory):
    client = TestClient(app_factory(with_dual_service=False))
    resp = client.post("/v1/embeddings", json={"input": "hi"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /v1/embeddings/multi (multi-vector)
# ---------------------------------------------------------------------------


def test_embeddings_multi_returns_matrix(client):
    resp = client.post(
        "/v1/embeddings/multi",
        json={"input": ["doc one", "doc two"], "input_type": "document"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimensions"] == 32
    assert len(body["data"]) == 2
    assert [d["index"] for d in body["data"]] == [0, 1]
    first = body["data"][0]
    assert first["num_vectors"] == 4
    assert len(first["vectors"]) == 4
    assert all(len(row) == 32 for row in first["vectors"])


# ---------------------------------------------------------------------------
# /v1/embeddings/images
# ---------------------------------------------------------------------------


def _tiny_image_b64() -> str:
    """A 1×1 PNG (8 bytes is enough for our hash-based mock)."""
    return base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")


def test_embeddings_images_multi_mode(client):
    resp = client.post(
        "/v1/embeddings/images",
        json={"images": [_tiny_image_b64()], "mode": "multi"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Multi mode returns MultiVectorEmbeddingResponse shape
    assert "dimensions" in body
    assert len(body["data"]) == 1
    assert body["data"][0]["num_vectors"] == 4


def test_embeddings_images_single_mode(client):
    """mode=single goes through the multimodal single-vector provider
    (real prod uses Qwen3-VL-Embedding-2B which is multimodal)."""
    resp = client.post(
        "/v1/embeddings/images",
        json={"images": [_tiny_image_b64()], "mode": "single"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert len(body["data"][0]["embedding"]) == 64


# ---------------------------------------------------------------------------
# /v1/score (MaxSim)
# ---------------------------------------------------------------------------


def test_score_maxsim_orders_descending(client):
    """High-similarity doc must rank above low-similarity doc."""
    query_vectors = [[1.0, 0.0], [0.0, 1.0]]
    document_vectors = [
        [[1.0, 0.0], [0.0, 1.0]],  # identical to query → high score
        [[-1.0, 0.0], [0.0, -1.0]],  # anti-correlated → low score
    ]
    resp = client.post(
        "/v1/score",
        json={"query_vectors": query_vectors, "document_vectors": document_vectors},
    )
    assert resp.status_code == 200
    scores = resp.json()["scores"]
    assert len(scores) == 2
    # Sorted descending — index 0 (identical) wins
    assert scores[0]["index"] == 0
    assert scores[0]["score"] > scores[1]["score"]


def test_score_requires_documents(client):
    resp = client.post(
        "/v1/score",
        json={"query_vectors": [[1.0]], "document_vectors": []},
    )
    assert resp.status_code == 400
