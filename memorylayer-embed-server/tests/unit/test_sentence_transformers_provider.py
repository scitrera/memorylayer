"""Unit tests for the sentence-transformers single-vector provider.

The provider is the embed server's DEFAULT single-vector backend, so these pin
the properties the rest of the stack relies on: 384-d unit-length vectors, batch
routing, off-loop execution, and a loud (not silent) reaction to a dimension
mismatch — dimension is baked into stored data, so a wrong width is expensive.

No model is downloaded here: ``SentenceTransformer`` is stubbed. The real model
is exercised by the integration lane.
"""

from __future__ import annotations

import sys
import types

import pytest

from memorylayer_embed_server.services.embedding.sentence_transformers import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformersEmbeddingProvider,
)


def _install_fake_st(monkeypatch, dim: int = 384, record: dict | None = None):
    """Install a fake ``sentence_transformers`` module exposing SentenceTransformer."""

    class FakeModel:
        device = "cpu"

        def __init__(self, model_name, device=None):
            if record is not None:
                record["model_name"] = model_name
                record["device"] = device

        def get_embedding_dimension(self):
            return dim

        def encode(self, texts, **kwargs):
            if record is not None:
                record["encode_kwargs"] = kwargs
                record["encode_texts"] = list(texts)
            # Unit-length rows of the requested width.
            return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return FakeModel


def test_defaults_are_minilm_384():
    """The documented default pair must stay in sync with the README/config."""
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert DEFAULT_EMBEDDING_DIMENSIONS == 384


async def test_embed_returns_single_vector_of_model_width(monkeypatch):
    _install_fake_st(monkeypatch, dim=384)
    provider = SentenceTransformersEmbeddingProvider(device="cpu")

    vector = await provider.embed("hello")

    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


async def test_embed_batch_routes_all_texts_in_one_encode(monkeypatch):
    record: dict = {}
    _install_fake_st(monkeypatch, dim=384, record=record)
    provider = SentenceTransformersEmbeddingProvider(device="cpu")

    out = await provider.embed_batch(["a", "b", "c"])

    assert [len(v) for v in out] == [384, 384, 384]
    # One encode() call carrying every text — not one call per text.
    assert record["encode_texts"] == ["a", "b", "c"]


async def test_embed_batch_empty_short_circuits_without_loading_model(monkeypatch):
    record: dict = {}
    _install_fake_st(monkeypatch, dim=384, record=record)
    provider = SentenceTransformersEmbeddingProvider(device="cpu")

    assert await provider.embed_batch([]) == []
    # Model never loaded: no download cost for a no-op call.
    assert "model_name" not in record


async def test_encode_requests_normalized_vectors(monkeypatch):
    """recall() treats a dot product as cosine, which requires unit vectors."""
    record: dict = {}
    _install_fake_st(monkeypatch, dim=384, record=record)
    provider = SentenceTransformersEmbeddingProvider(device="cpu")

    await provider.embed("hello")

    assert record["encode_kwargs"]["normalize_embeddings"] is True
    assert record["encode_kwargs"]["show_progress_bar"] is False


async def test_model_dimension_wins_over_config_and_warns(monkeypatch, caplog):
    """A configured dimension that disagrees with the model must not be honored.

    Writing vectors at the configured-but-wrong width would poison the store;
    the model is the authority and the mismatch has to be visible.
    """
    _install_fake_st(monkeypatch, dim=384)
    provider = SentenceTransformersEmbeddingProvider(device="cpu", output_dimensions=1024)

    with caplog.at_level("WARNING"):
        await provider.preload()

    assert provider.dimensions == 384
    assert any("384" in r.message and "1024" in r.message for r in caplog.records)


async def test_passes_configured_model_and_device(monkeypatch):
    record: dict = {}
    _install_fake_st(monkeypatch, dim=768, record=record)
    provider = SentenceTransformersEmbeddingProvider(model_name="some/other-model", device="cpu")

    await provider.preload()

    assert record["model_name"] == "some/other-model"
    assert record["device"] == "cpu"
    assert provider.dimensions == 768


async def test_missing_dependency_names_the_install(monkeypatch):
    """The default provider's most likely failure is a missing extra."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    provider = SentenceTransformersEmbeddingProvider(device="cpu")

    with pytest.raises(ImportError, match=r"memorylayer-embed-server\[local\]"):
        await provider.embed("hello")


async def test_legacy_sentence_transformers_dimension_api(monkeypatch):
    """sentence-transformers 3.x/4.x only expose get_sentence_embedding_dimension()."""

    class LegacyModel:
        device = "cpu"

        def __init__(self, model_name, device=None):
            pass

        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):
            return [[1.0] + [0.0] * 383 for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = LegacyModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    provider = SentenceTransformersEmbeddingProvider(device="cpu")
    await provider.preload()

    assert provider.dimensions == 384


async def test_model_loaded_once_across_concurrent_calls(monkeypatch):
    """Concurrent first-use must not load (or download) the model twice."""
    loads = []

    class CountingModel:
        device = "cpu"

        def __init__(self, model_name, device=None):
            loads.append(model_name)

        def get_embedding_dimension(self):
            return 384

        def encode(self, texts, **kwargs):
            return [[1.0] + [0.0] * 383 for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = CountingModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    import asyncio

    provider = SentenceTransformersEmbeddingProvider(device="cpu")
    await asyncio.gather(*(provider.embed(f"text {i}") for i in range(5)))

    assert len(loads) == 1
