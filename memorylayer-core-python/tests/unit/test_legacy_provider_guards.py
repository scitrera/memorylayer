"""Unit tests for the hard-error guards on removed provider names.

When the heavy in-process providers (``local``, ``colpali``,
``qwen3-vl``) were retired in favour of ``embed_server``, we put a
``ValueError`` guard at the plugin ``is_enabled`` boundary so that
operators on stale configs fail fast with a clear migration message
rather than silently ending up with no provider registered.
"""

from __future__ import annotations

import pytest

from memorylayer_server.config import (
    assert_supported_embedding_provider,
    assert_supported_reranker_provider,
)


@pytest.mark.parametrize("legacy_name", ["local", "colpali", "qwen3-vl"])
def test_legacy_embedding_provider_names_raise(legacy_name):
    with pytest.raises(ValueError, match=r"embed_server"):
        assert_supported_embedding_provider(legacy_name)


@pytest.mark.parametrize("supported", ["openai", "google", "embed_server", "mock"])
def test_supported_embedding_names_pass(supported):
    # Should not raise.
    assert_supported_embedding_provider(supported)


@pytest.mark.parametrize("legacy_name", ["local", "qwen3-vl"])
def test_legacy_reranker_provider_names_raise(legacy_name):
    with pytest.raises(ValueError, match=r"embed_server"):
        assert_supported_reranker_provider(legacy_name)


@pytest.mark.parametrize("supported", ["llm", "hyde", "rrf", "embed_server", "none"])
def test_supported_reranker_names_pass(supported):
    assert_supported_reranker_provider(supported)
