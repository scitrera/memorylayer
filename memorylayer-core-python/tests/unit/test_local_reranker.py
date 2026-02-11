"""Unit tests for local sentence-transformers CrossEncoder reranker."""
import math
import pytest
from unittest.mock import MagicMock, patch

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class TestLocalRerankerProvider:
    """Tests for LocalRerankerProvider."""

    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.reranker.local.provider import LocalRerankerProvider
        return LocalRerankerProvider(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    def test_default_model(self, provider):
        assert provider.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, provider):
        result = await provider.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_returns_scores(self, provider):
        """Scores are returned in same order as input documents."""
        raw_scores = np.array([2.5, -1.0, 0.5])

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=raw_scores)
        provider._model = mock_model

        scores = await provider.rerank(
            "test query",
            ["doc1", "doc2", "doc3"],
        )

        assert len(scores) == 3
        # Verify sigmoid normalization
        assert scores[0] == pytest.approx(_sigmoid(2.5), rel=1e-5)
        assert scores[1] == pytest.approx(_sigmoid(-1.0), rel=1e-5)
        assert scores[2] == pytest.approx(_sigmoid(0.5), rel=1e-5)

        # Verify pairs were passed correctly
        mock_model.predict.assert_called_once()
        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [
            ("test query", "doc1"),
            ("test query", "doc2"),
            ("test query", "doc3"),
        ]

    @pytest.mark.asyncio
    async def test_rerank_scores_in_range(self, provider):
        """All scores should be between 0 and 1 after sigmoid."""
        raw_scores = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=raw_scores)
        provider._model = mock_model

        scores = await provider.rerank("query", ["a", "b", "c", "d", "e"])

        for score in scores:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_with_instruction(self, provider):
        """Instruction is prepended to query."""
        raw_scores = np.array([1.0])

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=raw_scores)
        provider._model = mock_model

        await provider.rerank(
            "test query",
            ["doc1"],
            instruction="Find relevant documents.",
        )

        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][0] == "Find relevant documents. test query"

    @pytest.mark.asyncio
    async def test_rerank_without_instruction(self, provider):
        """Without instruction, query is used as-is."""
        raw_scores = np.array([1.0])

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=raw_scores)
        provider._model = mock_model

        await provider.rerank("test query", ["doc1"])

        pairs = mock_model.predict.call_args[0][0]
        assert pairs[0][0] == "test query"

    @pytest.mark.asyncio
    async def test_preload_loads_model(self, provider):
        mock_model = MagicMock()
        with patch.object(provider, '_get_model', return_value=mock_model) as mock_get:
            await provider.preload()
            mock_get.assert_called_once()

    def test_lazy_client_import_error(self, provider):
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            provider._model = None
            with pytest.raises(ImportError, match="sentence-transformers package not installed"):
                provider._get_model()


class TestSigmoid:
    """Tests for sigmoid normalization function."""

    def test_sigmoid_zero(self):
        from memorylayer_server.services.reranker.local.provider import _sigmoid
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        from memorylayer_server.services.reranker.local.provider import _sigmoid
        assert _sigmoid(10.0) == pytest.approx(1.0, abs=1e-4)

    def test_sigmoid_large_negative(self):
        from memorylayer_server.services.reranker.local.provider import _sigmoid
        assert _sigmoid(-10.0) == pytest.approx(0.0, abs=1e-4)


class TestLocalRerankerProviderPlugin:
    """Tests for LocalRerankerProviderPlugin."""

    def test_plugin_provider_name(self):
        from memorylayer_server.services.reranker.local.provider import LocalRerankerProviderPlugin
        from memorylayer_server.config import RerankerProviderType
        plugin = LocalRerankerProviderPlugin()
        assert plugin.PROVIDER_NAME == RerankerProviderType.LOCAL

    def test_plugin_name(self):
        from memorylayer_server.services.reranker.local.provider import LocalRerankerProviderPlugin
        plugin = LocalRerankerProviderPlugin()
        assert 'local' in plugin.name().lower()
