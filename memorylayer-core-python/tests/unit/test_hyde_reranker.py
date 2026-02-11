"""Tests for HyDE (Hypothetical Document Embeddings) reranker provider."""

from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import RerankerProviderType
from memorylayer_server.services.reranker.hyde.provider import (
    HyDERerankerProvider,
    HyDERerankerProviderPlugin,
    HYDE_PROMPT_TEMPLATE,
)
from memorylayer_server.utils import cosine_similarity
from memorylayer_server.services.llm import EXT_LLM_SERVICE
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE


# --- Fixtures ---

@pytest.fixture
def mock_v():
    """Provide a Variables instance for test provider construction."""
    return Variables()


@pytest.fixture
def mock_llm_service():
    """Mock LLM service with synthesize method."""
    service = AsyncMock()
    service.synthesize = AsyncMock(return_value="This is a hypothetical answer about the topic.")
    return service


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service with embed and embed_batch methods."""
    service = AsyncMock()
    # Hypothetical answer embedding
    service.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    # Document embeddings - first doc is similar, second is orthogonal
    service.embed_batch = AsyncMock(return_value=[
        [0.9, 0.1, 0.0],  # Similar to hyp
        [0.0, 0.0, 1.0],  # Orthogonal to hyp
        [0.5, 0.5, 0.0],  # Partially similar
    ])
    return service


@pytest.fixture
def provider(mock_v, mock_llm_service, mock_embedding_service):
    """HyDE reranker with mocked services."""
    return HyDERerankerProvider(
        v=mock_v,
        llm_service=mock_llm_service,
        embedding_service=mock_embedding_service,
    )


# --- Cosine similarity tests ---

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        sim = cosine_similarity([1.0, 0.0, 0.0], [0.9, 0.1, 0.0])
        assert 0.9 < sim <= 1.0

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_both_zero_vectors(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# --- Provider tests ---

class TestHyDERerankerProvider:
    @pytest.mark.asyncio
    async def test_rerank_returns_correct_count(self, provider):
        docs = ["doc one", "doc two", "doc three"]
        scores = await provider.rerank("test query", docs)
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, provider, mock_llm_service):
        scores = await provider.rerank("test query", [])
        assert scores == []
        # Should not call LLM for empty docs
        mock_llm_service.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_scores_in_range(self, provider):
        docs = ["doc one", "doc two", "doc three"]
        scores = await provider.rerank("test query", docs)
        for score in scores:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_similar_doc_scores_higher(self, provider):
        docs = ["similar doc", "orthogonal doc", "partial doc"]
        scores = await provider.rerank("test query", docs)
        # First doc (similar embedding) should score higher than second (orthogonal)
        assert scores[0] > scores[1]

    @pytest.mark.asyncio
    async def test_rerank_calls_llm_synthesize(self, provider, mock_llm_service):
        docs = ["doc one"]
        await provider.rerank("my query", docs)
        mock_llm_service.synthesize.assert_called_once()
        call_kwargs = mock_llm_service.synthesize.call_args
        assert "my query" in call_kwargs.kwargs.get("prompt", call_kwargs.args[0] if call_kwargs.args else "")

    @pytest.mark.asyncio
    async def test_rerank_calls_embed_for_hypothetical(self, provider, mock_embedding_service):
        docs = ["doc one"]
        await provider.rerank("test query", docs)
        # Should embed the hypothetical answer
        mock_embedding_service.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_calls_embed_batch_for_documents(self, provider, mock_embedding_service):
        docs = ["doc one", "doc two", "doc three"]
        await provider.rerank("test query", docs)
        mock_embedding_service.embed_batch.assert_called_once_with(docs)

    @pytest.mark.asyncio
    async def test_rerank_with_instruction(self, provider, mock_llm_service):
        docs = ["doc one"]
        await provider.rerank("my query", docs, instruction="Find scientific papers")
        call_kwargs = mock_llm_service.synthesize.call_args
        prompt = call_kwargs.kwargs.get("prompt", call_kwargs.args[0] if call_kwargs.args else "")
        assert "Find scientific papers" in prompt
        assert "my query" in prompt

    @pytest.mark.asyncio
    async def test_rerank_llm_failure_returns_uniform_scores(
            self, mock_v, mock_embedding_service
    ):
        """When LLM fails, should return uniform 0.5 scores."""
        llm = AsyncMock()
        llm.synthesize = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        provider = HyDERerankerProvider(
            v=mock_v,
            llm_service=llm,
            embedding_service=mock_embedding_service,
        )
        docs = ["doc one", "doc two"]
        scores = await provider.rerank("test query", docs)
        assert scores == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_rerank_embedding_failure_returns_uniform_scores(
            self, mock_v, mock_llm_service
    ):
        """When embedding fails, should return uniform 0.5 scores."""
        emb = AsyncMock()
        emb.embed = AsyncMock(side_effect=RuntimeError("Embedding unavailable"))
        provider = HyDERerankerProvider(
            v=mock_v,
            llm_service=mock_llm_service,
            embedding_service=emb,
        )
        docs = ["doc one", "doc two"]
        scores = await provider.rerank("test query", docs)
        assert scores == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_negative_cosine_clamped_to_zero(self, mock_v, mock_llm_service):
        """Negative cosine similarity should be clamped to 0.0."""
        emb = AsyncMock()
        emb.embed = AsyncMock(return_value=[1.0, 0.0])
        emb.embed_batch = AsyncMock(return_value=[[-1.0, 0.0]])  # Opposite direction
        provider = HyDERerankerProvider(
            v=mock_v,
            llm_service=mock_llm_service,
            embedding_service=emb,
        )
        scores = await provider.rerank("test", ["opposite doc"])
        assert scores[0] == 0.0

    @pytest.mark.asyncio
    async def test_custom_max_tokens_and_temperature(self, mock_v, mock_llm_service, mock_embedding_service):
        provider = HyDERerankerProvider(
            v=mock_v,
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            max_tokens=100,
            temperature=0.3,
        )
        await provider.rerank("test query", ["doc"])
        call_kwargs = mock_llm_service.synthesize.call_args.kwargs
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["temperature"] == 0.3


# --- Plugin tests ---

class TestHyDERerankerPlugin:
    def test_provider_name(self):
        plugin = HyDERerankerProviderPlugin()
        assert plugin.PROVIDER_NAME == RerankerProviderType.HYDE

    def test_plugin_name_contains_hyde(self):
        plugin = HyDERerankerProviderPlugin()
        name = plugin.name()
        assert "hyde" in name.lower() or "HYDE" in name

    def test_plugin_declares_dependencies(self):
        plugin = HyDERerankerProviderPlugin()
        v = Variables()
        deps = plugin.get_dependencies(v)
        assert EXT_LLM_SERVICE in deps
        assert EXT_EMBEDDING_SERVICE in deps


# --- Prompt template tests ---

class TestHyDEPromptTemplate:
    def test_prompt_contains_query_placeholder(self):
        assert "{query}" in HYDE_PROMPT_TEMPLATE

    def test_prompt_formatted_correctly(self):
        result = HYDE_PROMPT_TEMPLATE.format(query="How do deep sea fish survive?")
        assert "How do deep sea fish survive?" in result
        assert "hypothetical answer" in result.lower()


# --- Integration-style test with rerank_with_indices ---

class TestHyDERerankerWithIndices:
    @pytest.mark.asyncio
    async def test_rerank_with_indices_sorted_by_score(self, provider):
        docs = ["similar doc", "orthogonal doc", "partial doc"]
        results = await provider.rerank_with_indices("test query", docs)
        # Results should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_with_indices_top_k(self, provider):
        docs = ["similar doc", "orthogonal doc", "partial doc"]
        results = await provider.rerank_with_indices("test query", docs, top_k=2)
        assert len(results) == 2
