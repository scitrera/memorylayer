"""Tests for Reciprocal Rank Fusion (RRF) reranker provider."""

from unittest.mock import AsyncMock

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.config import RerankerProviderType
from memorylayer_server.services.reranker.rrf.provider import (
    RRFRerankerProvider,
    RRFRerankerProviderPlugin,
    decompose_query,
    compute_rrf_scores,
    _extract_keywords,
    _split_sentences,
    DEFAULT_RRF_K,
    DEFAULT_RRF_MIN_QUERIES,
    MEMORYLAYER_RERANKER_RRF_K,
)
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE


# --- Fixtures ---

@pytest.fixture
def mock_v():
    """Provide a Variables instance for test provider construction."""
    return Variables()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service with embed and embed_batch methods."""
    service = AsyncMock()
    # Default: return simple 3D embeddings
    # Sub-query embeddings (2 sub-queries)
    # Document embeddings (3 docs)
    service.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
    service.embed_batch = AsyncMock(side_effect=_default_embed_batch)
    return service


def _default_embed_batch(texts):
    """Generate distinct embeddings for sub-queries vs documents."""
    result = []
    for i, text in enumerate(texts):
        if len(texts) <= 3:
            # Small batch: likely sub-queries, give varied embeddings
            vectors = [
                [1.0, 0.0, 0.0],
                [0.7, 0.7, 0.0],
                [0.0, 1.0, 0.0],
            ]
            result.append(vectors[i % len(vectors)])
        else:
            # Larger batch: likely documents
            # Create a spread of vectors
            import math
            angle = (2 * math.pi * i) / len(texts)
            result.append([math.cos(angle), math.sin(angle), 0.0])
    return result


@pytest.fixture
def provider(mock_v, mock_embedding_service):
    """RRF reranker with mocked services."""
    return RRFRerankerProvider(
        v=mock_v,
        embedding_service=mock_embedding_service,
    )


# --- Query decomposition tests ---

class TestDecomposeQuery:
    def test_single_word_query(self):
        """Single word should produce at least the original query."""
        result = decompose_query("authentication")
        assert len(result) >= 1
        assert "authentication" in result

    def test_multi_sentence_query(self):
        """Multi-sentence queries should be split into sentences."""
        query = "How does authentication work? What protocols are used?"
        result = decompose_query(query)
        assert query in result  # Full query included
        assert len(result) >= 2  # At least full query + one sentence

    def test_keywords_extracted(self):
        """Should produce a keywords-only variant."""
        query = "How does the authentication system handle tokens?"
        result = decompose_query(query)
        # Should have the full query + keywords variant at minimum
        assert len(result) >= 2
        # Keywords variant should not contain stopwords
        keywords_found = any(
            'authentication' in sq and 'the' not in sq.split()
            for sq in result if sq != query
        )
        assert keywords_found

    def test_with_instruction(self):
        """Instruction should be prepended to the full query."""
        result = decompose_query("auth tokens", instruction="Find docs about")
        assert any("Find docs about" in sq for sq in result)

    def test_deduplication(self):
        """Duplicate sub-queries should be removed."""
        result = decompose_query("test")
        # All entries should be unique (case-insensitive)
        normalized = [sq.strip().lower() for sq in result]
        assert len(normalized) == len(set(normalized))

    def test_min_queries_with_instruction_fallback(self):
        """When instruction is present, raw query added if below min_queries."""
        result = decompose_query("x", instruction="find", min_queries=2)
        assert len(result) >= 2

    def test_semicolon_splits_sentences(self):
        """Semicolons should act as sentence boundaries."""
        query = "auth tokens; session management"
        result = decompose_query(query)
        assert len(result) >= 2


class TestExtractKeywords:
    def test_removes_stopwords(self):
        result = _extract_keywords("how does the system work")
        assert "how" not in result
        assert "does" not in result
        assert "the" not in result
        assert "system" in result
        assert "work" in result

    def test_empty_string(self):
        assert _extract_keywords("") == ""

    def test_all_stopwords(self):
        result = _extract_keywords("the a an is are")
        assert result == ""

    def test_preserves_content_words(self):
        result = _extract_keywords("machine learning algorithms")
        assert "machine" in result
        assert "learning" in result
        assert "algorithms" in result

    def test_single_char_words_removed(self):
        """Single character words should be filtered out."""
        result = _extract_keywords("I like a b c programming")
        assert "programming" in result
        # Single chars 'b', 'c' should be removed (len <= 1)
        words = result.split()
        assert all(len(w) > 1 for w in words)


class TestSplitSentences:
    def test_single_sentence(self):
        result = _split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_period_split(self):
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2
        assert "First sentence" in result
        assert "Second sentence." in result or "Second sentence" in result

    def test_question_mark_split(self):
        result = _split_sentences("What is this? How does it work?")
        assert len(result) == 2

    def test_semicolon_split(self):
        result = _split_sentences("Part one; part two")
        assert len(result) == 2

    def test_empty_string(self):
        result = _split_sentences("")
        assert result == []

    def test_whitespace_only(self):
        result = _split_sentences("   ")
        assert result == []


# --- RRF score computation tests ---

class TestComputeRRFScores:
    def test_single_ranking(self):
        """Single ranking should produce valid scores."""
        rankings = [[0, 1, 2]]  # Doc 0 is best
        scores = compute_rrf_scores(rankings, 3)
        assert len(scores) == 3
        assert scores[0] > scores[1] > scores[2]

    def test_scores_in_range(self):
        """All scores should be in [0, 1]."""
        rankings = [[2, 0, 1], [0, 1, 2]]
        scores = compute_rrf_scores(rankings, 3)
        for score in scores:
            assert 0.0 <= score <= 1.0

    def test_perfect_agreement(self):
        """When all rankings agree, top doc should get max score (1.0)."""
        rankings = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
        scores = compute_rrf_scores(rankings, 3)
        assert scores[0] == pytest.approx(1.0)

    def test_empty_rankings(self):
        scores = compute_rrf_scores([], 3)
        assert scores == []

    def test_zero_documents(self):
        scores = compute_rrf_scores([[0, 1]], 0)
        assert scores == []

    def test_two_rankings_fusion(self):
        """Fusion of two different rankings should boost docs ranked high in both."""
        # Ranking 1: doc0 > doc1 > doc2
        # Ranking 2: doc2 > doc0 > doc1
        rankings = [[0, 1, 2], [2, 0, 1]]
        scores = compute_rrf_scores(rankings, 3)
        # doc0 appears at rank 1 and rank 2 -> high fusion score
        # doc2 appears at rank 3 and rank 1 -> also decent
        # doc1 appears at rank 2 and rank 3 -> lowest
        assert scores[1] < scores[0]  # doc0 > doc1
        assert scores[1] < scores[2]  # doc2 > doc1

    def test_custom_k(self):
        """Different k values should produce different score distributions."""
        rankings = [[0, 1, 2]]
        scores_k1 = compute_rrf_scores(rankings, 3, k=1)
        scores_k100 = compute_rrf_scores(rankings, 3, k=100)
        # With lower k, the gap between top and bottom should be larger
        gap_k1 = scores_k1[0] - scores_k1[2]
        gap_k100 = scores_k100[0] - scores_k100[2]
        assert gap_k1 > gap_k100

    def test_out_of_bounds_indices_ignored(self):
        """Document indices outside valid range should not cause errors."""
        rankings = [[0, 5, 1]]  # Index 5 is out of bounds for 3 docs
        scores = compute_rrf_scores(rankings, 3)
        assert len(scores) == 3
        # Only doc 0 and doc 1 should have non-zero scores
        assert scores[0] > 0
        assert scores[1] > 0
        assert scores[2] == 0.0


# --- Provider tests ---

class TestRRFRerankerProvider:
    @pytest.mark.asyncio
    async def test_rerank_returns_correct_count(self, provider, mock_embedding_service):
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],  # sub-query embeddings
            [[0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],  # doc embeddings
        ])
        docs = ["doc one", "doc two", "doc three"]
        scores = await provider.rerank("test query", docs)
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, provider, mock_embedding_service):
        scores = await provider.rerank("test query", [])
        assert scores == []
        mock_embedding_service.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_scores_in_range(self, provider, mock_embedding_service):
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        ])
        docs = ["doc one", "doc two", "doc three"]
        scores = await provider.rerank("test query", docs)
        for score in scores:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_rerank_similar_doc_scores_higher(self, provider, mock_embedding_service):
        """Document most similar to query embeddings should score highest."""
        # Sub-queries both point toward [1, 0, 0]
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],  # sub-query embeddings
            [[0.95, 0.05, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # docs
        ])
        docs = ["similar doc", "orthogonal doc", "another orthogonal"]
        scores = await provider.rerank("test query", docs)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    @pytest.mark.asyncio
    async def test_rerank_calls_embed_batch_twice(self, provider, mock_embedding_service):
        """Should call embed_batch once for sub-queries, once for documents."""
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0]],
        ])
        await provider.rerank("test query", ["doc one"])
        assert mock_embedding_service.embed_batch.call_count == 2

    @pytest.mark.asyncio
    async def test_rerank_with_instruction(self, provider, mock_embedding_service):
        """Instruction should be included in sub-queries."""
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0]],
        ])
        await provider.rerank("my query", ["doc"], instruction="Find scientific papers")
        # First embed_batch call is sub-queries
        sub_queries_arg = mock_embedding_service.embed_batch.call_args_list[0][0][0]
        assert any("Find scientific papers" in sq for sq in sub_queries_arg)

    @pytest.mark.asyncio
    async def test_rerank_embedding_failure_returns_uniform_scores(self, mock_v):
        """When embedding fails, should return uniform 0.5 scores."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(side_effect=RuntimeError("Embedding unavailable"))
        provider = RRFRerankerProvider(v=mock_v, embedding_service=emb)
        docs = ["doc one", "doc two"]
        scores = await provider.rerank("test query", docs)
        assert scores == [0.5, 0.5]

    @pytest.mark.asyncio
    async def test_custom_rrf_k(self, mock_v, mock_embedding_service):
        """Custom k value should be used in RRF computation."""
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
        ])
        provider = RRFRerankerProvider(
            v=mock_v, embedding_service=mock_embedding_service, rrf_k=1,
        )
        scores = await provider.rerank("test", ["doc1", "doc2"])
        assert len(scores) == 2
        # With k=1, score gaps should be larger
        assert abs(scores[0] - scores[1]) > 0

    @pytest.mark.asyncio
    async def test_rerank_single_document(self, provider, mock_embedding_service):
        """Single document should get a valid score."""
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0]],
        ])
        scores = await provider.rerank("test query", ["only doc"])
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0


# --- Integration-style test with rerank_with_indices ---

class TestRRFRerankerWithIndices:
    @pytest.mark.asyncio
    async def test_rerank_with_indices_sorted_by_score(self, provider, mock_embedding_service):
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        ])
        docs = ["similar doc", "orthogonal doc", "partial doc"]
        results = await provider.rerank_with_indices("test query", docs)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_with_indices_top_k(self, provider, mock_embedding_service):
        mock_embedding_service.embed_batch = AsyncMock(side_effect=[
            [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]],
            [[0.9, 0.1, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]],
        ])
        docs = ["similar doc", "orthogonal doc", "partial doc"]
        results = await provider.rerank_with_indices("test query", docs, top_k=2)
        assert len(results) == 2


# --- Plugin tests ---

class TestRRFRerankerPlugin:
    def test_provider_name(self):
        plugin = RRFRerankerProviderPlugin()
        assert plugin.PROVIDER_NAME == RerankerProviderType.RRF

    def test_plugin_name_contains_rrf(self):
        plugin = RRFRerankerProviderPlugin()
        name = plugin.name()
        assert "rrf" in name.lower()

    def test_plugin_declares_embedding_dependency(self):
        plugin = RRFRerankerProviderPlugin()
        v = Variables()
        deps = plugin.get_dependencies(v)
        assert EXT_EMBEDDING_SERVICE in deps

    def test_plugin_has_no_llm_dependency(self):
        """RRF should not depend on LLM service."""
        from memorylayer_server.services.llm import EXT_LLM_SERVICE
        plugin = RRFRerankerProviderPlugin()
        v = Variables()
        deps = plugin.get_dependencies(v)
        assert EXT_LLM_SERVICE not in deps
