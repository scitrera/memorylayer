"""Unit tests for EmbeddingService."""
import pytest
from memorylayer_server.services.embedding import EmbeddingService

# Mock provider default dimensions
MOCK_EMBEDDING_DIMENSIONS = 384


class TestEmbedding:
    """Tests for embedding generation."""

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self, embedding_service: EmbeddingService):
        """Test that embed returns correct dimension vector."""
        embedding = await embedding_service.embed("Test text")

        assert embedding is not None
        assert len(embedding) == MOCK_EMBEDDING_DIMENSIONS
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, embedding_service: EmbeddingService):
        """Test that same text produces same embedding."""
        text = "Deterministic test"

        emb1 = await embedding_service.embed(text)
        emb2 = await embedding_service.embed(text)

        assert emb1 == emb2

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedding_service: EmbeddingService):
        """Test batch embedding generation."""
        texts = ["First text", "Second text", "Third text"]

        embeddings = await embedding_service.embed_batch(texts)

        assert len(embeddings) == 3
        assert all(len(e) == MOCK_EMBEDDING_DIMENSIONS for e in embeddings)

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Same vector = similarity 1
        assert EmbeddingService.cosine_similarity(vec1, vec2) == pytest.approx(1.0)

        # Orthogonal vectors = similarity 0
        assert EmbeddingService.cosine_similarity(vec1, vec3) == pytest.approx(0.0)
