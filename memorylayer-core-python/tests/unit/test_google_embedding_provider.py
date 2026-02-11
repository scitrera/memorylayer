"""Unit tests for Google GenAI embedding provider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGoogleEmbeddingProvider:
    """Tests for GoogleEmbeddingProvider."""

    @pytest.fixture
    def provider(self):
        from memorylayer_server.services.embedding.google import GoogleEmbeddingProvider
        return GoogleEmbeddingProvider(
            api_key="test-key",
            model="gemini-embedding-001",
            dimensions=768,
        )

    def test_default_model(self, provider):
        assert provider.model == "gemini-embedding-001"

    def test_dimensions(self, provider):
        assert provider.dimensions == 768

    def test_output_dimensionality(self, provider):
        assert provider._output_dimensionality == 768

    @pytest.mark.asyncio
    async def test_embed(self, provider):
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_aio_models = AsyncMock()
        mock_aio_models.embed_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        with patch.object(provider, '_get_config', return_value=MagicMock()):
            result = await provider.embed("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_aio_models.embed_content.assert_called_once()
        call_kwargs = mock_aio_models.embed_content.call_args[1]
        assert call_kwargs["model"] == "gemini-embedding-001"
        assert call_kwargs["contents"] == "test text"

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider):
        mock_emb1 = MagicMock()
        mock_emb1.values = [0.1, 0.2, 0.3]
        mock_emb2 = MagicMock()
        mock_emb2.values = [0.4, 0.5, 0.6]
        mock_emb3 = MagicMock()
        mock_emb3.values = [0.7, 0.8, 0.9]

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb1, mock_emb2, mock_emb3]

        mock_aio_models = AsyncMock()
        mock_aio_models.embed_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        texts = ["first", "second", "third"]
        with patch.object(provider, '_get_config', return_value=MagicMock()):
            result = await provider.embed_batch(texts)

        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]
        call_kwargs = mock_aio_models.embed_content.call_args[1]
        assert call_kwargs["contents"] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_floats(self, provider):
        """Ensure values are converted to plain list[float]."""
        mock_embedding = MagicMock()
        # Simulate numpy-like array that needs list() conversion
        mock_embedding.values = (0.1, 0.2, 0.3)

        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]

        mock_aio_models = AsyncMock()
        mock_aio_models.embed_content = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.aio.models = mock_aio_models
        provider._client = mock_client

        with patch.object(provider, '_get_config', return_value=MagicMock()):
            result = await provider.embed("test")

        assert isinstance(result, list)

    def test_lazy_client_import_error(self, provider):
        with patch.dict('sys.modules', {'google': None, 'google.genai': None}):
            provider._client = None
            with pytest.raises(ImportError, match="google-genai package not installed"):
                provider._get_client()


class TestGoogleEmbeddingProviderPlugin:
    """Tests for GoogleEmbeddingProviderPlugin."""

    def test_plugin_provider_name(self):
        from memorylayer_server.services.embedding.google import GoogleEmbeddingProviderPlugin
        from memorylayer_server.config import EmbeddingProviderType
        plugin = GoogleEmbeddingProviderPlugin()
        assert plugin.PROVIDER_NAME == EmbeddingProviderType.GOOGLE

    def test_plugin_name(self):
        from memorylayer_server.services.embedding.google import GoogleEmbeddingProviderPlugin
        plugin = GoogleEmbeddingProviderPlugin()
        assert 'GOOGLE' in plugin.name() or 'google' in plugin.name()
