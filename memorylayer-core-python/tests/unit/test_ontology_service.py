"""Unit tests for OntologyService relationship classification."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from memorylayer_server.services.ontology.default import DefaultOntologyService
from memorylayer_server.services.ontology.base import BASE_ONTOLOGY


class TestClassifyRelationshipPrefixMatching:
    """Tests for prefix matching on truncated LLM responses."""

    def _make_service(self, llm_response_content: str) -> DefaultOntologyService:
        """Create a DefaultOntologyService with a mocked LLM provider."""
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = llm_response_content
        mock_llm.complete.return_value = mock_response

        return DefaultOntologyService(v=None, llm_service=mock_llm)

    @pytest.mark.asyncio
    async def test_exact_match(self):
        """Exact valid relationship type should be returned directly."""
        service = self._make_service("builds_on")
        result = await service.classify_relationship("content A", "content B")
        assert result == "builds_on"

    @pytest.mark.asyncio
    async def test_truncated_built_prefix(self):
        """Truncated 'built_' should prefix-match to 'built_upon_by'."""
        service = self._make_service("built_")
        result = await service.classify_relationship("content A", "content B")
        # 'built_' is a unique prefix of 'built_upon_by'
        assert result == "built_upon_by"

    @pytest.mark.asyncio
    async def test_truncated_referenced_prefix(self):
        """Truncated 'referenced_' should prefix-match to 'referenced_by'."""
        service = self._make_service("referenced_")
        result = await service.classify_relationship("content A", "content B")
        assert result == "referenced_by"

    @pytest.mark.asyncio
    async def test_empty_string_falls_back(self):
        """Empty string should fall back to related_to."""
        service = self._make_service("")
        result = await service.classify_relationship("content A", "content B")
        assert result == "related_to"

    @pytest.mark.asyncio
    async def test_ambiguous_prefix_falls_back(self):
        """A prefix matching multiple types should fall back to related_to."""
        # 'replace' matches both 'replaces' and 'replaced_by'
        service = self._make_service("replace")
        result = await service.classify_relationship("content A", "content B")
        assert result == "related_to"

    @pytest.mark.asyncio
    async def test_quotes_stripped(self):
        """Quotes around the response should be stripped."""
        service = self._make_service('"causes"')
        result = await service.classify_relationship("content A", "content B")
        assert result == "causes"

    @pytest.mark.asyncio
    async def test_trailing_period_stripped(self):
        """Trailing period should be stripped."""
        service = self._make_service("similar_to.")
        result = await service.classify_relationship("content A", "content B")
        assert result == "similar_to"

    @pytest.mark.asyncio
    async def test_no_llm_provider_returns_related_to(self):
        """Without an LLM provider, should return related_to."""
        service = DefaultOntologyService(v=None, llm_service=None)
        result = await service.classify_relationship("content A", "content B")
        assert result == "related_to"

    @pytest.mark.asyncio
    async def test_nonsense_response_falls_back(self):
        """Completely invalid response should fall back to related_to."""
        service = self._make_service("this_is_not_a_relationship")
        result = await service.classify_relationship("content A", "content B")
        assert result == "related_to"

    @pytest.mark.asyncio
    async def test_unique_prefix_depended(self):
        """'depended_on' should prefix-match to 'depended_on_by'."""
        service = self._make_service("depended_on")
        result = await service.classify_relationship("content A", "content B")
        # 'depended_on' is a prefix of 'depended_on_by' only (not 'depends_on')
        assert result == "depended_on_by"
