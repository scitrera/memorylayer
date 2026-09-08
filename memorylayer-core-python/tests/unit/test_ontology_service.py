"""Unit tests for OntologyService relationship classification."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.ontology.default import DefaultOntologyService


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


class TestClassifyRelationshipsBatch:
    """Tests for the single-call batched classifier."""

    def _make_service(self, llm_response_content: str) -> DefaultOntologyService:
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = llm_response_content
        mock_llm.complete.return_value = mock_response
        service = DefaultOntologyService(v=None, llm_service=mock_llm)
        service._mock_llm = mock_llm  # expose for call-count assertions
        return service

    @pytest.mark.asyncio
    async def test_single_call_maps_all_candidates(self):
        """Every candidate is classified from ONE llm.complete invocation."""
        service = self._make_service("0: causes\n1: supports\n2: refines")
        result = await service.classify_relationships_batch(
            content_a="anchor",
            candidates=[("a", "ca"), ("b", "cb"), ("c", "cc")],
        )
        assert result == {"a": "causes", "b": "supports", "c": "refines"}
        assert service._mock_llm.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_bracketed_indices_and_prose_tolerated(self):
        """Surrounding prose and [n] index formatting still parse."""
        service = self._make_service("Here you go:\n[0]: builds_on\n[1]: related_to\n")
        result = await service.classify_relationships_batch(
            content_a="anchor",
            candidates=[("a", "ca"), ("b", "cb")],
        )
        assert result == {"a": "builds_on", "b": "related_to"}

    @pytest.mark.asyncio
    async def test_missing_and_invalid_lines_default_to_related_to(self):
        """Unparsed / out-of-range / invalid entries default to related_to."""
        service = self._make_service("0: causes\n1: not_a_real_type\n5: supports")
        result = await service.classify_relationships_batch(
            content_a="anchor",
            candidates=[("a", "ca"), ("b", "cb"), ("c", "cc")],
        )
        assert result["a"] == "causes"
        assert result["b"] == "related_to"  # invalid type
        assert result["c"] == "related_to"  # never mentioned

    @pytest.mark.asyncio
    async def test_truncated_type_prefix_matches(self):
        """A truncated but unique type prefix resolves like the single-call path."""
        service = self._make_service("0: built_")
        result = await service.classify_relationships_batch(
            content_a="anchor",
            candidates=[("a", "ca"), ("b", "cb")],
        )
        assert result["a"] == "built_upon_by"

    @pytest.mark.asyncio
    async def test_no_llm_uniform_related_to(self):
        """Without an LLM the batch is uniformly related_to and makes no call."""
        service = DefaultOntologyService(v=None, llm_service=None)
        result = await service.classify_relationships_batch(
            content_a="anchor",
            candidates=[("a", "ca"), ("b", "cb")],
        )
        assert result == {"a": "related_to", "b": "related_to"}

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        """No candidates -> empty result, no call."""
        service = self._make_service("")
        result = await service.classify_relationships_batch(content_a="anchor", candidates=[])
        assert result == {}
        assert service._mock_llm.complete.await_count == 0


# A condensed version of a real reply observed in production: the model reasons
# at length, lands on the right answer, and emits the reasoning as its response.
# Every edge classified this way silently became related_to.
_THINKING_REPLY = """The user wants to classify the relationship between A and B.
Content A states the museum has a lakefront building.
Content B adds that it has been there for over 100 years.
Does B refine A? Yes, B elaborates on A by adding the duration.
So A -> B would be refined_by.
refined_by"""


class TestReasoningModelReplies:
    """A thinking model must not silently degrade every classification.

    Classification asks for one label, so reasoning is both wasted cost and an
    active hazard: the chain of thought lands in `content`, matches no type, and
    falls back to related_to with only a log line to show for it.
    """

    def _make_service(self, content: str) -> DefaultOntologyService:
        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = content
        mock_llm.complete.return_value = mock_response
        service = DefaultOntologyService(v=None, llm_service=mock_llm)
        service._mock_llm = mock_llm
        return service

    @pytest.mark.asyncio
    async def test_reasoning_is_disabled_on_the_request(self):
        service = self._make_service("similar_to")
        await service.classify_relationship("a", "b")

        request = service._mock_llm.complete.await_args.args[0]
        assert request.reasoning_effort == "none"

    @pytest.mark.asyncio
    async def test_batch_request_also_disables_reasoning(self):
        # More acute in the batch path: one leaked chain of thought costs the
        # whole batch rather than a single edge.
        service = self._make_service("0: similar_to")
        await service.classify_relationships_batch(
            content_a="anchor", candidates=[("a", "ca")],
        )

        request = service._mock_llm.complete.await_args.args[0]
        assert request.reasoning_effort == "none"

    @pytest.mark.asyncio
    async def test_verdict_is_recovered_from_a_reasoned_reply(self):
        # Defence in depth: a model that reasons anyway (or a profile pointing at
        # one that cannot disable it) must still be parsed, not silently dropped.
        service = self._make_service(_THINKING_REPLY)
        assert await service.classify_relationship("a", "b") == "refined_by"

    @pytest.mark.asyncio
    async def test_think_tags_are_stripped(self):
        service = self._make_service("<think>maybe similar_to? no.</think>\nbuilds_on")
        assert await service.classify_relationship("a", "b") == "builds_on"

    @pytest.mark.asyncio
    async def test_unclosed_think_block_does_not_swallow_everything(self):
        # A reply truncated mid-thought has no closing tag; that must degrade to
        # the honest fallback rather than raise or return a fragment.
        service = self._make_service("<think>still deciding between")
        assert await service.classify_relationship("a", "b") == "related_to"

    def test_batch_parser_ignores_indices_inside_reasoning(self):
        # The batch parser scans for "<index>: <type>" lines, and reasoning
        # about candidates contains lines of exactly that shape. Left in, they
        # parse as verdicts -- confidently wrong beats an honest fallback.
        service = self._make_service("")
        raw = (
            "<think>\n"
            "0: this one looks like it could be causes\n"
            "1: and this one contradicts\n"
            "</think>\n"
            "0: similar_to\n"
            "1: part_of\n"
        )
        parsed = service._parse_batch_response(raw, service.base_ontology, 2)
        assert parsed == {0: "similar_to", 1: "part_of"}
