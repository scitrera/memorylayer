"""
LLM-based Reranker Provider.

Uses an LLM service to score document relevance via prompting.
This is a fallback when dedicated reranker models are unavailable.
"""

from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from ....config import RerankerProviderType
from ..base import RerankerProvider, RerankerProviderPluginBase
from ...llm import EXT_LLM_SERVICE


RERANK_PROMPT_TEMPLATE = """You are a relevance scoring assistant. Score how relevant each document is to the given query.

Query: {query}

Documents to score:
{documents}

For each document, output a relevance score from 0.0 to 1.0 where:
- 0.0 = completely irrelevant
- 0.5 = somewhat relevant
- 1.0 = highly relevant

Output ONLY a JSON array of scores in the same order as the documents, like: [0.8, 0.3, 0.9]
No other text or explanation."""


class LLMRerankerProvider(RerankerProvider):
    """
    LLM-based reranker that uses prompting to score relevance.

    This is a fallback provider when dedicated reranker models are unavailable.
    It's slower and more expensive than dedicated models but works with any LLM.
    """

    def __init__(self, v: Variables = None, llm_service=None):
        super().__init__(v)
        self._llm_service = llm_service

    @property
    def llm_service(self):
        """Get LLM service, either injected or from extension."""
        if self._llm_service is None:
            self._llm_service = get_extension(EXT_LLM_SERVICE)
        return self._llm_service

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: Optional[str] = None,
    ) -> list[float]:
        """
        Score documents by relevance to query using LLM.

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional task-specific instruction (appended to query)

        Returns:
            List of relevance scores (0-1) for each document
        """
        if not documents:
            return []

        self.logger.debug("LLM reranking %d documents", len(documents))

        # Format documents for prompt
        docs_text = "\n".join(
            f"[{i+1}] {doc[:500]}{'...' if len(doc) > 500 else ''}"
            for i, doc in enumerate(documents)
        )

        # Build query with optional instruction
        full_query = query
        if instruction:
            full_query = f"{instruction}\n\n{query}"

        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=full_query,
            documents=docs_text,
        )

        try:
            response = await self.llm_service.synthesize(prompt, profile="reranker")

            # Parse JSON array from response
            import json
            import re

            # Extract JSON array from response
            match = re.search(r'\[[\d.,\s]+\]', response)
            if match:
                scores = json.loads(match.group())
                # Ensure we have the right number of scores
                if len(scores) == len(documents):
                    # Clamp scores to 0-1 range
                    return [max(0.0, min(1.0, float(s))) for s in scores]

            # Fallback: return uniform scores
            self.logger.warning("Failed to parse LLM reranker response, using uniform scores")
            return [0.5] * len(documents)

        except Exception as e:
            self.logger.error("LLM reranking failed: %s", e)
            # Return uniform scores on error
            return [0.5] * len(documents)


class LLMRerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin for LLM-based reranker provider."""

    PROVIDER_NAME = RerankerProviderType.LLM

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return LLMRerankerProvider(v=v)

    def get_dependencies(self, v: Variables):
        return (EXT_LLM_SERVICE,)
