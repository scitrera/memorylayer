"""
HyDE (Hypothetical Document Embeddings) Reranker Provider.

Uses an LLM to generate a hypothetical answer to the query, then computes
embedding similarity between the hypothetical answer and each candidate document.

Based on: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
https://arxiv.org/abs/2212.10496

The intuition is that a hypothetical answer (even if imperfect) will be closer
in embedding space to relevant documents than the original short query would be.
"""

from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from ....config import RerankerProviderType
from ....utils import cosine_similarity
from ..base import RerankerProvider, RerankerProviderPluginBase
from ...llm import EXT_LLM_SERVICE, LLMService
from ...embedding import EXT_EMBEDDING_SERVICE, EmbeddingService

# Environment variable names
MEMORYLAYER_RERANKER_HYDE_MAX_TOKENS = 'MEMORYLAYER_RERANKER_HYDE_MAX_TOKENS'
MEMORYLAYER_RERANKER_HYDE_TEMPERATURE = 'MEMORYLAYER_RERANKER_HYDE_TEMPERATURE'

# Defaults
DEFAULT_HYDE_MAX_TOKENS = 2048
DEFAULT_HYDE_TEMPERATURE = 0.7

HYDE_PROMPT_TEMPLATE = """Generate a hypothetical answer to the user's query \
by using your own knowledge. Assume that you know everything about the said topic. \
Do not use factual information, instead use placeholders to complete your answer. \
Your answer should feel like it has been written by a human.

query: {query}"""


class HyDERerankerProvider(RerankerProvider):
    """
    HyDE-based reranker using LLM + embedding similarity.

    Given a query, generates a hypothetical answer using an LLM, then embeds
    the hypothetical answer and each document. Documents are scored by cosine
    similarity to the hypothetical answer embedding.

    This approach can outperform direct query-document similarity because the
    hypothetical answer is closer in embedding space to relevant documents.

    Requires both LLM and embedding services to be configured.
    """

    def __init__(
            self,
            v: Variables,
            llm_service: LLMService,
            embedding_service: EmbeddingService,
            max_tokens: int = DEFAULT_HYDE_MAX_TOKENS,
            temperature: float = DEFAULT_HYDE_TEMPERATURE,
    ):
        super().__init__(v)
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def _generate_hypothetical_answer(self, query: str, instruction: Optional[str] = None) -> str:
        """Generate a hypothetical answer to the query using the LLM."""
        full_query = query
        if instruction:
            full_query = f"{instruction}\n\n{query}"

        prompt = HYDE_PROMPT_TEMPLATE.format(query=full_query)

        hypothetical_answer = await self.llm_service.synthesize(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            profile="reranker",
        )

        return hypothetical_answer

    async def rerank(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """
        Score documents by HyDE similarity.

        1. Generate a hypothetical answer to the query using the LLM
        2. Embed the hypothetical answer
        3. Embed each document
        4. Score each document by cosine similarity to the hypothetical answer

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional task-specific instruction

        Returns:
            List of relevance scores (0-1) for each document, same order as input
        """
        if not documents:
            return []

        self.logger.debug("HyDE reranking %d documents", len(documents))

        try:
            # Step 1: Generate hypothetical answer
            hypothetical = await self._generate_hypothetical_answer(query, instruction)
            self.logger.debug(
                "Generated hypothetical answer: %d chars",
                len(hypothetical),
            )

            # Step 2: Embed hypothetical answer
            hyp_embedding = await self.embedding_service.embed(hypothetical)

            # Step 3: Embed all documents in batch
            doc_embeddings = await self.embedding_service.embed_batch(documents)

            # Step 4: Compute cosine similarity scores
            scores = []
            for doc_emb in doc_embeddings:
                sim = cosine_similarity(hyp_embedding, doc_emb)
                # Clamp to 0-1 (cosine sim can be negative)
                scores.append(max(0.0, min(1.0, sim)))

            return scores

        except Exception as e:
            self.logger.error("HyDE reranking failed: %s", e)
            # Return uniform scores on error
            return [0.5] * len(documents)


class HyDERerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin for HyDE reranker provider."""

    PROVIDER_NAME = RerankerProviderType.HYDE

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return HyDERerankerProvider(
            v=v,
            llm_service=get_extension(EXT_LLM_SERVICE, v),
            embedding_service=get_extension(EXT_EMBEDDING_SERVICE, v),
            max_tokens=v.environ(
                MEMORYLAYER_RERANKER_HYDE_MAX_TOKENS,
                default=DEFAULT_HYDE_MAX_TOKENS,
                type_fn=int,
            ),
            temperature=v.environ(
                MEMORYLAYER_RERANKER_HYDE_TEMPERATURE,
                default=DEFAULT_HYDE_TEMPERATURE,
                type_fn=float,
            ),
        )

    def get_dependencies(self, v: Variables):
        return (EXT_LLM_SERVICE, EXT_EMBEDDING_SERVICE)
