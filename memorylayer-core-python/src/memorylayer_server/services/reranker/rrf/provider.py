"""
Reciprocal Rank Fusion (RRF) Reranker Provider.

Decomposes queries into multiple sub-queries using text processing (no LLM),
embeds each sub-query, ranks documents by cosine similarity per sub-query,
and fuses the rankings using the RRF formula.

Based on: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009)

This approach provides a cheaper alternative to HyDE (no LLM call required)
while still improving over raw single-query embedding similarity by capturing
multiple facets of the query.
"""

import re
from logging import Logger
from typing import Optional

from scitrera_app_framework import Variables, get_extension

from ....config import RerankerProviderType
from ....utils import cosine_similarity
from ..base import RerankerProvider, RerankerProviderPluginBase
from ...embedding import EXT_EMBEDDING_SERVICE, EmbeddingService

# Environment variable names
MEMORYLAYER_RERANKER_RRF_K = 'MEMORYLAYER_RERANKER_RRF_K'
MEMORYLAYER_RERANKER_RRF_MIN_QUERIES = 'MEMORYLAYER_RERANKER_RRF_MIN_QUERIES'

# Defaults
DEFAULT_RRF_K = 60
DEFAULT_RRF_MIN_QUERIES = 2

# Common English stopwords for keyword extraction
_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
    'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'how',
    'when', 'where', 'why', 'not', 'no', 'nor', 'so', 'if', 'then',
    'than', 'too', 'very', 'just', 'about', 'above', 'after', 'again',
    'all', 'also', 'am', 'any', 'because', 'before', 'between', 'both',
    'each', 'few', 'more', 'most', 'other', 'over', 'own', 'same',
    'some', 'such', 'up', 'down', 'out', 'off', 'only', 'into',
})

# Sentence boundary pattern
_SENTENCE_SPLIT = re.compile(r'[.?!;]\s+')


def _extract_keywords(text: str) -> str:
    """Extract content words by removing stopwords.

    Args:
        text: Input text

    Returns:
        Space-joined content words, or empty string if none remain.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return ' '.join(keywords)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at sentence boundaries.

    Args:
        text: Input text

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def decompose_query(
        query: str,
        instruction: Optional[str] = None,
        min_queries: int = DEFAULT_RRF_MIN_QUERIES,
) -> list[str]:
    """Decompose a query into multiple sub-queries for multi-query RRF.

    Generates sub-queries using text processing (no LLM required):
    1. Full original query (with instruction if provided)
    2. Individual sentences (if multi-sentence)
    3. Keywords-only variant (stopwords removed)

    Args:
        query: The search query
        instruction: Optional task-specific instruction to prepend
        min_queries: Minimum number of sub-queries to produce

    Returns:
        List of sub-query strings (deduplicated, at least min_queries if possible)
    """
    full_query = query
    if instruction:
        full_query = f"{instruction} {query}"

    sub_queries = [full_query]

    # Split multi-sentence queries
    sentences = _split_sentences(query)
    if len(sentences) > 1:
        for sentence in sentences:
            if sentence != query.strip():
                sub_queries.append(sentence)

    # Keywords-only variant
    keywords = _extract_keywords(query)
    if keywords and keywords != query.lower().strip():
        sub_queries.append(keywords)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for sq in sub_queries:
        normalized = sq.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(sq)

    # If we still don't have enough, add the raw query without instruction
    if len(unique) < min_queries and instruction and query not in unique:
        unique.append(query)

    return unique


def compute_rrf_scores(
        rankings: list[list[int]],
        num_documents: int,
        k: int = DEFAULT_RRF_K,
) -> list[float]:
    """Compute Reciprocal Rank Fusion scores from multiple rankings.

    RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranking i

    Scores are normalized to [0, 1] by dividing by the theoretical
    maximum (N / (k + 1)) where N is the number of rankings.

    Args:
        rankings: List of rankings, each a list of document indices
                  ordered by relevance (best first)
        num_documents: Total number of documents
        k: RRF constant (default 60). Higher values reduce the impact
           of high rankings vs low rankings.

    Returns:
        List of normalized RRF scores indexed by document position.
    """
    if not rankings or num_documents == 0:
        return []

    scores = [0.0] * num_documents
    num_rankings = len(rankings)

    for ranking in rankings:
        for rank_position, doc_idx in enumerate(ranking):
            if 0 <= doc_idx < num_documents:
                scores[doc_idx] += 1.0 / (k + rank_position + 1)

    # Normalize to [0, 1]
    max_possible = num_rankings / (k + 1)
    if max_possible > 0:
        scores = [s / max_possible for s in scores]

    return scores


class RRFRerankerProvider(RerankerProvider):
    """
    Reciprocal Rank Fusion reranker using embedding-only multi-query.

    Decomposes the query into multiple sub-queries via text processing,
    embeds each sub-query and all documents, ranks documents by cosine
    similarity for each sub-query, and fuses the rankings with the RRF
    formula.

    Advantages over HyDE:
    - No LLM call required (only embedding service)
    - Lower latency and cost
    - Captures multiple query facets through decomposition

    Requires only the embedding service to be configured.
    """

    def __init__(
            self,
            v: Variables,
            embedding_service: EmbeddingService,
            rrf_k: int = DEFAULT_RRF_K,
            min_queries: int = DEFAULT_RRF_MIN_QUERIES,
    ):
        super().__init__(v)
        self.embedding_service = embedding_service
        self.rrf_k = rrf_k
        self.min_queries = min_queries

    async def rerank(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """
        Score documents by multi-query RRF fusion.

        1. Decompose query into sub-queries (text processing, no LLM)
        2. Embed each sub-query
        3. Embed all documents in batch
        4. For each sub-query, rank documents by cosine similarity
        5. Fuse rankings with RRF formula

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional task-specific instruction

        Returns:
            List of relevance scores (0-1) for each document, same order as input
        """
        if not documents:
            return []

        self.logger.debug("RRF reranking %d documents", len(documents))

        try:
            # Step 1: Decompose query
            sub_queries = decompose_query(query, instruction, self.min_queries)
            self.logger.debug(
                "Decomposed query into %d sub-queries", len(sub_queries),
            )

            # Step 2: Embed sub-queries
            query_embeddings = await self.embedding_service.embed_batch(sub_queries)

            # Step 3: Embed all documents
            doc_embeddings = await self.embedding_service.embed_batch(documents)

            # Step 4: For each sub-query, rank documents by cosine similarity
            rankings = []
            for q_emb in query_embeddings:
                similarities = [
                    cosine_similarity(q_emb, d_emb)
                    for d_emb in doc_embeddings
                ]
                # Sort document indices by similarity (descending)
                ranking = sorted(
                    range(len(documents)),
                    key=lambda i: similarities[i],
                    reverse=True,
                )
                rankings.append(ranking)

            # Step 5: Fuse rankings with RRF
            scores = compute_rrf_scores(rankings, len(documents), self.rrf_k)

            return scores

        except Exception as e:
            self.logger.error("RRF reranking failed: %s", e)
            # Return uniform scores on error
            return [0.5] * len(documents)


class RRFRerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin for RRF reranker provider."""

    PROVIDER_NAME = RerankerProviderType.RRF

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return RRFRerankerProvider(
            v=v,
            embedding_service=get_extension(EXT_EMBEDDING_SERVICE, v),
            rrf_k=v.environ(
                MEMORYLAYER_RERANKER_RRF_K,
                default=DEFAULT_RRF_K,
                type_fn=int,
            ),
            min_queries=v.environ(
                MEMORYLAYER_RERANKER_RRF_MIN_QUERIES,
                default=DEFAULT_RRF_MIN_QUERIES,
                type_fn=int,
            ),
        )

    def get_dependencies(self, v: Variables):
        return (EXT_EMBEDDING_SERVICE,)
