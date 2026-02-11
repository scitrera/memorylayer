"""Local reranker provider using sentence-transformers CrossEncoder."""
import math
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..base import RerankerProvider, RerankerProviderPluginBase
from ....config import RerankerProviderType

MEMORYLAYER_RERANKER_LOCAL_MODEL = 'MEMORYLAYER_RERANKER_LOCAL_MODEL'
DEFAULT_RERANKER_LOCAL_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def _sigmoid(x: float) -> float:
    """Apply sigmoid to normalize raw logits to 0-1 range."""
    return 1.0 / (1.0 + math.exp(-x))


class LocalRerankerProvider(RerankerProvider):
    """Local reranker using sentence-transformers CrossEncoder.

    Uses a cross-encoder model to score query-document pairs.
    Cross-encoders process the query and document together, producing
    more accurate relevance scores than bi-encoder similarity.
    """

    def __init__(
            self,
            v: Variables = None,
            model_name: str = DEFAULT_RERANKER_LOCAL_MODEL,
    ):
        super().__init__(v)
        self.model_name = model_name
        self._model = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized LocalRerankerProvider: model=%s", model_name
        )

    def _get_model(self):
        """Lazy-load the CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self.logger.info("Loading CrossEncoder model: %s", self.model_name)
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def preload(self):
        """Preload the CrossEncoder model."""
        self._get_model()

    async def rerank(
            self,
            query: str,
            documents: list[str],
            instruction: Optional[str] = None,
    ) -> list[float]:
        """Score documents by relevance to query using CrossEncoder.

        Args:
            query: The search query
            documents: List of document texts to score
            instruction: Optional instruction (prepended to query if provided)

        Returns:
            List of relevance scores (0-1) for each document, same order as input
        """
        if not documents:
            return []

        model = self._get_model()

        effective_query = query
        if instruction:
            effective_query = f"{instruction} {query}"

        self.logger.debug(
            "Reranking %d documents for query: %s chars",
            len(documents), len(effective_query),
        )

        # CrossEncoder expects list of (query, document) pairs
        pairs = [(effective_query, doc) for doc in documents]
        raw_scores = model.predict(pairs)

        # Normalize raw logits to 0-1 via sigmoid
        scores = [_sigmoid(float(s)) for s in raw_scores]

        return scores


class LocalRerankerProviderPlugin(RerankerProviderPluginBase):
    """Plugin for local CrossEncoder reranker."""
    PROVIDER_NAME = RerankerProviderType.LOCAL

    def initialize(self, v: Variables, logger: Logger) -> RerankerProvider:
        return LocalRerankerProvider(
            v=v,
            model_name=v.environ(
                MEMORYLAYER_RERANKER_LOCAL_MODEL,
                default=DEFAULT_RERANKER_LOCAL_MODEL,
            ),
        )
