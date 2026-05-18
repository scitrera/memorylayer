"""MaxSim scoring endpoint (low priority)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from scitrera_app_framework import Plugin, Variables

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.embedding import ScoreRequest, ScoreResponse, ScoreResult
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(prefix="/v1", tags=['score'])


@router.post("/score", response_model=ScoreResponse)
async def compute_score(
    request: ScoreRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> ScoreResponse:
    """
    Compute MaxSim scores between query and document multi-vectors.

    Used for late interaction retrieval scoring with ColPali-style
    multi-vector embeddings.
    """
    dual_service = v.get('dual_embedding_service', default=None)
    if dual_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not configured",
        )

    if not request.document_vectors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_vectors must not be empty",
        )

    logger.debug(
        "Computing MaxSim scores: query_vecs=%d, documents=%d",
        len(request.query_vectors), len(request.document_vectors)
    )

    try:
        from memorylayer_server.services.embedding._maxsim import MultiVectorEmbedding

        query_mv = MultiVectorEmbedding(vectors=request.query_vectors)

        scores = []
        for idx, doc_vecs in enumerate(request.document_vectors):
            doc_mv = MultiVectorEmbedding(vectors=doc_vecs)
            score = dual_service.maxsim_score(query_mv, doc_mv)
            scores.append(ScoreResult(index=idx, score=score))

        # Sort by score descending
        scores.sort(key=lambda s: s.score, reverse=True)

    except Exception as e:
        logger.error("MaxSim scoring failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {e}",
        )

    return ScoreResponse(scores=scores)


class ScoreAPIPlugin(Plugin):
    """Plugin to register score API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
