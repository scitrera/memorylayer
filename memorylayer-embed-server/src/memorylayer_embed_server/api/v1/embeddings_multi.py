"""Multi-vector (ColPali) embedding endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from scitrera_app_framework import Plugin, Variables

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.embedding import (
    MultiVectorEmbeddingRequest, MultiVectorEmbeddingResponse, MultiVectorData,
)
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(prefix="/v1", tags=['embeddings-multi'])


@router.post("/embeddings/multi", response_model=MultiVectorEmbeddingResponse)
async def create_multi_embeddings(
    request: MultiVectorEmbeddingRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> MultiVectorEmbeddingResponse:
    """
    Generate multi-vector embeddings (ColPali late interaction format).

    Returns multiple vectors per input, suitable for MaxSim scoring
    and late interaction retrieval.
    """
    dual_service = v.get('dual_embedding_service', default=None)
    if dual_service is None or not dual_service.has_multi_vector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Multi-vector embedding service not configured",
        )

    # Normalize input to list
    if isinstance(request.input, str):
        texts = [request.input]
    else:
        texts = request.input

    if not texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input must not be empty",
        )

    logger.debug("Generating multi-vector embeddings for %d text(s)", len(texts))

    try:
        if len(texts) == 1:
            multi_vec = await dual_service.embed_text_multivector(texts[0])
            multi_vecs = [multi_vec]
        else:
            multi_vecs = await dual_service.embed_batch_multivector(texts)
    except Exception as e:
        logger.error("Multi-vector embedding generation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-vector embedding generation failed: {e}",
        )

    data = [
        MultiVectorData(
            index=idx,
            vectors=mv.vectors,
            num_vectors=mv.num_vectors,
        )
        for idx, mv in enumerate(multi_vecs)
    ]

    return MultiVectorEmbeddingResponse(
        data=data,
        model=dual_service.multi_vector_model_name,
        dimensions=dual_service.multi_vector_dimensions,
    )


class EmbeddingsMultiAPIPlugin(Plugin):
    """Plugin to register multi-vector embeddings API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
