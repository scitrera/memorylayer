"""OpenAI-compatible single-vector embedding endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from scitrera_app_framework import Plugin, Variables

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.embedding import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(prefix="/v1", tags=["embeddings"])


@router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    request: EmbeddingRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EmbeddingResponse:
    """
    Generate single-vector embeddings (OpenAI-compatible).

    Accepts text input (string or list of strings) and returns
    embeddings in OpenAI API format.
    """
    dual_service = v.get("dual_embedding_service", default=None)
    if dual_service is None or not dual_service.has_single_vector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Single-vector embedding service not configured",
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

    logger.debug("Generating single-vector embeddings for %d text(s)", len(texts))

    try:
        if len(texts) == 1:
            embedding = await dual_service.embed(texts[0])
            embeddings = [embedding]
        else:
            embeddings = await dual_service.embed_batch(texts)
    except Exception as e:
        logger.error("Embedding generation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {e}",
        )

    data = [
        EmbeddingData(
            embedding=emb,
            index=idx,
        )
        for idx, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        data=data,
        model=dual_service.single_vector_model_name,
        usage=EmbeddingUsage(
            prompt_tokens=0,  # vLLM offline doesn't easily expose token counts
            total_tokens=0,
        ),
    )


class EmbeddingsAPIPlugin(Plugin):
    """Plugin to register embeddings API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
