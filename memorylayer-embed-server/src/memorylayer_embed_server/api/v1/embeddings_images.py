"""ColPali image embedding endpoint (single-vector and multi-vector)."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from scitrera_app_framework import Plugin, Variables

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.embedding import (
    EmbeddingData, EmbeddingResponse, EmbeddingUsage,
    ImageEmbeddingRequest, MultiVectorData, MultiVectorEmbeddingResponse,
)
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(prefix="/v1/embeddings/images", tags=["embeddings"])


@router.post("", response_model=EmbeddingResponse | MultiVectorEmbeddingResponse)
async def create_image_embeddings(
    request: ImageEmbeddingRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> EmbeddingResponse | MultiVectorEmbeddingResponse:
    """
    Generate embeddings for images via ColPali.

    Accepts a list of base64-encoded images and returns either:
    - Single-vector embeddings (mode='single'): one flat vector per image,
      compatible with the OpenAI embeddings response format.
    - Multi-vector embeddings (mode='multi'): multiple vectors per image in
      ColPali late-interaction format, suitable for MaxSim scoring.
    """
    dual_service = v.get('dual_embedding_service', default=None)

    if not request.images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="images must not be empty",
        )

    if request.mode not in ("single", "multi"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="mode must be 'single' or 'multi'",
        )

    if request.mode == "multi":
        if dual_service is None or not dual_service.has_multi_vector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Multi-vector embedding service not configured",
            )

        logger.debug(
            "Generating multi-vector image embeddings for %d image(s)", len(request.images)
        )

        try:
            multi_vecs = await dual_service.embed_images_batch_multivector(request.images)
        except Exception as e:
            logger.error("Multi-vector image embedding generation failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Multi-vector image embedding generation failed: {e}",
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

    else:  # mode == "single"
        if dual_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Embedding service not configured",
            )

        logger.debug(
            "Generating single-vector image embeddings for %d image(s)", len(request.images)
        )

        try:
            embeddings = []
            for image in request.images:
                embedding = await dual_service.embed_image_single(image)
                embeddings.append(embedding)
        except Exception as e:
            logger.error("Single-vector image embedding generation failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Single-vector image embedding generation failed: {e}",
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
                prompt_tokens=0,
                total_tokens=0,
            ),
        )


class EmbeddingsImagesAPIPlugin(Plugin):
    """Plugin to register image embeddings API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def is_enabled(self, v: Variables) -> bool:
        return False

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router
