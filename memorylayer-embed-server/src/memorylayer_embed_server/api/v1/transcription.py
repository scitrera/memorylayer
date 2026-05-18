"""Transcription API endpoint."""

import base64
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from scitrera_app_framework import Plugin, Variables

from ...lifecycle.fastapi import get_logger, get_variables_dep
from ...models.transcription import (
    ModelAttempt,
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionResult,
    TranscriptionStats,
)
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(prefix="/v1", tags=["transcription"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    request: TranscriptionRequest,
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> TranscriptionResponse:
    """
    Transcribe page images to markdown.

    Accepts base64-encoded page images and returns markdown transcriptions
    using a cascade of models (GLM-OCR primary, Gemini Flash fallback).
    """
    cascade = v.get("cascade_transcriber", default=None)
    if cascade is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transcription service not configured",
        )

    # Decode base64 images
    image_bytes_list = []
    for idx, image_b64 in enumerate(request.images):
        try:
            image_bytes_list.append(base64.b64decode(image_b64))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 in image at index {idx}: {e}",
            )

    logger.info("Transcribing %d page(s)", len(image_bytes_list))

    # Run cascade transcription
    page_results = await cascade.transcribe_pages(
        images=image_bytes_list,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens,
    )

    # Convert to response models
    results = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_latency_ms = 0.0
    successful = 0
    failed = 0

    for page in page_results:
        attempts = [
            ModelAttempt(
                model=a.model,
                provider=a.provider,
                success=a.success,
                tokens_in=a.tokens_in,
                tokens_out=a.tokens_out,
                latency_ms=a.latency_ms,
                finish_reason=a.finish_reason,
                error=a.error,
            )
            for a in page.attempts
        ]

        results.append(
            TranscriptionResult(
                page_index=page.page_index,
                content=page.content,
                success=page.success,
                model_used=page.model_used,
                provider_used=page.provider_used,
                attempts=attempts,
            )
        )

        if page.success:
            successful += 1
        else:
            failed += 1

        # Accumulate stats from all attempts
        for a in page.attempts:
            total_tokens_in += a.tokens_in
            total_tokens_out += a.tokens_out
            total_latency_ms += a.latency_ms

    stats = TranscriptionStats(
        total_pages=len(page_results),
        successful_pages=successful,
        failed_pages=failed,
        total_tokens_in=total_tokens_in,
        total_tokens_out=total_tokens_out,
        total_latency_ms=total_latency_ms,
    )

    return TranscriptionResponse(results=results, stats=stats)


class TranscriptionAPIPlugin(Plugin):
    """Plugin to register transcription API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
