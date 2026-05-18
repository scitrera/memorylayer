"""Health check endpoints for MemoryLayer Embed Server."""

import logging
from typing import Dict

from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from scitrera_app_framework import Plugin, Variables

from ..lifecycle.fastapi import get_logger, get_variables_dep
from . import EXT_MULTI_API_ROUTERS

router = APIRouter(tags=['health'])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check(
    v: Variables = Depends(get_variables_dep),
    logger: logging.Logger = Depends(get_logger),
) -> JSONResponse:
    """Readiness check verifying service availability."""
    checks = {
        "status": "ready",
        "services": {},
    }

    # Check transcription cascade
    cascade = v.get('cascade_transcriber', default=None)
    if cascade:
        checks["services"]["transcription"] = {
            "status": "available",
            "providers": len(cascade.providers),
        }
    else:
        checks["services"]["transcription"] = {"status": "not_configured"}

    # Check embedding service
    dual_service = v.get('dual_embedding_service', default=None)
    if dual_service:
        checks["services"]["embedding"] = {
            "single_vector": "available" if dual_service.has_single_vector else "not_configured",
            "multi_vector": "available" if dual_service.has_multi_vector else "not_configured",
        }
    else:
        checks["services"]["embedding"] = {"status": "not_configured"}
        checks["status"] = "not_ready"

    # LLM routing service (optional; only present when
    # MEMORYLAYER_EMBED_LLM_ENABLED=true and profiles were configured).
    llm_svc = v.get('llm_routing_service', default=None)
    if llm_svc is not None:
        checks["services"]["llm"] = {
            "status": "available",
            "profiles": sorted(llm_svc.profiles.keys()),
            "default_profile": llm_svc.default_profile,
        }
    else:
        checks["services"]["llm"] = {"status": "not_configured"}

    # Allow optional extensions (e.g. visual-tokenizer enterprise overlay)
    # to contribute health entries without coupling OSS to specific services.
    for callable_ in v.get('health_check_callables', default=[]):
        try:
            callable_(checks)
        except Exception as e:  # noqa: BLE001 - non-fatal, log and continue
            logger.warning("Health check extension failed: %s", e)

    status_code = (
        status.HTTP_200_OK
        if checks["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(content=checks, status_code=status_code)


@router.get("/health/load")
async def load_check(
    v: Variables = Depends(get_variables_dep),
) -> JSONResponse:
    """Per-instance load summary for upstream load balancers.

    Returns JSON shaped so a load-aware LB can route to the
    least-utilised embed-server replica:

    ::

        {
          "providers": {
            "colpali_multi_vector": {
              "in_flight": 1,
              "max_concurrent": 4,
              "utilization": 0.25
            }
          },
          "utilization": 0.25
        }

    Top-level ``utilization`` is the max across providers in [0, 1].
    Missing / not-configured providers are omitted.
    """
    providers: Dict[str, Dict[str, float]] = {}

    dual_service = v.get('dual_embedding_service', default=None)
    if dual_service is not None:
        multi = getattr(dual_service, '_multi_vector', None)
        if multi is not None and hasattr(multi, 'get_load_snapshot'):
            providers['colpali_multi_vector'] = multi.get_load_snapshot()

    # Merge LLM profile snapshots (one entry per profile, keyed ``llm_<name>``).
    llm_svc = v.get('llm_routing_service', default=None)
    if llm_svc is not None:
        try:
            providers.update(llm_svc.get_load_snapshot())
        except Exception:  # noqa: BLE001 - non-fatal; missing one profile shouldn't break health
            pass

    top = max(
        (p.get('utilization', 0.0) for p in providers.values()),
        default=0.0,
    )
    return JSONResponse(content={'providers': providers, 'utilization': top}, status_code=200)


@router.get("/health/gpu")
async def gpu_health(
    v: Variables = Depends(get_variables_dep),
) -> JSONResponse:
    """GPU health check with memory usage information."""
    gpu_monitor = v.get('gpu_monitor', default=None)
    if gpu_monitor is None:
        return JSONResponse(
            content={"error": "GPU monitor not configured"},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    gpu_status = gpu_monitor.get_gpu_status()
    status_code = (
        status.HTTP_200_OK
        if gpu_status.get("available", False)
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(content=gpu_status, status_code=status_code)


class HealthAPIPlugin(Plugin):
    """Plugin to register health API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # multi-extension pattern

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
