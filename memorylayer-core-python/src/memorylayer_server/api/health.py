"""Health check endpoints for MemoryLayer.ai API."""
import logging

from typing import Dict

from fastapi import APIRouter, status, Depends
from fastapi.responses import JSONResponse
from scitrera_app_framework import Plugin, Variables

from ..lifecycle.fastapi import get_logger
from . import EXT_MULTI_API_ROUTERS

router = APIRouter(tags=['health'])


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        dict: Health status
    """
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check(logger: logging.Logger = Depends(get_logger), ) -> JSONResponse:
    """
    Readiness check endpoint verifying database and cache connectivity.

    Returns:
        JSONResponse: Readiness status with service checks
    """
    checks = {
        "status": "ready",
        "services": {},
    }

    # Check database connectivity
    try:
        from memorylayer_server.services.storage import get_storage_backend
        storage = get_storage_backend()
        is_healthy = await storage.health_check()
        checks["services"]["database"] = "connected" if is_healthy else "disconnected"
        if not is_healthy:
            checks["status"] = "not_ready"
    except Exception as e:
        logger.error("Database connectivity check failed: %s", e)
        checks["services"]["database"] = "disconnected"
        checks["status"] = "not_ready"

    # Cache is optional and not yet configured via plugin
    checks["services"]["cache"] = "not_configured"

    status_code = (
        status.HTTP_200_OK
        if checks["status"] == "ready"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(content=checks, status_code=status_code)


class HealthAPIPlugin(Plugin):
    """Plugin to register health API routes."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        return False  # disable "single" extension for a multi-extension plugin

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        return router

    def is_multi_extension(self, v: Variables) -> bool:
        return True
