"""OpenAI-compatible chat / completions / models routes.

Registered only when ``MEMORYLAYER_EMBED_LLM_ENABLED=true`` and at least
one LLM profile is configured. The routes are transparent proxies:
they read the OpenAI-shape request body, pick a target profile by the
``model`` field, and forward the request to that profile's ``vllm serve``
subprocess via httpx (streaming SSE through when ``stream=true``).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from scitrera_app_framework.api import Plugin, Variables

from ...config import EMBED_SERVER_LLM_ENABLED, DEFAULT_EMBED_SERVER_LLM_ENABLED
from ...lifecycle.fastapi import get_variables_dep
from ...services.llm.router import LLMRoutingService, UnknownModelError
from .. import EXT_MULTI_API_ROUTERS

router = APIRouter(tags=["llm"])


def _get_routing_service(v: Variables) -> LLMRoutingService:
    svc = v.get("llm_routing_service", default=None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "LLM service is not configured on this embed-server. "
                "Set MEMORYLAYER_EMBED_LLM_ENABLED=true and declare profiles via "
                "MEMORYLAYER_EMBED_LLM_PROFILES."
            ),
        )
    return svc


async def _route_request(
    request: Request,
    v: Variables,
    *,
    endpoint: str,  # "chat" | "completions"
) -> JSONResponse | StreamingResponse:
    svc = _get_routing_service(v)

    try:
        payload = await request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid JSON body: {e}",
        ) from e
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be a JSON object",
        )

    model = payload.get("model")
    try:
        provider = svc.resolve(model)
    except UnknownModelError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "message": str(e),
                    "type": "model_not_found",
                    "param": "model",
                    "available_models": e.available,
                }
            },
        ) from e

    stream = bool(payload.get("stream", False))

    if endpoint == "chat":
        result = await provider.chat_completions(payload, stream=stream)
    else:
        result = await provider.completions(payload, stream=stream)

    if stream:
        return StreamingResponse(result, media_type="text/event-stream")
    return JSONResponse(content=result)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    v: Variables = Depends(get_variables_dep),
):
    return await _route_request(request, v, endpoint="chat")


@router.post("/v1/completions")
async def completions(
    request: Request,
    v: Variables = Depends(get_variables_dep),
):
    return await _route_request(request, v, endpoint="completions")


@router.get("/v1/models")
async def list_models(
    v: Variables = Depends(get_variables_dep),
) -> JSONResponse:
    svc = _get_routing_service(v)
    return JSONResponse(content={"object": "list", "data": svc.list_models()})


class LLMChatRoutePlugin(Plugin):
    """Register the OpenAI-compatible LLM routes when LLM hosting is enabled."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_MULTI_API_ROUTERS

    def is_enabled(self, v: Variables) -> bool:
        # ext_parse_bool would be cleaner but the plugin contract just needs a bool.
        raw = v.environ(EMBED_SERVER_LLM_ENABLED, default=str(DEFAULT_EMBED_SERVER_LLM_ENABLED))
        return str(raw).strip().lower() in ("true", "1", "yes", "on")

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        logger.info("Registering OpenAI-compatible LLM routes on embed-server")
        return router
