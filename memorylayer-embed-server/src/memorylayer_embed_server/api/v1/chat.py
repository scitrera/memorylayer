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
from scitrera_app_framework import ext_parse_bool
from scitrera_app_framework.api import Plugin, Variables

from ...config import DEFAULT_EMBED_SERVER_LLM_ENABLED, EMBED_SERVER_LLM_ENABLED
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
        # MUST be False for a multi-extension router plugin. Returning True makes
        # this the single-extension winner for the router extension point, which
        # causes get_extensions() to return THIS router for every registered
        # name — shadowing all sibling routers (transcription, embeddings,
        # score, visual-tokenizer).
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        # Gate enablement HERE, not in initialize(). A multi-extension router that
        # contributes only when a flag is set must report that via
        # is_multi_extension so the framework simply omits it from the extension
        # registry when disabled. Gating in initialize() by returning None is
        # unsafe: the router consumer (lifecycle/fastapi.py) iterates
        # get_extensions() values and a None poisons app.include_router(None),
        # which silently drops every sibling router registered after it — the
        # embeddings/score/images 404 regression seen once LLM hosting moved off
        # the embed server (EMBED_SERVER_LLM_ENABLED=false).
        return v.environ(
            EMBED_SERVER_LLM_ENABLED,
            default=DEFAULT_EMBED_SERVER_LLM_ENABLED,
            type_fn=ext_parse_bool,
        )

    def initialize(self, v: Variables, logger: logging.Logger) -> object | None:
        # Enablement is already decided by is_multi_extension(); just contribute
        # the router.
        logger.info("Registering OpenAI-compatible LLM routes on embed-server")
        return router
