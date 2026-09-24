"""Opt-in fail-closed startup and live readiness for dedicated model appliances."""

import asyncio
import json
import logging
import time

import httpx
from scitrera_app_framework import ext_parse_bool


def required_providers(v):
    if not v.environ("MEMORYLAYER_EMBED_REQUIRE_ALL_MODELS", default=False, type_fn=ext_parse_bool):
        return []
    dual = v.get("dual_embedding_service", default=None)
    cascade = v.get("cascade_transcriber", default=None)
    from ..profiles import service_profile
    profile = service_profile()
    providers = []
    if profile != "transcription":
        providers += [getattr(dual, "_single_vector", None), getattr(dual, "_multi_vector", None)]
    if profile != "embedding":
        ocr = list(getattr(cascade, "providers", []))
        if not ocr:
            raise RuntimeError("Required OCR models must be configured")
        if profile == "transcription" and {p.PROVIDER_NAME for p in ocr} != {"unlimited-ocr", "deepseek-ocr"}:
            raise RuntimeError("Transcription profile requires Unlimited-OCR and DeepSeek-OCR2")
        providers += ocr
    providers += list(v.get("required_model_resources", default=[]))
    if any(provider is None for provider in providers):
        raise RuntimeError("Required single-vector and multivector models must be configured")
    return providers


async def check_required_models(v):
    providers = required_providers(v)
    async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
        for provider in providers:
            runner = getattr(provider, "_runner", None)
            if runner is None:
                if not getattr(provider, "is_ready", False):
                    raise RuntimeError("Required model resource is not ready")
            else:
                if not runner.is_running:
                    raise RuntimeError("Required model engine is not running")
                response = await client.get(runner.health_url)
                response.raise_for_status()


async def preload_required_models(v):
    """Start every required engine, preserving the all-ready startup boundary."""
    providers = required_providers(v)
    if not providers:
        return providers
    parallel = v.environ("MEMORYLAYER_EMBED_PARALLEL_STARTUP", default=False, type_fn=ext_parse_bool)
    # OCR has the longest warmup. Start it first in either mode.
    from ..profiles import service_profile
    ordered = providers[2:] + providers[:2] if service_profile() == "combined" else providers
    logger = logging.getLogger(__name__)
    started = time.monotonic()
    timings = {}

    async def preload(provider):
        runner = getattr(provider, "_runner", None)
        name = getattr(runner, "model_name", type(provider).__name__)
        began = time.monotonic()
        logger.info("Starting required model: %s", name)
        await provider.preload()
        timings[name] = {
            "started_seconds": began - started,
            "ready_seconds": time.monotonic() - started,
            **getattr(runner, "startup_metrics", {}),
        }
        logger.info("Required model ready: %s (%.2f seconds)", name, time.monotonic() - began)

    if parallel:
        # TaskGroup waits for cancelled siblings' launch cleanup before the
        # lifespan shuts services down. No partial readiness on failure.
        async with asyncio.TaskGroup() as group:
            for provider in ordered:
                group.create_task(preload(provider))
    else:
        for provider in ordered:
            await preload(provider)
    await check_required_models(v)
    report = {"mode": "parallel" if parallel else "sequential",
              "seconds": time.monotonic() - started, "models": timings}
    v.set("required_model_startup", report)
    logger.info("Required model startup complete: %s", json.dumps(report, sort_keys=True))
    return providers
