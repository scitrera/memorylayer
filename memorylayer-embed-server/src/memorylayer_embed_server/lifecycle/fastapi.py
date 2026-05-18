"""FastAPI application factory with lifespan for model warm-up."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from logging import Logger

from fastapi import FastAPI, Request
from scitrera_app_framework import (
    Plugin, Variables, get_logger as _saf_get_logger, get_variables as _saf_get_variables,
    get_extension as _saf_get_extension, ext_parse_bool,
)
from scitrera_app_framework.core.plugins import init_all_plugins as _saf_init_all_plugins

from .. import __version__
from ..config import (
    EMBED_SERVER_PRELOAD_MODELS, DEFAULT_EMBED_SERVER_PRELOAD_MODELS,
    EMBED_SERVER_LLM_PRELOAD, DEFAULT_EMBED_SERVER_LLM_PRELOAD,
)
from ..services.embedding.dual_service import EXT_DUAL_EMBEDDING_SERVICE
from ..api import EXT_MULTI_API_ROUTERS

EXT_FASTAPI_SERVER = 'embed-server-fastapi-server'


async def get_variables_dep(request: Request) -> Variables:
    """Dependency to get Variables instance from request."""
    return request.app.state.v


async def get_logger(request: Request) -> Logger:
    """Dependency to get logger from Variables."""
    logger: Logger = _saf_get_logger(request.app.state.v)
    return logger


class FastApiPlugin(Plugin):
    """Configure routes for the FastAPI application."""

    def extension_point_name(self, v: Variables) -> str:
        return EXT_FASTAPI_SERVER

    def initialize(self, v, logger) -> object | None:
        logger.info('Initializing Embed Server FastAPI App')

        # noinspection PyShadowingNames
        @asynccontextmanager
        async def lifespan_context(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan context manager with model warm-up."""
            from ..dependencies import initialize_services, shutdown_services

            nonlocal v
            await initialize_services(v)

            # store app in variables for access in services
            v.set('app', app)

            # store variables in app state
            app.state.v = v

            # preload models if configured
            preload = v.environ(
                EMBED_SERVER_PRELOAD_MODELS,
                default=DEFAULT_EMBED_SERVER_PRELOAD_MODELS,
                type_fn=ext_parse_bool,
            )
            if preload:
                logger.info("Preloading models during startup")
                try:
                    # Preload transcription providers
                    cascade = v.get('cascade_transcriber', default=None)
                    if cascade:
                        await cascade.preload()
                        logger.info("Transcription providers preloaded")

                    # Preload embedding providers
                    dual_service = v.get('dual_embedding_service', default=None)
                    if dual_service:
                        await dual_service.preload()
                        logger.info("Embedding providers preloaded")

                    # Visual tokenizer (and any other extension service) handles
                    # its own preload via async_ready in its plugin definition.
                except Exception as e:
                    logger.warning("Model preload failed (non-fatal): %s", e)

            # LLM subprocesses have their own preload flag (independent of
            # ``EMBED_SERVER_PRELOAD_MODELS``) because spawning N chat models
            # at boot is much heavier than warming an embedding cache.
            llm_preload = v.environ(
                EMBED_SERVER_LLM_PRELOAD,
                default=DEFAULT_EMBED_SERVER_LLM_PRELOAD,
                type_fn=ext_parse_bool,
            )
            if llm_preload:
                llm_svc = v.get('llm_routing_service', default=None)
                if llm_svc is not None:
                    logger.info("Preloading LLM profiles during startup")
                    try:
                        await llm_svc.preload()
                        logger.info("LLM profiles preloaded")
                    except Exception as e:  # noqa: BLE001 - non-fatal; lazy-start on first request
                        logger.warning("LLM preload failed (non-fatal): %s", e)

            try:
                yield
            finally:
                await shutdown_services(v)

        app = FastAPI(
            title="MemoryLayer Embed Server",
            description="Stateless transcription and embedding GPU server",
            version=__version__,
            lifespan=lifespan_context,
        )

        # Map the GPU queue-timeout exception from any ColPali code path
        # to a 503 with Retry-After so callers can shed load gracefully.
        try:
            from ..services.embedding.colpali import ColPaliQueueTimeoutError
            from fastapi import Request
            from fastapi.responses import JSONResponse

            @app.exception_handler(ColPaliQueueTimeoutError)
            async def _colpali_queue_timeout_handler(  # noqa: ARG001 - signature mandated
                request: Request, exc: ColPaliQueueTimeoutError,
            ) -> JSONResponse:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": str(exc),
                        "wait_seconds": exc.wait_seconds,
                        "max_concurrent": exc.max_concurrent,
                    },
                    headers={"Retry-After": "1"},
                )
        except ImportError:
            # ColPali module not importable (e.g. tests with mocks only) —
            # the handler is irrelevant in that case.
            pass

        @app.get("/")
        async def root() -> dict:
            """Root endpoint providing API information."""
            return {
                "name": "MemoryLayer Embed Server",
                "version": __version__,
                "description": "Stateless transcription and embedding GPU server",
            }

        # Register API routers from multi-extension point
        from scitrera_app_framework import get_extensions
        try:
            routers = get_extensions(EXT_MULTI_API_ROUTERS, v)
            if routers:
                for router in routers.values():
                    app.include_router(router)
                logger.info("Registered %d API routers", len(routers))
        except Exception as e:
            logger.debug("No API routers registered yet (will be available after init): %s", e)

        return app


def fastapi_app_factory(v: Variables = None) -> FastAPI:
    """Factory function to create FastAPI app instance."""
    v: Variables = _saf_get_variables(v)

    # explicitly ensure that all plugins are initialized
    _saf_init_all_plugins(v, async_enabled=False)

    app: FastAPI = _saf_get_extension(EXT_FASTAPI_SERVER, v)
    return app
