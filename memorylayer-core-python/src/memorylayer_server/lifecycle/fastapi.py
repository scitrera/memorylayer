from contextlib import asynccontextmanager
from typing import AsyncGenerator
from logging import Logger

from fastapi import FastAPI, Request
from scitrera_app_framework import (
    Plugin, Variables, get_logger as _saf_get_logger, get_variables as _saf_get_variables, get_extension as _saf_get_extension
)
from scitrera_app_framework.core.plugins import init_all_plugins as _saf_init_all_plugins

from .. import __version__

EXT_FASTAPI_SERVER = 'memorylayer-server-fastapi-server'


async def get_variables_dep(request: Request) -> Variables:
    """Dependency to get Variables instance from request."""
    return request.app.state.v


async def get_logger(request: Request) -> Logger:
    """Dependency to get logger from Variables."""
    logger: Logger = _saf_get_logger(request.app.state.v)
    return logger


class FastApiPlugin(Plugin):
    """
    Configure routes for the FastAPI application.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_FASTAPI_SERVER

    def initialize(self, v, logger) -> object | None:
        logger.info('Initializing FastAPI App')

        # noinspection PyShadowingNames
        @asynccontextmanager
        async def lifespan_context(app: FastAPI) -> AsyncGenerator[None, None]:
            """Application lifespan context manager."""
            from ..dependencies import initialize_services, shutdown_services

            nonlocal v
            await initialize_services(v)

            # store app in variables for access in services/plugins
            v.set('app', app)

            # store variables in app state
            app.state.v = v

            try:
                yield
            finally:
                await shutdown_services(v)

        app = FastAPI(
            title="memorylayer.ai",
            description="API-first memory infrastructure for LLM-powered agents",
            version=__version__,
            lifespan=lifespan_context,
        )

        @app.get("/")
        async def root() -> dict:
            """
            Root endpoint providing API information.

            Returns:
                dict: API name and version
            """
            return {
                "name": "MemoryLayer.ai",
                "version": __version__,
                "description": "API-first memory infrastructure for LLM-powered agents",
            }

        return app


def fastapi_app_factory(v: Variables = None) -> FastAPI:
    """Factory function to create FastAPI app instance."""
    v: Variables = _saf_get_variables(v)

    # explicitly ensure that all plugins are initialized (async parts will be handled by lifespan context)
    _saf_init_all_plugins(v, async_enabled=False)

    app: FastAPI = _saf_get_extension(EXT_FASTAPI_SERVER, v)
    return app
