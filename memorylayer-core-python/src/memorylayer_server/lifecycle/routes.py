from typing import Iterable

from scitrera_app_framework import get_extensions, Plugin, Variables as Variables
from ..api import EXT_MULTI_API_ROUTERS
from .fastapi import EXT_FASTAPI_SERVER
from .cors import EXT_CORS

EXT_ROUTES = 'memorylayer-server-fastapi-routes'


class RoutesPlugin(Plugin):
    """
    Configure routes for the FastAPI application.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_ROUTES

    def initialize(self, v, logger) -> object | None:
        logger.info('Initializing Routes')
        app = self.get_extension(EXT_FASTAPI_SERVER, v)

        # Register API routers -- requires that we run after all API router plugins are registered!
        for ext_name, router in get_extensions(EXT_MULTI_API_ROUTERS, v).items():
            logger.info('Adding API router from extension: %s', ext_name)
            app.include_router(router)

        return

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (
            EXT_FASTAPI_SERVER,
            EXT_CORS,
        )
