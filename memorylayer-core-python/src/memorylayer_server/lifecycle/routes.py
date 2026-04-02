from typing import Iterable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from scitrera_app_framework import get_extensions, Plugin, Variables as Variables
from ..api import EXT_MULTI_API_ROUTERS
from .fastapi import EXT_FASTAPI_SERVER
from .cors import EXT_CORS

EXT_ROUTES = 'memorylayer-server-fastapi-routes'

# Prefixes that are enterprise-only.  When no enterprise plugin registers
# the real router the fallback below will respond with 501 (Not Implemented)
# so SDKs can distinguish "endpoint doesn't exist" from "resource not found".
_ENTERPRISE_PREFIXES = ("/v1/documents", "/v1/datasets")


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
        registered_prefixes: set[str] = set()
        for ext_name, router in get_extensions(EXT_MULTI_API_ROUTERS, v).items():
            logger.debug('Adding API router from extension: %s', ext_name)
            app.include_router(router)
            if hasattr(router, 'prefix'):
                registered_prefixes.add(router.prefix)

        # Register 501 fallback routes for enterprise prefixes that have no
        # real router.  This lets SDKs distinguish "route not implemented"
        # (501) from "resource not found" (404).
        for prefix in _ENTERPRISE_PREFIXES:
            if prefix not in registered_prefixes:
                logger.debug(
                    'Registering 501 fallback for enterprise prefix: %s', prefix,
                )
                fallback = APIRouter()

                @fallback.api_route(
                    prefix + "/{path:path}",
                    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                    include_in_schema=False,
                )
                @fallback.api_route(
                    prefix,
                    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                    include_in_schema=False,
                )
                async def _enterprise_not_implemented(request: Request) -> JSONResponse:
                    return JSONResponse(
                        status_code=501,
                        content={
                            "detail": "This endpoint requires MemoryLayer Enterprise.",
                        },
                    )

                app.include_router(fallback)

        return

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (
            EXT_FASTAPI_SERVER,
            EXT_CORS,
        )
