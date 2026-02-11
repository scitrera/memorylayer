from typing import Iterable

from fastapi.middleware.cors import CORSMiddleware
from scitrera_app_framework import Variables as Variables
from scitrera_app_framework.api import Plugin, ext_parse_bool, ext_parse_csv

from .fastapi import EXT_FASTAPI_SERVER

MEMORYLAYER_SERVER_CORS_ALLOW_ORIGINS = 'MEMORYLAYER_SERVER_CORS_ALLOW_ORIGINS'
MEMORYLAYER_SERVER_CORS_ALLOW_CREDENTIALS = 'MEMORYLAYER_SERVER_CORS_ALLOW_CREDENTIALS'
MEMORYLAYER_SERVER_CORS_ALLOW_METHODS = 'MEMORYLAYER_SERVER_CORS_ALLOW_METHODS'
MEMORYLAYER_SERVER_CORS_ALLOW_HEADERS = 'MEMORYLAYER_SERVER_CORS_ALLOW_HEADERS'

DEFAULT_CORS_ALLOW_ORIGINS = ['*']
DEFAULT_CORS_ALLOW_CREDENTIALS = True
DEFAULT_CORS_ALLOW_METHODS = ['*']
DEFAULT_CORS_ALLOW_HEADERS = ['*']

EXT_CORS = 'memorylayer-server-fastapi-middleware-cors'


class CORSMiddlewarePlugin(Plugin):
    """
    Configure CORS middleware for the FastAPI application.
    """

    def extension_point_name(self, v: Variables) -> str:
        return EXT_CORS

    def initialize(self, v, logger) -> object | None:
        app = self.get_extension(EXT_FASTAPI_SERVER, v)

        allow_origins = v.environ(MEMORYLAYER_SERVER_CORS_ALLOW_ORIGINS,
                                  default=DEFAULT_CORS_ALLOW_ORIGINS, type_fn=ext_parse_csv)
        allow_credentials = v.environ(MEMORYLAYER_SERVER_CORS_ALLOW_CREDENTIALS,
                                      default=DEFAULT_CORS_ALLOW_CREDENTIALS, type_fn=ext_parse_bool)
        allow_methods = v.environ(MEMORYLAYER_SERVER_CORS_ALLOW_METHODS,
                                  default=DEFAULT_CORS_ALLOW_METHODS, type_fn=ext_parse_csv)
        allow_headers = v.environ(MEMORYLAYER_SERVER_CORS_ALLOW_HEADERS,
                                  default=DEFAULT_CORS_ALLOW_HEADERS, type_fn=ext_parse_csv)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=allow_methods,
            allow_headers=allow_headers,
        )

        return

    def get_dependencies(self, v: Variables) -> Iterable[str] | None:
        return (EXT_FASTAPI_SERVER,)
