"""
FastAPI application stub for memorylayer.ai

This provides compatibility with typical uvicorn/gunicorn deployment setups.
"""

from memorylayer_server.lifecycle.fastapi import fastapi_app_factory, get_logger, get_variables_dep

app = fastapi_app_factory(v=None)

__all__ = (
    'app', 'get_logger', 'get_variables_dep',
)
