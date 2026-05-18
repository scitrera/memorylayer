"""FastAPI application stub for MemoryLayer Embed Server.

This provides compatibility with typical uvicorn/gunicorn deployment setups.
"""

from memorylayer_embed_server.lifecycle.fastapi import fastapi_app_factory, get_logger, get_variables_dep

app = fastapi_app_factory(v=None)

__all__ = ("app", "get_logger", "get_variables_dep")
