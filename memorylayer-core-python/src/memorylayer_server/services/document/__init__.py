"""Document ingestion service — upload, parse, and extract memories from documents.

Also hosts the embed-server REST client plugin base, relocated from the
enterprise package in Phase 3 of the Aether convergence so OSS deployments
can talk to the (now OSS) embed server.
"""

from scitrera_app_framework import Variables, get_extension
from scitrera_app_framework.api import Plugin, enabled_option_pattern

from ...config import (
    DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE,
    MEMORYLAYER_EMBED_SERVER_SERVICE,
)
from .._constants import EXT_DOCUMENT_SERVICE, EXT_EMBED_SERVER_CLIENT
from .base import DocumentService, DocumentServicePluginBase


# noinspection PyAbstractClass
class EmbedServerClientPluginBase(Plugin):
    """Base plugin for the embed-server REST client extension."""

    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_EMBED_SERVER_CLIENT}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_EMBED_SERVER_CLIENT

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_EMBED_SERVER_SERVICE, self_attr="PROVIDER_NAME")

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(
            MEMORYLAYER_EMBED_SERVER_SERVICE,
            DEFAULT_MEMORYLAYER_EMBED_SERVER_SERVICE,
        )


def get_embed_server_client(v: Variables = None):
    """Get the embed-server REST client extension instance."""
    return get_extension(EXT_EMBED_SERVER_CLIENT, v)


__all__ = [
    "DocumentService",
    "DocumentServicePluginBase",
    "EXT_DOCUMENT_SERVICE",
    "EXT_EMBED_SERVER_CLIENT",
    "EmbedServerClientPluginBase",
    "get_embed_server_client",
]
