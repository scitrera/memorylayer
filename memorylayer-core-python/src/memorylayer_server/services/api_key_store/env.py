"""Environment-backed API key store (OSS default).

Resolves keys from the process environment via the framework ``Variables``
instance. This is the open-source default and preserves the historical
behaviour of reading provider keys straight from env vars.
"""

from __future__ import annotations

from logging import Logger

from scitrera_app_framework import Variables

from .base import ApiKeyStore, ApiKeyStorePluginBase


class EnvApiKeyStore(ApiKeyStore):
    """Resolve API keys from the environment (via :class:`Variables`)."""

    def __init__(self, v: Variables = None):
        self._v = v

    async def get_api_key(self, name: str, *, default: str | None = None) -> str | None:
        if self._v is not None:
            val = self._v.environ(name, default=None)
            if val:
                return val
        return default


class EnvApiKeyStorePlugin(ApiKeyStorePluginBase):
    """Plugin selected when ``MEMORYLAYER_API_KEY_STORE=env`` (OSS default)."""

    PROVIDER_NAME = "env"

    def initialize(self, v: Variables, logger: Logger) -> EnvApiKeyStore:
        return EnvApiKeyStore(v)
