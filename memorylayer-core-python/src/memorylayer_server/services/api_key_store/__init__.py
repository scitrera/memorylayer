"""API key store service package.

Exposes the :class:`ApiKeyStore` abstraction, the :class:`ApiKeyRef` helper
providers use to resolve a key lazily, and the OSS default ``env`` store.
Plugins in this package are auto-discovered by
``register_package_plugins(services.__package__, recursive=True)``.
"""

from .._constants import EXT_API_KEY_STORE
from .base import (
    DEFAULT_MEMORYLAYER_API_KEY_STORE,
    DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL,
    MEMORYLAYER_API_KEY_STORE,
    MEMORYLAYER_API_KEY_STORE_TTL,
    ApiKeyRef,
    ApiKeyStore,
    ApiKeyStorePluginBase,
    get_api_key_store,
    resolve_api_key_ref,
)
from .env import EnvApiKeyStore, EnvApiKeyStorePlugin

__all__ = (
    "EXT_API_KEY_STORE",
    "ApiKeyStore",
    "ApiKeyRef",
    "ApiKeyStorePluginBase",
    "EnvApiKeyStore",
    "EnvApiKeyStorePlugin",
    "resolve_api_key_ref",
    "get_api_key_store",
    "MEMORYLAYER_API_KEY_STORE",
    "DEFAULT_MEMORYLAYER_API_KEY_STORE",
    "MEMORYLAYER_API_KEY_STORE_TTL",
    "DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL",
)
