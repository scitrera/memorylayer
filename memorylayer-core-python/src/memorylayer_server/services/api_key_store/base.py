"""Pluggable API key store abstraction.

Providers (LLM, embedding, ...) resolve their provider API keys through an
:class:`ApiKeyStore` instead of reading the environment directly. The OSS
default (``env``) reads from the process environment; enterprise deployments
can plug in an Aether-backed store that pulls per-tenant keys from secure KV
(with an env fallback), enabling dynamic/rotated keys without a restart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_API_KEY_STORE
from .._plugin_factory import make_service_plugin_base

# Provider-selection + tuning constants.
MEMORYLAYER_API_KEY_STORE = "MEMORYLAYER_API_KEY_STORE"
DEFAULT_MEMORYLAYER_API_KEY_STORE = "env"

# Cache TTL (seconds) for stores that resolve against a remote secret source.
# The env store ignores it (environment reads are free).
MEMORYLAYER_API_KEY_STORE_TTL = "MEMORYLAYER_API_KEY_STORE_TTL"
DEFAULT_MEMORYLAYER_API_KEY_STORE_TTL = 60.0


class ApiKeyStore(ABC):
    """Resolves a provider API key by name.

    The ``name`` is the raw secret / env-var name (e.g. ``OPENAI_API_KEY``).
    The store knows nothing about providers or their default names -- callers
    choose the name they want resolved.
    """

    @abstractmethod
    async def get_api_key(self, name: str, *, default: str | None = None) -> str | None:
        """Return the key registered under ``name``, or ``default`` if unresolved.

        Returns ``None`` (rather than raising) when the key cannot be resolved,
        so callers can degrade to keyless behaviour.
        """
        raise NotImplementedError


class ApiKeyRef:
    """A bound ``(store, name)`` pair; :meth:`resolve` returns the current key.

    Providers hold one of these instead of a static key string so a rotated
    key is picked up on the next resolve (bounded by the store's own caching).
    """

    __slots__ = ("_store", "_name", "_default")

    def __init__(self, store: ApiKeyStore, name: str, *, default: str | None = None):
        self._store = store
        self._name = name
        self._default = default

    @property
    def name(self) -> str:
        return self._name

    async def resolve(self) -> str | None:
        return await self._store.get_api_key(self._name, default=self._default)


def resolve_api_key_ref(
    *,
    api_key: str | None,
    api_key_name: str | None,
    api_key_store: ApiKeyStore | None,
    default_name: str,
) -> tuple[str | None, ApiKeyRef | None]:
    """Decide static-key vs store-resolution for a provider.

    Precedence:
        1. A literal ``api_key`` wins and **skips the store** (static escape hatch).
        2. Else resolve ``api_key_name or default_name`` through the store.
        3. Else (no store) -> no key.

    Returns ``(static_key, ref)`` where at most one is non-None.
    """
    if api_key:
        return api_key, None
    if api_key_store is None:
        return None, None
    return None, ApiKeyRef(api_key_store, api_key_name or default_name)


def get_api_key_store(v: Variables) -> ApiKeyStore:
    """Return the configured :class:`ApiKeyStore` extension.

    Falls back to a process-local :class:`EnvApiKeyStore` when the extension is
    not registered (e.g. unit tests that build a provider directly, or a
    deployment that has the store disabled) so callers always get a usable
    store rather than an error.
    """
    try:
        return get_extension(EXT_API_KEY_STORE, v)
    except Exception:
        # Lazy import avoids a base<->env import cycle.
        from .env import EnvApiKeyStore

        return EnvApiKeyStore(v)


# noinspection PyAbstractClass
ApiKeyStorePluginBase = make_service_plugin_base(
    ext_name=EXT_API_KEY_STORE,
    config_key=MEMORYLAYER_API_KEY_STORE,
    default_value=DEFAULT_MEMORYLAYER_API_KEY_STORE,
)
