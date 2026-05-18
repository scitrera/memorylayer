"""Deprecated compatibility shim for ``memorylayer_server.services.aether_agent``.

The implementation moved to :mod:`memorylayer_server.services.aether_service`
in the Phase 1 Aether convergence: MemoryLayer's in-process Aether
connection now registers as a Service principal (``sv::memorylayer::*``)
rather than an Agent, and the data-plane ``on_message`` dispatch is gone
in favour of REST.

This module re-exports the new module's public surface under the old names
for one release so external callers (and the enterprise re-export) keep
importing without immediate breakage.  Callers should update to
``memorylayer_server.services.aether_service`` directly.

Importing names from this module emits a ``DeprecationWarning`` (via the
module-level ``__getattr__`` hook) so callers see exactly which name needs
migrating.
"""

from __future__ import annotations

import warnings as _warnings

from memorylayer_server.services._constants import (
    EXT_AETHER_AGENT_SERVICE as _EXT_AETHER_AGENT_SERVICE,
)
from memorylayer_server.services._constants import (  # noqa: F401
    EXT_AETHER_SERVICE_CONNECTION as _EXT_AETHER_SERVICE_CONNECTION,
)
from memorylayer_server.services.aether_service import (
    AETHER_API_KEY as _AETHER_API_KEY,
)
from memorylayer_server.services.aether_service import (
    AETHER_API_KEY_FILE as _AETHER_API_KEY_FILE,
)
from memorylayer_server.services.aether_service import (
    AETHER_AUTH as _AETHER_AUTH,
)
from memorylayer_server.services.aether_service import (
    AETHER_GATEWAY_ADDR as _AETHER_GATEWAY_ADDR,
)
from memorylayer_server.services.aether_service import (
    AETHER_SERVICE_SPECIFIER as _AETHER_SERVICE_SPECIFIER,
)
from memorylayer_server.services.aether_service import (
    AETHER_TLS_CA_CERT as _AETHER_TLS_CA_CERT,
)
from memorylayer_server.services.aether_service import (
    AETHER_TLS_CLIENT_CERT as _AETHER_TLS_CLIENT_CERT,
)
from memorylayer_server.services.aether_service import (
    AETHER_TLS_CLIENT_KEY as _AETHER_TLS_CLIENT_KEY,
)
from memorylayer_server.services.aether_service import (
    AETHER_TLS_ENABLED as _AETHER_TLS_ENABLED,
)
from memorylayer_server.services.aether_service import (
    AETHER_WORKSPACE as _AETHER_WORKSPACE,
)
from memorylayer_server.services.aether_service import (
    DEFAULT_AETHER_GATEWAY_ADDR as _DEFAULT_AETHER_GATEWAY_ADDR,
)
from memorylayer_server.services.aether_service import (
    DEFAULT_AETHER_WORKSPACE as _DEFAULT_AETHER_WORKSPACE,
)
from memorylayer_server.services.aether_service import (
    DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION as _DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION,
)
from memorylayer_server.services.aether_service import (
    MEMORYLAYER_AETHER_SERVICE_CONNECTION as _MEMORYLAYER_AETHER_SERVICE_CONNECTION,
)

# Re-export the canonical module's public surface under both new and legacy
# names so importers using either set of identifiers keep working.  We do
# this eagerly (rather than lazily) so the framework's recursive plugin
# discovery picks up the canonical class object — the deprecation warning
# fires only on direct attribute access from external code (via the
# ``__getattr__`` hook below).
from memorylayer_server.services.aether_service import (  # noqa: F401
    AetherServiceConnection as _AetherServiceConnection,
)
from memorylayer_server.services.aether_service import (
    AetherServiceConnectionPlugin as _AetherServiceConnectionPlugin,
)

# Legacy / canonical name → underlying object table.  All names accessible
# via this module funnel through ``__getattr__`` so each lookup emits a
# single deprecation warning pointing at the new import path.
_EXPORTS = {
    # Canonical names (kept here so existing ``from aether_agent import ...``
    # for the new identifiers also works during the transition window).
    "AetherServiceConnection": _AetherServiceConnection,
    "AetherServiceConnectionPlugin": _AetherServiceConnectionPlugin,
    "EXT_AETHER_SERVICE_CONNECTION": _EXT_AETHER_SERVICE_CONNECTION,
    "MEMORYLAYER_AETHER_SERVICE_CONNECTION": _MEMORYLAYER_AETHER_SERVICE_CONNECTION,
    "DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION": _DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION,
    "AETHER_SERVICE_SPECIFIER": _AETHER_SERVICE_SPECIFIER,
    # Legacy aliases — same objects under their old names.
    "AetherAgentService": _AetherServiceConnection,
    "AetherAgentServicePlugin": _AetherServiceConnectionPlugin,
    "EXT_AETHER_AGENT_SERVICE": _EXT_AETHER_AGENT_SERVICE,
    "MEMORYLAYER_AETHER_AGENT_SERVICE": "MEMORYLAYER_AETHER_AGENT_SERVICE",
    "DEFAULT_MEMORYLAYER_AETHER_AGENT_SERVICE": "aether-agent",
    "AETHER_AGENT_SPECIFIER": "AETHER_AGENT_SPECIFIER",
    # Shared config knobs (canonical names).
    "AETHER_GATEWAY_ADDR": _AETHER_GATEWAY_ADDR,
    "DEFAULT_AETHER_GATEWAY_ADDR": _DEFAULT_AETHER_GATEWAY_ADDR,
    "AETHER_WORKSPACE": _AETHER_WORKSPACE,
    "DEFAULT_AETHER_WORKSPACE": _DEFAULT_AETHER_WORKSPACE,
    "AETHER_API_KEY": _AETHER_API_KEY,
    "AETHER_API_KEY_FILE": _AETHER_API_KEY_FILE,
    "AETHER_AUTH": _AETHER_AUTH,
    "AETHER_TLS_ENABLED": _AETHER_TLS_ENABLED,
    "AETHER_TLS_CA_CERT": _AETHER_TLS_CA_CERT,
    "AETHER_TLS_CLIENT_CERT": _AETHER_TLS_CLIENT_CERT,
    "AETHER_TLS_CLIENT_KEY": _AETHER_TLS_CLIENT_KEY,
}


def __getattr__(name: str):
    """Lazy attribute access with a deprecation warning.

    Defined as a module-level ``__getattr__`` (PEP 562) so consumers that
    import names from this module still resolve them, but each lookup is
    flagged with a ``DeprecationWarning`` pointing at the new module path.
    """
    if name in _EXPORTS:
        _warnings.warn(
            "memorylayer_server.services.aether_agent is deprecated; import from memorylayer_server.services.aether_service instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = tuple(_EXPORTS)
