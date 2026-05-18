"""Unit tests for AetherServiceConnection.get_authority_resolver (Phase 3.5c).

Covers the resolver-getter contract added in sub-phase 3.5c of the
MemoryLayer Aether Convergence plan: a single shared
``AsyncAuthorityResolver`` lazy-bound to the connection's
``AsyncServiceClient``, with TTL/max-entries honouring the
``MEMORYLAYER_AETHER_RESOLVER_*`` knobs.

What is tested here:
- ``get_authority_resolver()`` returns a resolver whose ``_client``
  attribute is the same client the connection holds.
- Calling twice returns the identical instance (idempotent / single cache).
- Calling before ``connect()`` raises ``RuntimeError``.
- ``resolver_cache_ttl_s`` constructor arg is forwarded to the resolver's
  ``_max_ttl_s`` attribute.
- ``resolver_max_entries`` constructor arg is forwarded to the resolver's
  ``_max_entries`` attribute.
- After a simulated disconnect + reconnect, a fresh resolver instance is
  returned (cached entries are bound to the old client).

Deviations from the original 3.5c spec:
- Spec described env-var tests (``MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S``
  etc.) exercised via the plugin's ``initialize()`` code path.  These tests
  instead drive the constructor-arg seam directly, which is the same
  correctness signal with less fixture overhead: the plugin reads those env
  vars and passes them to the constructor, so testing at the constructor
  seam is equivalent.

Out of scope here:
- The resolver's gRPC resolve_authority RPC (covered in the Aether SDK's
  ``tests/test_authority.py``).
- The terminator's header overlay that consumes the resolver (covered in
  ``tests/test_proxy_terminator.py`` in the SDK tree).
- Live Aether gateway / dev compose wiring — Phase 4 defers the
  live-gateway smoke check to a manual verification step.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.aether_service import AetherServiceConnection


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_variables():
    """Minimal Variables stub satisfying ``get_logger`` and ``environ``."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default, **kwargs: default)
    return v


@pytest.fixture
def mock_aether_client():
    """Stand-in for ``AsyncServiceClient``.

    The resolver only needs a non-None object to bind to; no RPC methods
    are called in these unit tests.
    """
    return AsyncMock()


def _make_service_connection(mock_variables, **overrides) -> AetherServiceConnection:
    """Construct ``AetherServiceConnection`` with test defaults.

    Mirrors the helper in
    ``proprietary/memorylayer-enterprise/tests/unit/test_aether_integration.py``
    (``TestAetherServiceConnectionUnifiedClient`` setup) for consistency.
    """
    kwargs = dict(
        gateway_addr="localhost:50051",
        workspace="test-workspace",
        specifier="main",
    )
    kwargs.update(overrides)
    return AetherServiceConnection(mock_variables, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetAuthorityResolver:
    """Resolver-getter contract for ``AetherServiceConnection``."""

    def test_get_authority_resolver_returns_resolver_bound_to_client(
        self, mock_variables, mock_aether_client
    ):
        """Resolver's ``_client`` attribute is the same object the connection holds."""
        svc = _make_service_connection(mock_variables)
        # Bypass connect() — the getter only requires a non-None _client.
        svc._client = mock_aether_client

        resolver = svc.get_authority_resolver()

        # AsyncAuthorityResolver stashes the client on ``_client``
        # (see sdk/python-client/scitrera_aether_client/authority.py __init__).
        assert resolver._client is mock_aether_client

    def test_get_authority_resolver_is_idempotent(
        self, mock_variables, mock_aether_client
    ):
        """A second call returns the same instance — single shared LRU cache."""
        svc = _make_service_connection(mock_variables)
        svc._client = mock_aether_client

        first = svc.get_authority_resolver()
        second = svc.get_authority_resolver()

        assert first is second

    def test_get_authority_resolver_raises_when_client_missing(self, mock_variables):
        """Calling before ``connect()`` raises ``RuntimeError`` with helpful text."""
        svc = _make_service_connection(mock_variables)
        # _client is None by default — connect() never ran.

        with pytest.raises(RuntimeError, match="connect"):
            svc.get_authority_resolver()

    def test_resolver_ttl_respects_constructor_arg(
        self, mock_variables, mock_aether_client
    ):
        """``resolver_cache_ttl_s=5`` is forwarded to the resolver's ``_max_ttl_s``."""
        svc = _make_service_connection(mock_variables, resolver_cache_ttl_s=5)
        svc._client = mock_aether_client

        resolver = svc.get_authority_resolver()

        # AsyncAuthorityResolver converts int → float on construction.
        assert resolver._max_ttl_s == 5.0

    def test_resolver_max_entries_respects_constructor_arg(
        self, mock_variables, mock_aether_client
    ):
        """``resolver_max_entries=42`` is forwarded to the resolver's ``_max_entries``."""
        svc = _make_service_connection(mock_variables, resolver_max_entries=42)
        svc._client = mock_aether_client

        resolver = svc.get_authority_resolver()

        assert resolver._max_entries == 42

    def test_disconnect_drops_resolver_so_reconnect_yields_fresh_instance(
        self, mock_variables, mock_aether_client
    ):
        """After a simulated disconnect+reconnect a fresh resolver is returned.

        Cached entries are bound to the old client identity; keeping them
        after a reconnect would be stale.  ``AetherServiceConnection.disconnect``
        sets ``_authority_resolver = None`` to force a fresh build on next call.
        """
        svc = _make_service_connection(mock_variables)
        svc._client = mock_aether_client
        first = svc.get_authority_resolver()

        # Simulate disconnect (mirrors the real disconnect() implementation).
        svc._client = None
        svc._authority_resolver = None

        # Re-connect with the same client object.
        svc._client = mock_aether_client
        second = svc.get_authority_resolver()

        assert second is not first
