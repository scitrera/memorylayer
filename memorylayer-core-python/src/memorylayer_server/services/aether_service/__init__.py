"""
Aether Service connection for MemoryLayer.

Owns the **single** ``AsyncServiceClient`` that connects MemoryLayer to the
Aether gateway as a Service principal.  All other Aether-dependent services
(tasks, rate limiting, token management) obtain the shared client from this
service rather than maintaining their own connections.

Identity: ``sv::memorylayer::{specifier}``

Phase 1 (Aether convergence): MemoryLayer used to register as an Agent
(``ag::_system::memorylayer::*``) and dispatch ``recall``/``remember``/
``search`` actions over an ``on_message`` handler.  Aether has a first-
class :class:`PrincipalService` type for workspace-less backends and the
REST surface (the same actions and more) is the canonical front door.
This module now registers as a Service and the agent message handler has
been retired — REST handles all data-plane operations.

Configuration (environment variables)
--------------------------------------
``MEMORYLAYER_AETHER_SERVICE_CONNECTION``
    Provider selection; set to ``aether-service`` to enable (default:
    ``disabled``).  Legacy ``MEMORYLAYER_AETHER_AGENT_SERVICE`` is still
    honoured for compatibility (with a deprecation warning).
``AETHER_GATEWAY_ADDR``
    Aether gateway gRPC address (default: ``localhost:50051``).
``AETHER_API_KEY``
    Single API key for all Aether operations (required unless
    ``AETHER_AUTH=none``).
``AETHER_AUTH``
    Set to ``none`` for local dev — skips key requirement and connectivity
    check.
``AETHER_SERVICE_SPECIFIER``
    Service specifier, typically the node/host name (default: hostname or
    ``main``).  Legacy ``AETHER_AGENT_SPECIFIER`` is still honoured.
``AETHER_WORKSPACE``
    Default Aether workspace for outbound messages and KV scoping.  Service
    principals are workspace-less but consumer services (tasks, rate-limit
    KV) still scope their operations against a workspace; this preserves
    the previous default of ``_system``.
``AETHER_TASKS_ENABLED``
    Enable/disable task scheduling via the shared client (default: ``true``).
``AETHER_RATELIMIT_ENABLED``
    Enable/disable rate limiting via the shared client (default: ``true``).
``MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S``
    Cache TTL (seconds) for the shared :class:`AsyncAuthorityResolver` used
    by the proxy-http terminator wiring (Phase 3.5c).  Default: ``60``.
``MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES``
    Max entries in the resolver's LRU cache.  Default: ``10000``.
``MEMORYLAYER_AETHER_REST_FRONT_DOOR``
    Selects how MemoryLayer's REST surface is reachable via Aether
    (Phase 2c).  Allowed values:

    * ``in_process`` (default) — register an in-process
      :class:`ProxyHttpTerminator` that dispatches inbound proxy envelopes
      into the FastAPI app via the ASGI bridge.  Single Aether
      connection, single process, no extra container.
    * ``disabled`` — do not register the terminator.  The Aether
      connection still services back-channel work (tasks, KV, tokens) but
      the REST surface is reachable only over plain HTTP (the legacy
      auth-proxy path).
    * ``sidecar`` — reserved for the contingency path (running the Go
      ``proxy-sidecar`` binary as a separate container).  In-process the
      effect is identical to ``disabled`` (the sidecar lives outside this
      process); we accept the value so deployments can express intent.
``MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY``
    OBO policy for the in-process terminator (Phase 2c).  Allowed values:

    * ``require_resolver`` (default, safe production) — on_behalf_of
      requests are rejected with ``ACL_DENIED`` unless the authority
      resolver returns a validated grant.  Downstream services see fully
      minted scope headers (``X-Auth-Max-Access-Level``,
      ``X-Auth-Workspace-Scope``, ``X-Auth-Audience-*``).
    * ``allow_partial`` — on_behalf_of requests proceed even when the
      resolver is absent or returns ``None``, with only the wire-derived
      header set.  Use during transition periods or for downstream
      services that don't enforce scope/audience.  Direct-mode requests
      are unaffected by this knob.
"""

from __future__ import annotations

import asyncio
import socket
import warnings
from collections.abc import Callable
from logging import Logger

from scitrera_app_framework import Variables, get_logger

from memorylayer_server.services._constants import EXT_AETHER_SERVICE_CONNECTION
from memorylayer_server.services._plugin_factory import make_service_plugin_base

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Provider selection (new + legacy alias).
MEMORYLAYER_AETHER_SERVICE_CONNECTION = "MEMORYLAYER_AETHER_SERVICE_CONNECTION"
DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION = "aether-service"

# Legacy env var names — checked as a fallback when the new names are unset.
_LEGACY_MEMORYLAYER_AETHER_AGENT_SERVICE = "MEMORYLAYER_AETHER_AGENT_SERVICE"
# The legacy provider value was 'aether-agent'; retained as an accepted alias
# below so existing deployments don't break the moment they upgrade.
_LEGACY_DEFAULT_AGENT_PROVIDER = "aether-agent"

AETHER_GATEWAY_ADDR = "AETHER_GATEWAY_ADDR"
DEFAULT_AETHER_GATEWAY_ADDR = "localhost:50051"

AETHER_WORKSPACE = "AETHER_WORKSPACE"
DEFAULT_AETHER_WORKSPACE = "_system"

AETHER_API_KEY = "AETHER_API_KEY"
AETHER_API_KEY_FILE = "AETHER_API_KEY_FILE"

AETHER_AUTH = "AETHER_AUTH"

# Service specifier (new) with legacy alias.
AETHER_SERVICE_SPECIFIER = "AETHER_SERVICE_SPECIFIER"
_LEGACY_AETHER_AGENT_SPECIFIER = "AETHER_AGENT_SPECIFIER"

# TLS configuration
AETHER_TLS_ENABLED = "AETHER_TLS_ENABLED"
AETHER_TLS_CA_CERT = "AETHER_TLS_CA_CERT"
AETHER_TLS_CLIENT_CERT = "AETHER_TLS_CLIENT_CERT"
AETHER_TLS_CLIENT_KEY = "AETHER_TLS_CLIENT_KEY"

# Hardcoded service implementation name.  Phase 1 keeps the implementation
# slot as ``memorylayer`` — the Aether identity slot itself is unchanged
# (only the principal *type* flips from Agent → Service).  Renaming the
# implementation would have churned every ACL and cert SAN; not worth it.
_SERVICE_IMPLEMENTATION = "memorylayer"

# Phase 3.5c: shared authority resolver knobs.
MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S = "MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S"
DEFAULT_MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S = 60
MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES = "MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES"
DEFAULT_MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES = 10_000

# Phase 2c: front-door selection + terminator OBO policy knobs.
MEMORYLAYER_AETHER_REST_FRONT_DOOR = "MEMORYLAYER_AETHER_REST_FRONT_DOOR"
DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR = "in_process"
_FRONT_DOOR_IN_PROCESS = "in_process"
_FRONT_DOOR_DISABLED = "disabled"
_FRONT_DOOR_SIDECAR = "sidecar"

MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY = "MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY"
DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY = "require_resolver"

# Path globs the in-process terminator accepts. Matches the Phase 2 plan;
# ``/`` (root metadata page) is intentionally NOT included so the metadata
# endpoint stays direct-only.
_DEFAULT_TERMINATOR_ALLOW_PATHS: tuple[str, ...] = (
    "/v1/*",
    "/healthz",
    "/v1/health/*",
    "/metrics",
)


def _default_specifier() -> str:
    """Return hostname or 'main' as the default service specifier."""
    try:
        return socket.gethostname() or "main"
    except Exception:
        return "main"


def _resolve_specifier(v: Variables) -> str:
    """Resolve the service specifier, honouring the legacy env var as a fallback.

    Reads ``AETHER_SERVICE_SPECIFIER`` first; if unset, falls back to the
    legacy ``AETHER_AGENT_SPECIFIER`` (with a deprecation warning).
    """
    new_value = v.environ(AETHER_SERVICE_SPECIFIER, None)
    if new_value:
        return new_value
    legacy_value = v.environ(_LEGACY_AETHER_AGENT_SPECIFIER, None)
    if legacy_value:
        warnings.warn(
            f"{_LEGACY_AETHER_AGENT_SPECIFIER} is deprecated; use {AETHER_SERVICE_SPECIFIER} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy_value
    return _default_specifier()


# ---------------------------------------------------------------------------
# Plugin base (generated via factory)
# ---------------------------------------------------------------------------

_AetherServiceConnectionPluginBase = make_service_plugin_base(
    ext_name=EXT_AETHER_SERVICE_CONNECTION,
    config_key=MEMORYLAYER_AETHER_SERVICE_CONNECTION,
    default_value=DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION,
    # No deps: tasks/rate-limit/tokens depend on us, not the other way round.
    dependencies=(),
)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AetherServiceConnection:
    """Unified Aether client owner for MemoryLayer.

    Connects to Aether as a Service principal (``sv::memorylayer::*``) and
    exposes the shared :class:`AsyncServiceClient` to dependent services:

    - Task assignment dispatch via ``on_task_assignment`` (registered by the
      Aether-backed task service).
    - KV operations for rate limiting (via :attr:`client`).
    - Token CRUD (via :attr:`client`).

    The previous ``on_message`` dispatch path (``recall``/``remember``/
    ``search``) has been removed — REST handles those operations now.
    """

    def __init__(
        self,
        v: Variables,
        *,
        gateway_addr: str = DEFAULT_AETHER_GATEWAY_ADDR,
        workspace: str = DEFAULT_AETHER_WORKSPACE,
        specifier: str = "main",
        credentials: dict | None = None,
        auth_mode: str | None = None,
        tls_enabled: bool = False,
        tls_ca_cert: str | None = None,
        tls_client_cert: str | None = None,
        tls_client_key: str | None = None,
        resolver_cache_ttl_s: int = DEFAULT_MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S,
        resolver_max_entries: int = DEFAULT_MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES,
        rest_front_door: str = DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR,
        terminator_obo_policy: str = DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY,
    ) -> None:
        self._v = v
        self._gateway_addr = gateway_addr
        # Service principals are workspace-less, but consumers (task service,
        # rate-limit KV) still scope their operations against a workspace.
        # We preserve the previous default of ``_system``.
        self._workspace = workspace
        self._specifier = specifier
        self._credentials = credentials
        self._auth_mode = auth_mode
        self._tls_enabled = tls_enabled
        self._tls_ca_cert = tls_ca_cert
        self._tls_client_cert = tls_client_cert
        self._tls_client_key = tls_client_key
        self._client = None
        self._task_assignment_handler: Callable | None = None
        # Phase 3.5c: lazy-constructed shared authority resolver. Materialised
        # on first :meth:`get_authority_resolver` call so connections that
        # never need OBO resolution do not allocate the cache.
        self._authority_resolver = None
        self._resolver_cache_ttl_s = resolver_cache_ttl_s
        self._resolver_max_entries = resolver_max_entries
        # Phase 2c: in-process REST-over-Aether terminator state.
        # ``_fastapi_app`` is populated via :meth:`attach_fastapi_app` once the
        # FastAPI app is fully built (routers + middleware registered). The
        # terminator is constructed and started only after BOTH the Aether
        # connection is up AND the app is attached, regardless of which
        # arrives first (Option A — event-based wiring).
        self._rest_front_door = rest_front_door
        self._terminator_obo_policy = terminator_obo_policy
        self._fastapi_app = None
        self._terminator = None
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized AetherServiceConnection (gateway=%s, workspace=%s, specifier=%s, tls=%s, rest_front_door=%s)",
            gateway_addr,
            workspace,
            specifier,
            tls_enabled,
            rest_front_door,
        )

    # ------------------------------------------------------------------
    # Client access for dependent services
    # ------------------------------------------------------------------

    @property
    def client(self):
        """Return the shared ``AsyncServiceClient`` instance, or ``None`` if not connected."""
        return self._client

    @property
    def workspace(self) -> str:
        """Return the configured Aether workspace (default scope for consumer services)."""
        return self._workspace

    # ------------------------------------------------------------------
    # Authority resolver (Phase 3.5c)
    # ------------------------------------------------------------------

    def get_authority_resolver(self):
        """Return a shared :class:`AsyncAuthorityResolver` bound to the Aether client.

        Lazy-constructed and cached on first call so subsequent callers
        receive the same instance — meaning a single in-process LRU cache
        is shared across the proxy-http terminator and any other consumer
        that wants to validate runtime authority grants.

        Phase 2c (terminator instantiation in MemoryLayer) is not yet
        landed; once it is, the wiring will look like::

            terminator = ProxyHttpTerminator(
                client=svc.client,
                handler=...,
                resolver=svc.get_authority_resolver(),
                obo_policy="require_resolver",
            )

        Raises:
            RuntimeError: when the connection has not been established yet
                (i.e., :meth:`connect` has not run).  Resolver construction
                requires a live ``AsyncServiceClient``.
        """
        if self._authority_resolver is not None:
            return self._authority_resolver
        if self._client is None:
            raise RuntimeError(
                "AetherServiceConnection.get_authority_resolver() called before connect(); resolver requires a live AsyncServiceClient."
            )
        # Delayed import: pulling the resolver in at module load would force
        # the whole aether SDK to initialise just to read env vars in
        # consumers that never use OBO resolution.
        from scitrera_aether_client.authority import AsyncAuthorityResolver

        self._authority_resolver = AsyncAuthorityResolver(
            self._client,
            max_ttl_s=self._resolver_cache_ttl_s,
            max_entries=self._resolver_max_entries,
        )
        self.logger.info(
            "Constructed shared AsyncAuthorityResolver (ttl=%ds, max_entries=%d)",
            self._resolver_cache_ttl_s,
            self._resolver_max_entries,
        )
        return self._authority_resolver

    # ------------------------------------------------------------------
    # REST-over-Aether terminator (Phase 2c)
    # ------------------------------------------------------------------

    async def attach_fastapi_app(self, app) -> None:
        """Attach the FastAPI app reference and (maybe) start the terminator.

        Called from the FastAPI lifespan startup hook AFTER the app is
        fully built (routers + middleware registered).  If the Aether
        connection is already up, the terminator is constructed and
        started here.  Otherwise the app reference is stashed and the
        terminator is started later when :meth:`connect` completes.
        """
        if self._fastapi_app is app:
            self.logger.debug("attach_fastapi_app: same app already attached; no-op")
            return
        self._fastapi_app = app
        self.logger.debug("FastAPI app attached to AetherServiceConnection")
        await self._maybe_start_terminator()

    async def _maybe_start_terminator(self) -> None:
        """Construct and start the in-process terminator if all preconditions are met.

        Preconditions:
        * Front-door selection is ``in_process`` (other values intentionally
          skip terminator registration — see module docstring).
        * The Aether client is connected.
        * The FastAPI app has been attached.
        * No terminator is already running.

        This method is invoked from both :meth:`connect` and
        :meth:`attach_fastapi_app`, whichever arrives last.  The check is
        idempotent — calling repeatedly without preconditions met is a
        no-op.
        """
        if self._terminator is not None:
            return
        if self._rest_front_door != _FRONT_DOOR_IN_PROCESS:
            # Disabled / sidecar — back-channel still works, just no
            # in-process REST terminator.
            return
        if self._client is None or self._fastapi_app is None:
            return

        # Delayed imports: pulling the SDK helpers at module load forces
        # the gRPC client surface to initialise before the framework is
        # ready (Aether SDK has its own grpc-side hooks in proxy.py).  We
        # only need them once the connection + app are both up.
        from scitrera_aether_client.proxy_terminator import ProxyHttpTerminator

        from .asgi_bridge import asgi_dispatch

        app = self._fastapi_app

        async def _handler(req):
            return await asgi_dispatch(app, req)

        try:
            terminator = ProxyHttpTerminator(
                client=self._client,
                handler=_handler,
                allow_paths=list(_DEFAULT_TERMINATOR_ALLOW_PATHS),
                header_mode="strict",
                resolver=self.get_authority_resolver(),
                obo_policy=self._terminator_obo_policy,
            )
            await terminator.start()
        except Exception:
            self.logger.error(
                "Failed to start ProxyHttpTerminator (REST-over-Aether front door will be unavailable)",
                exc_info=True,
            )
            return

        self._terminator = terminator
        self.logger.info(
            "ProxyHttpTerminator registered (allow_paths=%s, obo_policy=%s)",
            list(_DEFAULT_TERMINATOR_ALLOW_PATHS),
            self._terminator_obo_policy,
        )

    async def _stop_terminator(self) -> None:
        """Stop the in-process terminator, if one was started."""
        terminator = self._terminator
        if terminator is None:
            return
        self._terminator = None
        try:
            await terminator.stop()
        except Exception:
            self.logger.warning(
                "Error stopping ProxyHttpTerminator",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Task assignment delegation
    # ------------------------------------------------------------------

    def set_task_assignment_handler(self, handler: Callable) -> None:
        """Register a callback for task assignments received on the shared client.

        Called by AetherTaskService during its initialization to receive
        task assignments dispatched through this client.
        """
        self._task_assignment_handler = handler
        self.logger.debug("Task assignment handler registered")

    async def _on_task_assignment(self, assignment) -> None:
        """Dispatch task assignments to a background task.

        Runs the handler in a separate ``asyncio.Task`` so that long-running
        task handlers don't block the SDK's receive loop.  This prevents
        timeouts on concurrent operations (e.g., ``upsert_schedule``) that
        wait for responses on the same gRPC stream.
        """
        if self._task_assignment_handler is not None:
            asyncio.create_task(self._run_task_handler(assignment))
        else:
            self.logger.warning(
                "Received task assignment but no handler registered (task_type=%s)",
                getattr(assignment, "task_type", "<unknown>"),
            )

    async def _run_task_handler(self, assignment) -> None:
        """Execute the task assignment handler with error protection."""
        try:
            await self._task_assignment_handler(assignment)
        except Exception:
            self.logger.error(
                "Unhandled error in task assignment handler (task_type=%s)",
                getattr(assignment, "task_type", "<unknown>"),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to the Aether gateway as a Service principal."""
        if self._client is not None:
            self.logger.debug("AetherServiceConnection already connected")
            return

        try:
            from scitrera_aether_client import AsyncServiceClient

            tls_kwargs = {}
            if self._tls_enabled:
                tls_kwargs["tls_enabled"] = True
                if self._tls_ca_cert:
                    tls_kwargs["tls_root_cert_path"] = self._tls_ca_cert
                if self._tls_client_cert:
                    tls_kwargs["tls_client_cert_path"] = self._tls_client_cert
                if self._tls_client_key:
                    tls_kwargs["tls_client_key_path"] = self._tls_client_key

            client = AsyncServiceClient(
                implementation=_SERVICE_IMPLEMENTATION,
                specifier=self._specifier,
                credentials=self._credentials,
                **tls_kwargs,
            )
            # No on_message handler — REST handles the data-plane surface.
            client.on_task_assignment = self._on_task_assignment
            await client.connect(self._gateway_addr)
            self._client = client
            self.logger.info(
                "Connected to Aether gateway at %s as sv.%s.%s",
                self._gateway_addr,
                _SERVICE_IMPLEMENTATION,
                self._specifier,
            )
        except Exception:
            self._client = None
            self.logger.error(
                "Failed to connect to Aether gateway at %s",
                self._gateway_addr,
                exc_info=True,
            )
            raise

        # Phase 2c: if the FastAPI app has already been attached (typical
        # ordering in the lifespan: connect happens during async_ready
        # which fires before the lifespan's post-init attach call only if
        # async_ready races startup; in normal flow the app attach happens
        # last). Either way, this is a no-op when the app isn't yet
        # attached — :meth:`attach_fastapi_app` will pick up the slack.
        await self._maybe_start_terminator()

    async def disconnect(self) -> None:
        """Disconnect from the Aether gateway."""
        # Stop the terminator first so it stops accepting new requests
        # before we tear down the underlying client connection.
        await self._stop_terminator()
        if self._client is not None:
            try:
                await self._client.close()
                self.logger.info("Disconnected AetherServiceConnection from Aether gateway")
            except Exception:
                self.logger.error("Error closing AetherServiceConnection client", exc_info=True)
            finally:
                self._client = None
                # Drop the shared resolver — its cached entries are bound to
                # the old client's identity/connection.  A subsequent connect
                # will yield a fresh resolver on next get_authority_resolver().
                self._authority_resolver = None

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the Aether client is active."""
        return self._client is not None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


def _read_key_file(path: str, logger: Logger) -> str | None:
    """Read an API key from a file, stripping whitespace."""
    try:
        with open(path) as f:
            key = f.read().strip()
        if key:
            logger.info("Loaded API key from file: %s", path)
            return key
        logger.warning("API key file is empty: %s", path)
    except FileNotFoundError:
        logger.warning("API key file not found: %s", path)
    except Exception:
        logger.error("Failed to read API key file: %s", path, exc_info=True)
    return None


class AetherServiceConnectionPlugin(_AetherServiceConnectionPluginBase):
    """Plugin that creates and manages an :class:`AetherServiceConnection` instance.

    Enabled when ``MEMORYLAYER_AETHER_SERVICE_CONNECTION=aether-service``.
    The legacy provider name ``aether-agent`` is also accepted (with a
    deprecation warning) so existing deployments don't break on upgrade.

    Lifecycle:
        ``initialize`` -- constructs the service (no I/O).
        ``async_ready`` -- validates config, connects to the Aether gateway (fail-fast).
        ``async_stopping`` -- disconnects from the gateway.
    """

    PROVIDER_NAME = "aether-service"

    def is_enabled(self, v: Variables) -> bool:
        # Honour the new provider key first.
        if super().is_enabled(v):
            return True
        # Legacy compatibility: ``MEMORYLAYER_AETHER_AGENT_SERVICE=aether-agent``
        # used to enable this service.  Continue to honour it, but warn.
        legacy = v.environ(_LEGACY_MEMORYLAYER_AETHER_AGENT_SERVICE, None)
        if legacy and legacy.lower() == _LEGACY_DEFAULT_AGENT_PROVIDER:
            warnings.warn(
                f"{_LEGACY_MEMORYLAYER_AETHER_AGENT_SERVICE}={_LEGACY_DEFAULT_AGENT_PROVIDER} "
                f"is deprecated; set {MEMORYLAYER_AETHER_SERVICE_CONNECTION}="
                f"{DEFAULT_MEMORYLAYER_AETHER_SERVICE_CONNECTION} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return True
        return False

    def initialize(self, v: Variables, logger: Logger) -> AetherServiceConnection:
        """Construct the service from environment config (no I/O)."""
        gateway_addr = v.environ(AETHER_GATEWAY_ADDR, DEFAULT_AETHER_GATEWAY_ADDR)
        workspace = v.environ(AETHER_WORKSPACE, DEFAULT_AETHER_WORKSPACE)
        specifier = _resolve_specifier(v)
        auth_mode = v.environ(AETHER_AUTH, None)

        # Resolve API key: env var directly, or read from file
        api_key = v.environ(AETHER_API_KEY, None)
        if not api_key:
            api_key_file = v.environ(AETHER_API_KEY_FILE, None)
            if api_key_file:
                api_key = _read_key_file(api_key_file, logger)

        if auth_mode and auth_mode.lower() == "none":
            credentials = None
        else:
            credentials = {"api_key": api_key} if api_key else None

        # TLS configuration
        tls_enabled = v.environ(AETHER_TLS_ENABLED, "false").lower() in ("true", "1", "yes")
        tls_ca_cert = v.environ(AETHER_TLS_CA_CERT, None) if tls_enabled else None
        tls_client_cert = v.environ(AETHER_TLS_CLIENT_CERT, None) if tls_enabled else None
        tls_client_key = v.environ(AETHER_TLS_CLIENT_KEY, None) if tls_enabled else None

        # Phase 3.5c: authority resolver knobs (used when the proxy-http
        # terminator is wired up later by Phase 2c).
        resolver_cache_ttl_s = int(
            v.environ(
                MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S,
                DEFAULT_MEMORYLAYER_AETHER_RESOLVER_CACHE_TTL_S,
            )
        )
        resolver_max_entries = int(
            v.environ(
                MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES,
                DEFAULT_MEMORYLAYER_AETHER_RESOLVER_MAX_ENTRIES,
            )
        )

        # Phase 2c: front-door + terminator OBO policy knobs.  Validate
        # values up-front so misconfiguration surfaces at boot instead of
        # at first request.
        rest_front_door = (
            v.environ(
                MEMORYLAYER_AETHER_REST_FRONT_DOOR,
                DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR,
            )
            or DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR
        ).lower()
        if rest_front_door not in (
            _FRONT_DOOR_IN_PROCESS,
            _FRONT_DOOR_DISABLED,
            _FRONT_DOOR_SIDECAR,
        ):
            logger.warning(
                "Unknown %s=%s; falling back to %s",
                MEMORYLAYER_AETHER_REST_FRONT_DOOR,
                rest_front_door,
                DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR,
            )
            rest_front_door = DEFAULT_MEMORYLAYER_AETHER_REST_FRONT_DOOR

        terminator_obo_policy = (
            v.environ(
                MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY,
                DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY,
            )
            or DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY
        )
        if terminator_obo_policy not in ("require_resolver", "allow_partial"):
            logger.warning(
                "Unknown %s=%s; falling back to %s",
                MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY,
                terminator_obo_policy,
                DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY,
            )
            terminator_obo_policy = DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_OBO_POLICY

        return AetherServiceConnection(
            v,
            gateway_addr=gateway_addr,
            workspace=workspace,
            specifier=specifier,
            credentials=credentials,
            auth_mode=auth_mode,
            tls_enabled=tls_enabled,
            tls_ca_cert=tls_ca_cert,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            resolver_cache_ttl_s=resolver_cache_ttl_s,
            resolver_max_entries=resolver_max_entries,
            rest_front_door=rest_front_door,
            terminator_obo_policy=terminator_obo_policy,
        )

    async def async_ready(self, v: Variables, logger: Logger, value: AetherServiceConnection) -> None:
        """Validate configuration, connect to the gateway, attach the FastAPI app.

        Fail-fast behaviour:
        - If ``AETHER_AUTH`` is not ``none`` and no API key is resolvable,
          raises with a clear error.
        - If connection fails, the exception propagates (fail-fast).
        - If ``AETHER_AUTH=none``, skips key requirement, logs warning.

        Phase 2c: after the connection is up, attach the FastAPI app from
        the framework's extension registry.  The FastAPI plugin's
        ``initialize()`` returns the app; routes/CORS/middleware plugins
        all declare ``EXT_FASTAPI_SERVER`` as a dependency so by the time
        any plugin's ``async_ready`` runs, the app is fully built (every
        sync ``initialize`` completes before ``async_plugins_ready``
        starts walking the startup order).  Failure to find the FastAPI
        app is non-fatal — the back-channel (tasks/KV/tokens) still
        works; only the in-process REST terminator is skipped.
        """
        auth_mode = value._auth_mode

        if auth_mode and auth_mode.lower() == "none":
            logger.warning("AETHER_AUTH=none: skipping API key requirement. This is intended for local development only.")
        else:
            # Check if credentials were resolved during initialize
            # (from AETHER_API_KEY or AETHER_API_KEY_FILE)
            if not value._credentials:
                raise RuntimeError(
                    "AETHER_API_KEY is required when Aether service connection is enabled. "
                    "Set AETHER_API_KEY, AETHER_API_KEY_FILE, or use AETHER_AUTH=none for local development."
                )

        await value.connect()
        logger.info(
            "Aether service connection ready (gateway=%s, identity=sv.%s.%s)",
            value._gateway_addr,
            _SERVICE_IMPLEMENTATION,
            value._specifier,
        )

        # Phase 2c: attach the FastAPI app so the in-process terminator can
        # spin up.  Delayed import to avoid a hard cyclic dependency
        # between services (this module) and lifecycle (the FastAPI app).
        try:
            from scitrera_app_framework import get_extension as _get_extension

            from memorylayer_server.lifecycle.fastapi import EXT_FASTAPI_SERVER

            fastapi_app = _get_extension(EXT_FASTAPI_SERVER, v)
        except Exception:
            logger.warning(
                "FastAPI app not available; skipping in-process terminator wiring (REST-over-Aether front door will be unavailable)",
                exc_info=True,
            )
            return

        if fastapi_app is None:
            logger.debug("FastAPI app extension is None; skipping in-process terminator wiring")
            return

        try:
            await value.attach_fastapi_app(fastapi_app)
        except Exception:
            logger.warning(
                "Failed to attach FastAPI app to AetherServiceConnection",
                exc_info=True,
            )

    async def async_stopping(self, v: Variables, logger: Logger, value: AetherServiceConnection) -> None:
        """Disconnect from the Aether gateway."""
        await value.disconnect()
