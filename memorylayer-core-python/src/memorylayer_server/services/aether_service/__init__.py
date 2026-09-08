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
``MEMORYLAYER_AETHER_LIVENESS_ENABLED``
    Enable the connection liveness watchdog that detects a silently dropped
    gateway connection (hard pod kill) and forces the SDK to reconnect.
    Default: ``true``.  The Aether SDK's ``auto_reconnect`` only fires on a
    *graceful* disconnect; without gRPC keepalive (which the SDK does not
    expose) a silent TCP drop is otherwise never noticed.
``MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S``
    Seconds between liveness probes (default: ``30``).
``MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S``
    Per-probe timeout in seconds (default: ``10``).
``MEMORYLAYER_AETHER_LIVENESS_FAILURES``
    Consecutive probe failures before forcing a reconnect (default: ``2``).
"""

from __future__ import annotations

import asyncio
import os
import socket
import warnings
from collections.abc import Callable
from logging import Logger

from scitrera_app_framework import Variables, ext_parse_bool, get_logger

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

# ProxyHttpTerminator header mode. "strict" makes the terminator re-mint the
# X-Auth-* trusted header set itself; "passthrough" makes it trust the headers
# the gateway already minted (the gateway is now the single minting point via
# identityheaders.MintIntoMap on every ProxyHttpRequest). Default stays
# "strict" for backward compatibility; deployments fronted by a minting gateway
# set "passthrough" so direct-service-call X-Auth-Workspace-Access is honored.
MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE = "MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE"
DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE = "strict"
_TERMINATOR_HEADER_MODES = ("strict", "passthrough")

# Connection liveness watchdog (silent-drop detection).
#
# The Aether SDK (``scitrera_aether_client``) supports ``auto_reconnect``
# (exponential backoff + GRACEFUL_DISCONNECT handling) but builds its gRPC
# channel WITHOUT keepalive options and exposes no channel-options surface to
# pass them in.  A *graceful* gateway disconnect is detected (the listen loop
# sees a stream error and reconnect fires), but a *silently dropped* TCP
# connection — e.g. a hard gateway pod kill — produces no stream error, so the
# SDK's receive loop blocks forever believing it is still connected and
# reconnect never triggers.  Symptom: after a gateway restart the service
# disappears from the gateway's connection list indefinitely, with no error
# logs ("it doesn't realize it got disconnected"), and only a pod restart
# recovers it.
#
# Since the SDK is an external package (not vendored in this repo) and offers
# no keepalive knob, we detect the dead connection ourselves with a periodic
# application-level liveness probe (a cheap ``kv_get`` round-trip).  When the
# probe fails repeatedly while the SDK still believes it is connected, we close
# the dead channel — which surfaces a stream error in the SDK's listen loop and
# lets its native ``auto_reconnect`` machinery re-establish the connection.
MEMORYLAYER_AETHER_LIVENESS_ENABLED = "MEMORYLAYER_AETHER_LIVENESS_ENABLED"
DEFAULT_MEMORYLAYER_AETHER_LIVENESS_ENABLED = True
MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S = "MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S"
DEFAULT_MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S = 30.0
MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S = "MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S"
DEFAULT_MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S = 10.0
MEMORYLAYER_AETHER_LIVENESS_FAILURES = "MEMORYLAYER_AETHER_LIVENESS_FAILURES"
DEFAULT_MEMORYLAYER_AETHER_LIVENESS_FAILURES = 2

# Max concurrent POOL task handlers (document ingestion/render/embed, decay,
# doc_verify, …) run on this connection. A doc_verify sweep can resume many docs
# at once; without a cap every assignment spawns an unbounded ``asyncio.Task`` so
# a burst runs ALL of them concurrently — pressuring the thread pool (the
# ``to_thread`` PDF rasterize is CPU-bound), the DB, and the embed backend, and
# starving the event loop / HTTP ``/livez`` probe. Most pool tasks are light
# (decay, doc_verify/kb_refresh sweeps); only document render/embed is heavy and
# is itself memory-bounded (batched per MEMORYLAYER_RENDER_BATCH_PAGES), so the
# cap is generous — it exists to bound a runaway burst, not to serialize work.
# ``<= 0`` disables the cap (unbounded — the legacy behavior).
MEMORYLAYER_TASK_CONCURRENCY = "MEMORYLAYER_TASK_CONCURRENCY"
DEFAULT_MEMORYLAYER_TASK_CONCURRENCY = 16

# Task-handler concurrency LANES.
#
# A single shared cap is not enough, because ``asyncio.Semaphore`` is strictly
# FIFO: whoever queues first runs first, regardless of what the work is. Fact
# decomposition fans out hard — one decomposed page yields ~35-40 facts, and
# each fact schedules its own ``generate_tiers`` + ``auto_enrich`` — so a
# document ingest routinely parks >1000 enrichment tasks in the queue. Under one
# cap the NEXT document_transcribe lands behind all of them and the document
# pipeline stalls for as long as the backlog takes to burn down, even though the
# backlog is not what anyone is waiting on.
#
# Splitting into lanes gives each class its own reserved slots, so document
# progress is never gated on enrichment backlog. Lanes are independent: total
# possible concurrency is the SUM of the caps, which is the point — the document
# lane's slots must not be consumable by fan-out work.
TASK_LANE_DOCUMENT = "document"
TASK_LANE_FANOUT = "fanout"
TASK_LANE_DEFAULT = "default"

# Memory fan-out: enqueued per-fact and unbounded in the document's page count,
# so this is the lane that floods. It keeps the historical cap.
MEMORYLAYER_TASK_CONCURRENCY_FANOUT = "MEMORYLAYER_TASK_CONCURRENCY_FANOUT"
DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_FANOUT = 16
# Document pipeline: bounded per document and latency-visible to callers, so it
# gets dedicated slots rather than competing for the shared pool.
MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT = "MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT"
DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT = 8

#: Task types whose volume scales with fact count rather than document count.
_FANOUT_TASK_TYPES = frozenset({
    "auto_enrich",
    "generate_tiers",
    "decompose_facts",
    "reindex_memory",
    "kb_update",
})

#: Prefixes of the document ingestion pipeline (document_render/_transcribe/
#: _embed/_finalize, doc_added, doc_verify).
_DOCUMENT_TASK_PREFIXES = ("document_", "doc_")


def resolve_task_lane(task_type: str | None) -> str:
    """Map a task type to its concurrency lane.

    Unknown types land in the default lane deliberately: a new task type gets
    ordinary shared capacity rather than silently borrowing the document lane's
    reserved slots.
    """
    if not task_type:
        return TASK_LANE_DEFAULT
    # Task types arrive NAMESPACED on the wire ("memorylayer-task.auto_enrich")
    # while the tables above hold bare names. Match on the last dot-segment so
    # both spellings route identically. Matching the raw value alone made every
    # lane inert in production: nothing matched, so all work -- document phases
    # and fan-out alike -- fell through to the default lane and starved each
    # other exactly as it had before lanes existed.
    name = task_type.rsplit(".", 1)[-1]
    if name in _FANOUT_TASK_TYPES:
        return TASK_LANE_FANOUT
    if name.startswith(_DOCUMENT_TASK_PREFIXES):
        return TASK_LANE_DOCUMENT
    return TASK_LANE_DEFAULT

# Grace (seconds) to let in-flight task handlers finish on disconnect before they
# are cancelled, and the timeout for the Aether client ``close()`` (which can
# block on a silently-dead channel). Together these BOUND shutdown so a handler
# awaiting a dead channel can't hang the pod past the k8s termination grace
# (-> SIGKILL/exit 137). Ingestion is resumable (doc_verify re-sweeps + ingestion
# is idempotent), so cancelling an in-flight render mid-shutdown is safe.
MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S = "MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S"
DEFAULT_MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S = 8.0
MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S = "MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S"
DEFAULT_MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S = 5.0

# Periodic INFO log of the in-flight POOL task count (+ per-type breakdown, and
# running-vs-waiting against the concurrency cap) WHILE tasks are running —
# observability for long ingestion bursts where individual handlers log little.
# Emits only while there ARE live tasks, plus one line when the last drains, so
# an idle worker stays quiet. ``<= 0`` disables.
MEMORYLAYER_TASK_MONITOR_INTERVAL_S = "MEMORYLAYER_TASK_MONITOR_INTERVAL_S"
DEFAULT_MEMORYLAYER_TASK_MONITOR_INTERVAL_S = 30.0

# Well-known key the liveness probe reads. Its value is irrelevant — a hit OR
# a miss both prove the round-trip works; only a timeout/error signals trouble.
_LIVENESS_PROBE_KEY = "__memorylayer_aether_liveness__"

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
        liveness_enabled: bool = DEFAULT_MEMORYLAYER_AETHER_LIVENESS_ENABLED,
        liveness_interval_s: float = DEFAULT_MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S,
        liveness_timeout_s: float = DEFAULT_MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S,
        liveness_failures: int = DEFAULT_MEMORYLAYER_AETHER_LIVENESS_FAILURES,
        task_concurrency: int = DEFAULT_MEMORYLAYER_TASK_CONCURRENCY,
        document_task_concurrency: int = DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT,
        fanout_task_concurrency: int = DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_FANOUT,
        shutdown_drain_timeout_s: float = DEFAULT_MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S,
        shutdown_close_timeout_s: float = DEFAULT_MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S,
        task_monitor_interval_s: float = DEFAULT_MEMORYLAYER_TASK_MONITOR_INTERVAL_S,
        consumes_pool_tasks: bool = True,
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
        # When False (in-process worker disabled → serve-only), the Aether
        # connection declares no_pool_consumer so the gateway never routes POOL
        # task assignments to this instance (they go to dedicated workers).
        self._consumes_pool_tasks = consumes_pool_tasks
        self._client = None
        self._task_assignment_handler: Callable | None = None
        # POOL task-handler concurrency cap + in-flight tracking. The semaphore
        # bounds concurrent handlers so a burst of assignments (e.g. a doc_verify
        # sweep) can't run all at once; the set lets disconnect() drain/cancel
        # stragglers within a bounded grace. ``task_concurrency <= 0`` => no cap.
        # (asyncio.Semaphore does not bind a loop at construction on 3.10+, so
        # building it here — possibly before the loop runs — is safe.)
        self._task_concurrency = task_concurrency
        self._task_semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(task_concurrency) if task_concurrency and task_concurrency > 0 else None
        )
        # Per-lane caps (see TASK_LANE_* above). The default lane reuses
        # ``_task_semaphore`` so the existing knob keeps its meaning and a
        # ``task_concurrency <= 0`` still disables capping for that lane.
        self._lane_concurrency = {
            TASK_LANE_DOCUMENT: document_task_concurrency,
            TASK_LANE_FANOUT: fanout_task_concurrency,
            TASK_LANE_DEFAULT: task_concurrency,
        }
        self._lane_semaphores: dict[str, asyncio.Semaphore | None] = {
            TASK_LANE_DOCUMENT: (
                asyncio.Semaphore(document_task_concurrency)
                if document_task_concurrency and document_task_concurrency > 0
                else None
            ),
            TASK_LANE_FANOUT: (
                asyncio.Semaphore(fanout_task_concurrency)
                if fanout_task_concurrency and fanout_task_concurrency > 0
                else None
            ),
            TASK_LANE_DEFAULT: self._task_semaphore,
        }
        self._inflight_tasks: set[asyncio.Task] = set()
        self._shutdown_drain_timeout_s = shutdown_drain_timeout_s
        self._shutdown_close_timeout_s = shutdown_close_timeout_s
        # Periodic in-flight task-count logger (started on connect, stopped on
        # disconnect). Handler tasks are named by their task_type so the log can
        # break the count down by type.
        self._task_monitor_interval_s = task_monitor_interval_s
        self._task_monitor_task: asyncio.Task | None = None
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
        # Connection liveness watchdog state (silent-drop detection — see the
        # module-level config constants for the full rationale).
        self._liveness_enabled = liveness_enabled
        self._liveness_interval_s = liveness_interval_s
        self._liveness_timeout_s = liveness_timeout_s
        self._liveness_failures = liveness_failures
        self._liveness_task: asyncio.Task | None = None
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

        # Per-deployment tenant for the terminator to stamp as X-Auth-Tenant-ID on
        # every minted request. MemoryLayer runs one deployment per tenant and the
        # platform framework injects SCITRERA_TENANT (= the tenant id); the cert's
        # specifier is NOT usable here (it is overridden per-pod for unique connection
        # identity). Without this, the Python terminator never carries a tenant and the
        # fail-closed AetherAuthenticationService rejects EVERY request with
        # "Missing X-Auth-Tenant-ID" (direct-aether path has no Go sidecar to mint it).
        tenant_id = self._v.environ("SCITRERA_TENANT", None)
        if not tenant_id:
            self.logger.warning(
                "SCITRERA_TENANT unset: ProxyHttpTerminator will not stamp "
                "X-Auth-Tenant-ID; fail-closed REST auth will reject requests"
            )

        # Header mode: "strict" re-mints the X-Auth-* set here; "passthrough"
        # trusts the gateway-minted headers (the gateway is the single minting
        # point via identityheaders.MintIntoMap). Default "strict" for
        # backward compatibility; invalid values warn and fall back to strict.
        header_mode = self._v.environ(
            MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE,
            DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE,
        )
        if header_mode not in _TERMINATOR_HEADER_MODES:
            self.logger.warning(
                "Invalid %s=%r (expected one of %s); falling back to %r",
                MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE,
                header_mode,
                _TERMINATOR_HEADER_MODES,
                DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE,
            )
            header_mode = DEFAULT_MEMORYLAYER_AETHER_TERMINATOR_HEADER_MODE

        try:
            terminator = ProxyHttpTerminator(
                client=self._client,
                handler=_handler,
                allow_paths=list(_DEFAULT_TERMINATOR_ALLOW_PATHS),
                header_mode=header_mode,
                resolver=self.get_authority_resolver(),
                obo_policy=self._terminator_obo_policy,
                tenant_id=tenant_id,
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
            # Track the task so disconnect() can drain/cancel it on shutdown
            # (otherwise a handler blocked on a dead channel hangs termination).
            # Name it by task_type so the task monitor can break the count down.
            task = asyncio.create_task(
                self._run_task_handler(assignment),
                name=str(getattr(assignment, "task_type", "task")),
            )
            self._inflight_tasks.add(task)
            task.add_done_callback(self._inflight_tasks.discard)
        else:
            self.logger.warning(
                "Received task assignment but no handler registered (task_type=%s)",
                getattr(assignment, "task_type", "<unknown>"),
            )

    async def _run_task_handler(self, assignment) -> None:
        """Execute the task assignment handler with error protection.

        Bounded by the semaphore of the task's LANE (see ``resolve_task_lane``)
        so a burst of pool assignments — e.g. a doc_verify sweep resuming many
        ``document_render`` jobs — doesn't run every handler at once and starve
        the event loop / thread pool.

        Lanes are what keep that bound from becoming head-of-line blocking: with
        one shared FIFO semaphore, thousands of queued enrichment tasks delay
        every later document task behind them.
        """
        lane = resolve_task_lane(getattr(assignment, "task_type", None))
        semaphore = self._lane_semaphores.get(lane)
        try:
            if semaphore is not None:
                async with semaphore:
                    await self._task_assignment_handler(assignment)
            else:
                await self._task_assignment_handler(assignment)
        except asyncio.CancelledError:
            # Cancelled by the shutdown drain — propagate quietly (not an error).
            raise
        except Exception:
            self.logger.error(
                "Unhandled error in task assignment handler (task_type=%s)",
                getattr(assignment, "task_type", "<unknown>"),
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Connection liveness watchdog (silent-drop detection)
    # ------------------------------------------------------------------

    def _start_liveness_watchdog(self) -> None:
        """Start the background liveness watchdog (idempotent)."""
        if not self._liveness_enabled:
            return
        if self._liveness_task is not None and not self._liveness_task.done():
            return
        self._liveness_task = asyncio.create_task(self._liveness_loop())
        self.logger.info(
            "Aether liveness watchdog started (interval=%.1fs, timeout=%.1fs, failures=%d)",
            self._liveness_interval_s,
            self._liveness_timeout_s,
            self._liveness_failures,
        )

    async def _stop_liveness_watchdog(self) -> None:
        """Stop the background liveness watchdog, if running."""
        task = self._liveness_task
        self._liveness_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.warning("Error stopping Aether liveness watchdog", exc_info=True)

    # ------------------------------------------------------------------
    # In-flight POOL task monitor (periodic count logging)
    # ------------------------------------------------------------------

    def _start_task_monitor(self) -> None:
        """Start the periodic in-flight task-count logger (idempotent)."""
        if self._task_monitor_interval_s <= 0:
            return
        if self._task_monitor_task is not None and not self._task_monitor_task.done():
            return
        self._task_monitor_task = asyncio.create_task(self._task_monitor_loop())
        self.logger.info(
            "POOL task monitor started (interval=%.1fs)", self._task_monitor_interval_s,
        )

    async def _stop_task_monitor(self) -> None:
        """Stop the periodic task-count logger, if running."""
        task = self._task_monitor_task
        self._task_monitor_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.warning("Error stopping POOL task monitor", exc_info=True)

    async def _task_monitor_loop(self) -> None:
        """Log the in-flight POOL task count at INFO while tasks are running.

        Reports the total live handler count, a per-``task_type`` breakdown, and
        (when a concurrency cap is set) an approximate running-vs-waiting split
        (``running ≈ min(n, cap)``, the rest parked on the semaphore). Silent
        while idle; emits a single line when the last task drains.
        """
        from collections import Counter

        was_active = False
        while True:
            try:
                await asyncio.sleep(self._task_monitor_interval_s)
            except asyncio.CancelledError:
                return
            live = [t for t in self._inflight_tasks if not t.done()]
            n = len(live)
            if n > 0:
                by_type = Counter(t.get_name() for t in live)
                breakdown = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
                # Running-vs-waiting is per LANE: each lane has its own cap, so a
                # single global cap would misreport (and hide exactly the
                # starvation lanes exist to prevent — a saturated fan-out lane
                # next to an idle document lane).
                by_lane = Counter(resolve_task_lane(t.get_name()) for t in live)
                lane_parts = []
                for lane, lane_n in sorted(by_lane.items()):
                    lane_cap = self._lane_concurrency.get(lane) or 0
                    if lane_cap > 0:
                        lane_running = min(lane_n, lane_cap)
                        lane_parts.append(
                            "%s ~%d/%d running, %d waiting"
                            % (lane, lane_running, lane_cap, lane_n - lane_running)
                        )
                    else:
                        lane_parts.append("%s %d unbounded" % (lane, lane_n))
                self.logger.info(
                    "POOL tasks in flight: %d (%s) [%s]",
                    n, "; ".join(lane_parts), breakdown,
                )
                was_active = True
            elif was_active:
                self.logger.info("POOL tasks in flight: 0 (all drained)")
                was_active = False

    def _client_believes_connected(self) -> bool:
        """Return ``True`` when the SDK thinks the stream is live and is not reconnecting.

        We only probe / force a reconnect when the SDK believes it is connected
        (``_connection_confirmed``) and is not already mid-reconnect
        (``_reconnecting``).  Those internals are best-effort — if a future SDK
        version drops them we fall back to "assume connected" so the probe still
        runs (a dead connection is the failure mode we must catch).
        """
        client = self._client
        if client is None:
            return False
        confirmed = getattr(client, "_connection_confirmed", True)
        reconnecting = getattr(client, "_reconnecting", False)
        return bool(confirmed) and not bool(reconnecting)

    async def _liveness_probe(self) -> bool:
        """Issue one cheap round-trip and return ``True`` iff the gateway answered.

        ``kv_get`` returns a ``KVResponse`` on success (hit OR miss) and ``None``
        on timeout, so a non-``None`` result proves the bidirectional stream is
        alive.  A silently dropped connection yields ``None`` (the request is
        queued but never answered) — the signal we use to detect the dead link.
        """
        client = self._client
        if client is None:
            return False
        try:
            resp = await asyncio.wait_for(
                client.kv_get(_LIVENESS_PROBE_KEY, scope="global", timeout=self._liveness_timeout_s),
                # Hard ceiling in case the SDK's own timeout is bypassed on a
                # wedged stream; keep it just above the SDK timeout.
                timeout=self._liveness_timeout_s + 5.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Any RPC error (e.g. channel already failing) counts as a failed
            # probe; the watchdog will force a reconnect after enough failures.
            return False
        return resp is not None

    async def _force_reconnect(self) -> None:
        """Close the dead channel so the SDK's auto_reconnect machinery fires.

        Closing the underlying gRPC channel surfaces a stream error in the
        SDK's listen loop, which (with ``auto_reconnect=True`` — the SDK
        default) triggers ``_attempt_reconnect``.  We deliberately do NOT call
        ``client.close()`` here (that sets the SDK's stop event and would tear
        the client down for good); we only nudge the transport so the SDK
        re-establishes it.
        """
        client = self._client
        if client is None:
            return
        channel = getattr(client, "channel", None)
        if channel is None:
            return
        try:
            await channel.close()
            self.logger.warning(
                "Aether liveness watchdog: closed dead channel to force reconnect "
                "(silent-drop detected; SDK auto_reconnect will re-establish)"
            )
        except Exception:
            self.logger.warning(
                "Aether liveness watchdog: error closing dead channel", exc_info=True
            )

    async def _liveness_loop(self) -> None:
        """Periodically probe the connection and force a reconnect when it is dead."""
        consecutive_failures = 0
        while True:
            try:
                await asyncio.sleep(self._liveness_interval_s)
            except asyncio.CancelledError:
                return

            if not self._client_believes_connected():
                # Not connected yet, or the SDK is already reconnecting — let it
                # work and reset our failure counter.
                consecutive_failures = 0
                continue

            try:
                ok = await self._liveness_probe()
            except asyncio.CancelledError:
                return

            if ok:
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            self.logger.warning(
                "Aether liveness probe failed (%d/%d)",
                consecutive_failures,
                self._liveness_failures,
            )
            if consecutive_failures >= self._liveness_failures:
                await self._force_reconnect()
                # Reset so we give the SDK time to reconnect before probing
                # again; the next loop also skips while ``_reconnecting``.
                consecutive_failures = 0

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
                consumes_pool_tasks=self._consumes_pool_tasks,
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

        # Start the liveness watchdog so a silently dropped connection (hard
        # gateway pod kill) is detected and the SDK's auto_reconnect is fired.
        self._start_liveness_watchdog()
        # Periodic in-flight task-count logging (observability during bursts).
        self._start_task_monitor()

    async def disconnect(self) -> None:
        """Disconnect from the Aether gateway."""
        # Stop the liveness watchdog first so it does not try to probe (or
        # force-reconnect) a connection we are deliberately tearing down.
        await self._stop_liveness_watchdog()
        await self._stop_task_monitor()
        # Stop the terminator next so it stops accepting new requests
        # before we tear down the underlying client connection.
        await self._stop_terminator()
        # Drain in-flight POOL task handlers within a bounded grace, then cancel
        # the stragglers, so a handler blocked on a (silently) dead channel can't
        # hang termination past the k8s grace (-> SIGKILL/exit 137).
        await self._drain_inflight_tasks()
        if self._client is not None:
            try:
                # ``close()`` can block on a dead channel — bound it so shutdown
                # stays within the termination grace.
                await asyncio.wait_for(self._client.close(), timeout=self._shutdown_close_timeout_s)
                self.logger.info("Disconnected AetherServiceConnection from Aether gateway")
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Aether client close() timed out after %.1fs; abandoning it",
                    self._shutdown_close_timeout_s,
                )
            except Exception:
                self.logger.error("Error closing AetherServiceConnection client", exc_info=True)
            finally:
                self._client = None
                # Drop the shared resolver — its cached entries are bound to
                # the old client's identity/connection.  A subsequent connect
                # will yield a fresh resolver on next get_authority_resolver().
                self._authority_resolver = None

    async def _drain_inflight_tasks(self) -> None:
        """Await in-flight task handlers up to the drain grace, then cancel the rest.

        Called on disconnect. Bounds how long shutdown waits on running handlers
        so a handler blocked on a dead channel can't stall termination; the
        cancelled work is safe to drop (ingestion is resumable + idempotent).
        """
        tasks = [t for t in self._inflight_tasks if not t.done()]
        if not tasks:
            return
        self.logger.info(
            "Draining %d in-flight task handler(s) on disconnect (grace=%.1fs)",
            len(tasks),
            self._shutdown_drain_timeout_s,
        )
        _done, pending = await asyncio.wait(tasks, timeout=self._shutdown_drain_timeout_s)
        if pending:
            self.logger.warning(
                "Cancelling %d in-flight task handler(s) that did not drain in %.1fs",
                len(pending),
                self._shutdown_drain_timeout_s,
            )
            for t in pending:
                t.cancel()
            # Let cancellation propagate; swallow the resulting CancelledErrors.
            await asyncio.gather(*pending, return_exceptions=True)

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


# Aether variables that indicate the operator actually intends to use Aether.
# Deliberately excludes anything with a usable default (e.g. AETHER_WORKSPACE):
# presence of those says nothing about intent.
_AETHER_INTENT_ENV_KEYS = (
    AETHER_GATEWAY_ADDR,
    AETHER_API_KEY,
    AETHER_API_KEY_FILE,
    AETHER_AUTH,
    AETHER_SERVICE_SPECIFIER,
    AETHER_TLS_ENABLED,
    AETHER_TLS_CA_CERT,
    AETHER_TLS_CLIENT_CERT,
    AETHER_TLS_CLIENT_KEY,
)


def _any_aether_env_configured() -> bool:
    """True when any Aether variable is set in the environment.

    Read from ``os.environ`` rather than ``Variables`` on purpose: this must
    distinguish "operator set it" from "a default was applied", and the defaults
    are exactly what would make the check always true.
    """
    return any(os.environ.get(key) for key in _AETHER_INTENT_ENV_KEYS)


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

        # Connection liveness watchdog knobs (silent-drop detection).
        # Boolean parsed with the same idiom as AETHER_TLS_ENABLED above; the
        # default (enabled) applies only when the env var is unset.
        liveness_enabled_raw = v.environ(MEMORYLAYER_AETHER_LIVENESS_ENABLED, None)
        if liveness_enabled_raw is None:
            liveness_enabled = DEFAULT_MEMORYLAYER_AETHER_LIVENESS_ENABLED
        else:
            liveness_enabled = liveness_enabled_raw.lower() in ("true", "1", "yes")
        liveness_interval_s = float(
            v.environ(
                MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S,
                DEFAULT_MEMORYLAYER_AETHER_LIVENESS_INTERVAL_S,
            )
        )
        liveness_timeout_s = float(
            v.environ(
                MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S,
                DEFAULT_MEMORYLAYER_AETHER_LIVENESS_TIMEOUT_S,
            )
        )
        liveness_failures = int(
            v.environ(
                MEMORYLAYER_AETHER_LIVENESS_FAILURES,
                DEFAULT_MEMORYLAYER_AETHER_LIVENESS_FAILURES,
            )
        )

        # POOL task-handler concurrency cap + bounded-shutdown knobs.
        task_concurrency = int(
            v.environ(MEMORYLAYER_TASK_CONCURRENCY, DEFAULT_MEMORYLAYER_TASK_CONCURRENCY)
        )
        document_task_concurrency = int(
            v.environ(
                MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT,
                DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_DOCUMENT,
            )
        )
        fanout_task_concurrency = int(
            v.environ(
                MEMORYLAYER_TASK_CONCURRENCY_FANOUT,
                DEFAULT_MEMORYLAYER_TASK_CONCURRENCY_FANOUT,
            )
        )
        shutdown_drain_timeout_s = float(
            v.environ(
                MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S,
                DEFAULT_MEMORYLAYER_SHUTDOWN_DRAIN_TIMEOUT_S,
            )
        )
        shutdown_close_timeout_s = float(
            v.environ(
                MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S,
                DEFAULT_MEMORYLAYER_SHUTDOWN_CLOSE_TIMEOUT_S,
            )
        )
        task_monitor_interval_s = float(
            v.environ(
                MEMORYLAYER_TASK_MONITOR_INTERVAL_S,
                DEFAULT_MEMORYLAYER_TASK_MONITOR_INTERVAL_S,
            )
        )

        # A serve-only server (in-process worker disabled) must NOT be an Aether
        # pool-task consumer: otherwise the gateway load-balances POOL task
        # assignments (e.g. session_cleanup) onto it by implementation, claims
        # them on its behalf, and — with no handler registered — they are
        # dropped and stick assigned until reconcile. Gate pool-consumer status
        # on the same MEMORYLAYER_TASKS_INPROCESS_WORKER flag the task service
        # uses to decide whether to register handlers, so server↔consumer stay
        # consistent. Local import avoids an import cycle with the task service.
        from ..tasks.aether import (
            DEFAULT_MEMORYLAYER_TASKS_INPROCESS_WORKER,
            MEMORYLAYER_TASKS_INPROCESS_WORKER,
        )
        consumes_pool_tasks = v.environ(
            MEMORYLAYER_TASKS_INPROCESS_WORKER,
            DEFAULT_MEMORYLAYER_TASKS_INPROCESS_WORKER,
            type_fn=ext_parse_bool,
        )

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
            liveness_enabled=liveness_enabled,
            liveness_interval_s=liveness_interval_s,
            liveness_timeout_s=liveness_timeout_s,
            liveness_failures=liveness_failures,
            task_concurrency=task_concurrency,
            document_task_concurrency=document_task_concurrency,
            fanout_task_concurrency=fanout_task_concurrency,
            shutdown_drain_timeout_s=shutdown_drain_timeout_s,
            shutdown_close_timeout_s=shutdown_close_timeout_s,
            task_monitor_interval_s=task_monitor_interval_s,
            consumes_pool_tasks=consumes_pool_tasks,
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

        # Standalone (typical OSS) deployments never configure Aether at all. Treat
        # "no Aether environment whatsoever" as "not in use" and stay quiet, rather
        # than failing the credential check on every start — that surfaced as an
        # alarming AETHER_API_KEY warning on a server that was working fine and had
        # never asked for Aether. Any Aether variable being set means the operator
        # DID intend to use it, so the fail-fast credential check below still runs.
        if not _any_aether_env_configured():
            logger.info(
                "Aether is not configured (no AETHER_* environment set); running standalone. "
                "Aether-dependent features (cross-service proxying, gRPC token API, task back-channel) are inactive. "
                "To enable, set AETHER_GATEWAY_ADDR and AETHER_API_KEY (or AETHER_AUTH=none for local development)."
            )
            return

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
