"""Caller-asserted attribution headers for outgoing LLM calls.

MemoryLayer routes its first-party LLM traffic through an OpenAI-compatible
MLflow AI Gateway that maps inbound ``X-Scitrera-<key>`` HTTP headers to
queryable trace metadata. This module holds the *ambient* attribution the
provider layer stamps on every call without threading it through every service
method signature:

* Static per-pod headers (``X-Scitrera-Source``, ``X-Scitrera-Tenant``) are
  built once at registry construction and installed on the provider client as
  ``default_headers`` — see ``services/llm/registry.py``.
* The dynamic per-task header (``X-Scitrera-Task-Id``) is carried in a
  :class:`contextvars.ContextVar` set by the task-execution chokepoint
  (``services/tasks/aether.py:_handle_task_assignment``) around the handler
  call. The OpenAI-compatible provider reads it and merges it into the outgoing
  request headers, so LLM calls made transitively (memory.remember -> extraction
  -> llm, ontology, tiering, ...) inherit the task attribution automatically.

Trust model = caller-stamped (no auth): MemoryLayer asserts only its OWN known
identity (source + its single-tenant slug + the Aether task it is running).

**These headers are first-party only.** They name the operator's internal tenant
to whoever receives them, and every provider here speaks the same OpenAI-compatible
protocol as public vendors — so nothing about the provider type says whether the
endpoint is ours. :func:`host_allows_identity` is the gate: a profile is stamped
only when its ``base_url`` host matches the configured allowlist, and both the
static headers and the per-task header ride on that single decision. It fails
closed twice over — an empty allowlist stamps nothing, and a profile with no
``base_url`` is by definition pointed at the vendor's own endpoint and is never
stamped.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit

# Header names (MLflow AI Gateway maps X-Scitrera-<key> -> trace metadata).
HEADER_SOURCE = "X-Scitrera-Source"
HEADER_TENANT = "X-Scitrera-Tenant"
HEADER_TASK_ID = "X-Scitrera-Task-Id"

# Static source value: MemoryLayer always asserts this as the origin service.
SOURCE_MEMORYLAYER = "memorylayer"

# Ambient Aether task id for the currently-executing task (None outside a task).
_current_task_id: ContextVar[str | None] = ContextVar("memorylayer_llm_task_id", default=None)


def get_current_task_id() -> str | None:
    """Return the Aether task id for the currently-executing task, if any."""
    return _current_task_id.get()


def task_attribution_headers() -> dict[str, str]:
    """Build the dynamic (per-task) attribution headers for the current context.

    Returns a dict with ``X-Scitrera-Task-Id`` when running inside an Aether
    task, else an empty dict. Static source/tenant headers are handled by the
    provider's ``default_headers`` and are intentionally NOT included here.
    """
    task_id = _current_task_id.get()
    if task_id:
        return {HEADER_TASK_ID: task_id}
    return {}


def parse_host_patterns(raw: str | None) -> tuple[str, ...]:
    """Parse the configured allowlist into normalized host patterns.

    Accepts a comma- or whitespace-separated list. Each entry may carry a scheme,
    port or path (``https://gw.internal:8443/v1``) — only the host is kept, so an
    operator can paste a profile's ``base_url`` verbatim and have it work.
    """
    if not raw:
        return ()
    parts = [p for chunk in raw.split(",") for p in chunk.split()]
    return tuple(dict.fromkeys(filter(None, (_normalize_pattern(p) for p in parts))))


def _normalize_pattern(raw: str) -> str:
    pattern = raw.strip().lower()
    if "://" in pattern:
        pattern = pattern.split("://", 1)[1]
    pattern = pattern.split("/", 1)[0]
    # Strip a port, but never mangle the bare "*" wildcard.
    if ":" in pattern:
        pattern = pattern.split(":", 1)[0]
    return pattern


def _host_of(base_url: str | None) -> str | None:
    """Lowercased hostname of ``base_url``, or None when it has none."""
    if not base_url or not base_url.strip():
        return None
    raw = base_url.strip()
    if "://" not in raw:
        # urlsplit would file a bare "host/path" under .path, leaving no hostname.
        raw = "//" + raw
    try:
        host = urlsplit(raw).hostname
    except ValueError:  # malformed URL (e.g. bad IPv6 literal) -> not allowed
        return None
    return host.lower() if host else None


def host_allows_identity(base_url: str | None, patterns: tuple[str, ...]) -> bool:
    """Whether a profile at ``base_url`` may be stamped with identity headers.

    Fails closed on both empty inputs. A missing ``base_url`` means the provider
    talks to its vendor default (``api.openai.com``, Fireworks) — external by
    definition, and never stamped.

    A pattern is either an exact host, ``*.suffix`` (matching any host under that
    suffix, but not the bare suffix itself), or ``*`` to allow everything — the
    last being an explicit escape hatch that defeats the point of the allowlist.
    """
    if not patterns:
        return False
    host = _host_of(base_url)
    if not host:
        return False
    for pattern in patterns:
        if pattern == "*":
            return True
        if pattern.startswith("*."):
            suffix = pattern[1:]  # "*.mt" -> ".mt"
            if host.endswith(suffix) and len(host) > len(suffix):
                return True
        elif host == pattern:
            return True
    return False


@contextmanager
def task_attribution(task_id: str | None):
    """Bind ``task_id`` as the ambient LLM task attribution for the block.

    A falsy ``task_id`` binds ``None`` (no task header stamped). The previous
    value is restored on exit, so nested tasks and the non-task path are safe.
    """
    token = _current_task_id.set(task_id or None)
    try:
        yield
    finally:
        _current_task_id.reset(token)
