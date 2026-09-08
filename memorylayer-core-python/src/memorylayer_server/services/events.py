"""OSS-core helper for emitting domain events to Aether's ``event::*`` plane.

Provides ``emit_event`` — the OSS-core equivalent of the proprietary
``doc_added._emit_event`` — so OSS task handlers can emit workflow events
(e.g. ``memorylayer.decompose_complete``) without depending on proprietary
code.  In an OSS-standalone deployment (no Aether connection, or a client
without ``send_event``) it no-ops cleanly.
"""
from __future__ import annotations

import json

from scitrera_app_framework import Variables, get_extension

from ._constants import EXT_AETHER_SERVICE_CONNECTION


async def emit_event(
    v: Variables, workspace_id: str, event_name: str, data: dict, logger,
) -> None:
    """Emit an event via Aether's ``send_event`` to the ``event::*`` plane.

    Best-effort; failure is logged but not raised, and the absence of an Aether
    connection (OSS-standalone) is a clean no-op.

    The payload is JSON (UTF-8), NOT msgpack: Aether's native workflow engine
    consumes the default ``event::*`` plane and ``json.Unmarshal``s each event
    into ``EventPayload{source_agent, event_names[], data, workspace}``.  We
    deliberately send to the broad (un-scoped) ``event::*`` topic so the
    workflow engine receives it; a workspace-scoped variant would not be seen.

    Args:
        v: Variables instance.
        workspace_id: Workspace the event pertains to.
        event_name: Fully-qualified event name (e.g. ``memorylayer.decompose_complete``).
        data: Event-specific payload; merged under a ``workspace_id`` key.
        logger: Logger for best-effort warnings.
    """
    try:
        agent_service = get_extension(EXT_AETHER_SERVICE_CONNECTION, v)
        client = getattr(agent_service, "client", None)

        if client is None or not hasattr(client, "send_event"):
            logger.debug(
                "Aether client unavailable or does not support send_event, "
                "skipping event emission",
            )
            return

        envelope = {
            "source_agent": "memorylayer",
            "workspace": workspace_id,
            "event_names": [event_name],
            "data": {"workspace_id": workspace_id, **data},
        }
        payload_bytes = json.dumps(envelope).encode("utf-8")
        await client.send_event(payload_bytes)
    except Exception:
        logger.warning(
            "Failed to emit event %s for workspace %s (best-effort)",
            event_name, workspace_id,
            exc_info=True,
        )
