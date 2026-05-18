"""Skills sync logic: compare local vs server state, return reconciliation action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SyncAction = Literal["push", "pull", "conflict", "in_sync"]


@dataclass(frozen=True)
class SyncResult:
    action: SyncAction
    reason: str
    server_manifest_hash: str
    server_bundle_hash: str


def compute_sync_action(
    server_manifest_hash: str,
    server_bundle_hash: str,
    client_manifest_hash: str,
    client_bundle_hash: str,
) -> SyncResult:
    """Determine what sync action is needed by comparing hashes.

    Hashes are SHA-256 strings. Empty string means "no data on that side".

    Decision table:
    - Both sides match -> in_sync
    - Client has no data (empty hashes) -> pull (server is authoritative)
    - Server has no data (empty hashes) -> push (client is authoritative)
    - Only manifest differs -> conflict (needs human resolution)
    - Only bundle differs -> conflict
    - Both differ -> conflict
    """
    manifest_match = server_manifest_hash == client_manifest_hash
    bundle_match = server_bundle_hash == client_bundle_hash

    if manifest_match and bundle_match:
        return SyncResult(
            action="in_sync",
            reason="Local and server are identical",
            server_manifest_hash=server_manifest_hash,
            server_bundle_hash=server_bundle_hash,
        )

    client_empty = not client_manifest_hash and not client_bundle_hash
    server_empty = not server_manifest_hash and not server_bundle_hash

    if client_empty:
        return SyncResult(
            action="pull",
            reason="No local state; server version available",
            server_manifest_hash=server_manifest_hash,
            server_bundle_hash=server_bundle_hash,
        )

    if server_empty:
        return SyncResult(
            action="push",
            reason="Server has no content; local version available",
            server_manifest_hash=server_manifest_hash,
            server_bundle_hash=server_bundle_hash,
        )

    # Both sides have data but differ
    parts = []
    if not manifest_match:
        parts.append("manifest")
    if not bundle_match:
        parts.append("bundle")
    reason = f"Conflict: {' and '.join(parts)} differ between local and server"
    return SyncResult(
        action="conflict",
        reason=reason,
        server_manifest_hash=server_manifest_hash,
        server_bundle_hash=server_bundle_hash,
    )
