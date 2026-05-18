"""MCP server sync logic — pure hash comparison, no I/O."""
from __future__ import annotations

from typing import Literal

SyncAction = Literal["push", "pull", "conflict", "in_sync"]


def compare_hashes(
    local_hash: str,
    server_hash: str,
) -> tuple[SyncAction, str]:
    """Compare local vs server manifest_hash and return (action, reason).

    Rules:
    - both equal → in_sync
    - local empty, server non-empty → pull (server has data, local doesn't)
    - local non-empty, server empty → push (local has data, server doesn't)
    - both non-empty and different → conflict (both sides have diverged)
    """
    local = local_hash.strip()
    server = server_hash.strip()

    if local == server:
        return "in_sync", "local and server configs are identical"
    if not local and server:
        return "pull", "server has config, local is empty"
    if local and not server:
        return "push", "local has config, server record is empty"
    return "conflict", "local and server configs have both changed"


def resolve_conflict(
    action: SyncAction,
    auto_resolve: Literal["prefer-local", "prefer-remote", "abort"] | None,
) -> tuple[SyncAction, str]:
    """Apply auto-resolution policy to a conflict action.

    Returns the resolved (action, reason). If no policy or not a conflict,
    returns the original action unchanged.
    """
    if action != "conflict" or not auto_resolve:
        return action, ""
    if auto_resolve == "prefer-local":
        return "push", "conflict auto-resolved: prefer-local"
    if auto_resolve == "prefer-remote":
        return "pull", "conflict auto-resolved: prefer-remote"
    # abort — caller must treat this as an error
    return "conflict", "conflict: auto-resolve=abort"
