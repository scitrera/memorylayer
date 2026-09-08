"""Stable SDK constants shared with the MemoryLayer server.

Each constant in this module has a server-side twin in
``memorylayer-core-python`` (``memorylayer_server/models/chat.py``).
The values MUST match — when bumping one, bump the other.
"""
from __future__ import annotations


USER_CHAT_HOME_WORKSPACE = "_user_chat"
"""Storage workspace sentinel for user-owned chat threads.

When a thread is created or appended-to with ``ownership='user'`` (the
SDK default), the SDK substitutes the caller-supplied workspace with
this sentinel so the thread row lives in the cross-workspace user
namespace. The caller's original workspace is preserved on per-message
metadata under :data:`MESSAGE_META_APP_WORKSPACE_KEY`.

MUST match ``memorylayer_server.models.chat.USER_CHAT_HOME_WORKSPACE``.
"""

MESSAGE_META_APP_WORKSPACE_KEY = "app_workspace"
"""Reserved top-level key on chat message ``metadata`` carrying the
originating app workspace at write time.

When the SDK substitutes a user-owned thread's storage workspace, it
also folds the caller's workspace_id into each message's metadata
under this key so per-message origin context isn't lost.

MUST match ``memorylayer_server.models.chat.MESSAGE_META_APP_WORKSPACE_KEY``.
"""


__all__ = [
    "USER_CHAT_HOME_WORKSPACE",
    "MESSAGE_META_APP_WORKSPACE_KEY",
]
