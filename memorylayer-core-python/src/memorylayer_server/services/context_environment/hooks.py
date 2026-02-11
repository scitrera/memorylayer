"""Persistence hooks for context environment state."""
from abc import ABC


class ContextPersistenceHook(ABC):
    """Hook for persisting sandbox state. No-op by default."""

    async def on_state_changed(self, session_id: str, state: dict) -> None:
        """Called when sandbox state changes after execution."""
        pass

    async def on_checkpoint(self, session_id: str, state: dict) -> None:
        """Called on explicit checkpoint request."""
        pass

    async def on_session_end(self, session_id: str, state: dict) -> None:
        """Called when a session environment is cleaned up."""
        pass

    async def on_session_restore(self, session_id: str) -> dict | None:
        """Called to restore state for a session. Returns state dict or None."""
        return None


class NoOpPersistenceHook(ContextPersistenceHook):
    """Default no-op persistence hook."""
    pass
