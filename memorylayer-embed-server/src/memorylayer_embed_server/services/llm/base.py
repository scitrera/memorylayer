"""Abstract LLM provider interface for the embed-server.

Each provider wraps a single chat-capable backend (a ``vllm serve``
subprocess today; openai_compat / llama.cpp could follow). Providers
are pure proxies — they don't translate request shapes; they forward
the OpenAI-compatible payload as-is so tool calls, multimodal,
response_format, and reasoning fields pass through whatever the
underlying model supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Wraps one chat-capable LLM backend.

    Implementations may stream incrementally (``stream=True``) or
    return a single JSON dict (``stream=False``). Lifecycle (process
    start / health-check / shutdown) is the implementation's concern.
    """

    # Profile name as declared in env config (``MEMORYLAYER_EMBED_LLM_PROFILES``).
    profile_name: str = ""

    # Underlying model identifier passed to ``vllm serve``.
    model_name: str = ""

    # Names the backend will answer to in addition to ``profile_name``.
    # Used by ``LLMRoutingService`` to build the model→profile alias map.
    served_names: list[str]

    @abstractmethod
    async def chat_completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ) -> Any:
        """Run a chat completion.

        For ``stream=False`` return the upstream JSON response (``dict``).
        For ``stream=True`` return an ``AsyncIterator[bytes]`` yielding
        SSE-formatted bytes verbatim from the upstream so the FastAPI
        route can pipe them through unchanged.
        """

    @abstractmethod
    async def completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ) -> Any:
        """Legacy text completions endpoint. Same return contract as ``chat_completions``."""

    @abstractmethod
    async def preload(self) -> None:
        """Eagerly start the subprocess + open clients (idempotent)."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Tear down the subprocess and any HTTP clients (idempotent)."""

    def get_load_snapshot(self) -> dict:
        """Return ``{in_flight, max_concurrent, utilization}`` for load reporting.

        Default implementation returns zeros — providers with concurrency
        accounting override this so ``/health/load`` reflects real state.
        """
        return {"in_flight": 0, "max_concurrent": 0, "utilization": 0.0}


__all__ = ["LLMProvider"]
