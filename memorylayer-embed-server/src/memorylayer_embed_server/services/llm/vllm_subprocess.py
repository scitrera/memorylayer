"""LLM provider that spawns ``vllm serve`` as a chat backend.

Mirror of ``services/embedding/vllm_subprocess.py`` but configured for
LLM (chat) workloads instead of embedding: no ``--runner pooling
--convert embed`` flags, and the wrapped subprocess advertises itself
via ``--served-model-name <profile>,<alias1>,...`` so the routing
layer can resolve incoming ``model`` strings.

We deliberately do NOT use the ``openai`` SDK here — instead, we
transparently proxy the raw HTTP request via ``httpx`` so tool calls,
``response_format``, multimodal (``image_url``) inputs, reasoning
fields, and any other OpenAI-API extension pass through unchanged.
"""

from __future__ import annotations

import asyncio
import json as _json
from collections.abc import AsyncIterator, Sequence
from logging import Logger
from typing import Any

from scitrera_app_framework import Variables, get_logger

from .._vllm_runner import VLLMSubprocessRunner
from .base import LLMProvider


class VLLMSubprocessLLMProvider(LLMProvider):
    """One vLLM subprocess hosting one LLM, with HTTP-proxy chat / completions."""

    def __init__(
        self,
        v: Variables | None = None,
        *,
        profile_name: str,
        model_name: str,
        host: str = "127.0.0.1",
        port: int,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.25,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        aliases: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
        cmd: str = "vllm",
        startup_timeout_sec: float = 600.0,
        request_timeout_sec: float = 600.0,
    ) -> None:
        self.profile_name = profile_name
        self.model_name = model_name
        self.host = host
        self.port = int(port)
        self.request_timeout_sec = float(request_timeout_sec)
        self.aliases = list(aliases) if aliases else []

        # Build the routing-name list. Dedup while preserving first-seen
        # order so the canonical profile name comes first.
        seen: set[str] = set()
        served: list[str] = []
        for name in (profile_name, model_name, *self.aliases):
            if name and name not in seen:
                seen.add(name)
                served.append(name)
        self.served_names = served

        self.logger = get_logger(v, name=self.__class__.__name__)

        self._runner = VLLMSubprocessRunner(
            role="llm",
            model_name=model_name,
            host=host,
            port=int(port),
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            served_model_names=served,
            extra_args=extra_args,
            cmd=cmd,
            startup_timeout_sec=startup_timeout_sec,
            logger=self.logger,
        )

        self._http = None  # lazy httpx.AsyncClient
        self._start_lock = asyncio.Lock()
        self._ready = False
        self._in_flight = 0

        # Test harness can flip to skip the actual ``vllm serve`` exec.
        self._skip_subprocess: bool = False

        self.logger.info(
            "Initialized VLLMSubprocessLLMProvider: profile=%s, model=%s, port=%d, served_names=%s",
            profile_name,
            model_name,
            port,
            served,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._runner.base_url

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_started(self):
        """Lazy start: spawn the subprocess (unless skipped) and open httpx."""
        if self._ready:
            return self._http
        async with self._start_lock:
            if self._ready:
                return self._http
            if not self._skip_subprocess:
                await self._runner.start()
            if self._http is None:
                import httpx

                self._http = httpx.AsyncClient(
                    base_url=self._runner.base_url,
                    timeout=self.request_timeout_sec,
                )
            self._ready = True
            return self._http

    async def preload(self) -> None:
        await self._ensure_started()

    async def shutdown(self) -> None:
        await self._runner.shutdown()
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
            self._http = None
        self._ready = False

    # ------------------------------------------------------------------
    # Proxy methods
    # ------------------------------------------------------------------

    async def chat_completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ) -> Any:
        return await self._forward("/chat/completions", payload, stream=stream)

    async def completions(
        self,
        payload: dict,
        *,
        stream: bool = False,
    ) -> Any:
        return await self._forward("/completions", payload, stream=stream)

    async def _forward(self, path: str, payload: dict, *, stream: bool):
        """Proxy ``POST <base_url>{path}`` returning either dict (json) or async byte iterator."""
        client = await self._ensure_started()

        # Make sure the upstream payload honors the streaming flag if
        # the caller set it via the URL path rather than the body.
        if stream:
            payload = {**payload, "stream": True}

        self._in_flight += 1
        if not stream:
            try:
                resp = await client.post(path, json=payload)
                resp.raise_for_status()
                return resp.json()
            finally:
                self._in_flight -= 1

        async def _aiter() -> AsyncIterator[bytes]:
            try:
                async with client.stream("POST", path, json=payload) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
            finally:
                # Decrement in the same task that incremented to keep
                # the counter accurate even on early-cancel.
                self._in_flight -= 1

        return _aiter()

    # ------------------------------------------------------------------
    # /v1/models metadata
    # ------------------------------------------------------------------

    def list_models(self) -> list[dict]:
        """Return ``/v1/models``-shaped entries for every routing name."""
        return [
            {
                "id": name,
                "object": "model",
                "owned_by": "memorylayer-embed-server",
                "profile": self.profile_name,
                "model": self.model_name,
            }
            for name in self.served_names
        ]

    # ------------------------------------------------------------------
    # Load reporting (for /health/load)
    # ------------------------------------------------------------------

    def get_load_snapshot(self) -> dict:
        # vLLM owns concurrency on its side; we treat ``in_flight`` as the
        # number of requests we've forwarded that haven't returned yet.
        # ``max_concurrent`` is reported as 0 to signal "no LB-side cap";
        # operators routing on this metric should treat 0 as "unlimited".
        return {
            "in_flight": self._in_flight,
            "max_concurrent": 0,
            "utilization": 0.0,
        }


# Helper for cleanly building from env config (used by dependencies.py).
def _shellsplit(s: str | None) -> list[str]:
    if not s:
        return []
    import shlex

    return shlex.split(s)


def _csv_split(s: str | None) -> list[str]:
    if not s:
        return []
    return [piece.strip() for piece in s.split(",") if piece.strip()]


def build_provider_from_env(
    v: Variables,
    logger: Logger,
    *,
    profile_name: str,
    port: int,
) -> VLLMSubprocessLLMProvider:
    """Read per-profile env vars and instantiate a provider.

    The port is supplied by the caller (typically allocated via
    ``find_free_port`` against the configured LLM port range).
    """
    upper = profile_name.upper()
    prefix = f"MEMORYLAYER_EMBED_LLM_PROFILE_{upper}_"

    def _env(suffix: str, default=None, type_fn=None):
        kwargs = {"default": default}
        if type_fn is not None:
            kwargs["type_fn"] = type_fn
        return v.environ(f"{prefix}{suffix}", **kwargs)

    model_name = _env("MODEL", default=None)
    if not model_name:
        raise RuntimeError(f"Missing required env var {prefix}MODEL for LLM profile {profile_name!r}")

    aliases = _csv_split(_env("ALIASES", default=""))
    dtype = _env("DTYPE", default="auto")
    max_model_len_str = _env("MAX_MODEL_LEN", default="")
    max_model_len = int(max_model_len_str) if max_model_len_str else None
    gpu_mem_util = _env("GPU_MEM_UTIL", default=0.25, type_fn=float)
    enforce_eager = _env(
        "ENFORCE_EAGER",
        default=False,
        type_fn=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
    )
    tp_size = _env("TENSOR_PARALLEL_SIZE", default=1, type_fn=int)
    startup_timeout = _env("STARTUP_TIMEOUT_SEC", default=600.0, type_fn=float)
    cmd = _env("CMD", default="vllm")
    host = _env("HOST", default="127.0.0.1")
    extra_args = _shellsplit(_env("EXTRA_ARGS", default=""))

    return VLLMSubprocessLLMProvider(
        v=v,
        profile_name=profile_name,
        model_name=model_name,
        host=host,
        port=port,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        enforce_eager=enforce_eager,
        tensor_parallel_size=tp_size,
        aliases=aliases,
        extra_args=extra_args,
        cmd=cmd,
        startup_timeout_sec=startup_timeout,
    )


# Re-export the JSON helper for tests that want to round-trip payloads.
_dump_json = _json.dumps
