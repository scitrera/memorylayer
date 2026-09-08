"""Launch ``vllm serve`` through sparkrun instead of spawning it ourselves.

``VLLMSubprocessRunner`` spawns the child, polls health, and tears it down.
sparkrun already does all three, already runs these exact models, and the launch
parameters already exist as recipes. This subclass keeps everything ABOVE the
process and replaces only the process lifecycle:

  inherited   wait_for_health, concurrency_slot, effective_max_concurrent,
              _maybe_capture_max_concurrency  — the vLLM-specific parts that have
              no sparkrun equivalent
  overridden  start, shutdown, is_running, pid, build_argv

The concurrency logic is the reason this is a subclass rather than a rewrite: we
size the client-side semaphore from vLLM's ``Maximum concurrency for N tokens per
request: X`` line so backpressure lands at our HTTP boundary rather than inside
vLLM's scheduler queue. sparkrun does not model that, so we keep parsing it — just
from its logfile rather than from a stderr pipe we own.

Requires the ``sparkrun`` extra. It is optional on purpose: this is a GPU-path
dependency and the CPU image must not carry it.

Selected with ``MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_sparkrun``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from ._vllm_runner import VLLMSubprocessRunner

__all__ = ["SparkrunVLLMRunner", "SparkrunUnavailable"]


class SparkrunUnavailable(RuntimeError):
    """Raised when the sparkrun extra is not installed."""


def _require_sparkrun():
    try:
        from sparkrun import api
        from sparkrun.core.recipe import Recipe
    except ImportError as exc:  # pragma: no cover - exercised via the plugin path
        raise SparkrunUnavailable(
            "The 'vllm_sparkrun' provider requires the sparkrun extra: "
            'pip install "memorylayer-embed-server[sparkrun]". '
            "Alternatively use MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_subprocess, "
            "which spawns vllm directly and has no extra dependency."
        ) from exc
    return api, Recipe


class SparkrunVLLMRunner(VLLMSubprocessRunner):
    """Process lifecycle backed by ``sparkrun.api`` with the ``local`` executor.

    The local executor ``setsid``-launches natively — no container, no SSH — so this
    works inside the embed-server's own container without a docker socket.
    """

    def __init__(
        self,
        *,
        recipe_path: str | Path,
        owner: str = "memorylayer-embed-server",
        reuse_existing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.recipe_path = Path(recipe_path)
        self.owner = owner
        self.reuse_existing = reuse_existing
        self._cluster_id: str | None = None
        self._log_task: asyncio.Task | None = None
        self._serve_command: str = ""

    # ------------------------------------------------------------------
    # State — the parent tracks an owned asyncio subprocess; we do not have one
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._cluster_id is not None

    @property
    def pid(self) -> int | None:
        # sparkrun owns the pidfile; the embed server is deliberately not the parent.
        return None

    def build_argv(self) -> list[str]:
        """The serve command sparkrun rendered, for parity with the parent's argv.

        Empty until :meth:`start` has resolved the recipe — the command is produced
        by sparkrun's runtime, not composed here.
        """
        return self._serve_command.split() if self._serve_command else []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _run_options(self, api):
        """Recipe overrides carrying our config onto sparkrun's launch."""
        _, Recipe = _require_sparkrun()
        recipe = Recipe.load(self.recipe_path)
        overrides: dict[str, Any] = {
            # The embed server assigns a loopback port per lane; the recipe's port
            # is a default, not the truth.
            "port": self.port,
            # Recipes bind 0.0.0.0 for cluster serving. This child is ours alone, so
            # keep it on loopback rather than exposing it inside the container.
            "host": self.host,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }
        if self.max_model_len is not None:
            overrides["max_model_len"] = self.max_model_len
        if self.dtype and self.dtype != "auto":
            overrides["dtype"] = self.dtype
        if self.tensor_parallel_size and self.tensor_parallel_size != 1:
            overrides["tensor_parallel"] = self.tensor_parallel_size

        return api.RunOptions(
            recipe=recipe,
            hosts=("localhost",),
            executor="local",
            overrides=overrides,
            follow=False,
            detached=True,
            # Never prompt: a server has no TTY, and an interactive trust prompt
            # would hang startup rather than fail it.
            trust=True,
            # Reuse an identical running job instead of launching a second copy —
            # what a restarted server wants, and what stops a crash-loop from
            # stacking vLLM instances on one GPU.
            ensure=self.reuse_existing,
            owner=self.owner,
        )

    async def start(self) -> None:
        if self.is_running:
            return
        api, _ = _require_sparkrun()

        # sparkrun's API is synchronous; keep it off the event loop.
        result = await asyncio.to_thread(api.run, self._run_options(api))

        self._cluster_id = result.cluster_id
        self._serve_command = result.serve_command or ""
        if result.serve_port:
            self.port = int(result.serve_port)

        self.logger.info(
            "sparkrun launched vllm (cluster_id=%s executor=%s runtime=%s port=%s)",
            result.cluster_id,
            result.executor,
            result.runtime,
            self.port,
        )

        self._log_task = asyncio.create_task(self._pump_logs())
        try:
            await self.wait_for_health()
        except Exception:
            # A failed startup must not leave a detached vLLM holding VRAM.
            await self.shutdown()
            raise

    async def _pump_logs(self) -> None:
        """Feed sparkrun's logfile into the inherited concurrency parser.

        The parent reads a stderr pipe it owns; sparkrun writes to a logfile and
        ``api.logs`` tails it. ``logs()`` is a BLOCKING iterator, so it runs on a
        worker thread and hands each line back to the loop rather than being
        iterated inline.
        """
        api, _ = _require_sparkrun()
        loop = asyncio.get_running_loop()
        cluster_id = self._cluster_id

        def _consume() -> None:
            for line in api.logs(cluster_id=cluster_id, follow=True, tail=200):
                text = getattr(line, "text", "") or ""
                loop.call_soon_threadsafe(self._maybe_capture_max_concurrency, text)

        try:
            await asyncio.to_thread(_consume)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Losing the log tail costs the vLLM-reported concurrency value, which
            # falls back to the configured/default one. Not worth failing a healthy
            # server over.
            self.logger.warning("sparkrun log tail stopped (%s); using fallback concurrency", exc)

    async def shutdown(self) -> None:
        if self._log_task is not None:
            self._log_task.cancel()
            try:
                await self._log_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._log_task = None

        if self._cluster_id is None:
            return
        api, _ = _require_sparkrun()
        cluster_id, self._cluster_id = self._cluster_id, None
        try:
            await asyncio.to_thread(api.stop, cluster_id=cluster_id)
            self.logger.info("sparkrun stopped vllm (cluster_id=%s)", cluster_id)
        except Exception as exc:  # noqa: BLE001
            # The workload is setsid-detached, so a failed stop leaks a process
            # holding GPU memory. Say so loudly with the id needed to clean up.
            self.logger.error(
                "sparkrun stop FAILED for cluster_id=%s (%s); vllm may still hold GPU "
                "memory — clean up with: sparkrun stop --cluster-id %s",
                cluster_id,
                exc,
                cluster_id,
            )


def default_recipe_path(name: str) -> Path:
    """Path to a recipe shipped inside the package."""
    return Path(__file__).resolve().parent.parent / "recipes" / name


def make_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
