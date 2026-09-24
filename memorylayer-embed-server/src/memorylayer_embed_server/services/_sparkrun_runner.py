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
import os
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
from pathlib import Path
from typing import Any

from ._vllm_runner import VLLMSubprocessRunner

__all__ = ["SparkrunVLLMRunner", "SparkrunUnavailable"]


class SparkrunUnavailableError(RuntimeError):
    """Raised when the sparkrun extra is not installed."""


SparkrunUnavailable = SparkrunUnavailableError  # compatibility alias


def _require_sparkrun():
    try:
        from sparkrun import api
        from sparkrun.application import initialize  # noqa: F401 - alpha API capability check
        from sparkrun.core.recipe import Recipe

        if not all(hasattr(api, name) for name in ("BuildOptions", "plan_build", "build")):
            raise ImportError("sparkrun independent build API is missing")
    except ImportError as exc:  # pragma: no cover - exercised via the plugin path
        raise SparkrunUnavailable(
            "The 'vllm_sparkrun' provider requires the API-capable sparkrun 0.4 alpha or later: "
            'pip install "memorylayer-embed-server[sparkrun]". '
            "Alternatively use MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER=vllm_subprocess, "
            "which spawns vllm directly and has no extra dependency."
        ) from exc
    return api, Recipe


@lru_cache(maxsize=1)
def sparkrun_context():
    from sparkrun.application import initialize

    return initialize(config_path=os.environ.get("MEMORYLAYER_EMBED_SPARKRUN_CONFIG"))



@lru_cache(maxsize=1)
def _controller_executor():
    # Sparkrun contexts and builder activation files are shared. Serialize API
    # operations, not the detached engines or their asynchronous health waits.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="sparkrun-controller")


async def _controller_call(fn, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(
        _controller_executor(), partial(fn, *args, **kwargs)
    )


async def _finish_cleanup(task):
    """Finish owned cleanup even if its caller receives another cancellation."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def configured_runner(*, recipe_env: str, **kwargs):
    """Select recipe lifecycle without duplicating a provider's HTTP client."""
    recipe = os.environ.get(recipe_env)
    if recipe:
        return SparkrunVLLMRunner(recipe_path=recipe, reuse_existing=False, **kwargs)
    return VLLMSubprocessRunner(**kwargs)


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
        self._owns_job = False
        self._sctx = None
        self._alive = False
        self._launch_task: asyncio.Task | None = None
        self.startup_metrics: dict[str, float] = {}

    # ------------------------------------------------------------------
    # State — the parent tracks an owned asyncio subprocess; we do not have one
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._cluster_id is not None and self._alive

    @property
    def pid(self) -> int | None:
        # sparkrun owns the pidfile; the embed server is deliberately not the parent.
        return None

    def build_argv(self) -> list[str]:
        """The serve command sparkrun rendered, for parity with the parent's argv.

        Empty until :meth:`start` has resolved the recipe — the command is produced
        by sparkrun's runtime, not composed here.
        """
        return shlex.split(self._serve_command) if self._serve_command else []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _run_options(self, api):
        """Recipe overrides carrying our config onto sparkrun's launch."""
        _, recipe_type = _require_sparkrun()
        recipe = recipe_type.load(self.recipe_path)
        if recipe.model != self.model_name:
            raise ValueError(f"Recipe model {recipe.model!r} does not match provider model {self.model_name!r}")
        # A recipe owns its full serving command, including model revision and
        # architecture. Carry eager mode and additional provider flags if absent.
        if recipe.command:
            flags = shlex.split(recipe.command.replace("\\\n", " "))
            extras = list(self.extra_args)
            if self.enforce_eager:
                extras.append("--enforce-eager")
            if self.architectures and "--hf-overrides" not in flags:
                import json
                extras += ["--hf-overrides", json.dumps({"architectures": self.architectures})]
            additions = []
            include = True
            for arg in extras:
                if arg.startswith("--"):
                    include = arg not in flags
                if include:
                    additions.append(arg)
            recipe.command += " " + shlex.join(additions)

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
        if self.tensor_parallel_size:
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
            sync_tuning=False,
            cache_dir=os.environ.get("HF_HOME"),
            local_cache_dir=os.environ.get("HF_HOME"),
            transfer_mode="local",
        )

    async def start(self) -> None:
        if self.is_running:
            return
        api, _ = _require_sparkrun()
        started = time.monotonic()
        # Initialize once on the event-loop thread, before submitting workers.
        self._sctx = sparkrun_context()
        self.startup_metrics = {}
        # Shield the entire launch AND ownership handoff, not just api.run.
        # Cancelling a thread await does not stop a detached native launch.
        self._launch_task = asyncio.create_task(self._launch(api))
        try:
            await asyncio.shield(self._launch_task)
            self.startup_metrics["launch_elapsed_seconds"] = time.monotonic() - started
            self._log_task = asyncio.create_task(self._pump_logs())
            await self.wait_for_health()
            self.startup_metrics["ready_elapsed_seconds"] = time.monotonic() - started
        except BaseException:
            cleanup = asyncio.create_task(self.shutdown())
            try:
                await _finish_cleanup(cleanup)
            except Exception:
                self.logger.exception("Failed to clean up model startup: %s", self.model_name)
            raise

    async def _launch(self, api) -> None:
        options = self._run_options(api)

        def launch():
            started = time.monotonic()
            result = api.run(options, sctx=self._sctx)
            return result, time.monotonic() - started

        result, elapsed = await _controller_call(launch)
        self.startup_metrics["controller_seconds"] = elapsed
        if result.rc != 0 or result.dry_run or not result.cluster_id:
            raise RuntimeError(f"sparkrun failed to launch {self.model_name}: rc={result.rc}")
        self._owns_job = not result.already_running
        self._cluster_id = result.cluster_id
        self._alive = True
        self._serve_command = result.serve_command or ""
        if result.serve_port:
            self.port = int(result.serve_port)
        self.logger.info(
            "sparkrun launched vllm (cluster_id=%s executor=%s runtime=%s port=%s)",
            result.cluster_id, result.executor, result.runtime, self.port,
        )

    async def _pump_logs(self) -> None:
        """Feed sparkrun's logfile into the inherited concurrency parser.

        The parent reads a stderr pipe it owns; sparkrun writes to a logfile and
        ``api.logs`` tails it. ``logs()`` is a BLOCKING iterator, so it runs on a
        worker thread and hands each line back to the loop rather than being
        iterated inline.
        """
        api, _ = _require_sparkrun()
        cluster_id = self._cluster_id

        def snapshot():
            return list(api.logs(cluster_id=cluster_id, follow=False, tail=200, sctx=self._sctx))

        previous: set[str] = set()
        try:
            while self._cluster_id:
                # Finite reads: cancelling the task never leaves an infinite
                # follow=True iterator alive in an executor thread.
                lines = [getattr(line, "text", "") or "" for line in await _controller_call(snapshot)]
                for text in lines:
                    if text and text not in previous:
                        self._maybe_capture_max_concurrency(text)
                        self.logger.info("[vllm:%s] %s", self.role, text)
                previous = set(lines)
                observed = await _controller_call(api.status, hosts=["localhost"], executor="local", sctx=self._sctx)
                if observed.for_host("localhost") is not None and not observed.observation_errors:
                    self._alive = cluster_id in observed.running_cluster_ids()
                    if not self._alive:
                        self.logger.error("vLLM engine exited: %s", cluster_id)
                        return
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.logger.warning("sparkrun log polling stopped: %s", exc)
            # A terminated workload may make logs() raise before status() runs.
            # Confirm the process is gone instead of treating loss of its log
            # stream as merely a missing concurrency hint.
            try:
                observed = await _controller_call(api.status, hosts=["localhost"], executor="local", sctx=self._sctx)
                if observed.for_host("localhost") is not None and not observed.observation_errors:
                    self._alive = cluster_id in observed.running_cluster_ids()
            except Exception:
                pass

    async def shutdown(self) -> None:
        # A sibling may have failed while this launch was queued/running. Await
        # ownership registration before deciding whether there is a job to stop.
        if self._launch_task is not None:
            try:
                await asyncio.shield(self._launch_task)
            except Exception:
                pass  # start() propagates the launch error; no successful handoff.
            self._launch_task = None
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
        cluster_id = self._cluster_id
        self._alive = False
        if not self._owns_job:
            self._cluster_id = None
            return
        try:
            result = await _controller_call(api.stop, cluster_id=cluster_id, sctx=self._sctx)
            if not result.success:
                raise RuntimeError(f"stop failed: {result.errors}")
            self._cluster_id = None
            self._owns_job = False
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
            raise


def default_recipe_path(name: str) -> Path:
    """Path to a recipe shipped inside the package."""
    return Path(__file__).resolve().parent.parent / "recipes" / name


def make_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
