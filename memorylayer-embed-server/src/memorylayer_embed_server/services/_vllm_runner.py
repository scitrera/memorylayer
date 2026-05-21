"""Shared subprocess lifecycle for ``vllm serve``.

Both the embedding-side (``services/embedding/vllm_subprocess.py``) and the
LLM-side (``services/llm/vllm_subprocess.py``) provider classes need to
spawn a child ``vllm serve``, wait for its ``/health`` endpoint, and tear
it down cleanly. The only meaningful difference is the argv: embedding
workloads pass ``--runner pooling --convert embed``; chat workloads pass
``--served-model-name`` plus an optional tensor-parallel size. Everything
else — process-group handling, stderr forwarding, health-poll loop,
SIGTERM-then-SIGKILL shutdown — is identical.

This module also exposes :func:`find_free_port`, a tiny helper that
auto-assigns each LLM profile a loopback port at boot so operators don't
have to.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import socket
import time
from collections.abc import Sequence

__all__ = ["VLLMSubprocessRunner", "find_free_port"]


_ROLE_EMBEDDING = "embedding"
_ROLE_LLM = "llm"
_ROLE_MULTI_VECTOR = "multi_vector"
_VALID_ROLES = (_ROLE_EMBEDDING, _ROLE_LLM, _ROLE_MULTI_VECTOR)


class VLLMSubprocessRunner:
    """Manage a single ``vllm serve`` child process.

    Parameters
    ----------
    role
        ``"embedding"`` adds ``--runner pooling --convert embed`` (mean-pool
        to a single vector per input). ``"multi_vector"`` adds just
        ``--runner pooling`` so the model's native architecture class
        (e.g. ``ColPaliForRetrieval``, ``ColModernVBertForRetrieval``,
        ``ColQwen3_5``) drives per-token output served on ``/pooling``.
        ``"llm"`` adds neither — standard chat/completion serving.
    architectures
        When set, passes ``--hf-overrides '{"architectures": [...]}'`` to
        vLLM. Required for ColPali checkpoints that ship without a populated
        ``architectures`` field (e.g. ``ModernVBERT/colmodernvbert-merged``),
        and for Qwen3-VL-Reranker (where the HF default points at the
        chat arch).
    served_model_names
        Forwarded to vLLM as ``--served-model-name`` (vLLM accepts
        multiple, advertising the model under each name). Useful for
        chat workloads where the routing layer wants ``model="qwen"``
        to resolve to ``Qwen/Qwen2.5-7B-Instruct``.
    extra_args
        Free-form passthrough appended after the standard flags. The
        operator owns ordering and correctness.
    """

    def __init__(
        self,
        *,
        role: str,
        model_name: str,
        host: str,
        port: int,
        dtype: str = "auto",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.25,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        served_model_names: Sequence[str] | None = None,
        extra_args: Sequence[str] | None = None,
        architectures: Sequence[str] | None = None,
        cmd: str = "vllm",
        startup_timeout_sec: float = 600.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got {role!r}")
        self.role = role
        self.model_name = model_name
        self.host = host
        self.port = int(port)
        self.dtype = dtype
        self.max_model_len = int(max_model_len) if max_model_len is not None else None
        self.gpu_memory_utilization = float(gpu_memory_utilization)
        self.enforce_eager = bool(enforce_eager)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.served_model_names = list(served_model_names) if served_model_names else []
        self.extra_args = list(extra_args) if extra_args else []
        self.architectures = list(architectures) if architectures else []
        self.cmd = cmd
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.logger = logger or logging.getLogger(__name__)

        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # URLs / state
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    # ------------------------------------------------------------------
    # Argv construction
    # ------------------------------------------------------------------

    def build_argv(self) -> list[str]:
        argv: list[str] = [
            self.cmd,
            "serve",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        if self.max_model_len is not None:
            argv.extend(["--max-model-len", str(self.max_model_len)])
        if self.tensor_parallel_size > 1:
            argv.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.role == _ROLE_EMBEDDING:
            # Mean-pool to a single vector per input.
            argv.extend(["--runner", "pooling", "--convert", "embed"])
        elif self.role == _ROLE_MULTI_VECTOR:
            # Native multi-vector / late-interaction output. The model's
            # arch class (ColPaliForRetrieval, ColModernVBertForRetrieval,
            # ColQwen3_5, …) drives the per-token output served on /pooling.
            # Do NOT add --convert embed here — that would mean-pool away
            # the multi-vector signal we want to preserve.
            argv.extend(["--runner", "pooling"])
        if self.architectures:
            import json as _json

            argv.extend(["--hf-overrides", _json.dumps({"architectures": list(self.architectures)})])
        if self.served_model_names:
            argv.append("--served-model-name")
            argv.extend(self.served_model_names)
        if self.enforce_eager:
            argv.append("--enforce-eager")
        if self.extra_args:
            argv.extend(self.extra_args)
        return argv

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _pipe_stderr(self) -> None:
        # Reads vllm's merged stdout+stderr stream (we redirect stderr to
        # stdout in start()). Drains continuously so the OS pipe buffer
        # never fills — a full pipe blocks vllm's next write and stalls
        # engine init.
        if self._process is None or self._process.stdout is None:
            return
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                self.logger.info(
                    "[vllm-subprocess role=%s] %s",
                    self.role,
                    line.decode(errors="replace").rstrip(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - best-effort
            self.logger.debug("vllm subprocess output reader stopped: %s", e)

    async def wait_for_health(self) -> None:
        import httpx

        deadline = time.monotonic() + self.startup_timeout_sec
        last_err: BaseException | None = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    raise RuntimeError(
                        f"vllm serve (role={self.role}) exited prematurely with code {self._process.returncode} during startup"
                    )
                try:
                    resp = await client.get(self.health_url)
                    if resp.status_code == 200:
                        return
                except Exception as e:  # noqa: BLE001 - polling
                    last_err = e
                await asyncio.sleep(2.0)
        raise RuntimeError(
            f"vllm serve (role={self.role}) did not become healthy within "
            f"{self.startup_timeout_sec:.0f}s at {self.health_url}: {last_err!r}"
        )

    async def start(self) -> None:
        """Spawn the subprocess and wait until /health returns 200.

        Idempotent — a no-op if the process is already running.
        """
        if self.is_running:
            return
        if shutil.which(self.cmd) is None:
            raise RuntimeError(
                f"vllm subprocess command {self.cmd!r} not found on PATH; "
                f"install memorylayer-embed-server[vllm] in the image or "
                f"override the *_CMD env var."
            )
        argv = self.build_argv()
        self.logger.info("Starting vllm subprocess (role=%s): %s", self.role, " ".join(argv))
        # Merge stdout into stderr so we only need one drain task — and
        # critically, so vllm's stdout writes (banner, some warnings)
        # cannot deadlock on a full OS pipe buffer when nothing is
        # draining the other stream. We hit this in test harnesses: vllm
        # would block ~64 KiB into its boot logging, engine init stalled,
        # memory pressure climbed, earlyoom eventually killed the process.
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._pipe_stderr())
        try:
            await self.wait_for_health()
        except BaseException:
            # wait_for_health raised — the subprocess (and any EngineCore
            # worker procs it spawned) didn't reach a usable state. Tear
            # them down before re-raising so we don't leak GPU memory.
            self.logger.warning(
                "vllm subprocess (role=%s) failed health check; tearing down to avoid orphans.",
                self.role,
            )
            try:
                await self.shutdown()
            except Exception as cleanup_err:  # noqa: BLE001 - log + continue
                self.logger.error(
                    "vllm subprocess (role=%s) cleanup after failed start raised: %s",
                    self.role,
                    cleanup_err,
                )
            raise
        self.logger.info(
            "vllm subprocess healthy at %s (role=%s, pid=%s)",
            self.health_url,
            self.role,
            self._process.pid,
        )

    async def shutdown(self) -> None:
        """SIGTERM the process group; escalate to SIGKILL after 10s. Idempotent.

        Uses the PID we recorded at spawn time as the PGID — with
        ``start_new_session=True`` the child is the session leader so
        PID == PGID. Looking it up via ``os.getpgid`` at shutdown time
        fails when the parent has been reaped, which loses our handle
        on still-alive EngineCore workers and leaks GPU memory.
        """
        proc = self._process
        # The parent may have already exited (exit code != None) but its
        # EngineCore worker processes can outlive it. Always attempt the
        # process-group kill on the spawn PID.
        if proc is not None:
            pgid = proc.pid  # spawn used start_new_session=True ⇒ pid == pgid
            self.logger.info(
                "Stopping vllm subprocess (role=%s, pid=%s, exit=%s)",
                self.role,
                pgid,
                proc.returncode,
            )
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                if proc.returncode is None:
                    proc.terminate()
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except TimeoutError:
                    self.logger.warning(
                        "vllm subprocess (role=%s) did not stop in 10s; SIGKILL",
                        self.role,
                    )
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        proc.kill()
                    await proc.wait()
            else:
                # Parent already exited but the group kill above still
                # reaches any orphan EngineCore workers. Give them a
                # moment to receive the signal and exit; escalate if
                # they don't.
                await asyncio.sleep(1.0)
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
            self.logger.info(
                "vllm subprocess stopped (role=%s, exit=%s)",
                self.role,
                proc.returncode,
            )
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()


def find_free_port(
    host: str = "127.0.0.1",
    *,
    low: int | None = None,
    high: int | None = None,
) -> int:
    """Find an unused TCP port on ``host``.

    With ``low`` and ``high`` set, scans the inclusive range and returns
    the first port that binds successfully. Otherwise asks the kernel for
    any free port via ``bind((host, 0))``.

    There is a TOCTOU window between this call and the consumer's actual
    bind. Callers should be prepared to retry on ``EADDRINUSE``.
    """
    if low is not None and high is not None:
        if low > high:
            raise ValueError(f"port range invalid: low={low} > high={high}")
        for port in range(low, high + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind((host, port))
                except OSError:
                    continue
                return port
        raise RuntimeError(f"no free port in range {low}-{high} on {host}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]
