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
from typing import Optional, Sequence


__all__ = ["VLLMSubprocessRunner", "find_free_port"]


_ROLE_EMBEDDING = "embedding"
_ROLE_LLM = "llm"
_VALID_ROLES = (_ROLE_EMBEDDING, _ROLE_LLM)


class VLLMSubprocessRunner:
    """Manage a single ``vllm serve`` child process.

    Parameters
    ----------
    role
        ``"embedding"`` adds ``--runner pooling --convert embed`` to the
        argv. ``"llm"`` does not.
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
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.25,
        enforce_eager: bool = False,
        tensor_parallel_size: int = 1,
        served_model_names: Optional[Sequence[str]] = None,
        extra_args: Optional[Sequence[str]] = None,
        cmd: str = "vllm",
        startup_timeout_sec: float = 600.0,
        logger: Optional[logging.Logger] = None,
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
        self.cmd = cmd
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.logger = logger or logging.getLogger(__name__)

        self._process: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None

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
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    # ------------------------------------------------------------------
    # Argv construction
    # ------------------------------------------------------------------

    def build_argv(self) -> list[str]:
        argv: list[str] = [
            self.cmd, "serve", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        if self.max_model_len is not None:
            argv.extend(["--max-model-len", str(self.max_model_len)])
        if self.tensor_parallel_size > 1:
            argv.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        if self.role == _ROLE_EMBEDDING:
            # vLLM 0.6+ pooling/embed flags.
            argv.extend(["--runner", "pooling", "--convert", "embed"])
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
        if self._process is None or self._process.stderr is None:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                self.logger.info(
                    "[vllm-subprocess role=%s] %s",
                    self.role, line.decode(errors="replace").rstrip(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - best-effort
            self.logger.debug("vllm subprocess stderr reader stopped: %s", e)

    async def wait_for_health(self) -> None:
        import httpx

        deadline = time.monotonic() + self.startup_timeout_sec
        last_err: Optional[BaseException] = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self._process is not None and self._process.returncode is not None:
                    raise RuntimeError(
                        f"vllm serve (role={self.role}) exited prematurely with code "
                        f"{self._process.returncode} during startup"
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
        self._process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._stderr_task = asyncio.create_task(self._pipe_stderr())
        await self.wait_for_health()
        self.logger.info(
            "vllm subprocess healthy at %s (role=%s, pid=%s)",
            self.health_url, self.role, self._process.pid,
        )

    async def shutdown(self) -> None:
        """SIGTERM the process group; escalate to SIGKILL after 10s. Idempotent."""
        proc = self._process
        if proc is not None and proc.returncode is None:
            self.logger.info(
                "Stopping vllm subprocess (role=%s, pid=%s)", self.role, proc.pid,
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.logger.warning(
                    "vllm subprocess (role=%s) did not stop in 10s; SIGKILL",
                    self.role,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    proc.kill()
                await proc.wait()
            self.logger.info(
                "vllm subprocess stopped (role=%s, exit=%s)",
                self.role, proc.returncode,
            )
        if self._stderr_task is not None and not self._stderr_task.done():
            self._stderr_task.cancel()


def find_free_port(
    host: str = "127.0.0.1",
    *,
    low: Optional[int] = None,
    high: Optional[int] = None,
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
