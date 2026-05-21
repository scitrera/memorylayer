"""Shared subprocess helper for the real-GPU integration tests.

The tests start the embed-server CLI as a child process on a free
loopback port and drive it over HTTP. This keeps the tests close to a
production deployment (real uvicorn, real FastAPI lifespan, real plugin
initialization) while being lighter than the docker-compose variant.

The helper resolves the ``memorylayer-embed`` CLI from the package's
local ``.venv`` so the tests do not require the caller to activate any
environment — ``pytest .venv/bin/pytest tests/...`` is enough.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

HOST = "127.0.0.1"

# .../oss/memorylayer-embed-server/tests/integration/_real_gpu_subprocess.py
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_VENV_CLI = _PACKAGE_ROOT / ".venv" / "bin" / "memorylayer-embed"


def find_free_port() -> int:
    """Bind an ephemeral port, then release it for the child to reuse."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def resolve_embed_cli() -> Path | None:
    """Prefer the package-local ``.venv``; fall back to ``$PATH``.

    Returns ``None`` when no CLI is reachable so callers can skip cleanly.
    """
    if _LOCAL_VENV_CLI.exists():
        return _LOCAL_VENV_CLI
    on_path = shutil.which("memorylayer-embed")
    return Path(on_path) if on_path else None


def wait_for_http(url: str, *, timeout_s: float, accept_status: tuple[int, ...] = (200,)) -> httpx.Response:
    """Poll ``url`` until one of ``accept_status`` is returned or timeout."""
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    last_response: httpx.Response | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=5.0)
            if r.status_code in accept_status:
                return r
            last_response = r
        except Exception as e:  # noqa: BLE001 - retryable polling
            last_exc = e
        time.sleep(1.0)
    body = last_response.text if last_response is not None else None
    status = last_response.status_code if last_response is not None else None
    raise TimeoutError(
        f"{url} did not return {accept_status} within {timeout_s:.0f}s "
        f"(last_status={status}, last_body={body!r}, last_exc={last_exc!r})"
    )


@contextmanager
def embed_server_subprocess(
    *,
    env_overrides: dict[str, str],
    boot_timeout_s: float = 120.0,
    log_path: Path | None = None,
) -> Iterator[str]:
    """Start the embed-server CLI on a free port and yield its base URL.

    ``env_overrides`` is layered on top of the caller's environment so
    HF cache + CUDA visibility carry through. Server stdout/stderr is
    streamed to ``log_path`` (or a tempfile under ``$TMPDIR``) so test
    failures have actionable context — ``cat`` it on a timeout to see
    why the model never came up.
    """
    cli = resolve_embed_cli()
    if cli is None:
        import pytest

        pytest.skip(
            "memorylayer-embed CLI not found — install the package into "
            "oss/memorylayer-embed-server/.venv (uv pip install -e ...) "
            "or onto $PATH."
        )

    port = int(env_overrides.pop("__PORT__", 0) or find_free_port())
    base_url = f"http://{HOST}:{port}"

    env = os.environ.copy()
    env.update(env_overrides)

    # Prepend the package-local venv's bin to PATH so child processes spawned
    # by the embed-server (notably ``vllm serve`` for the vLLM multi-vector
    # and rerank providers) resolve to the venv-installed binaries even when
    # the test was launched via ``.venv/bin/pytest`` without the venv being
    # activated.
    cli_bin = cli.parent
    existing_path = env.get("PATH", "")
    if str(cli_bin) not in existing_path.split(os.pathsep):
        env["PATH"] = f"{cli_bin}{os.pathsep}{existing_path}" if existing_path else str(cli_bin)

    if log_path is None:
        tmp = os.environ.get("TMPDIR", "/tmp")
        log_path = Path(tmp) / f"embed-server-test-{port}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1, encoding="utf-8", errors="replace")
    log_fh.write(f"# embed-server subprocess starting on {base_url}\n")
    log_fh.write(f"# cli = {cli}\n")
    log_fh.write(f"# env overrides: {env_overrides!r}\n\n")
    log_fh.flush()

    proc = subprocess.Popen(
        [str(cli), "-v", "serve", "--host", HOST, "--port", str(port)],
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )

    try:
        try:
            wait_for_http(f"{base_url}/health", timeout_s=boot_timeout_s)
        except Exception:
            # Make sure the log is flushed so the assertion message in the
            # test surfaces the boot failure context.
            log_fh.flush()
            sys.stderr.write(f"\n--- embed-server boot log ({log_path}) ---\n")
            sys.stderr.write(log_path.read_text(errors="replace"))
            sys.stderr.write("\n--- end embed-server boot log ---\n")
            raise

        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        log_fh.close()
        # Keep the log file by default — failure diagnosis is the main
        # reason these tests exist, and discarding logs on teardown means
        # any failure that needs the log requires re-running with logs
        # kept (doubling test time + GPU pressure). Operators / CI can
        # set MEMORYLAYER_EMBED_TEST_DISCARD_LOGS=1 to clean up.
        if os.environ.get("MEMORYLAYER_EMBED_TEST_DISCARD_LOGS") == "1":
            try:
                log_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        else:
            sys.stderr.write(f"\n[embed-server-test fixture] log kept at: {log_path}\n")
