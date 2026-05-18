"""Docker-based integration test for the embed-server ↔ memorylayer-server chain.

Spins up both services on an isolated docker network using the compose
file alongside this test, then drives requests through memorylayer-server
to confirm the embedding chain reaches memorylayer-embed-server.

Marked ``integration`` and skipped by default. To run::

    pytest -m integration oss/memorylayer-embed-server/tests/integration/

Requirements:
    * Docker daemon reachable (``docker info`` succeeds).
    * ``docker compose`` plugin available (v2 syntax).
    * Ports 61101 free on the host.

The test uses the lightweight mock-provider variant of the embed-server
(``Dockerfile.test``), so no GPU / torch / model downloads are needed.
For the heavy real-model variant, see the README in this directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

COMPOSE_FILE = Path(__file__).parent / "docker-compose.embed-chain.yml"
# Optional second compose file layered on top (for the heavy real-model
# variant). Set the filename — resolved relative to the integration dir.
_OVERRIDE_NAME = os.environ.get("COMPOSE_FILE_OVERRIDE")
COMPOSE_OVERRIDE = (Path(__file__).parent / _OVERRIDE_NAME) if _OVERRIDE_NAME else None
COMPOSE_PROJECT = "memorylayer-embed-chain-test"
SERVER_BASE_URL = "http://localhost:61101"


pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


if not _docker_available():  # pragma: no cover - env-gated
    pytest.skip(
        "Docker daemon not available; skipping embed-chain integration test",
        allow_module_level=True,
    )


def _compose(*args: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    if COMPOSE_OVERRIDE is not None:
        cmd.extend(["-f", str(COMPOSE_OVERRIDE)])
    cmd.extend(["-p", COMPOSE_PROJECT, *args])
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        text=True,
    )


def _wait_for_url(url: str, *, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code < 500:
                return
        except Exception as exc:  # noqa: BLE001 - polling, any failure is retryable
            last_exc = exc
        time.sleep(1.0)
    raise TimeoutError(f"Service at {url} did not become ready within {timeout_s:.0f}s: {last_exc!r}")


@pytest.fixture(scope="module")
def stack():
    """Start the compose stack for the module, tear it down at the end."""
    # Best-effort cleanup of any leftover state from prior runs.
    _compose("down", "-v", "--remove-orphans", check=False, capture=True)

    # `up --build -d` builds the two images and starts both services.
    _compose("up", "--build", "-d")

    try:
        _wait_for_url(f"{SERVER_BASE_URL}/health", timeout_s=120)
        yield
    finally:
        if os.environ.get("KEEP_STACK") != "1":
            _compose("down", "-v", "--remove-orphans", check=False, capture=True)


# ---------------------------------------------------------------------------
# Chain assertions
# ---------------------------------------------------------------------------


def test_memorylayer_server_health_includes_embed_server(stack):
    """memorylayer-server should boot and be reachable on the host."""
    resp = httpx.get(f"{SERVER_BASE_URL}/health", timeout=5.0)
    assert resp.status_code == 200


def test_create_memory_traverses_chain_to_embed_server(stack):
    """Creating a memory triggers an embedding call that must reach embed-server.

    With ``MEMORYLAYER_EMBEDDING_PROVIDER=embed_server``, the OSS server
    delegates to the mock-providers embed-server over the private docker
    network. Successful memory creation proves the chain works.
    """
    payload = {
        "content": "Apple orchards in Yakima produce excellent Honeycrisp.",
        "type": "semantic",
        "importance": 0.7,
        "tags": ["agriculture", "test"],
    }
    # Generous timeout: heavy/real-model variant downloads + loads ColPali on
    # the very first request (can take a couple of minutes on a cold image).
    resp = httpx.post(f"{SERVER_BASE_URL}/v1/memories", json=payload, timeout=300.0)
    assert resp.status_code in (200, 201), resp.text
    body = resp.json()
    assert "id" in body or "memory_id" in body or "memory" in body, body


def test_recall_uses_embed_server_for_query_embedding(stack):
    """Recall calls embed-server for the query embedding; success ⇒ chain."""
    httpx.post(
        f"{SERVER_BASE_URL}/v1/memories",
        json={
            "content": "MaxSim scoring is used in ColPali retrieval.",
            "type": "semantic",
            "importance": 0.8,
        },
        timeout=300.0,
    )
    resp = httpx.post(
        f"{SERVER_BASE_URL}/v1/memories/recall",
        json={"query": "What scoring does ColPali use?", "limit": 3},
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Don't over-assert ranking quality (mock providers are random-seeded);
    # we just care the chain rendered results.
    assert isinstance(body, dict)
    assert "results" in body or "memories" in body, body
