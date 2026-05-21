"""Real-GPU integration tests for the vLLM-backed multi-vector path.

Boots ``memorylayer-embed`` with ``EMBED_SERVER_MULTI_VECTOR_PROVIDER=vllm_subprocess``
and ``ModernVBERT/colmodernvbert-merged`` (~1 GB unquantized, the merged
checkpoint vLLM can load — the LoRA-adapter ``colmodernvbert`` repo can't
be served by vLLM). The embed-server in turn spawns a child
``vllm serve --runner pooling`` subprocess with the
``ColModernVBertForRetrieval`` architecture override, then this test
drives every multi-vector HTTP surface end-to-end:

* ``POST /v1/embeddings/multi``       — text → multi-vector via /pooling
* ``POST /v1/embeddings`` (mean-pool) — single vector compat shim
* ``POST /v1/score``                  — MaxSim self-vs-other

Marked ``slow`` + ``integration``. Opt in explicitly:

.. code-block:: shell

    cd oss/memorylayer-embed-server
    .venv/bin/pytest -m "slow and integration" \\
        tests/integration/test_real_gpu_vllm_multivector.py -v

vLLM cold-start can take a couple of minutes on a fresh image (engine
init + CUDA graph compile, plus model download on cold HF cache). The
boot timeout in the fixture accounts for that.
"""

from __future__ import annotations

import os

import httpx
import pytest

# Heavy deps — gate cleanly.
pytest.importorskip("vllm")
pytest.importorskip("colpali_engine")  # for the client-side pooler
pytest.importorskip("torch")

from ._real_gpu_subprocess import embed_server_subprocess  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]


FIRST_REQUEST_TIMEOUT_S = float(os.environ.get("MEMORYLAYER_EMBED_TEST_REQUEST_TIMEOUT", "600"))
WARM_REQUEST_TIMEOUT_S = 60.0
VLLM_BOOT_TIMEOUT_S = float(os.environ.get("MEMORYLAYER_EMBED_TEST_BOOT_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# Module fixture: one vLLM-backed embed-server for every test in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    """Spawn embed-server with the vLLM multi-vector provider selected.

    The embed-server's startup will in turn fork ``vllm serve --runner
    pooling`` for ``ModernVBERT/colmodernvbert-merged``; that double-
    subprocess cold-start is what the long boot timeout pays for.
    """
    env = {
        "MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER": "vllm_subprocess",
        # Use the merged ColModernVBert checkpoint (~1 GB unquantized).
        # The default ``ModernVBERT/colmodernvbert`` is a LoRA adapter
        # vLLM can't load; the provider auto-upgrades it but be explicit.
        "MEMORYLAYER_EMBEDDING_COLPALI_MODEL": "ModernVBERT/colmodernvbert-merged",
        # The merged repo's config.json doesn't set ``architectures``;
        # tell vLLM how to route the model.
        "MEMORYLAYER_EMBEDDING_VLLM_MV_ARCHITECTURES": "ColModernVBertForRetrieval",
        # Reuse the multi-vec provider as single-vec via mean-pool so we
        # can exercise /v1/embeddings against the same subprocess and
        # avoid needing a second vLLM profile in the test.
        "MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE": "true",
        # Keep the test honest about per-token pooling — leave it OFF so
        # we can verify the raw multi-vector shape comes out of vLLM
        # before pooling factors enter the picture. A separate test below
        # turns it on and asserts the count drops.
        "MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR": "1",
        # Cap vLLM's GPU appetite so it cohabits the GPU with whatever
        # else is running (the existing tests + any sidecar). The runner
        # forwards this via --gpu-memory-utilization.
        # Test-only conservative GPU memory budget — 0.1 of 120 GiB unified
        # memory ≈ 12 GiB, leaves headroom for other test subprocesses to
        # coexist on the same GPU.
        "MEMORYLAYER_EMBEDDING_VLLM_GPU_MEM_UTIL": "0.1",
        # Disable services we are not exercising in this module.
        "MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED": "false",
        "MEMORYLAYER_EMBED_LLM_ENABLED": "false",
        # Lazy-load: model load + vLLM warm-up happens during /health/ready,
        # not embed-server /health.
        "MEMORYLAYER_EMBED_PRELOAD_MODELS": "false",
    }
    with embed_server_subprocess(env_overrides=env, boot_timeout_s=VLLM_BOOT_TIMEOUT_S) as url:
        yield url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(base_url: str, path: str, payload: dict, *, timeout: float) -> httpx.Response:
    return httpx.post(f"{base_url}{path}", json=payload, timeout=timeout)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_multi_vector_via_vllm(base_url: str) -> None:
    r = httpx.get(f"{base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["services"]["embedding"]["multi_vector"] == "available"
    # USE_MULTI_FOR_SINGLE=true reuses the multi-vec provider for /v1/embeddings.
    assert body["services"]["embedding"]["single_vector"] == "available"


# ---------------------------------------------------------------------------
# Multi-vector text — the headline path
# ---------------------------------------------------------------------------


def test_vllm_embeddings_multi_single_text(base_url: str) -> None:
    """First real request — covers ColPali vLLM cold-start + /pooling parse."""
    r = _post(
        base_url,
        "/v1/embeddings/multi",
        {"input": "apple orchards in Yakima", "input_type": "query"},
        timeout=FIRST_REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["index"] == 0
    assert item["num_vectors"] > 1, "ColPali via vLLM must emit multiple token vectors"
    assert len(item["vectors"]) == item["num_vectors"]
    assert all(len(v) == body["dimensions"] for v in item["vectors"])
    assert body["dimensions"] > 0


def test_vllm_embeddings_multi_batch(base_url: str) -> None:
    r = _post(
        base_url,
        "/v1/embeddings/multi",
        {
            "input": [
                "the quick brown fox",
                "MaxSim late interaction scoring",
                "apple orchards",
            ],
            "input_type": "document",
        },
        timeout=WARM_REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 3
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    for item in body["data"]:
        assert item["num_vectors"] >= 1
        assert all(len(v) == body["dimensions"] for v in item["vectors"])


# ---------------------------------------------------------------------------
# MaxSim — self-vs-other sanity (proves projection head is preserved)
# ---------------------------------------------------------------------------


def test_vllm_maxsim_score_self_beats_unrelated(base_url: str) -> None:
    multi = _post(
        base_url,
        "/v1/embeddings/multi",
        {
            "input": [
                "apple orchards in Yakima",
                "apple orchards in Yakima",
                "quantum chromodynamics lecture notes",
            ],
            "input_type": "document",
        },
        timeout=WARM_REQUEST_TIMEOUT_S,
    ).json()

    q_vecs = multi["data"][0]["vectors"]
    d_same = multi["data"][1]["vectors"]
    d_other = multi["data"][2]["vectors"]

    r = _post(
        base_url,
        "/v1/score",
        {
            "query_vectors": q_vecs,
            "document_vectors": [d_same, d_other],
        },
        timeout=30.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scores"][0]["index"] == 0
    assert body["scores"][0]["score"] > body["scores"][1]["score"]


# ---------------------------------------------------------------------------
# Single-vector compat shim (mean-pool of multi-vector)
# ---------------------------------------------------------------------------


def test_vllm_embeddings_single_mean_pooled(base_url: str) -> None:
    r = _post(
        base_url,
        "/v1/embeddings",
        {"input": "apple orchards in Yakima"},
        timeout=WARM_REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 1
    vec = body["data"][0]["embedding"]
    assert isinstance(vec, list) and len(vec) > 0
    assert all(isinstance(x, float) for x in vec)
