"""Real-GPU integration tests for the ColPali multi-vector embedding path.

Boots ``memorylayer-embed`` in a child process on a free loopback port
with the real ColPali provider (``ModernVBERT/colmodernvbert`` — the
smallest supported model, MIT licensed) and exercises every embedding
HTTP endpoint:

* ``POST /v1/embeddings``            — mean-pooled single-vector from ColPali
* ``POST /v1/embeddings/multi``      — native ColPali multi-vector (late
                                       interaction)
* ``POST /v1/embeddings/images``     — both ``single`` and ``multi`` modes
* ``POST /v1/score``                 — MaxSim scoring

Marked ``slow`` + ``integration`` — opt in explicitly:

.. code-block:: shell

    cd oss/memorylayer-embed-server
    .venv/bin/pytest -m "slow and integration" tests/integration/test_real_gpu_embeddings.py

The first embedding request triggers ColPali weights download + warm-up,
so the per-request timeouts inside the tests are generous. Override the
ColPali model via ``MEMORYLAYER_EMBEDDING_COLPALI_MODEL`` to test a
different family (e.g. ColQwen2.5).
"""

from __future__ import annotations

import base64
import io
import os

import httpx
import numpy as np
import pytest

# Gate the whole module on real model deps and integration-test markers.
pytest.importorskip("colpali_engine")
pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

from ._real_gpu_subprocess import embed_server_subprocess  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# Each ColPali call goes through model load on the first request — keep
# this generous so flaky weight downloads do not kill the run.
FIRST_REQUEST_TIMEOUT_S = float(os.environ.get("MEMORYLAYER_EMBED_TEST_REQUEST_TIMEOUT", "600"))
WARM_REQUEST_TIMEOUT_S = 120.0


# ---------------------------------------------------------------------------
# Module fixture: one ColPali subprocess for every test in this module.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    """Spawn the embed-server with real ColPali wired for both modes.

    ``EMBED_SERVER_USE_MULTI_FOR_SINGLE=true`` means the ColPali provider
    fulfills /v1/embeddings (mean-pooled) AND /v1/embeddings/multi. That
    is the exact deployment shape we want to validate: one GPU, one
    model, two surfaces.
    """
    env = {
        # Wire ColPali to serve both single- and multi-vector requests.
        # Pin colpali_inprocess so this test keeps exercising the in-process
        # colpali-engine path even after the server default flips to vllm.
        "MEMORYLAYER_EMBED_MULTI_VECTOR_PROVIDER": "colpali_inprocess",
        "MEMORYLAYER_EMBED_USE_MULTI_FOR_SINGLE": "true",
        "MEMORYLAYER_EMBED_SINGLE_VECTOR_PROVIDER": "colpali",
        "MEMORYLAYER_EMBEDDING_COLPALI_MODEL": os.environ.get(
            "MEMORYLAYER_EMBEDDING_COLPALI_MODEL",
            "ModernVBERT/colmodernvbert",
        ),
        # Pin pool_factor=1 so the assertion that single-vec == mean(multi-vec)
        # holds — pooling rearranges tokens before the mean would be taken.
        "MEMORYLAYER_EMBEDDING_COLPALI_POOL_FACTOR": "1",
        "MEMORYLAYER_EMBEDDING_DEVICE": os.environ.get(
            "MEMORYLAYER_EMBEDDING_DEVICE",
            "cuda",
        ),
        # Lazy-load: keep startup fast and let the first request trigger
        # the (large) download + warm-up. Tests poll /health first then
        # accept long timeouts on the first embedding call.
        "MEMORYLAYER_EMBED_PRELOAD_MODELS": "false",
        # Disable services not exercised here to keep the subprocess
        # focused (less GPU memory, faster startup).
        "MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED": "false",
        "MEMORYLAYER_EMBED_LLM_ENABLED": "false",
    }
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _post(base_url: str, path: str, payload: dict, *, timeout: float) -> httpx.Response:
    return httpx.post(f"{base_url}{path}", json=payload, timeout=timeout)


def _synthetic_png(size: tuple[int, int] = (64, 64)) -> bytes:
    """Build a tiny RGB PNG entirely in memory; no fixtures on disk."""
    img = PIL_Image.new("RGB", size, color=(220, 200, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_data_url(size: tuple[int, int] = (64, 64)) -> str:
    """Encode a synthetic PNG as a ``data:image/png;base64,...`` URL.

    The embed-server's ``load_image_bytes`` helper has a heuristic that
    treats short bare-base64 strings as a file path. The data-URL form
    is unambiguous and matches how callers in production wrap images.
    """
    payload = base64.b64encode(_synthetic_png(size)).decode()
    return f"data:image/png;base64,{payload}"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_multi_vector_available(base_url: str) -> None:
    r = httpx.get(f"{base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["services"]["embedding"]["multi_vector"] == "available"
    # USE_MULTI_FOR_SINGLE=true means the same provider also serves single.
    assert body["services"]["embedding"]["single_vector"] == "available"


# ---------------------------------------------------------------------------
# Multi-vector — native ColPali output
# ---------------------------------------------------------------------------


def test_embeddings_multi_single_text(base_url: str) -> None:
    """First real GPU call — triggers ColPali model load + warm-up."""
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
    assert item["num_vectors"] > 1, "ColPali must emit multiple token vectors"
    assert len(item["vectors"]) == item["num_vectors"]
    # Per-vector dimension is ColPali's projection dim (128 for ModernVBert).
    assert all(len(v) == body["dimensions"] for v in item["vectors"])
    assert body["dimensions"] > 0


def test_embeddings_multi_batch(base_url: str) -> None:
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
# Single-vector — mean-pooled from ColPali multi-vector
# ---------------------------------------------------------------------------


def test_embeddings_single_mean_pooled(base_url: str) -> None:
    """ColPali's ``.embed()`` returns mean-of-multi-vector.

    With ``USE_MULTI_FOR_SINGLE=true`` the /v1/embeddings endpoint is
    served by ColPali via that mean-pool path — the response is one flat
    vector of dimension == ColPali's projection dim.
    """
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


def test_single_vector_equals_mean_of_multi_vector(base_url: str) -> None:
    """The /v1/embeddings (single) result must equal the mean of the
    multi-vector tokens — i.e. the two surfaces are consistent."""
    text = "apple orchards in Yakima"

    single = _post(
        base_url,
        "/v1/embeddings",
        {"input": text},
        timeout=WARM_REQUEST_TIMEOUT_S,
    ).json()
    multi = _post(
        base_url,
        "/v1/embeddings/multi",
        {"input": text, "input_type": "query"},
        timeout=WARM_REQUEST_TIMEOUT_S,
    ).json()

    single_vec = np.array(single["data"][0]["embedding"], dtype=np.float32)
    multi_vecs = np.array(multi["data"][0]["vectors"], dtype=np.float32)
    expected = multi_vecs.mean(axis=0)

    np.testing.assert_allclose(single_vec, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Image embeddings — single and multi modes
# ---------------------------------------------------------------------------


def test_image_embeddings_multi_mode(base_url: str) -> None:
    r = _post(
        base_url,
        "/v1/embeddings/images",
        {"images": [_png_data_url()], "mode": "multi"},
        timeout=FIRST_REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["num_vectors"] > 1
    assert all(len(v) == body["dimensions"] for v in item["vectors"])


def test_image_embeddings_single_mode(base_url: str) -> None:
    r = _post(
        base_url,
        "/v1/embeddings/images",
        {"images": [_png_data_url()], "mode": "single"},
        timeout=WARM_REQUEST_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 1
    vec = body["data"][0]["embedding"]
    assert isinstance(vec, list) and len(vec) > 0


# ---------------------------------------------------------------------------
# MaxSim score — self-vs-other sanity
# ---------------------------------------------------------------------------


def test_maxsim_score_self_beats_unrelated(base_url: str) -> None:
    """A query scored against itself outranks an unrelated document."""
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
    # Scores are sorted descending by the route; the top result must be
    # the self-document (index 0).
    assert body["scores"][0]["index"] == 0
    assert body["scores"][0]["score"] > body["scores"][1]["score"]
