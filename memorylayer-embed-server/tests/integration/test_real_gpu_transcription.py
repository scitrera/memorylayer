"""Real-GPU integration test for the transcription cascade.

Boots ``memorylayer-embed`` in a child process with the transcription
cascade enabled (GLM-OCR primary; DeepSeek-OCR-2 secondary; Gemini only
if ``google-genai`` is installed). Embedding services are mocked so the
subprocess only loads the OCR model — keeps GPU memory contained.

Drives a synthetic page image through ``POST /v1/transcribe`` and
asserts:

* the route returns a well-formed response with one result per page,
* at least one provider attempt was made (cascade was actually invoked),
* on success, the active provider is one of the real OCR models (i.e.
  the cascade reached a GPU model, not just a Gemini-API fallback).

Marked ``slow`` + ``integration``. Skip cleanly if ``transformers`` is
absent from the test venv.

The GLM-OCR model download is multi-GB on a cold HF cache — request
timeouts are intentionally large. Override the model via
``MEMORYLAYER_EMBED_GLM_OCR_MODEL`` for a faster alternative.
"""

from __future__ import annotations

import base64
import io
import os

import httpx
import pytest

# Real OCR requires transformers + torch in the test venv.
pytest.importorskip("transformers")
pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

from PIL import ImageDraw  # noqa: E402

from ._real_gpu_subprocess import embed_server_subprocess  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]


FIRST_REQUEST_TIMEOUT_S = float(os.environ.get("MEMORYLAYER_EMBED_TEST_REQUEST_TIMEOUT", "900"))


# ---------------------------------------------------------------------------
# Subprocess fixture: transcription cascade enabled, embeddings mocked.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_url() -> str:
    env = {
        # Mock embeddings so this subprocess doesn't compete with the
        # ColPali test for GPU memory and warm-up time.
        "MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS": "true",
        "MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED": "true",
        "MEMORYLAYER_EMBED_LLM_ENABLED": "false",
        "MEMORYLAYER_EMBED_PRELOAD_MODELS": "false",
        "MEMORYLAYER_EMBED_GLM_OCR_MODEL": os.environ.get(
            "MEMORYLAYER_EMBED_GLM_OCR_MODEL",
            "zai-org/GLM-OCR",
        ),
        # Constrain output so the first call finishes in a reasonable
        # window — full 16384 max_tokens is overkill for a 1-line image.
        "MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS": os.environ.get(
            "MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS",
            "512",
        ),
        # Test-only conservative GPU memory budget. GLM-OCR weights +
        # KV cache need ~18 GiB to boot; DeepSeek-OCR-2 is smaller.
        "MEMORYLAYER_EMBED_GLM_OCR_VLLM_GPU_MEM_UTIL": "0.15",
        "MEMORYLAYER_EMBED_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL": "0.1",
    }
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_page(text: str, size: tuple[int, int] = (640, 200)) -> bytes:
    """Render ``text`` onto a white page image as a PNG byte string.

    The default PIL bitmap font is small but readable to a vision-LLM —
    no TrueType dependency required. Background is white, foreground
    black so OCR contrast is unambiguous.
    """
    img = PIL_Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Wrap text manually so it fits the canvas (default font ~ 6x11 px).
    y = 20
    for line in text.split("\n"):
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 18
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------


def test_health_reports_transcription_available(base_url: str) -> None:
    r = httpx.get(f"{base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    transcription = body["services"]["transcription"]
    assert transcription["status"] == "available"
    assert transcription["providers"] >= 1, "expected at least one OCR provider"


# ---------------------------------------------------------------------------
# Transcription — drives the cascade on a synthetic page
# ---------------------------------------------------------------------------


def test_transcribe_synthetic_page(base_url: str) -> None:
    """End-to-end: the route accepts a page, runs OCR on the GPU, and
    returns one result. The first request triggers the model download +
    warm-up, hence the large timeout."""
    page_bytes = _synthetic_page(
        "MemoryLayer GPU transcription test.\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Apple orchards in Yakima."
    )
    payload = {
        "images": [base64.b64encode(page_bytes).decode()],
        # Keep the system prompt short so test runtime is dominated by
        # vision encoding, not text decoding.
        "system_prompt": "Transcribe the document to plain text. No commentary.",
        "max_tokens": 256,
    }

    r = httpx.post(f"{base_url}/v1/transcribe", json=payload, timeout=FIRST_REQUEST_TIMEOUT_S)
    assert r.status_code == 200, r.text
    body = r.json()

    # Shape assertions — these hold regardless of cascade outcome.
    assert len(body["results"]) == 1
    page = body["results"][0]
    assert page["page_index"] == 0
    assert page["attempts"], "cascade must record at least one attempt"

    # Aggregate stats are always populated.
    stats = body["stats"]
    assert stats["total_pages"] == 1
    assert stats["successful_pages"] + stats["failed_pages"] == 1
    assert stats["total_latency_ms"] > 0

    # The local-GPU OCR providers we expect the cascade to reach.
    # Provider names come from each class's PROVIDER_NAME (note: DeepSeek's
    # PROVIDER_NAME is "deepseek-ocr", not "deepseek-ocr-2").
    gpu_providers = {"glm-ocr", "deepseek-ocr"}
    attempted_providers = {a["provider"] for a in page["attempts"]}
    assert gpu_providers & attempted_providers, (
        f"cascade should attempt at least one GPU OCR provider, "
        f"got attempts: {attempted_providers}"
    )

    if page["success"]:
        # When the GPU model succeeded, we should see content and a
        # GPU-provider attribution.
        assert page["provider_used"] in gpu_providers, (
            f"unexpected provider_used={page['provider_used']!r}"
        )
        assert page["content"].strip(), "successful transcription must return content"
    else:
        # Soft failure: surface the per-provider errors so test output is
        # actionable. We still asserted the cascade *ran* above; failures
        # here typically indicate model-side issues (OOM, bad weights),
        # not a bug in the route.
        errors = [(a["provider"], a.get("error")) for a in page["attempts"]]
        pytest.fail(f"transcription failed for synthetic page; per-provider errors: {errors}")
