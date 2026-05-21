"""Per-provider transcription tests + cascade fall-through.

The existing ``test_real_gpu_transcription.py`` exercises the cascade as a
whole and lets GLM-OCR (the primary) handle the request. This module
isolates each provider so we get coverage of the DeepSeek-OCR-2 and
Gemini code paths individually, plus a fall-through test that verifies
the cascade actually moves on when the primary fails.

Each isolated test spawns its own embed-server subprocess with all other
providers disabled via the ``MEMORYLAYER_EMBED_{PROVIDER}_ENABLED`` env
vars. That guarantees the cascade has exactly one configured member,
so ``provider_used`` in the response is unambiguously the one under
test.

Marked ``slow`` + ``integration``. The Gemini-backed tests skip cleanly
when no ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` is present — that's a
real missing dependency, not a preference. Everything else runs by
default so ``pytest -m "slow and integration"`` exercises the whole
matrix hands-off.
"""

from __future__ import annotations

import base64
import io
import os

import httpx
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")

from PIL import ImageDraw  # noqa: E402

from ._real_gpu_subprocess import embed_server_subprocess  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]


FIRST_REQUEST_TIMEOUT_S = float(os.environ.get("MEMORYLAYER_EMBED_TEST_REQUEST_TIMEOUT", "900"))


# ---------------------------------------------------------------------------
# Helpers (kept inline; mirror test_real_gpu_transcription.py for clarity)
# ---------------------------------------------------------------------------


def _synthetic_page(text: str, size: tuple[int, int] = (640, 200)) -> bytes:
    img = PIL_Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in text.split("\n"):
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 18
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _post_transcribe(base_url: str, *, system_prompt: str | None = None, max_tokens: int = 256) -> dict:
    page_bytes = _synthetic_page(
        "MemoryLayer per-provider transcription test.\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Apple orchards in Yakima."
    )
    payload: dict = {"images": [base64.b64encode(page_bytes).decode()], "max_tokens": max_tokens}
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    r = httpx.post(f"{base_url}/v1/transcribe", json=payload, timeout=FIRST_REQUEST_TIMEOUT_S)
    assert r.status_code == 200, r.text
    return r.json()


def _common_env_only_enable(*, glm: bool, deepseek: bool, gemini: bool) -> dict[str, str]:
    """Build the env override map that pins exactly one cascade provider.

    The vLLM GPU memory cap is fixed at 0.1 in the test harness. On a
    DGX-Spark-class 120 GiB unified-memory chip that's ~12 GiB —
    sufficient for both GLM-OCR (~9 GiB) and DeepSeek-OCR-2 — and on
    a typical 24 GiB inference card that maps to ~50% utilization,
    which is the production sweet spot when OCR shares the GPU with
    the embedding + multi-vec subprocesses.
    """
    return {
        # Mock embeddings so this subprocess only loads OCR models.
        "MEMORYLAYER_EMBED_USE_MOCK_PROVIDERS": "true",
        "MEMORYLAYER_EMBED_TRANSCRIPTION_ENABLED": "true",
        "MEMORYLAYER_EMBED_LLM_ENABLED": "false",
        "MEMORYLAYER_EMBED_PRELOAD_MODELS": "false",
        "MEMORYLAYER_EMBED_GLM_OCR_ENABLED": "true" if glm else "false",
        "MEMORYLAYER_EMBED_DEEPSEEK_OCR_ENABLED": "true" if deepseek else "false",
        "MEMORYLAYER_EMBED_GEMINI_ENABLED": "true" if gemini else "false",
        # Cap tokens so each test finishes quickly.
        "MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS": os.environ.get("MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS", "256"),
        "MEMORYLAYER_EMBED_DEEPSEEK_OCR_MAX_TOKENS": os.environ.get("MEMORYLAYER_EMBED_DEEPSEEK_OCR_MAX_TOKENS", "256"),
        "MEMORYLAYER_EMBED_GEMINI_MAX_TOKENS": os.environ.get("MEMORYLAYER_EMBED_GEMINI_MAX_TOKENS", "256"),
        # Test-only GPU budget. GLM-OCR boots at 0.15 (~18 GiB);
        # DeepSeek-OCR-2 needs 0.2 (~24 GiB) — its engine-init phase
        # (logits-processor warmup + KV-cache allocation) is more
        # memory-greedy than the weight size alone implies.
        "MEMORYLAYER_EMBED_GLM_OCR_VLLM_GPU_MEM_UTIL": "0.15",
        "MEMORYLAYER_EMBED_DEEPSEEK_OCR_VLLM_GPU_MEM_UTIL": "0.2",
    }


# ---------------------------------------------------------------------------
# GLM-OCR only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def glm_only_base_url() -> str:
    env = _common_env_only_enable(glm=True, deepseek=False, gemini=False)
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


def test_glm_ocr_only_health_lists_single_provider(glm_only_base_url: str) -> None:
    r = httpx.get(f"{glm_only_base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["services"]["transcription"]["providers"] == 1


def test_glm_ocr_only_transcribes_synthetic_page(glm_only_base_url: str) -> None:
    body = _post_transcribe(glm_only_base_url)
    page = body["results"][0]
    assert page["attempts"], "must record at least one attempt"
    assert {a["provider"] for a in page["attempts"]} == {"glm-ocr"}, (
        f"isolated-GLM-OCR test must hit only the glm-ocr provider, got "
        f"{{a['provider'] for a in page['attempts']}}"
    )
    if page["success"]:
        assert page["provider_used"] == "glm-ocr"
        assert page["content"].strip()
    else:
        errors = [(a["provider"], a.get("error")) for a in page["attempts"]]
        pytest.fail(f"GLM-OCR alone failed; attempts: {errors}")


# ---------------------------------------------------------------------------
# DeepSeek-OCR-2 only
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deepseek_only_base_url() -> str:
    env = _common_env_only_enable(glm=False, deepseek=True, gemini=False)
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


def test_deepseek_ocr_only_health_lists_single_provider(deepseek_only_base_url: str) -> None:
    r = httpx.get(f"{deepseek_only_base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["services"]["transcription"]["providers"] == 1


def test_deepseek_ocr_only_transcribes_synthetic_page(deepseek_only_base_url: str) -> None:
    body = _post_transcribe(deepseek_only_base_url)
    page = body["results"][0]
    assert page["attempts"], "must record at least one attempt"
    assert {a["provider"] for a in page["attempts"]} == {"deepseek-ocr"}
    if page["success"]:
        assert page["provider_used"] == "deepseek-ocr"
        assert page["content"].strip()
    else:
        errors = [(a["provider"], a.get("error")) for a in page["attempts"]]
        pytest.fail(f"DeepSeek-OCR-2 alone failed; attempts: {errors}")


# ---------------------------------------------------------------------------
# Gemini only — needs GOOGLE_API_KEY
# ---------------------------------------------------------------------------


_HAS_GOOGLE_API_KEY = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


@pytest.fixture(scope="module")
def gemini_only_base_url() -> str:
    if not _HAS_GOOGLE_API_KEY:
        pytest.skip("Gemini test requires GOOGLE_API_KEY (or GEMINI_API_KEY) in the environment.")
    pytest.importorskip("google.genai")
    env = _common_env_only_enable(glm=False, deepseek=False, gemini=True)
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


def test_gemini_only_health_lists_single_provider(gemini_only_base_url: str) -> None:
    r = httpx.get(f"{gemini_only_base_url}/health/ready", timeout=10.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["services"]["transcription"]["providers"] == 1


def test_gemini_only_transcribes_synthetic_page(gemini_only_base_url: str) -> None:
    body = _post_transcribe(gemini_only_base_url)
    page = body["results"][0]
    assert page["attempts"], "must record at least one attempt"
    assert {a["provider"] for a in page["attempts"]} == {"gemini"}
    if page["success"]:
        assert page["provider_used"] == "gemini"
        assert page["content"].strip()
    else:
        errors = [(a["provider"], a.get("error")) for a in page["attempts"]]
        pytest.fail(f"Gemini alone failed; attempts: {errors}")


# ---------------------------------------------------------------------------
# Cascade fall-through — GLM-OCR fails (token cap) → next provider runs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cascade_fallthrough_base_url() -> str:
    """GLM-OCR + Gemini both enabled, GLM-OCR forced to fail via max_tokens=1."""
    if not _HAS_GOOGLE_API_KEY:
        pytest.skip(
            "Cascade fall-through test needs Gemini as the second provider; "
            "set GOOGLE_API_KEY to run it."
        )
    pytest.importorskip("google.genai")
    env = _common_env_only_enable(glm=True, deepseek=False, gemini=True)
    # Force GLM-OCR to hit its token cap immediately so the cascade has to
    # fall through to Gemini. Gemini gets a generous cap.
    env["MEMORYLAYER_EMBED_GLM_OCR_MAX_TOKENS"] = "1"
    env["MEMORYLAYER_EMBED_GEMINI_MAX_TOKENS"] = "512"
    with embed_server_subprocess(env_overrides=env) as url:
        yield url


def test_cascade_falls_through_when_primary_fails(cascade_fallthrough_base_url: str) -> None:
    body = _post_transcribe(cascade_fallthrough_base_url, max_tokens=None)
    page = body["results"][0]
    providers_attempted = [a["provider"] for a in page["attempts"]]
    # GLM-OCR must have been tried first and failed.
    assert providers_attempted[0] == "glm-ocr"
    glm_attempt = page["attempts"][0]
    assert glm_attempt["success"] is False
    # The cap should produce a length finish (the value is heuristic in
    # the provider; accept either an explicit length signal or any error).
    assert glm_attempt["finish_reason"] in {"length", "max_tokens"} or glm_attempt["error"]
    # And the cascade must have continued on to Gemini.
    assert "gemini" in providers_attempted
    assert page["success"], f"expected gemini fallback to succeed; attempts: {page['attempts']}"
    assert page["provider_used"] == "gemini"
