"""Unit tests for the Unlimited-OCR transcription provider.

Unlimited-OCR only produces output when four things are true at once: the
user text is exactly ``<image>document parsing.``, the text block leads the
image block, ``skip_special_tokens`` is false, and the n-gram logits
processor gets its per-request args. Any one of them silently returns an
empty transcript on a real GPU, which makes them expensive to catch late —
so each is pinned here against a stubbed chat client, plus the inverse
(the grounded raw output is unwrapped back to markdown).

The subprocess is stubbed via ``_skip_subprocess = True`` and a fake
``AsyncOpenAI``-shaped client, so nothing here needs vLLM or a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from memorylayer_embed_server.services.transcription.base import strip_grounding_tokens
from memorylayer_embed_server.services.transcription.vllm_transcription import (
    UNLIMITED_OCR_NGRAM_SIZE,
    UNLIMITED_OCR_PROMPT,
    UNLIMITED_OCR_WINDOW_SIZE,
    VLLMTranscriptionProvider,
    build_deepseek_ocr_vllm_provider,
    build_unlimited_ocr_vllm_provider,
)

PAGE_BYTES = b"\x89PNG\r\n\x1a\nfake-page-bytes"


def _build(**overrides) -> VLLMTranscriptionProvider:
    defaults = dict(
        v=None,
        logger=None,
        model_name="baidu/Unlimited-OCR",
        max_tokens=4096,
        port=18012,
        gpu_memory_utilization=0.15,
        startup_timeout_sec=5.0,
        cmd="vllm",
    )
    defaults.update(overrides)
    provider = build_unlimited_ocr_vllm_provider(**defaults)
    provider._skip_subprocess = True
    return provider


def _stub_client(content: str, *, finish_reason: str = "stop") -> AsyncMock:
    """A minimal stand-in for ``AsyncOpenAI`` capturing the create() kwargs."""
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
    )
    create = AsyncMock(return_value=response)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


async def _transcribe(provider: VLLMTranscriptionProvider, client, **kwargs):
    provider._client = client
    provider._ready = True
    return await provider.transcribe_page(image_data=PAGE_BYTES, **kwargs)


# ---------------------------------------------------------------------------
# Serving recipe — vllm serve flags
# ---------------------------------------------------------------------------


def test_argv_carries_the_arch_specific_logits_processor():
    argv = _build()._runner.build_argv()
    assert "--logits_processors" in argv
    assert "vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor" in argv
    # Not the DeepSeek one — the module path is per-arch.
    assert "vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor" not in argv


def test_argv_disables_caches_that_ocr_cannot_reuse():
    argv = _build()._runner.build_argv()
    assert "--no-enable-prefix-caching" in argv
    assert argv[argv.index("--mm-processor-cache-gb") + 1] == "0"


def test_argv_trusts_remote_code():
    """The arch loads via trust_remote_code; the runner adds it for every role."""
    assert "--trust-remote-code" in _build()._runner.build_argv()


def test_provider_name_is_distinct_from_siblings():
    assert _build().PROVIDER_NAME == "unlimited-ocr"
    deepseek = build_deepseek_ocr_vllm_provider(
        v=None,
        logger=None,
        model_name="deepseek-ai/DeepSeek-OCR-2",
        max_tokens=4096,
        port=18011,
        gpu_memory_utilization=0.15,
        startup_timeout_sec=5.0,
        cmd="vllm",
    )
    assert deepseek.PROVIDER_NAME == "deepseek-ocr"


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_uses_the_fixed_prompt_with_text_before_image():
    provider = _build()
    client = _stub_client("hello")
    await _transcribe(provider, client, system_prompt="")

    kwargs = client.chat.completions.create.await_args.kwargs
    parts = kwargs["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert parts[0]["text"] == UNLIMITED_OCR_PROMPT
    assert parts[0]["text"].startswith("<image>")
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_caller_system_prompt_does_not_leak_into_the_prompt():
    """A cascade-supplied markdown system prompt would break this model."""
    provider = _build()
    client = _stub_client("hello")
    await _transcribe(provider, client, system_prompt="You are a document transcription assistant.")

    parts = client.chat.completions.create.await_args.kwargs["messages"][0]["content"]
    assert parts[0]["text"] == UNLIMITED_OCR_PROMPT


@pytest.mark.asyncio
async def test_request_pins_skip_special_tokens_and_ngram_xargs():
    provider = _build()
    client = _stub_client("hello")
    await _transcribe(provider, client, system_prompt="")

    extra_body = client.chat.completions.create.await_args.kwargs["extra_body"]
    assert extra_body["skip_special_tokens"] is False
    assert extra_body["vllm_xargs"] == {
        "ngram_size": UNLIMITED_OCR_NGRAM_SIZE,
        "window_size": UNLIMITED_OCR_WINDOW_SIZE,
    }


@pytest.mark.asyncio
async def test_window_size_override_reaches_the_request():
    """Multi-page / PDF input wants window_size=1024."""
    provider = _build(window_size=1024)
    client = _stub_client("hello")
    await _transcribe(provider, client, system_prompt="")

    extra_body = client.chat.completions.create.await_args.kwargs["extra_body"]
    assert extra_body["vllm_xargs"]["window_size"] == 1024


@pytest.mark.asyncio
async def test_generic_provider_keeps_image_first_and_sends_no_extra_body():
    """The Unlimited-OCR knobs must not change the chat-templated providers."""
    provider = build_deepseek_ocr_vllm_provider(
        v=None,
        logger=None,
        model_name="deepseek-ai/DeepSeek-OCR-2",
        max_tokens=4096,
        port=18011,
        gpu_memory_utilization=0.15,
        startup_timeout_sec=5.0,
        cmd="vllm",
    )
    provider._skip_subprocess = True
    client = _stub_client("hello")
    await _transcribe(provider, client, system_prompt="Be a transcriber.")

    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["messages"][0]["content"][0]["type"] == "image_url"
    assert "extra_body" not in kwargs


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grounded_output_is_unwrapped_to_markdown():
    raw = (
        "<|ref|># Quarterly Report<|/ref|><|det|>[[10, 12, 480, 44]]<|/det|>\n"
        "<|ref|>Revenue rose 12%.<|/ref|><|det|>[[10, 60, 512, 88]]<|/det|>"
    )
    provider = _build()
    attempt = await _transcribe(provider, _stub_client(raw), system_prompt="")

    assert attempt.success is True
    assert attempt.content == "# Quarterly Report\nRevenue rose 12%."
    assert "<|" not in attempt.content
    assert attempt.provider == "unlimited-ocr"
    assert attempt.tokens_in == 11
    assert attempt.tokens_out == 22


@pytest.mark.asyncio
async def test_output_that_is_only_grounding_markup_is_reported_as_empty():
    """A page of pure coordinate boxes is a failure, not a blank success."""
    provider = _build()
    attempt = await _transcribe(provider, _stub_client("<|det|>[[1, 2, 3, 4]]<|/det|>"), system_prompt="")

    assert attempt.success is False
    assert attempt.error == "Empty content after cleaning"


@pytest.mark.asyncio
async def test_token_limit_is_still_surfaced_as_a_cascade_failure():
    provider = _build()
    attempt = await _transcribe(
        provider,
        _stub_client("<|ref|>partial<|/ref|>", finish_reason="length"),
        system_prompt="",
    )

    assert attempt.success is False
    assert "Token limit reached" in attempt.error


# ---------------------------------------------------------------------------
# strip_grounding_tokens
# ---------------------------------------------------------------------------


def test_strip_grounding_tokens_keeps_referenced_text_drops_boxes():
    raw = "<|ref|>Title<|/ref|><|det|>[[0, 0, 1, 1]]<|/det|> tail"
    assert strip_grounding_tokens(raw) == "Title tail"


def test_strip_grounding_tokens_handles_multiline_spans():
    raw = "<|ref|>line one\nline two<|/ref|><|det|>[[0, 0, 1, 1]]<|/det|>"
    assert strip_grounding_tokens(raw) == "line one\nline two"


def test_strip_grounding_tokens_drops_unpaired_markers_from_truncated_output():
    """Output cut at max_tokens can end mid-span; don't leak the opener."""
    assert strip_grounding_tokens("<|ref|>Truncated heading") == "Truncated heading"


def test_strip_grounding_tokens_drops_residual_special_tokens():
    """skip_special_tokens=False leaves the EOS marker in the decoded text."""
    assert strip_grounding_tokens("<|ref|>Body<|/ref|><|end of sentence|>") == "Body"
    assert strip_grounding_tokens("<|ref|>Body<|/ref|><｜end▁of▁sentence｜>") == "Body"


def test_strip_grounding_tokens_leaves_ordinary_markdown_alone():
    md = "# Heading\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n`x <| y` stays"
    assert strip_grounding_tokens(md) == md
