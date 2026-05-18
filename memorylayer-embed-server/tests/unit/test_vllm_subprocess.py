"""Unit tests for the out-of-process vLLM provider.

We don't start a real ``vllm serve`` here — we stub the subprocess
+ httpx + the OpenAI async client so the tests stay fast and
hermetic. The behaviours we care about:

* Dispatcher routes ``EMBED_SERVER_SINGLE_VECTOR_PROVIDER=vllm_subprocess``
  to ``VLLMSubprocessEmbeddingProviderPlugin``.
* ``_ensure_started`` is idempotent (only one subprocess spawned even
  under concurrent first calls).
* ``embed`` / ``embed_batch`` round-trip through the OpenAI client.
* ``embed_image`` / ``embed_multimodal(text+image)`` POST a chat-style
  ``messages`` payload to ``/v1/embeddings`` (vLLM's VLM extension).
* ``shutdown`` is a no-op when no subprocess has been started.
"""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_embed_server.dependencies import _init_single_vector_provider
from memorylayer_embed_server.services.embedding.vllm_subprocess import (
    VLLMSubprocessEmbeddingProvider,
)


def _logger():
    return MagicMock(name="logger")


def _make_provider(**overrides) -> VLLMSubprocessEmbeddingProvider:
    """Provider with ``_skip_subprocess`` flipped so we never actually exec."""
    defaults = dict(
        v=None,
        model_name="test/mock",
        host="127.0.0.1",
        port=18099,
        startup_timeout_sec=5.0,
    )
    defaults.update(overrides)
    p = VLLMSubprocessEmbeddingProvider(**defaults)
    p._skip_subprocess = True
    return p


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_routes_vllm_subprocess(monkeypatch):
    sentinel = object()
    fake_mod = types.ModuleType(
        "memorylayer_embed_server.services.embedding.vllm_subprocess",
    )

    class _FakePlugin:
        def initialize(self, v, logger):
            return sentinel

    fake_mod.VLLMSubprocessEmbeddingProviderPlugin = _FakePlugin
    monkeypatch.setitem(
        sys.modules,
        "memorylayer_embed_server.services.embedding.vllm_subprocess",
        fake_mod,
    )

    provider = _init_single_vector_provider(v=None, logger=_logger(), kind="vllm_subprocess")
    assert provider is sentinel


def test_dispatcher_unknown_kind_message_mentions_vllm_subprocess():
    logger = _logger()
    _init_single_vector_provider(v=None, logger=logger, kind="not-a-provider")
    msg = " ".join(str(c) for c in logger.warning.call_args_list)
    assert "vllm_subprocess" in msg


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_ensure_started_creates_one_client_under_concurrency(monkeypatch):
    """Two parallel first calls produce one OpenAI client."""
    p = _make_provider()
    fake_client = MagicMock(name="AsyncOpenAI")
    fake_ctor = MagicMock(return_value=fake_client)

    fake_openai_mod = types.ModuleType("openai")
    fake_openai_mod.AsyncOpenAI = fake_ctor
    monkeypatch.setitem(sys.modules, "openai", fake_openai_mod)

    # Two callers race the start lock — exactly one should construct the client.
    await asyncio.gather(p._ensure_started(), p._ensure_started())
    assert fake_ctor.call_count == 1
    assert p._client is fake_client


async def test_shutdown_is_safe_without_subprocess():
    p = _make_provider()
    # Never started — shutdown should not blow up.
    await p.shutdown()


# ---------------------------------------------------------------------------
# Embedding round-trip via mocked OpenAI client
# ---------------------------------------------------------------------------


def _make_openai_response(vectors: list[list[float]]):
    """Mimic the shape of openai.types.CreateEmbeddingResponse."""
    return MagicMock(
        data=[MagicMock(embedding=v) for v in vectors],
    )


async def test_embed_calls_openai_with_single_input(monkeypatch):
    p = _make_provider()
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=_make_openai_response([[0.1, 0.2, 0.3]]))
    p._client = fake_client
    p._ready = True

    result = await p.embed("hello")

    assert result == [0.1, 0.2, 0.3]
    fake_client.embeddings.create.assert_awaited_once_with(input=["hello"], model="test/mock")


async def test_embed_batch_preserves_order():
    p = _make_provider()
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(
        return_value=_make_openai_response([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    )
    p._client = fake_client
    p._ready = True

    out = await p.embed_batch(["a", "b", "c"])
    assert out == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    fake_client.embeddings.create.assert_awaited_once_with(input=["a", "b", "c"], model="test/mock")


async def test_embed_batch_empty_short_circuits():
    p = _make_provider()
    p._client = MagicMock()
    p._client.embeddings.create = AsyncMock()
    p._ready = True
    assert await p.embed_batch([]) == []
    p._client.embeddings.create.assert_not_called()


async def test_embed_respects_output_dimensions_truncation():
    p = _make_provider(output_dimensions=2)
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=_make_openai_response([[0.1, 0.2, 0.3, 0.4]]))
    p._client = fake_client
    p._ready = True
    out = await p.embed("hi")
    assert out == [0.1, 0.2]


# ---------------------------------------------------------------------------
# Multimodal — explicitly unsupported
# ---------------------------------------------------------------------------


class _StubMultimodalHttp:
    """Stand-in for ``httpx.AsyncClient`` recording the multimodal POSTs."""

    def __init__(self, payload_embedding: list[float]):
        self.payload_embedding = payload_embedding
        self.calls: list[tuple[str, dict]] = []

    async def post(self, path, json):
        self.calls.append((path, json))
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={
            "data": [{"embedding": list(self.payload_embedding), "index": 0}],
        })
        return resp

    async def aclose(self):
        pass


async def test_embed_image_posts_chat_messages():
    """Image-only call → chat messages payload with one image_url part."""
    p = _make_provider()
    p._mm_http = _StubMultimodalHttp(payload_embedding=[0.1, 0.2, 0.3])
    p._ready = True  # skip subprocess startup

    out = await p.embed_image(b"\x89PNG\r\n\x1a\n")

    assert out == [0.1, 0.2, 0.3]
    assert len(p._mm_http.calls) == 1
    path, payload = p._mm_http.calls[0]
    assert path == "/embeddings"
    assert payload["model"] == "test/mock"
    messages = payload["messages"]
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Image-only: exactly one content part, of type image_url, with data: URL.
    assert len(content) == 1
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_embed_multimodal_text_only_delegates_to_embed():
    """Text-only multimodal call routes through the cheap /embeddings input path."""
    p = _make_provider()
    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=_make_openai_response([[0.7, 0.8]]))
    p._client = fake_client
    p._ready = True

    result = await p.embed_multimodal(text="just text, no image")
    assert result == [0.7, 0.8]


async def test_embed_multimodal_with_image_posts_chat_messages_with_both_parts():
    """text + image → messages payload with both content parts in order."""
    p = _make_provider()
    p._mm_http = _StubMultimodalHttp(payload_embedding=[0.9, 1.0])
    p._ready = True

    out = await p.embed_multimodal(text="describe this", image=b"img-bytes")

    assert out == [0.9, 1.0]
    payload = p._mm_http.calls[0][1]
    content = payload["messages"][0]["content"]
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_embed_multimodal_requires_one_of_text_or_image():
    p = _make_provider()
    with pytest.raises(ValueError):
        await p.embed_multimodal()


def test_image_to_data_url_accepts_existing_data_url():
    """A pre-encoded data URL passes through unchanged."""
    same = "data:image/png;base64,AAAA"
    from memorylayer_embed_server.services.embedding.vllm_subprocess import (
        VLLMSubprocessEmbeddingProvider as _P,
    )
    assert _P._image_to_data_url(same) == same


def test_image_to_data_url_passes_through_http_url():
    """HTTP/HTTPS URLs are forwarded so vllm can fetch them server-side."""
    from memorylayer_embed_server.services.embedding.vllm_subprocess import (
        VLLMSubprocessEmbeddingProvider as _P,
    )
    url = "https://example.com/cat.png"
    assert _P._image_to_data_url(url) == url


# ---------------------------------------------------------------------------
# Argv construction
# ---------------------------------------------------------------------------


def test_argv_contains_modern_pooling_flags():
    p = _make_provider(model_name="Qwen/Qwen3-Embedding-0.6B")
    argv = p._build_vllm_argv()
    # vllm serve <model>
    assert argv[0:3] == ["vllm", "serve", "Qwen/Qwen3-Embedding-0.6B"]
    assert "--runner" in argv and argv[argv.index("--runner") + 1] == "pooling"
    assert "--convert" in argv and argv[argv.index("--convert") + 1] == "embed"
    assert "--trust-remote-code" in argv


def test_argv_includes_enforce_eager_when_set():
    p = _make_provider(enforce_eager=True)
    assert "--enforce-eager" in p._build_vllm_argv()


def test_argv_excludes_enforce_eager_by_default():
    p = _make_provider(enforce_eager=False)
    assert "--enforce-eager" not in p._build_vllm_argv()
