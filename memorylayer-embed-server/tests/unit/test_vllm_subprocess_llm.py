"""Unit tests for :class:`VLLMSubprocessLLMProvider`.

We stub the subprocess via ``_skip_subprocess = True`` and inject a fake
``httpx.AsyncClient`` so we can exercise the proxy paths (non-streaming
JSON and streaming SSE) without launching a real vLLM child.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from memorylayer_embed_server.services.llm.vllm_subprocess import (
    VLLMSubprocessLLMProvider,
)


def _provider(**overrides) -> VLLMSubprocessLLMProvider:
    defaults = dict(
        v=None,
        profile_name="qwen",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        host="127.0.0.1",
        port=18101,
        startup_timeout_sec=5.0,
        aliases=["qwen-7b"],
    )
    defaults.update(overrides)
    p = VLLMSubprocessLLMProvider(**defaults)
    p._skip_subprocess = True
    return p


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_served_names_dedup_and_preserve_order():
    p = _provider(aliases=["qwen-7b", "qwen", "Qwen/Qwen2.5-7B-Instruct"])
    # Profile name first, then model, then aliases, no duplicates.
    assert p.served_names[0] == "qwen"
    assert "qwen-7b" in p.served_names
    assert "Qwen/Qwen2.5-7B-Instruct" in p.served_names
    assert len(p.served_names) == len(set(p.served_names))


def test_argv_is_llm_role_no_pooling_flags():
    p = _provider()
    argv = p._runner.build_argv()
    assert "--runner" not in argv
    assert "--convert" not in argv
    assert "--served-model-name" in argv


def test_list_models_returns_one_entry_per_served_name():
    p = _provider(aliases=["qwen-7b"])
    models = p.list_models()
    ids = [m["id"] for m in models]
    assert "qwen" in ids
    assert "qwen-7b" in ids
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    for m in models:
        assert m["profile"] == "qwen"
        assert m["model"] == "Qwen/Qwen2.5-7B-Instruct"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class _StubHttp:
    """Minimal stand-in for httpx.AsyncClient supporting both ``post`` and ``stream``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.json_response: dict = {"id": "ok"}
        self.stream_chunks: list[bytes] = []
        self.aclose_called = False

    async def post(self, path, json):
        self.calls.append(("post", {"path": path, "json": json}))
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=self.json_response)
        return resp

    def stream(self, method, path, *, json):
        self.calls.append((method.lower() + "-stream", {"path": path, "json": json}))
        chunks = list(self.stream_chunks)

        class _Ctx:
            async def __aenter__(_self):  # noqa: N805
                resp = MagicMock()
                resp.raise_for_status = MagicMock()

                async def _aiter() -> AsyncIterator[bytes]:
                    for c in chunks:
                        yield c

                resp.aiter_bytes = lambda: _aiter()
                return resp

            async def __aexit__(_self, exc_type, exc, tb):  # noqa: N805
                return False

        return _Ctx()

    async def aclose(self):
        self.aclose_called = True


@pytest.mark.asyncio
async def test_ensure_started_lazy_and_idempotent(monkeypatch):
    """Two parallel first calls only construct one httpx client."""
    p = _provider()
    constructed = []

    class _Captor:
        def __init__(self, base_url, timeout):
            constructed.append((base_url, timeout))
            self._stub = _StubHttp()

        async def post(self, path, json):
            return await self._stub.post(path, json)

        def stream(self, method, path, *, json):
            return self._stub.stream(method, path, json=json)

        async def aclose(self):
            await self._stub.aclose()

    # Patch httpx.AsyncClient that the provider lazily imports.
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Captor)

    await asyncio.gather(p._ensure_started(), p._ensure_started())
    assert len(constructed) == 1
    assert constructed[0][0] == "http://127.0.0.1:18101/v1"


@pytest.mark.asyncio
async def test_shutdown_safe_without_subprocess():
    p = _provider()
    # Never started — shutdown should not blow up.
    await p.shutdown()


# ---------------------------------------------------------------------------
# Proxy: non-streaming and streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_non_streaming_forwards_payload():
    p = _provider()
    stub = _StubHttp()
    stub.json_response = {"id": "chatcmpl-1", "choices": []}
    p._http = stub
    p._ready = True  # skip the full _ensure_started path

    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    result = await p.chat_completions(payload, stream=False)
    assert result == {"id": "chatcmpl-1", "choices": []}
    method, args = stub.calls[0]
    assert method == "post"
    assert args["path"] == "/chat/completions"
    assert args["json"]["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_chat_completions_streaming_yields_chunks():
    p = _provider()
    stub = _StubHttp()
    stub.stream_chunks = [b'data: {"a":1}\n\n', b"data: [DONE]\n\n"]
    p._http = stub
    p._ready = True

    aiter_result = await p.chat_completions({"model": "qwen"}, stream=True)
    received: list[bytes] = []
    async for chunk in aiter_result:
        received.append(chunk)
    assert received == [b'data: {"a":1}\n\n', b"data: [DONE]\n\n"]
    method, args = stub.calls[0]
    assert method == "post-stream"
    # stream=True should have been merged into the upstream payload.
    assert args["json"]["stream"] is True


@pytest.mark.asyncio
async def test_completions_routes_to_completions_endpoint():
    p = _provider()
    stub = _StubHttp()
    p._http = stub
    p._ready = True

    await p.completions({"model": "qwen", "prompt": "hi"}, stream=False)
    method, args = stub.calls[0]
    assert method == "post"
    assert args["path"] == "/completions"


@pytest.mark.asyncio
async def test_in_flight_increments_during_request():
    p = _provider()
    stub = _StubHttp()
    p._http = stub
    p._ready = True

    assert p._in_flight == 0
    await p.chat_completions({"model": "qwen"}, stream=False)
    # After completion, counter is decremented.
    assert p._in_flight == 0


@pytest.mark.asyncio
async def test_load_snapshot_reports_in_flight():
    p = _provider()
    p._in_flight = 3
    snap = p.get_load_snapshot()
    assert snap["in_flight"] == 3
    assert "utilization" in snap


# ---------------------------------------------------------------------------
# build_provider_from_env
# ---------------------------------------------------------------------------


class _FakeVars:
    """Mimic ``scitrera_app_framework.Variables.environ`` and ``.get`` lookups."""

    def __init__(self, env: dict[str, str]):
        self._env = env

    def environ(self, key, *, default=None, type_fn=None):
        val = self._env.get(key, default)
        if type_fn is not None and val is not None and val is not default:
            try:
                return type_fn(val)
            except Exception:
                return default
        return val

    def get(self, key, default=None):
        # ``get_logger`` consults Variables.get for the main logger key —
        # returning None tells the framework to fall back to a default logger.
        return default


def test_build_provider_from_env_reads_per_profile_fields():
    from memorylayer_embed_server.services.llm.vllm_subprocess import build_provider_from_env

    env = {
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_ALIASES": "qwen-7b, qwen2.5",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_DTYPE": "bfloat16",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_MAX_MODEL_LEN": "16384",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_GPU_MEM_UTIL": "0.4",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_ENFORCE_EAGER": "true",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_TENSOR_PARALLEL_SIZE": "2",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_EXTRA_ARGS": "--quantization fp8",
        "MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_CMD": "/usr/local/bin/vllm",
    }
    provider = build_provider_from_env(
        _FakeVars(env),
        MagicMock(),
        profile_name="qwen",
        port=18101,
    )
    assert provider.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert provider.aliases == ["qwen-7b", "qwen2.5"]
    argv = provider._runner.build_argv()
    assert "--dtype" in argv and argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--max-model-len" in argv and argv[argv.index("--max-model-len") + 1] == "16384"
    assert "--gpu-memory-utilization" in argv
    assert "--tensor-parallel-size" in argv and argv[argv.index("--tensor-parallel-size") + 1] == "2"
    assert "--enforce-eager" in argv
    assert "--quantization" in argv and argv[argv.index("--quantization") + 1] == "fp8"
    assert argv[0] == "/usr/local/bin/vllm"


def test_build_provider_from_env_missing_model_raises():
    from memorylayer_embed_server.services.llm.vllm_subprocess import build_provider_from_env

    with pytest.raises(RuntimeError, match="MEMORYLAYER_EMBED_LLM_PROFILE_QWEN_MODEL"):
        build_provider_from_env(
            _FakeVars({}),
            MagicMock(),
            profile_name="qwen",
            port=18101,
        )
