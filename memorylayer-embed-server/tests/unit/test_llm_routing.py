"""Unit tests for ``LLMRoutingService`` and the ``/v1/chat/completions`` route.

The router is a pure-Python alias map + lifecycle fanout, so it tests
without spinning up real httpx clients or subprocesses. The route tests
construct a tiny FastAPI app and drive it via ``TestClient`` with a
fake provider stub.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from memorylayer_embed_server.api.v1.chat import router as llm_router
from memorylayer_embed_server.services.llm.base import LLMProvider
from memorylayer_embed_server.services.llm.router import (
    LLMRoutingService,
    UnknownModelError,
)


class _FakeProvider(LLMProvider):
    """In-memory stand-in for a real provider."""

    def __init__(self, profile_name: str, model_name: str, *, aliases=None):
        self.profile_name = profile_name
        self.model_name = model_name
        self.aliases = list(aliases) if aliases else []
        self.served_names = list({profile_name, model_name, *self.aliases})
        self.chat_calls: list[tuple[dict, bool]] = []
        self.completions_calls: list[tuple[dict, bool]] = []
        self.preloaded = False
        self.shut_down = False
        self.chat_response: dict | list[bytes] = {"id": "chat-1", "choices": []}
        self.completions_response: dict | list[bytes] = {"id": "compl-1", "choices": []}

    async def chat_completions(self, payload, *, stream=False):
        self.chat_calls.append((payload, stream))
        if stream:
            chunks = list(self.chat_response)  # type: ignore[arg-type]

            async def _iter():
                for c in chunks:
                    yield c

            return _iter()
        return self.chat_response

    async def completions(self, payload, *, stream=False):
        self.completions_calls.append((payload, stream))
        if stream:
            chunks = list(self.completions_response)  # type: ignore[arg-type]

            async def _iter():
                for c in chunks:
                    yield c

            return _iter()
        return self.completions_response

    async def preload(self) -> None:
        self.preloaded = True

    async def shutdown(self) -> None:
        self.shut_down = True

    def list_models(self) -> list[dict]:
        return [
            {"id": n, "object": "model", "profile": self.profile_name, "model": self.model_name, "owned_by": "memorylayer-embed-server"}
            for n in self.served_names
        ]

    def get_load_snapshot(self) -> dict:
        return {"in_flight": len(self.chat_calls), "max_concurrent": 0, "utilization": 0.0}


# ---------------------------------------------------------------------------
# LLMRoutingService — alias resolution
# ---------------------------------------------------------------------------


def _service_with_two_profiles() -> LLMRoutingService:
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct", aliases=["qwen-7b", "qwen2.5"])
    llama = _FakeProvider("llama", "meta-llama/Llama-3.1-8B-Instruct", aliases=["llama-8b"])
    return LLMRoutingService(profiles={"qwen": qwen, "llama": llama}, default_profile="qwen")


def test_resolve_by_profile_name():
    svc = _service_with_two_profiles()
    assert svc.resolve("qwen").profile_name == "qwen"
    assert svc.resolve("llama").profile_name == "llama"


def test_resolve_by_alias():
    svc = _service_with_two_profiles()
    assert svc.resolve("qwen-7b").profile_name == "qwen"
    assert svc.resolve("llama-8b").profile_name == "llama"


def test_resolve_by_underlying_model_name():
    svc = _service_with_two_profiles()
    assert svc.resolve("Qwen/Qwen2.5-7B-Instruct").profile_name == "qwen"
    assert svc.resolve("meta-llama/Llama-3.1-8B-Instruct").profile_name == "llama"


def test_resolve_is_case_insensitive():
    svc = _service_with_two_profiles()
    assert svc.resolve("QWEN").profile_name == "qwen"
    assert svc.resolve("Qwen-7B").profile_name == "qwen"


def test_resolve_unknown_falls_back_to_default():
    svc = _service_with_two_profiles()
    assert svc.resolve("does-not-exist").profile_name == "qwen"
    assert svc.resolve(None).profile_name == "qwen"


def test_resolve_unknown_raises_when_no_default():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    svc = LLMRoutingService(profiles={"qwen": qwen})  # no default_profile
    with pytest.raises(UnknownModelError) as exc:
        svc.resolve("nope")
    assert "qwen" in exc.value.available


def test_invalid_default_profile_raises_at_construction():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(ValueError):
        LLMRoutingService(profiles={"qwen": qwen}, default_profile="missing")


def test_list_models_contains_every_routing_name():
    svc = _service_with_two_profiles()
    ids = sorted(m["id"] for m in svc.list_models())
    # Each profile's served_names contributes; combined set covers profile + aliases + model.
    assert "qwen" in ids
    assert "qwen-7b" in ids
    assert "qwen2.5" in ids
    assert "Qwen/Qwen2.5-7B-Instruct" in ids
    assert "llama" in ids
    assert "llama-8b" in ids


# ---------------------------------------------------------------------------
# Lifecycle fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preload_calls_each_profile():
    svc = _service_with_two_profiles()
    await svc.preload()
    for provider in svc.profiles.values():
        assert provider.preloaded is True


@pytest.mark.asyncio
async def test_shutdown_calls_each_profile():
    svc = _service_with_two_profiles()
    await svc.shutdown()
    for provider in svc.profiles.values():
        assert provider.shut_down is True


@pytest.mark.asyncio
async def test_preload_failure_does_not_block_siblings():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    llama = _FakeProvider("llama", "meta-llama/Llama-3.1-8B-Instruct")
    qwen.preload = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    svc = LLMRoutingService(profiles={"qwen": qwen, "llama": llama})
    await svc.preload()
    assert llama.preloaded is True


def test_load_snapshot_keyed_by_profile():
    svc = _service_with_two_profiles()
    snap = svc.get_load_snapshot()
    assert "llm_qwen" in snap
    assert "llm_llama" in snap


# ---------------------------------------------------------------------------
# FastAPI route — POST /v1/chat/completions + GET /v1/models
# ---------------------------------------------------------------------------


def _build_test_app(svc: LLMRoutingService | None) -> TestClient:
    app = FastAPI()

    class _State:
        pass

    state = _State()
    state.v = MagicMock()
    state.v.get = MagicMock(return_value=svc)

    app.state.v = state.v
    app.include_router(llm_router)
    return TestClient(app)


def test_chat_completions_non_streaming_round_trip():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    qwen.chat_response = {"id": "chatcmpl-test", "choices": [{"message": {"content": "hi"}}]}
    svc = LLMRoutingService(profiles={"qwen": qwen}, default_profile="qwen")

    client = _build_test_app(svc)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "messages": [{"role": "user", "content": "hi?"}]},
    )
    assert r.status_code == 200
    assert r.json()["id"] == "chatcmpl-test"
    assert qwen.chat_calls[0][0]["messages"] == [{"role": "user", "content": "hi?"}]
    assert qwen.chat_calls[0][1] is False  # stream=False


def test_chat_completions_routes_by_alias():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct", aliases=["qwen-7b"])
    llama = _FakeProvider("llama", "meta-llama/Llama-3.1-8B-Instruct")
    svc = LLMRoutingService(profiles={"qwen": qwen, "llama": llama})

    client = _build_test_app(svc)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen-7b", "messages": [{"role": "user", "content": "x"}]},
    )
    assert r.status_code == 200
    # qwen handled, llama didn't.
    assert len(qwen.chat_calls) == 1
    assert len(llama.chat_calls) == 0


def test_chat_completions_streaming_emits_sse():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    qwen.chat_response = [
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    svc = LLMRoutingService(profiles={"qwen": qwen}, default_profile="qwen")

    client = _build_test_app(svc)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.content
    assert b'"hi"' in body
    assert b'" there"' in body
    assert b"[DONE]" in body
    # stream=True flag forwarded to provider.
    assert qwen.chat_calls[0][1] is True


def test_chat_completions_unknown_model_returns_404():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    svc = LLMRoutingService(profiles={"qwen": qwen})  # no default
    client = _build_test_app(svc)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": []},
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["error"]["type"] == "model_not_found"
    assert "qwen" in detail["error"]["available_models"]


def test_chat_completions_without_routing_service_returns_503():
    client = _build_test_app(svc=None)
    r = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 503


def test_list_models_endpoint():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct", aliases=["qwen-7b"])
    llama = _FakeProvider("llama", "meta-llama/Llama-3.1-8B-Instruct")
    svc = LLMRoutingService(profiles={"qwen": qwen, "llama": llama})
    client = _build_test_app(svc)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = sorted(item["id"] for item in data["data"])
    for expected in ("qwen", "qwen-7b", "Qwen/Qwen2.5-7B-Instruct", "llama"):
        assert expected in ids


def test_completions_endpoint_routes_to_provider():
    qwen = _FakeProvider("qwen", "Qwen/Qwen2.5-7B-Instruct")
    qwen.completions_response = {"id": "cmpl-1", "choices": [{"text": "ok"}]}
    svc = LLMRoutingService(profiles={"qwen": qwen}, default_profile="qwen")
    client = _build_test_app(svc)
    r = client.post("/v1/completions", json={"model": "qwen", "prompt": "hi"})
    assert r.status_code == 200
    assert r.json()["id"] == "cmpl-1"
    assert qwen.completions_calls[0][0]["prompt"] == "hi"
