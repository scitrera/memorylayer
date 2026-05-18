"""Tests for OBO (on-behalf-of) authority support in MemoryLayerClient."""

import asyncio

import pytest
import respx
from httpx import Request, Response

from memorylayer import AuthorityContext, MemoryLayerClient, PrincipalRef

BASE_URL = "http://test.memorylayer.ai"

MEMORY_RESPONSE = {
    "memory": {
        "id": "mem_123",
        "workspace_id": "ws_test",
        "content": "test content",
        "type": "semantic",
        "importance": 0.5,
        "tags": [],
        "metadata": {},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }
}

RECALL_RESPONSE = {
    "memories": [],
    "total_count": 0,
}

REFLECT_RESPONSE = {
    "reflection": "test reflection",
    "source_memories": [],
    "confidence": 0.9,
}


@pytest.fixture
def client() -> MemoryLayerClient:
    return MemoryLayerClient(base_url=BASE_URL, api_key="test_key", workspace_id="ws_default")


# --- acting_for sends correct X-Aether-* headers ---


@pytest.mark.asyncio
@respx.mock
async def test_acting_for_sends_obo_headers(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=MEMORY_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories").mock(side_effect=capture)

    async with client:
        async with client.acting_for("g_abc", subject=("user", "alice")) as alice:
            await alice.remember("hello")

    assert len(captured) == 1
    req = captured[0]
    assert req.headers["X-Aether-Grant-ID"] == "g_abc"
    assert req.headers["X-Aether-Authority-Mode"] == "on_behalf_of"
    assert req.headers["X-Aether-Subject-Type"] == "user"
    assert req.headers["X-Aether-Subject-ID"] == "alice"


# --- for_workspace nested proxy ---


@pytest.mark.asyncio
@respx.mock
async def test_for_workspace_nested_proxy(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=RECALL_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories/recall").mock(side_effect=capture)

    async with client:
        async with client.acting_for("g_xyz", subject=("user", "bob")) as bob:
            bob_inner = bob.for_workspace("ws_inner")
            await bob_inner.recall("some query")

    assert len(captured) == 1
    req = captured[0]
    assert req.headers["X-Aether-Grant-ID"] == "g_xyz"
    assert req.headers["X-Aether-Subject-ID"] == "bob"
    # workspace_id baked into payload
    import json

    body = json.loads(req.content)
    assert body["workspace_id"] == "ws_inner"


# --- concurrent acting_for blocks — no cross-talk ---


@pytest.mark.asyncio
@respx.mock
async def test_concurrent_acting_for_no_crosstalk(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=MEMORY_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories").mock(side_effect=capture)

    async with client:

        async def do_remember(grant_id: str, subject: str) -> None:
            async with client.acting_for(grant_id, subject=("user", subject)) as proxy:
                await proxy.remember(f"content for {subject}")

        await asyncio.gather(
            do_remember("g_1", "alice"),
            do_remember("g_2", "bob"),
            do_remember("g_3", "carol"),
        )

    assert len(captured) == 3
    subjects = {req.headers["X-Aether-Subject-ID"] for req in captured}
    grants = {req.headers["X-Aether-Grant-ID"] for req in captured}
    assert subjects == {"alice", "bob", "carol"}
    assert grants == {"g_1", "g_2", "g_3"}

    # Each request carries its own grant — no mixing
    for req in captured:
        sid = req.headers["X-Aether-Subject-ID"]
        gid = req.headers["X-Aether-Grant-ID"]
        expected = {"alice": "g_1", "bob": "g_2", "carol": "g_3"}
        assert gid == expected[sid], f"cross-talk: {sid} got grant {gid}"


# --- per-call authority= on remember/recall/reflect ---


@pytest.mark.asyncio
@respx.mock
async def test_per_call_authority_remember(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=MEMORY_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories").mock(side_effect=capture)

    authority = AuthorityContext(grant_id="g_percall", subject=PrincipalRef("user", "dave"))

    async with client:
        await client.remember("per-call content", authority=authority)

    assert len(captured) == 1
    req = captured[0]
    assert req.headers["X-Aether-Grant-ID"] == "g_percall"
    assert req.headers["X-Aether-Subject-ID"] == "dave"


@pytest.mark.asyncio
@respx.mock
async def test_per_call_authority_recall(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=RECALL_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories/recall").mock(side_effect=capture)

    authority = AuthorityContext(grant_id="g_recall", subject=PrincipalRef("user", "eve"))

    async with client:
        await client.recall("query", authority=authority)

    req = captured[0]
    assert req.headers["X-Aether-Grant-ID"] == "g_recall"
    assert req.headers["X-Aether-Subject-ID"] == "eve"


@pytest.mark.asyncio
@respx.mock
async def test_per_call_authority_reflect(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=REFLECT_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories/reflect").mock(side_effect=capture)

    authority = AuthorityContext(grant_id="g_reflect", subject=PrincipalRef("service", "sv.foo"))

    async with client:
        await client.reflect("query", authority=authority)

    req = captured[0]
    assert req.headers["X-Aether-Grant-ID"] == "g_reflect"
    assert req.headers["X-Aether-Subject-Type"] == "service"
    assert req.headers["X-Aether-Subject-ID"] == "sv.foo"


# --- default_authority constructor param ---


@pytest.mark.asyncio
@respx.mock
async def test_default_authority_on_constructor() -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=MEMORY_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories").mock(side_effect=capture)

    authority = AuthorityContext(grant_id="g_default", subject=PrincipalRef("user", "frank"))
    client = MemoryLayerClient(base_url=BASE_URL, api_key="key", default_authority=authority)

    async with client:
        await client.remember("content 1")
        await client.remember("content 2")

    assert len(captured) == 2
    for req in captured:
        assert req.headers["X-Aether-Grant-ID"] == "g_default"
        assert req.headers["X-Aether-Subject-ID"] == "frank"


# --- backward compat: no authority → no X-Aether-* headers ---


@pytest.mark.asyncio
@respx.mock
async def test_no_authority_no_obo_headers(client: MemoryLayerClient) -> None:
    captured: list[Request] = []

    def capture(request: Request) -> Response:
        captured.append(request)
        return Response(200, json=MEMORY_RESPONSE)

    respx.post(f"{BASE_URL}/v1/memories").mock(side_effect=capture)

    async with client:
        await client.remember("plain content")

    req = captured[0]
    assert "X-Aether-Grant-ID" not in req.headers
    assert "X-Aether-Authority-Mode" not in req.headers
    assert "X-Aether-Subject-Type" not in req.headers
    assert "X-Aether-Subject-ID" not in req.headers
