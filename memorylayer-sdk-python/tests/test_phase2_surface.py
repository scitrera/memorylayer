"""Unit tests for the Phase-2 client-surface additions.

Covers (async + sync where applicable):
- Tokens: create/list/get/delete/revoke (/v1/tokens)
- list_memories (GET /v1/memories) + iterate_memories auto-pagination
- Entity ops: list/get/resolve/merge (gated; 501 -> EnterpriseRequiredError)
- Association ops: update (PATCH) / delete (DELETE)
- traverse_graph (POST /v1/memories/{id}/traverse)
- recall new params (offset/event_*/time_order/include_global*)
- HTTP transport retry-with-backoff for idempotent requests
"""

import httpx
import pytest
import respx
from httpx import Response

from memorylayer import (
    EnterpriseRequiredError,
    MemoryLayerClient,
    SyncMemoryLayerClient,
    TokenCreateResult,
    TokenInfo,
)
from memorylayer._transport.http import HttpTransport


@pytest.fixture
def base_url() -> str:
    return "http://test.memorylayer.ai"


@pytest.fixture
def client(base_url: str) -> MemoryLayerClient:
    return MemoryLayerClient(base_url=base_url, api_key="k", workspace_id="ws_test")


@pytest.fixture
def sync_cli(base_url: str) -> SyncMemoryLayerClient:
    return SyncMemoryLayerClient(base_url=base_url, api_key="k", workspace_id="ws_test")


def _mem(mem_id: str) -> dict:
    return {
        "id": mem_id,
        "workspace_id": "ws_test",
        "content": "c",
        "type": "semantic",
        "importance": 0.5,
        "tags": [],
        "metadata": {},
        "access_count": 0,
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }


# ------------------------------------------------------------------ #
# Tokens
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_create_token(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.post(f"{base_url}/v1/tokens").mock(
        return_value=Response(
            201,
            json={
                "id": "tok_1",
                "name": "ci",
                "principal_type": "User",
                "workspace_patterns": ["*"],
                "scopes": ["*"],
                "created_at": "2026-01-26T10:00:00Z",
                "expires_at": None,
                "revoked": False,
                "token": "secret-plaintext",
            },
        )
    )
    async with client:
        result = await client.create_token("ci", expires_in_days=30)
    assert isinstance(result, TokenCreateResult)
    assert result.id == "tok_1"
    assert result.token == "secret-plaintext"
    sent = route.calls.last.request
    assert b'"expires_in_days":30' in sent.content


@pytest.mark.asyncio
@respx.mock
async def test_list_get_delete_revoke_token(client: MemoryLayerClient, base_url: str) -> None:
    token_json = {
        "id": "tok_1",
        "name": "ci",
        "workspace_patterns": ["*"],
        "scopes": ["*"],
        "created_at": "2026-01-26T10:00:00Z",
    }
    respx.get(f"{base_url}/v1/tokens").mock(return_value=Response(200, json={"tokens": [token_json]}))
    respx.get(f"{base_url}/v1/tokens/tok_1").mock(return_value=Response(200, json=token_json))
    respx.delete(f"{base_url}/v1/tokens/tok_1").mock(return_value=Response(204))
    revoke_route = respx.post(f"{base_url}/v1/tokens/tok_1/revoke").mock(return_value=Response(204))

    async with client:
        tokens = await client.list_tokens()
        assert len(tokens) == 1 and isinstance(tokens[0], TokenInfo)
        one = await client.get_token("tok_1")
        assert one.id == "tok_1"
        await client.delete_token("tok_1")
        await client.revoke_token("tok_1")
    assert revoke_route.called


@respx.mock
def test_sync_create_and_revoke_token(sync_cli: SyncMemoryLayerClient, base_url: str) -> None:
    respx.post(f"{base_url}/v1/tokens").mock(
        return_value=Response(
            201,
            json={
                "id": "tok_2",
                "name": "x",
                "workspace_patterns": ["*"],
                "scopes": ["*"],
                "created_at": "2026-01-26T10:00:00Z",
                "token": "plain",
            },
        )
    )
    respx.post(f"{base_url}/v1/tokens/tok_2/revoke").mock(return_value=Response(204))
    with sync_cli:
        created = sync_cli.create_token("x")
        assert created.token == "plain"
        sync_cli.revoke_token("tok_2")


# ------------------------------------------------------------------ #
# list_memories + iterate_memories
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_list_memories(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.get(f"{base_url}/v1/memories").mock(
        return_value=Response(200, json={"memories": [_mem("m1"), _mem("m2")], "total_count": 2})
    )
    async with client:
        page = await client.list_memories(limit=10, offset=0, tag="pref")
    assert page.total_count == 2
    assert [m.id for m in page.memories] == ["m1", "m2"]
    assert route.calls.last.request.url.params["tag"] == "pref"


@pytest.mark.asyncio
@respx.mock
async def test_iterate_memories_paginates(client: MemoryLayerClient, base_url: str) -> None:
    # page_size=2: first call full page, second call partial -> stop
    pages = [
        Response(200, json={"memories": [_mem("m1"), _mem("m2")], "total_count": 3}),
        Response(200, json={"memories": [_mem("m3")], "total_count": 3}),
    ]
    respx.get(f"{base_url}/v1/memories").mock(side_effect=pages)
    seen = []
    async with client:
        async for mem in client.iterate_memories(page_size=2):
            seen.append(mem.id)
    assert seen == ["m1", "m2", "m3"]


@respx.mock
def test_sync_iterate_memories(sync_cli: SyncMemoryLayerClient, base_url: str) -> None:
    pages = [
        Response(200, json={"memories": [_mem("a"), _mem("b")], "total_count": 3}),
        Response(200, json={"memories": [_mem("c")], "total_count": 3}),
    ]
    respx.get(f"{base_url}/v1/memories").mock(side_effect=pages)
    with sync_cli:
        ids = [m.id for m in sync_cli.iterate_memories(page_size=2)]
    assert ids == ["a", "b", "c"]


# ------------------------------------------------------------------ #
# Entities (gated)
# ------------------------------------------------------------------ #


def _entity(eid: str) -> dict:
    return {
        "id": eid,
        "workspace_id": "ws_test",
        "entity_type": "person",
        "canonical_name": "Alice",
        "normalized_name": "alice",
        "aliases": [],
        "confidence": 1.0,
        "provenance": {},
        "status": "active",
        "created_at": "2026-01-26T10:00:00Z",
        "updated_at": "2026-01-26T10:00:00Z",
    }


@pytest.mark.asyncio
@respx.mock
async def test_list_entities(client: MemoryLayerClient, base_url: str) -> None:
    respx.get(f"{base_url}/v1/entities").mock(
        return_value=Response(200, json={"entities": [_entity("e1")], "total_count": 1})
    )
    async with client:
        entities = await client.list_entities()
    assert entities[0].id == "e1"
    assert entities[0].canonical_name == "Alice"


@pytest.mark.asyncio
@respx.mock
async def test_entities_gated_501_raises_enterprise(client: MemoryLayerClient, base_url: str) -> None:
    respx.get(f"{base_url}/v1/entities").mock(
        return_value=Response(501, json={"detail": "Entity registry is not enabled"})
    )
    async with client:
        with pytest.raises(EnterpriseRequiredError):
            await client.list_entities()


@pytest.mark.asyncio
@respx.mock
async def test_resolve_entity_404_returns_none(client: MemoryLayerClient, base_url: str) -> None:
    respx.get(f"{base_url}/v1/entities/resolve").mock(return_value=Response(404, json={"detail": "no match"}))
    async with client:
        result = await client.resolve_entity("Nobody")
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_resolve_entity_match(client: MemoryLayerClient, base_url: str) -> None:
    respx.get(f"{base_url}/v1/entities/resolve").mock(
        return_value=Response(200, json={"resolution": {"entity": _entity("e1"), "matched_via": "exact", "score": 1.0}})
    )
    async with client:
        result = await client.resolve_entity("Alice")
    assert result is not None
    assert result.matched_via == "exact"
    assert result.entity.id == "e1"


@pytest.mark.asyncio
@respx.mock
async def test_merge_entities(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.post(f"{base_url}/v1/entities/merge").mock(
        return_value=Response(200, json={"entity": _entity("target")})
    )
    async with client:
        survivor = await client.merge_entities("src", "target", reason="dup")
    assert survivor.id == "target"
    assert b'"reason":"dup"' in route.calls.last.request.content


# ------------------------------------------------------------------ #
# Associations: update + delete
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_update_association(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.patch(f"{base_url}/v1/memories/m1/associations/a1").mock(return_value=Response(204))
    async with client:
        ok = await client.update_association("m1", "a1", strength=0.9)
    assert ok is True
    assert b'"strength":0.9' in route.calls.last.request.content


@pytest.mark.asyncio
@respx.mock
async def test_update_association_404_returns_false(client: MemoryLayerClient, base_url: str) -> None:
    respx.patch(f"{base_url}/v1/memories/m1/associations/a1").mock(
        return_value=Response(404, json={"detail": "Association not found"})
    )
    async with client:
        ok = await client.update_association("m1", "a1", strength=0.9)
    assert ok is False


@pytest.mark.asyncio
@respx.mock
async def test_delete_association(client: MemoryLayerClient, base_url: str) -> None:
    respx.delete(f"{base_url}/v1/memories/m1/associations/a1").mock(return_value=Response(204))
    async with client:
        ok = await client.delete_association("m1", "a1")
    assert ok is True


@respx.mock
def test_sync_update_and_delete_association(sync_cli: SyncMemoryLayerClient, base_url: str) -> None:
    respx.patch(f"{base_url}/v1/memories/m1/associations/a1").mock(return_value=Response(204))
    respx.delete(f"{base_url}/v1/memories/m1/associations/a1").mock(return_value=Response(204))
    with sync_cli:
        assert sync_cli.update_association("m1", "a1", metadata={"x": 1}) is True
        assert sync_cli.delete_association("m1", "a1") is True


# ------------------------------------------------------------------ #
# traverse_graph
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_traverse_graph(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.post(f"{base_url}/v1/memories/m1/traverse").mock(
        return_value=Response(
            200,
            json={"paths": [], "total_paths": 0, "unique_nodes": ["m1"], "query_latency_ms": 3},
        )
    )
    async with client:
        result = await client.traverse_graph("m1", max_depth=3)
    assert result["unique_nodes"] == ["m1"]
    assert b'"max_depth":3' in route.calls.last.request.content


# ------------------------------------------------------------------ #
# recall new params
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_recall_new_params_sent(client: MemoryLayerClient, base_url: str) -> None:
    route = respx.post(f"{base_url}/v1/memories/recall").mock(
        return_value=Response(200, json={"memories": [], "total_count": 0})
    )
    async with client:
        await client.recall(
            "q",
            offset=5,
            event_after="2026-01-01T00:00:00Z",
            event_before="2026-02-01T00:00:00Z",
            time_order="desc",
            include_global=False,
            include_global_user=True,
        )
    body = route.calls.last.request.content
    assert b'"offset":5' in body
    assert b'"event_after"' in body
    assert b'"event_before"' in body
    assert b'"time_order":"desc"' in body
    assert b'"include_global":false' in body
    assert b'"include_global_user":true' in body


# ------------------------------------------------------------------ #
# delete_context
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
@respx.mock
async def test_delete_context(client: MemoryLayerClient, base_url: str) -> None:
    # client fixture has workspace_id="ws_test"; pass explicit workspace_id to verify routing
    route = respx.delete(f"{base_url}/v1/workspaces/ws_1/contexts/ctx_1").mock(return_value=Response(204))
    async with client:
        await client.delete_context("ctx_1", workspace_id="ws_1")
    assert route.called


@pytest.mark.asyncio
@respx.mock
async def test_delete_context_uses_client_workspace(client: MemoryLayerClient, base_url: str) -> None:
    # When workspace_id is omitted, falls back to client.workspace_id ("ws_test")
    route = respx.delete(f"{base_url}/v1/workspaces/ws_test/contexts/ctx_2").mock(return_value=Response(204))
    async with client:
        await client.delete_context("ctx_2")
    assert route.called


@respx.mock
def test_sync_delete_context(sync_cli: SyncMemoryLayerClient, base_url: str) -> None:
    route = respx.delete(f"{base_url}/v1/workspaces/ws_1/contexts/ctx_1").mock(return_value=Response(204))
    with sync_cli:
        sync_cli.delete_context("ctx_1", workspace_id="ws_1")
    assert route.called


@respx.mock
def test_sync_delete_context_uses_client_workspace(sync_cli: SyncMemoryLayerClient, base_url: str) -> None:
    # sync_cli fixture has workspace_id="ws_test"
    route = respx.delete(f"{base_url}/v1/workspaces/ws_test/contexts/ctx_2").mock(return_value=Response(204))
    with sync_cli:
        sync_cli.delete_context("ctx_2")
    assert route.called


# ------------------------------------------------------------------ #
# Transport retries
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_transport_retries_idempotent_then_succeeds(monkeypatch) -> None:
    """A GET that 503s twice then 200s should succeed transparently."""
    transport = HttpTransport(base_url="http://x/v1", max_retries=3, backoff_factor=0.0)

    calls = {"n": 0}

    async def fake_request(method, path, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, request=httpx.Request(method, "http://x/v1" + path))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, "http://x/v1" + path))

    monkeypatch.setattr(transport._client, "request", fake_request)
    resp = await transport.request("GET", "/memories")
    assert resp.status_code == 200
    assert calls["n"] == 3
    await transport.aclose()


@pytest.mark.asyncio
async def test_transport_does_not_retry_post(monkeypatch) -> None:
    """POST is not idempotent and must be attempted exactly once."""
    transport = HttpTransport(base_url="http://x/v1", max_retries=3, backoff_factor=0.0)

    calls = {"n": 0}

    async def fake_request(method, path, **kwargs):
        calls["n"] += 1
        return httpx.Response(503, request=httpx.Request(method, "http://x/v1" + path))

    monkeypatch.setattr(transport._client, "request", fake_request)
    resp = await transport.request("POST", "/memories")
    assert resp.status_code == 503
    assert calls["n"] == 1
    await transport.aclose()


@pytest.mark.asyncio
async def test_transport_honors_retry_after(monkeypatch) -> None:
    """Retry-After numeric value is honored over the default backoff."""
    transport = HttpTransport(base_url="http://x/v1", max_retries=2, backoff_factor=0.0)

    slept: list[float] = []

    async def fake_sleep(d):
        slept.append(d)

    calls = {"n": 0}

    async def fake_request(method, path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "1.5"},
                request=httpx.Request(method, "http://x/v1" + path),
            )
        return httpx.Response(200, json={}, request=httpx.Request(method, "http://x/v1" + path))

    import memorylayer._transport.http as http_mod

    monkeypatch.setattr(http_mod.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(transport._client, "request", fake_request)
    resp = await transport.request("GET", "/memories")
    assert resp.status_code == 200
    assert slept == [1.5]
    await transport.aclose()
