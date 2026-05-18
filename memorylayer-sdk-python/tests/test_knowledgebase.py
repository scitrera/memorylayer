"""Tests for MemoryLayer knowledgebase namespace (async + sync)."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from memorylayer import (
    KbArticle,
    KbGraphAnalysis,
    KbMetadata,
    MemoryLayerClient,
    SyncMemoryLayerClient,
)
from memorylayer.knowledgebase import KnowledgebaseAPI, SyncKnowledgebaseAPI

BASE_URL = "http://test.memorylayer.ai"


_ARTICLE_PAYLOAD = {
    "id": "community-3",
    "article_type": "community",
    "title": "Cluster about coding preferences",
    "content_md": "# Cluster\n\n[[index]] [[entity-foo|Foo]]\n",
    "metadata": {"size": 7},
    "generated_at": "2026-04-01T12:00:00Z",
}

_KB_PAYLOAD = {
    "workspace_id": "ws_test",
    "article_count": 12,
    "community_count": 5,
    "generated_at": "2026-04-01T12:00:00Z",
    "stats": {
        "node_count": 80,
        "edge_count": 132,
        "community_count": 5,
        "density": 0.041,
        "avg_degree": 3.3,
        "max_degree": 9,
        "god_node_count": 2,
    },
}

_GRAPH_PAYLOAD = {
    "analysis": {
        "snapshot": {
            "workspace_id": "ws_test",
            "context_id": None,
            "node_count": 80,
            "edge_count": 132,
            "includes_rpg": False,
        },
        "communities": [
            {
                "id": 0,
                "memory_ids": ["m1", "m2"],
                "size": 2,
                "cohesion_score": 0.5,
                "central_node_ids": ["m1"],
                "label": "Cluster 0",
            },
        ],
        "central_nodes": [
            {"memory_id": "m1", "degree": 4, "betweenness": 0.2, "community_id": 0},
        ],
        "bridges": [
            {
                "source_community_id": 0,
                "target_community_id": 1,
                "memory_id_source": "m1",
                "memory_id_target": "m9",
                "relationship_type": "related",
                "strength": 0.7,
            },
        ],
        "stats": _KB_PAYLOAD["stats"],
    },
    "cached": False,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def async_client() -> MemoryLayerClient:
    return MemoryLayerClient(
        base_url=BASE_URL,
        api_key="test_key",
        workspace_id="ws_test",
    )


@pytest.fixture
def sync_client_obj() -> SyncMemoryLayerClient:
    return SyncMemoryLayerClient(
        base_url=BASE_URL,
        api_key="test_key",
        workspace_id="ws_test",
    )


# ---------------------------------------------------------------------------
# Namespace wiring
# ---------------------------------------------------------------------------


def test_kb_namespace_attached(async_client: MemoryLayerClient) -> None:
    assert isinstance(async_client.kb, KnowledgebaseAPI)


def test_sync_kb_namespace_attached(sync_client_obj: SyncMemoryLayerClient) -> None:
    assert isinstance(sync_client_obj.kb, SyncKnowledgebaseAPI)


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_kb_get_metadata(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase").mock(
        return_value=Response(200, json=_KB_PAYLOAD),
    )
    async with async_client:
        kb = await async_client.kb.get()
    assert route.called
    assert isinstance(kb, KbMetadata)
    assert kb.workspace_id == "ws_test"
    assert kb.article_count == 12
    assert kb.stats is not None
    assert kb.stats.density == pytest.approx(0.041)
    # default workspace_id flows through as a query param
    assert route.calls[0].request.url.params.get("workspace_id") == "ws_test"


@pytest.mark.asyncio
@respx.mock
async def test_kb_list_articles(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase/articles").mock(
        return_value=Response(200, json={"articles": [_ARTICLE_PAYLOAD], "total": 1}),
    )
    async with async_client:
        articles = await async_client.kb.list_articles(article_type="community", limit=50)
    assert route.called
    params = route.calls[0].request.url.params
    assert params.get("article_type") == "community"
    assert params.get("limit") == "50"
    assert params.get("workspace_id") == "ws_test"
    assert len(articles) == 1
    assert isinstance(articles[0], KbArticle)
    assert articles[0].id == "community-3"


@pytest.mark.asyncio
@respx.mock
async def test_kb_get_article(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase/articles/community-3").mock(
        return_value=Response(200, json=_ARTICLE_PAYLOAD),
    )
    async with async_client:
        article = await async_client.kb.get_article("community-3")
    assert route.called
    assert article.id == "community-3"
    assert "[[index]]" in article.content_md


@pytest.mark.asyncio
@respx.mock
async def test_kb_generate(async_client: MemoryLayerClient) -> None:
    route = respx.post(f"{BASE_URL}/v1/knowledgebase/generate").mock(
        return_value=Response(200, json=_KB_PAYLOAD),
    )
    async with async_client:
        kb = await async_client.kb.generate(regenerate=True, max_communities=10)
    assert route.called
    body = route.calls[0].request.read()
    import json
    payload = json.loads(body)
    assert payload["regenerate"] is True
    assert payload["max_communities"] == 10
    assert payload["workspace_id"] == "ws_test"
    assert isinstance(kb, KbMetadata)


@pytest.mark.asyncio
@respx.mock
async def test_kb_get_graph_analysis(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase/graph").mock(
        return_value=Response(200, json=_GRAPH_PAYLOAD),
    )
    async with async_client:
        graph = await async_client.kb.get_graph_analysis()
    assert route.called
    assert isinstance(graph, KbGraphAnalysis)
    assert graph.snapshot.workspace_id == "ws_test"
    assert len(graph.communities) == 1
    assert graph.bridges[0].strength == pytest.approx(0.7)


@pytest.mark.asyncio
@respx.mock
async def test_kb_get_graph_analysis_empty(async_client: MemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/knowledgebase/graph").mock(
        return_value=Response(200, json={"analysis": None, "cached": False}),
    )
    async with async_client:
        graph = await async_client.kb.get_graph_analysis()
    assert graph is None


@pytest.mark.asyncio
@respx.mock
async def test_kb_export_vault_returns_bytes(async_client: MemoryLayerClient) -> None:
    zip_bytes = b"PK\x03\x04...fake-zip..."
    respx.get(f"{BASE_URL}/v1/knowledgebase/export").mock(
        return_value=Response(
            200, content=zip_bytes, headers={"content-type": "application/zip"},
        ),
    )
    async with async_client:
        data = await async_client.kb.export_vault()
    assert data == zip_bytes


@pytest.mark.asyncio
@respx.mock
async def test_kb_get_community(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase/graph/communities/3").mock(
        return_value=Response(200, json={
            "id": 3,
            "memory_ids": ["m1"],
            "size": 1,
            "cohesion_score": 0.0,
            "central_node_ids": [],
            "label": "Some topic",
        }),
    )
    async with async_client:
        community = await async_client.kb.get_community(3)
    assert route.called
    assert community.id == 3
    assert community.label == "Some topic"


@pytest.mark.asyncio
@respx.mock
async def test_kb_obo_proxy_passes_authority(async_client: MemoryLayerClient) -> None:
    route = respx.get(f"{BASE_URL}/v1/knowledgebase").mock(
        return_value=Response(200, json=_KB_PAYLOAD),
    )
    async with async_client:
        async with async_client.acting_for("g_abc", subject=("user", "alice")) as proxy:
            await proxy.kb.get()
    assert route.called
    headers = route.calls[0].request.headers
    assert headers.get("X-Aether-Grant-ID") == "g_abc"
    assert headers.get("X-Aether-Authority-Mode") == "on_behalf_of"
    assert headers.get("X-Aether-Subject-Type") == "user"
    assert headers.get("X-Aether-Subject-ID") == "alice"


# ---------------------------------------------------------------------------
# Sync API
# ---------------------------------------------------------------------------


@respx.mock
def test_sync_kb_get_metadata(sync_client_obj: SyncMemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/knowledgebase").mock(
        return_value=Response(200, json=_KB_PAYLOAD),
    )
    with sync_client_obj as client:
        kb = client.kb.get()
    assert kb.workspace_id == "ws_test"


@respx.mock
def test_sync_kb_list_articles(sync_client_obj: SyncMemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/knowledgebase/articles").mock(
        return_value=Response(200, json={"articles": [_ARTICLE_PAYLOAD], "total": 1}),
    )
    with sync_client_obj as client:
        articles = client.kb.list_articles()
    assert articles[0].id == "community-3"
