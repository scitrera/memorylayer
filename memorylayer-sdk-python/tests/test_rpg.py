"""Request-contract tests for the Python SDK RPG namespace."""

import json

import pytest
import respx
from httpx import Response

from memorylayer import (
    MemoryLayerClient,
    RpgAPI,
    RpgNodeInput,
    RpgSyncResult,
    SyncMemoryLayerClient,
    SyncRpgAPI,
)

BASE_URL = "http://test.memorylayer.ai"


def _node() -> RpgNodeInput:
    return RpgNodeInput(
        node_id="src/main.py",
        node_type="rpg_file",
        path="src/main.py",
        name="main.py",
        language="python",
    )


@pytest.mark.asyncio
@respx.mock
async def test_async_rpg_sync_serializes_server_contract() -> None:
    route = respx.post(f"{BASE_URL}/v1/rpg/sync").mock(return_value=Response(200, json={"nodes_created": 1, "source_commit": "abc123"}))
    client = MemoryLayerClient(base_url=BASE_URL, api_key="key", workspace_id="ws_1")

    assert isinstance(client.rpg, RpgAPI)
    async with client:
        result = await client.rpg.sync(
            [_node()],
            [],
            full_sync=True,
            source_commit="abc123",
            context_id="rpg-task-42",
        )

    assert isinstance(result, RpgSyncResult)
    assert result.nodes_created == 1
    body = json.loads(route.calls.last.request.content)
    assert body["nodes"][0]["node_id"] == "src/main.py"
    assert body["nodes"][0]["node_type"] == "rpg_file"
    assert body["full_sync"] is True
    assert body["context_id"] == "rpg-task-42"


@pytest.mark.asyncio
@respx.mock
async def test_async_rpg_subgraph_encodes_list_filters() -> None:
    route = respx.get(f"{BASE_URL}/v1/rpg/subgraph").mock(
        return_value=Response(
            200,
            json={"nodes": [], "edges": [], "root_id": "src", "depth": 2},
        )
    )
    client = MemoryLayerClient(base_url=BASE_URL, api_key="key", workspace_id="ws_1")

    async with client:
        result = await client.rpg.get_subgraph(
            node_id="src",
            depth=2,
            node_types=["rpg_file", "rpg_class"],
            relationship_types=["contains", "imports"],
        )

    assert result.root_id == "src"
    params = route.calls.last.request.url.params
    assert params["node_id"] == "src"
    assert params["node_types"] == "rpg_file,rpg_class"
    assert params["relationship_types"] == "contains,imports"


@pytest.mark.asyncio
async def test_rpg_subgraph_requires_a_root() -> None:
    client = MemoryLayerClient(base_url=BASE_URL, api_key="key", workspace_id="ws_1")
    with pytest.raises(ValueError, match="path or node_id"):
        await client.rpg.get_subgraph()


@respx.mock
def test_sync_rpg_namespace_and_delete_nodes() -> None:
    route = respx.delete(f"{BASE_URL}/v1/rpg/nodes").mock(return_value=Response(200, json={"deleted": 1, "not_found": 0}))
    client = SyncMemoryLayerClient(base_url=BASE_URL, api_key="key", workspace_id="ws_1")

    assert isinstance(client.rpg, SyncRpgAPI)
    with client:
        result = client.rpg.delete_nodes(["src/main.py"], context_id="rpg-task-42")

    assert result.deleted == 1
    body = json.loads(route.calls.last.request.content)
    assert body == {"node_ids": ["src/main.py"], "context_id": "rpg-task-42"}


@respx.mock
def test_sync_rpg_delete_overlay_encodes_context_path_segment() -> None:
    route = respx.delete(f"{BASE_URL}/v1/rpg/overlays/rpg-task-feature%2Fapi").mock(
        return_value=Response(
            200,
            json={"context_id": "rpg-task-feature/api", "deleted": 1},
        )
    )
    client = SyncMemoryLayerClient(base_url=BASE_URL, api_key="key", workspace_id="ws_1")

    with client:
        result = client.rpg.delete_overlay("rpg-task-feature/api")

    assert result.deleted == 1
    assert route.called
    assert route.calls.last.request.url.raw_path.endswith(b"/rpg/overlays/rpg-task-feature%2Fapi")
