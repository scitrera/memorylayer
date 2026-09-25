"""Regression tests for parsing the server's wrapped single-resource responses.

The server returns ``{"session": {...}}`` (SessionResponse / SessionStartResponse)
and ``{"thread": {...}}`` (ThreadResponse) envelopes. The clients must unwrap
them, while still accepting a bare object for backward compatibility.
"""

import pytest
import respx
from httpx import Response

from memorylayer import MemoryLayerClient, SyncMemoryLayerClient

BASE_URL = "http://test.memorylayer.ai"

SESSION = {
    "id": "sess_123",
    "workspace_id": "ws_test",
    "metadata": {},
    "expires_at": "2026-01-26T11:00:00Z",
    "created_at": "2026-01-26T10:00:00Z",
}

THREAD = {
    "id": "thr_1",
    "workspace_id": "ws_test",
    "tenant_id": "_default",
    "context_id": "_default",
    "metadata": {},
    "message_count": 0,
    "last_decomposed_index": 0,
    "created_at": "2026-01-26T10:00:00Z",
    "updated_at": "2026-01-26T10:00:00Z",
    "ownership": "workspace",
}


@pytest.fixture
def async_client() -> MemoryLayerClient:
    return MemoryLayerClient(base_url=BASE_URL, api_key="test_key", workspace_id="ws_test")


@pytest.fixture
def sync_client() -> SyncMemoryLayerClient:
    return SyncMemoryLayerClient(base_url=BASE_URL, api_key="test_key", workspace_id="ws_test")


@pytest.mark.parametrize("body", [{"session": SESSION}, SESSION], ids=["wrapped", "bare"])
@pytest.mark.asyncio
@respx.mock
async def test_async_get_session_parses_response(async_client: MemoryLayerClient, body: dict) -> None:
    respx.get(f"{BASE_URL}/v1/sessions/sess_123").mock(return_value=Response(200, json=body))

    async with async_client:
        session = await async_client.get_session("sess_123")

    assert session.id == "sess_123"
    assert session.workspace_id == "ws_test"


@pytest.mark.parametrize("body", [{"session": SESSION}, SESSION], ids=["wrapped", "bare"])
@respx.mock
def test_sync_get_session_parses_response(sync_client: SyncMemoryLayerClient, body: dict) -> None:
    respx.get(f"{BASE_URL}/v1/sessions/sess_123").mock(return_value=Response(200, json=body))

    with sync_client:
        session = sync_client.get_session("sess_123")

    assert session.id == "sess_123"


@respx.mock
def test_sync_create_session_parses_wrapped_response(sync_client: SyncMemoryLayerClient) -> None:
    respx.post(f"{BASE_URL}/v1/sessions").mock(
        return_value=Response(200, json={"session": SESSION, "briefing": None}),
    )

    with sync_client:
        session = sync_client.create_session(ttl_seconds=3600)

    assert session.id == "sess_123"


@pytest.mark.parametrize("body", [{"thread": THREAD}, THREAD], ids=["wrapped", "bare"])
@pytest.mark.asyncio
@respx.mock
async def test_async_thread_methods_parse_response(async_client: MemoryLayerClient, body: dict) -> None:
    respx.post(f"{BASE_URL}/v1/threads").mock(return_value=Response(201, json=body))
    respx.get(f"{BASE_URL}/v1/threads/thr_1").mock(return_value=Response(200, json=body))
    respx.put(f"{BASE_URL}/v1/threads/thr_1").mock(return_value=Response(200, json=body))

    async with async_client:
        created = await async_client.create_thread(ownership="workspace")
        fetched = await async_client.get_thread("thr_1")
        updated = await async_client.update_thread("thr_1", title="Renamed")

    assert created.id == fetched.id == updated.id == "thr_1"


@pytest.mark.parametrize("body", [{"thread": THREAD}, THREAD], ids=["wrapped", "bare"])
@respx.mock
def test_sync_thread_methods_parse_response(sync_client: SyncMemoryLayerClient, body: dict) -> None:
    respx.post(f"{BASE_URL}/v1/threads").mock(return_value=Response(201, json=body))
    respx.get(f"{BASE_URL}/v1/threads/thr_1").mock(return_value=Response(200, json=body))

    with sync_client:
        created = sync_client.create_thread(ownership="workspace")
        fetched = sync_client.get_thread("thr_1")

    assert created.id == fetched.id == "thr_1"


WORKSPACE = {
    "id": "ws_test",
    "tenant_id": "_default",
    "name": "Test",
    "created_at": "2026-01-26T10:00:00Z",
    "updated_at": "2026-01-26T10:00:00Z",
}

BRIEFING = {"workspace_summary": {"total_memories": 3}}

ASSOCIATION = {
    "id": "assoc_1",
    "workspace_id": "ws_test",
    "source_id": "mem_1",
    "target_id": "mem_2",
    "relationship": "related_to",
    "strength": 0.5,
    "metadata": {},
    "created_at": "2026-01-26T10:00:00Z",
}

DOCUMENT = {
    "id": "doc_1",
    "workspace_id": "ws_test",
    "filename": "a.pdf",
    "document_type": "pdf",
    "content_hash": "abc",
    "size_bytes": 10,
    "status": "pending",
    "created_at": "2026-01-26T10:00:00Z",
}

JOB = {"id": "job_1", "workspace_id": "ws_test", "status": "pending", "created_at": "2026-01-26T10:00:00Z"}


def _mock_misc_routes(workspace_body: dict, briefing_body: dict, association_body: dict, reprocess_body: dict) -> None:
    respx.post(f"{BASE_URL}/v1/workspaces").mock(return_value=Response(200, json=workspace_body))
    respx.get(f"{BASE_URL}/v1/workspaces/ws_test").mock(return_value=Response(200, json=workspace_body))
    respx.get(url__startswith=f"{BASE_URL}/v1/sessions/briefing").mock(return_value=Response(200, json=briefing_body))
    respx.post(f"{BASE_URL}/v1/memories/mem_1/associate").mock(return_value=Response(200, json=association_body))
    respx.post(f"{BASE_URL}/v1/documents/doc_1/reprocess").mock(return_value=Response(200, json=reprocess_body))


MISC_CASES = [
    pytest.param(
        {"workspace": WORKSPACE},
        {"briefing": BRIEFING},
        {"association": ASSOCIATION},
        {"document": DOCUMENT, "job": JOB},
        id="wrapped",
    ),
    pytest.param(WORKSPACE, BRIEFING, ASSOCIATION, JOB, id="bare"),
]


@pytest.mark.parametrize(("ws", "briefing", "assoc", "reprocess"), MISC_CASES)
@pytest.mark.asyncio
@respx.mock
async def test_async_misc_methods_parse_response(
    async_client: MemoryLayerClient, ws: dict, briefing: dict, assoc: dict, reprocess: dict
) -> None:
    _mock_misc_routes(ws, briefing, assoc, reprocess)

    async with async_client:
        created = await async_client.create_workspace("Test")
        fetched = await async_client.get_workspace("ws_test")
        got_briefing = await async_client.get_briefing()
        association = await async_client.associate("mem_1", "mem_2", "related_to")
        job = await async_client.reprocess_document("doc_1")

    assert created.id == fetched.id == "ws_test"
    assert got_briefing.workspace_summary == {"total_memories": 3}
    assert association.id == "assoc_1"
    assert job.id == "job_1"


@pytest.mark.parametrize(("ws", "briefing", "assoc", "reprocess"), MISC_CASES)
@respx.mock
def test_sync_misc_methods_parse_response(
    sync_client: SyncMemoryLayerClient, ws: dict, briefing: dict, assoc: dict, reprocess: dict
) -> None:
    _mock_misc_routes(ws, briefing, assoc, reprocess)

    with sync_client:
        created = sync_client.create_workspace("Test")
        fetched = sync_client.get_workspace("ws_test")
        got_briefing = sync_client.get_briefing()
        association = sync_client.associate("mem_1", "mem_2", "related_to")
        job = sync_client.reprocess_document("doc_1")

    assert created.id == fetched.id == "ws_test"
    assert got_briefing.workspace_summary == {"total_memories": 3}
    assert association.id == "assoc_1"
    assert job.id == "job_1"


@pytest.mark.asyncio
@respx.mock
async def test_async_list_documents_uses_server_total(async_client: MemoryLayerClient) -> None:
    respx.get(url__startswith=f"{BASE_URL}/v1/documents").mock(
        return_value=Response(200, json={"documents": [DOCUMENT], "total": 7, "limit": 1, "offset": 0}),
    )

    async with async_client:
        docs, total = await async_client.list_documents(limit=1)

    assert [d.id for d in docs] == ["doc_1"]
    assert total == 7


@respx.mock
def test_sync_list_documents_uses_server_total(sync_client: SyncMemoryLayerClient) -> None:
    respx.get(url__startswith=f"{BASE_URL}/v1/documents").mock(
        return_value=Response(200, json={"documents": [DOCUMENT], "total": 7, "limit": 1, "offset": 0}),
    )

    with sync_client:
        docs, total = sync_client.list_documents(limit=1)

    assert [d.id for d in docs] == ["doc_1"]
    assert total == 7
