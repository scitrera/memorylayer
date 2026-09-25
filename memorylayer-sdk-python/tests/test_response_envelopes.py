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
