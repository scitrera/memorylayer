"""Tests for the chat-thread workspace substitution in MemoryLayerClient.

Covers the SDK-side enforcement of the user-owned thread convention:

* ``create_thread(ownership='user')`` substitutes the caller's
  ``workspace_id`` to :data:`USER_CHAT_HOME_WORKSPACE`.
* ``append_messages(ownership='user')`` does the same AND folds the
  caller's workspace_id into each message's metadata under
  :data:`MESSAGE_META_APP_WORKSPACE_KEY`.
* ``ownership='workspace'`` is pass-through (legacy behavior).
* A warning is logged on substitution so callers can spot stale workspace
  threading without breaking.
"""
from __future__ import annotations

import logging

import pytest
import respx
from httpx import Response

from memorylayer import (
    MESSAGE_META_APP_WORKSPACE_KEY,
    MemoryLayerClient,
    USER_CHAT_HOME_WORKSPACE,
)


@pytest.fixture
def base_url() -> str:
    return "http://test.memorylayer.ai"


@pytest.fixture
def client(base_url: str) -> MemoryLayerClient:
    return MemoryLayerClient(base_url=base_url, api_key="test_key")


def _thread_response(workspace_id: str, thread_id: str = "thr_1", ownership: str = "user") -> dict:
    return {
        "id": thread_id,
        "workspace_id": workspace_id,
        "tenant_id": "_default",
        "user_id": None,
        "context_id": "_default",
        "observer_id": None,
        "subject_id": None,
        "title": None,
        "metadata": {},
        "scope": None,
        "ownership": ownership,
        "message_count": 0,
        "last_decomposed_at": None,
        "last_decomposed_index": 0,
        "expires_at": None,
        "created_at": "2026-05-28T18:00:00Z",
        "updated_at": "2026-05-28T18:00:00Z",
    }


def _messages_response() -> dict:
    return {
        "messages": [],
        "thread_id": "thr_1",
        "new_message_count": 0,
    }


# ── create_thread ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_create_thread_user_owned_substitutes_workspace(
    client: MemoryLayerClient, base_url: str, caplog: pytest.LogCaptureFixture,
) -> None:
    route = respx.post(f"{base_url}/v1/threads").mock(
        return_value=Response(200, json=_thread_response(USER_CHAT_HOME_WORKSPACE)),
    )

    with caplog.at_level(logging.WARNING, logger="memorylayer.client"):
        async with client:
            await client.create_thread(workspace_id="ws-app", ownership="user")

    sent = route.calls.last.request.read()
    import json as _json
    body = _json.loads(sent)
    assert body["workspace_id"] == USER_CHAT_HOME_WORKSPACE
    assert body["ownership"] == "user"
    # Warning fires because ws-app != sentinel
    assert any("overrides workspace_id" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_create_thread_user_owned_sentinel_no_warning(
    client: MemoryLayerClient, base_url: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """When the caller already passes the sentinel, no warning fires."""
    respx.post(f"{base_url}/v1/threads").mock(
        return_value=Response(200, json=_thread_response(USER_CHAT_HOME_WORKSPACE)),
    )

    with caplog.at_level(logging.WARNING, logger="memorylayer.client"):
        async with client:
            await client.create_thread(
                workspace_id=USER_CHAT_HOME_WORKSPACE, ownership="user",
            )
    assert not any("overrides workspace_id" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_create_thread_workspace_ownership_passes_through(
    client: MemoryLayerClient, base_url: str,
) -> None:
    """ownership='workspace' keeps the caller's workspace as-is (legacy)."""
    route = respx.post(f"{base_url}/v1/threads").mock(
        return_value=Response(200, json=_thread_response("ws-app", ownership="workspace")),
    )

    async with client:
        await client.create_thread(workspace_id="ws-app", ownership="workspace")

    import json as _json
    body = _json.loads(route.calls.last.request.read())
    assert body["workspace_id"] == "ws-app"
    assert body["ownership"] == "workspace"


# ── append_messages ───────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_append_messages_user_owned_substitutes_and_folds_metadata(
    client: MemoryLayerClient, base_url: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """User-owned append: storage workspace becomes the sentinel; the
    caller's app workspace is preserved on each message's metadata."""
    route = respx.post(f"{base_url}/v1/threads/thr_1/messages").mock(
        return_value=Response(201, json=_messages_response()),
    )

    with caplog.at_level(logging.WARNING, logger="memorylayer.client"):
        async with client:
            await client.append_messages(
                "thr_1",
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello", "metadata": {"x": 1}},
                ],
                workspace_id="ws-app",
                ownership="user",
            )

    # URL carries the substituted storage workspace
    sent_url = str(route.calls.last.request.url)
    assert f"workspace_id={USER_CHAT_HOME_WORKSPACE}" in sent_url

    # Body messages each carry the originating workspace on metadata
    import json as _json
    body = _json.loads(route.calls.last.request.read())
    assert body["messages"][0]["metadata"][MESSAGE_META_APP_WORKSPACE_KEY] == "ws-app"
    assert body["messages"][1]["metadata"][MESSAGE_META_APP_WORKSPACE_KEY] == "ws-app"
    # User-supplied metadata key not overwritten
    assert body["messages"][1]["metadata"]["x"] == 1

    assert any("overrides workspace_id" in r.message for r in caplog.records)


@pytest.mark.asyncio
@respx.mock
async def test_append_messages_user_owned_sentinel_no_fold_no_warning(
    client: MemoryLayerClient, base_url: str, caplog: pytest.LogCaptureFixture,
) -> None:
    """If the caller passed the sentinel directly, no fold + no warning."""
    route = respx.post(f"{base_url}/v1/threads/thr_1/messages").mock(
        return_value=Response(201, json=_messages_response()),
    )

    with caplog.at_level(logging.WARNING, logger="memorylayer.client"):
        async with client:
            await client.append_messages(
                "thr_1",
                [{"role": "user", "content": "hi"}],
                workspace_id=USER_CHAT_HOME_WORKSPACE,
                ownership="user",
            )

    assert not any("overrides workspace_id" in r.message for r in caplog.records)
    import json as _json
    body = _json.loads(route.calls.last.request.read())
    # No fold — message metadata not augmented
    assert MESSAGE_META_APP_WORKSPACE_KEY not in body["messages"][0].get("metadata", {})


@pytest.mark.asyncio
@respx.mock
async def test_append_messages_workspace_ownership_passes_through(
    client: MemoryLayerClient, base_url: str,
) -> None:
    """ownership='workspace' keeps the caller's workspace as-is (legacy)."""
    route = respx.post(f"{base_url}/v1/threads/thr_1/messages").mock(
        return_value=Response(201, json=_messages_response()),
    )

    async with client:
        await client.append_messages(
            "thr_1",
            [{"role": "user", "content": "hi"}],
            workspace_id="ws-app",
            ownership="workspace",
        )

    sent_url = str(route.calls.last.request.url)
    assert "workspace_id=ws-app" in sent_url
    # No fold for workspace-owned
    import json as _json
    body = _json.loads(route.calls.last.request.read())
    assert MESSAGE_META_APP_WORKSPACE_KEY not in body["messages"][0].get("metadata", {})


@pytest.mark.asyncio
@respx.mock
async def test_append_messages_obo_proxy_forwards_ownership(
    client: MemoryLayerClient, base_url: str,
) -> None:
    """The _OBOProxy shim forwards the new ownership kwarg unchanged."""
    route = respx.post(f"{base_url}/v1/threads/thr_1/messages").mock(
        return_value=Response(201, json=_messages_response()),
    )

    async with client:
        async with client.acting_for("g_xyz", subject=("user", "alice")) as obo:
            await obo.append_messages(
                "thr_1",
                [{"role": "user", "content": "hi"}],
                workspace_id="ws-app",
                ownership="user",
            )

    sent_url = str(route.calls.last.request.url)
    assert f"workspace_id={USER_CHAT_HOME_WORKSPACE}" in sent_url
    import json as _json
    body = _json.loads(route.calls.last.request.read())
    assert body["messages"][0]["metadata"][MESSAGE_META_APP_WORKSPACE_KEY] == "ws-app"
