"""Bounded role concurrency with stable batch indexes and no lost text."""
import asyncio
from unittest.mock import MagicMock
import pytest
from memorylayer_server.services.document.embed_client import EmbedServerClient


async def test_role_limits_are_shared_across_documents_and_roles_are_independent():
    client = EmbedServerClient("http://embed", logger=MagicMock(), image_batch_size=1,
        image_concurrency=3, text_concurrency=3, transcription_concurrency=2,
        text_batch_size=1)
    active = {"embedding": 0, "transcription": 0}
    peak = dict(active)
    together = False
    async def send(method, path, payload):
        nonlocal together
        role = "transcription" if path == "/v1/transcribe" else "embedding"
        active[role] += 1
        peak[role] = max(peak[role], active[role])
        together |= all(active.values())
        value = int(payload.get("input", payload.get("images"))[0])
        try:
            await asyncio.sleep(.005 * (5 - value % 5))
            if role == "transcription":
                return {"results": [{"page_index": 0, "success": value != 3, "content": str(value)}]}
            return {"data": [{"index": 0, "embedding": [float(value)]}]}
        finally:
            active[role] -= 1
    client.request_json = send
    values = [str(i) for i in range(8)]
    results = await asyncio.gather(client.embed_texts(values), client.embed_texts(values),
        client.embed_images(values), client.transcribe_pages(values), client.transcribe_pages(values))
    assert peak == {"embedding": 3, "transcription": 2}
    assert together and active == {"embedding": 0, "transcription": 0}
    for result in results[:3]:
        assert result == [[float(i)] for i in range(8)]
    for result in results[3:]:
        assert [p["page_index"] for p in result["results"]] == list(range(8))
        assert not result["results"][3]["success"]


async def test_text_fanout_cancellation_releases_slots():
    client = EmbedServerClient("http://embed", logger=MagicMock(), text_concurrency=2, text_batch_size=1)
    started = asyncio.Event()
    active = 0
    async def send(*args):
        nonlocal active
        active += 1
        if active == 2:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            active -= 1
    client.request_json = send
    task = asyncio.create_task(client.embed_texts(["a", "b", "c"]))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0
    assert client._embedding_slots._value == 2


@pytest.mark.parametrize("entries", [[{"index": 0, "embedding": [1]}] * 2, []])
async def test_text_missing_or_duplicate_indexes_fail(entries):
    client = EmbedServerClient("http://embed", logger=MagicMock(), text_batch_size=2, text_concurrency=2)
    async def send(*args):
        return {"data": entries}
    client.request_json = send
    with pytest.raises(ValueError, match="text index"):
        await client.embed_texts(["a", "b"])


@pytest.mark.parametrize("name", ["text_concurrency", "transcription_concurrency"])
@pytest.mark.parametrize("value", [0, 33, True, "8"])
def test_invalid_limits_rejected(name, value):
    with pytest.raises(ValueError):
        EmbedServerClient("http://embed", **{name: value})


async def test_deepseek_first_retries_only_failed_pages_and_preserves_evidence():
    client = EmbedServerClient("http://embed", logger=MagicMock(),
        transcription_providers=["deepseek-ocr", "unlimited-ocr"])
    calls = []
    async def send(method, path, payload):
        calls.append(payload)
        primary = payload["provider"] == "deepseek-ocr"
        entries = []
        for i, page in enumerate(payload["images"]):
            success = not (primary and page == "bad")
            entries.append({"page_index": i, "success": success, "content": page,
                "raw_content": "grounded:" + page,
                "output_contract": "deepseek_ocr" if primary else "unlimited_ocr",
                "provider_used": payload["provider"],
                "attempts": [{"provider": payload["provider"], "success": success}]})
        return {"results": list(reversed(entries)), "stats": {"total_tokens_in": 10}}
    client.request_json = send
    result = await client.transcribe_pages(["good", "bad", "other"])
    assert [(p["provider"], p["images"]) for p in calls] == [
        ("deepseek-ocr", ["good", "bad", "other"]), ("unlimited-ocr", ["bad"])]
    assert [p["page_index"] for p in result["results"]] == [0, 1, 2]
    assert result["results"][0]["output_contract"] == "deepseek_ocr"
    assert result["results"][1]["raw_content"] == "grounded:bad"
    assert len(result["results"][1]["attempts"]) == 2
    assert result["stats"]["successful_pages"] == 3
    assert result["stats"]["total_tokens_in"] == 20


async def test_ambiguous_transcription_transport_failure_does_not_try_another_provider():
    import httpx
    client = EmbedServerClient("http://embed", logger=MagicMock(),
        transcription_providers=["deepseek-ocr", "unlimited-ocr"])
    calls = []
    async def send(method, path, payload):
        calls.append(payload)
        raise httpx.ReadError("response lost")
    client.request_json = send
    with pytest.raises(httpx.ReadError):
        await client.transcribe_pages(["page"])
    assert len(calls) == 1


async def test_both_providers_failed_remains_failed():
    client = EmbedServerClient("http://embed", logger=MagicMock(),
        transcription_providers=["deepseek-ocr", "unlimited-ocr"])
    async def send(method, path, payload):
        return {"results": [{"page_index": 0, "success": False, "attempts": [{"provider": payload["provider"]}]}]}
    client.request_json = send
    result = await client.transcribe_pages(["page"])
    assert result["stats"]["failed_pages"] == 1
    assert not result["results"][0]["success"]
    assert len(result["results"][0]["attempts"]) == 2
