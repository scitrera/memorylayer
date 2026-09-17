"""Long page transcripts must fit request limits without losing tail content."""
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.document.embed_client import EmbedServerClient, _split_embedding_text


@pytest.mark.parametrize("text,budget", [("abc def ghi", 5), ("你好 café 🌍 " * 5, 7), ("x" * 100, 9)])
def test_chunks_preserve_every_character_within_utf8_budget(text, budget):
    chunks = _split_embedding_text(text, budget)
    assert "".join(chunks) == text
    assert all(0 < len(chunk.encode("utf-8")) <= budget for chunk in chunks)


def test_impossibly_small_utf8_budget_fails_without_discarding_character():
    with pytest.raises(ValueError, match="UTF-8 character"):
        _split_embedding_text("🌍", 3)


async def test_long_text_is_pooled_and_short_neighbors_keep_their_own_vectors():
    client = EmbedServerClient("http://embed", logger=MagicMock(), text_batch_size=2, text_batch_bytes=4)
    vectors = {"a": [4., 0.], "abcd": [1., 0.], "é": [0., 1.], "z": [0., 3.]}
    async def send(method, path, payload):
        assert path == "/v1/embeddings" and payload["dimensions"] == 2
        assert sum(len(t.encode("utf-8")) for t in payload["input"]) <= 4
        return {"data": [{"index": i, "embedding": vectors[t]}
                         for i,t in reversed(list(enumerate(payload["input"]))) ]}
    client.request_json = AsyncMock(side_effect=send)
    actual = await client.embed_texts(["a", "abcdé", "z"], dimensions=2)
    assert actual[0] == [4., 0.] and actual[2] == [0., 3.]
    assert actual[1] == pytest.approx([4 / math.sqrt(20), 2 / math.sqrt(20)])
    assert [t for call in client.request_json.call_args_list for t in call.args[2]["input"]] == ["a", "abcd", "é", "z"]


@pytest.mark.parametrize("entries", [[], [{"index": 0, "embedding": [1.]}] * 2,
                                    [{"index": 0, "embedding": [1.]}, {"index": True, "embedding": [1.]}]])
async def test_missing_duplicate_and_noninteger_indices_cannot_shift_page_vectors(entries):
    client = EmbedServerClient("http://embed", logger=MagicMock())
    client.request_json = AsyncMock(return_value={"data": entries})
    with pytest.raises(ValueError, match="exactly one"):
        await client.embed_texts(["page one", "page two"])


@pytest.mark.parametrize("vectors", [[[1.], [1., 2.]], [[1.], [float("nan")]], [[1.], [-1.]]])
async def test_invalid_chunk_vectors_fail_closed(vectors):
    client = EmbedServerClient("http://embed", logger=MagicMock(), text_batch_bytes=1)
    client.request_json = AsyncMock(side_effect=[{"data": [{"index": 0, "embedding": v}]} for v in vectors])
    with pytest.raises(ValueError, match="embedding|dimensions"):
        await client.embed_texts(["ab"])
