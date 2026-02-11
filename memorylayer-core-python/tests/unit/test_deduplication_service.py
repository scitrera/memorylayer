"""Tests for DeduplicationService."""
import pytest

from memorylayer_server.services.deduplication import DeduplicationAction
from memorylayer_server.models import RememberInput
from memorylayer_server.utils import compute_content_hash


@pytest.mark.asyncio
async def test_check_duplicate_new_memory(deduplication_service):
    """Test that new unique content returns CREATE."""
    result = await deduplication_service.check_duplicate(
        content="test content for dedup",
        content_hash="unique_hash_abc123",
        embedding=[0.1] * 384,
        workspace_id="ws-dedup-1"
    )

    assert result.action == DeduplicationAction.CREATE
    assert result.reason == "New unique memory"


@pytest.mark.asyncio
async def test_check_duplicate_exact_match(storage_backend, deduplication_service):
    """Test that exact hash match returns SKIP."""
    # Pre-populate storage with an existing memory
    existing_content = "existing content for exact match test"
    existing_hash = compute_content_hash(existing_content)
    existing_embedding = [0.5] * 384

    existing_input = RememberInput(content=existing_content)
    existing_memory = await storage_backend.create_memory("ws-dedup-2", existing_input)
    # Add embedding
    await storage_backend.update_memory("ws-dedup-2", existing_memory.id, embedding=existing_embedding)

    # Try to add duplicate with same content (same hash)
    result = await deduplication_service.check_duplicate(
        content=existing_content,
        content_hash=existing_hash,
        embedding=existing_embedding,
        workspace_id="ws-dedup-2"
    )

    assert result.action == DeduplicationAction.SKIP
    assert result.existing_memory_id == existing_memory.id
    assert result.similarity_score == 1.0
    assert "Exact content duplicate" in result.reason


@pytest.mark.asyncio
async def test_check_duplicate_semantic_match(storage_backend, deduplication_service):
    """Test that high semantic similarity returns UPDATE."""
    # Create an existing memory with a specific embedding
    existing_embedding = [0.9] * 384

    existing_input = RememberInput(content="similar content here for semantic test")
    existing_memory = await storage_backend.create_memory("ws-dedup-3", existing_input)
    # Add embedding
    await storage_backend.update_memory("ws-dedup-3", existing_memory.id, embedding=existing_embedding)

    # Try to add semantically similar content (same embedding = cosine similarity 1.0)
    # Different hash but same embedding
    result = await deduplication_service.check_duplicate(
        content="similar content there for semantic test",
        content_hash="new_hash_789",
        embedding=existing_embedding,  # Same embedding = similarity 1.0
        workspace_id="ws-dedup-3"
    )

    assert result.action == DeduplicationAction.UPDATE
    assert result.existing_memory_id == existing_memory.id
    assert result.similarity_score >= 0.95


@pytest.mark.asyncio
async def test_deduplicate_batch(deduplication_service):
    """Test batch deduplication."""
    candidates = [
        ("batch content 1", "batch_hash_1", [0.1] * 384),
        ("batch content 2", "batch_hash_2", [0.2] * 384),
        ("batch content 3", "batch_hash_3", [0.3] * 384),
    ]

    results = await deduplication_service.deduplicate_batch(
        candidates=candidates,
        workspace_id="ws-dedup-batch"
    )

    assert len(results) == 3
    # All should be CREATE since workspace is empty
    assert all(r.action == DeduplicationAction.CREATE for r in results)
