"""
Unit tests for SQLite storage backend.

Tests storage layer functionality from the spec (Section 4, 8):
- SQLite backend operations
- Vector search (with and without sqlite-vec)
- Full-text search
- Association storage and graph traversal
- Workspace storage
- Session persistence storage
"""
import hashlib
import pytest
from datetime import datetime, timedelta, timezone

from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.models.memory import RememberInput, MemoryType, MemorySubtype
from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.workspace import Workspace
from memorylayer_server.models.session import Session, WorkingMemory

# Embedding dimension constant (must match mock embedding provider)
EMBEDDING_DIM = 384



# ============================================================================
# SQLite Backend Tests
# ============================================================================


@pytest.mark.asyncio
class TestSQLiteBackendLifecycle:
    """Test connection lifecycle and health checks."""

    async def test_connect_disconnect(self, temp_db_path):
        """Test connect() and disconnect() operations."""
        backend = SQLiteStorageBackend(str(temp_db_path))

        # Initially not connected
        assert await backend.health_check() is False

        # Connect
        await backend.connect()
        assert backend._connection is not None

        # Health check after connect
        assert await backend.health_check() is True

        # Disconnect
        await backend.disconnect()

    async def test_health_check_returns_true_when_connected(self, storage_backend):
        """Test health_check() returns True when connected."""
        assert await storage_backend.health_check() is True

    async def test_health_check_returns_false_when_disconnected(self, temp_db_path):
        """Test health_check() returns False when not connected."""
        backend = SQLiteStorageBackend(str(temp_db_path))
        assert await backend.health_check() is False


@pytest.mark.asyncio
class TestMemoryOperations:
    """Test core memory CRUD operations."""

    async def test_create_memory_stores_correctly(self, storage_backend, workspace_id):
        """Test create_memory() stores memory correctly."""
        input_data = RememberInput(
            content="Python is the preferred backend language",
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.PREFERENCE,
            importance=0.8,
            tags=["programming", "languages"],
            metadata={"source": "conversation", "confidence": 0.95}
        )

        memory = await storage_backend.create_memory(workspace_id, input_data)

        # Verify returned memory
        assert memory.id.startswith("mem_")
        assert memory.workspace_id == workspace_id
        assert memory.content == "Python is the preferred backend language"
        assert memory.type == MemoryType.SEMANTIC
        assert memory.subtype == MemorySubtype.PREFERENCE
        assert memory.importance == 0.8
        assert memory.tags == ["languages", "programming"]  # Sorted and normalized
        assert memory.metadata == {"source": "conversation", "confidence": 0.95}
        assert memory.content_hash == hashlib.sha256(input_data.content.encode()).hexdigest()
        assert memory.access_count == 0
        assert memory.created_at is not None
        assert memory.updated_at is not None

    async def test_get_memory_retrieves_by_id(self, storage_backend, workspace_id):
        """Test get_memory() retrieves by ID."""
        # Create a memory
        input_data = RememberInput(
            content="Test memory content",
            type=MemoryType.EPISODIC,
            importance=0.7
        )
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Retrieve it
        retrieved = await storage_backend.get_memory(workspace_id, created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == "Test memory content"
        assert retrieved.type == MemoryType.EPISODIC
        # Access count should increment on read
        assert retrieved.access_count == 1

    async def test_get_memory_increments_access_count(self, storage_backend, workspace_id):
        """Test that get_memory() increments access count on each retrieval."""
        # Create a memory
        input_data = RememberInput(content="Track access count", importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Access multiple times
        mem1 = await storage_backend.get_memory(workspace_id, created.id)
        assert mem1.access_count == 1

        mem2 = await storage_backend.get_memory(workspace_id, created.id)
        assert mem2.access_count == 2

        mem3 = await storage_backend.get_memory(workspace_id, created.id)
        assert mem3.access_count == 3

    async def test_get_memory_returns_none_for_nonexistent(self, storage_backend, workspace_id):
        """Test get_memory() returns None for non-existent ID."""
        result = await storage_backend.get_memory(workspace_id, "mem_nonexistent")
        assert result is None

    async def test_get_memory_returns_none_for_deleted(self, storage_backend, workspace_id):
        """Test get_memory() returns None for soft-deleted memory."""
        # Create and soft delete
        input_data = RememberInput(content="To be deleted", importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)
        await storage_backend.delete_memory(workspace_id, created.id, hard=False)

        # Should not retrieve
        result = await storage_backend.get_memory(workspace_id, created.id)
        assert result is None

    async def test_update_memory_modifies_fields(self, storage_backend, workspace_id):
        """Test update_memory() modifies fields correctly."""
        # Create a memory
        input_data = RememberInput(
            content="Original content",
            importance=0.5,
            tags=["tag1"]
        )
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Update multiple fields
        updated = await storage_backend.update_memory(
            workspace_id,
            created.id,
            importance=0.9,
            tags=["tag1", "tag2", "tag3"],
            metadata={"updated": True}
        )

        assert updated is not None
        assert updated.id == created.id
        assert updated.importance == 0.9
        assert updated.tags == ["tag1", "tag2", "tag3"]
        assert updated.metadata == {"updated": True}
        # Note: updated_at is set (SQLite datetime() truncates microseconds, so can't reliably compare)
        assert updated.updated_at is not None
        # Internal get_memory() in update_memory uses track_access=False, so no increment
        assert updated.access_count == 0

    async def test_update_memory_with_embedding(self, storage_backend, workspace_id):
        """Test update_memory() can store embeddings."""
        # Create a memory
        input_data = RememberInput(content="Memory with embedding", importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Add embedding
        embedding = ([0.1, 0.2, 0.3, 0.4, 0.5] * 77)[:EMBEDDING_DIM]  # 384-dim vector
        updated = await storage_backend.update_memory(
            workspace_id,
            created.id,
            embedding=embedding
        )

        assert updated.embedding is not None
        assert len(updated.embedding) == EMBEDDING_DIM
        # Check approximate equality due to float serialization
        for i, val in enumerate(updated.embedding):
            assert abs(val - embedding[i]) < 0.0001

    async def test_delete_memory_soft_delete(self, storage_backend, workspace_id):
        """Test delete_memory() soft delete (sets deleted_at)."""
        # Create a memory
        input_data = RememberInput(content="Soft delete test", importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Soft delete
        result = await storage_backend.delete_memory(workspace_id, created.id, hard=False)
        assert result is True

        # Should not be retrievable
        retrieved = await storage_backend.get_memory(workspace_id, created.id)
        assert retrieved is None

    async def test_delete_memory_hard_delete(self, storage_backend, workspace_id):
        """Test delete_memory() hard delete (removes from DB)."""
        # Create a memory
        input_data = RememberInput(content="Hard delete test", importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Hard delete
        result = await storage_backend.delete_memory(workspace_id, created.id, hard=True)
        assert result is True

        # Should not exist in DB at all
        retrieved = await storage_backend.get_memory(workspace_id, created.id)
        assert retrieved is None

    async def test_delete_memory_returns_false_for_nonexistent(self, storage_backend, workspace_id):
        """Test delete_memory() returns False for non-existent memory."""
        result = await storage_backend.delete_memory(workspace_id, "mem_nonexistent", hard=False)
        assert result is False

    async def test_get_memory_by_hash_for_deduplication(self, storage_backend, workspace_id):
        """Test get_memory_by_hash() for deduplication."""
        content = "Unique content for deduplication"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Create memory
        input_data = RememberInput(content=content, importance=0.5)
        created = await storage_backend.create_memory(workspace_id, input_data)

        # Find by hash
        found = await storage_backend.get_memory_by_hash(workspace_id, content_hash)

        assert found is not None
        assert found.id == created.id
        assert found.content == content
        assert found.content_hash == content_hash

    async def test_get_memory_by_hash_returns_none_for_nonexistent(self, storage_backend, workspace_id):
        """Test get_memory_by_hash() returns None for non-existent hash."""
        fake_hash = hashlib.sha256(b"nonexistent").hexdigest()
        result = await storage_backend.get_memory_by_hash(workspace_id, fake_hash)
        assert result is None


# ============================================================================
# Vector Search Tests
# ============================================================================


@pytest.mark.asyncio
class TestVectorSearch:
    """Test vector similarity search functionality.

    Uses class_workspace_id for isolation from other test classes.
    """

    async def test_search_memories_with_embedding_similarity(self, storage_backend, class_workspace_id):
        """Test search_memories() with embedding similarity."""
        workspace_id = class_workspace_id  # Local alias for cleaner code
        # Create memories with embeddings
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Python programming", importance=0.8)
        )
        await storage_backend.update_memory(
            workspace_id, mem1.id,
            embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1)
        )

        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="JavaScript development", importance=0.7)
        )
        await storage_backend.update_memory(
            workspace_id, mem2.id,
            embedding=[0.9, 0.1] + [0.0] * (EMBEDDING_DIM - 2)
        )

        mem3 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Database design", importance=0.6)
        )
        await storage_backend.update_memory(
            workspace_id, mem3.id,
            embedding=[0.0, 0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 3)
        )

        # Search with query embedding similar to mem1
        query_embedding = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=10,
            min_relevance=0.5
        )

        # Should return memories ordered by relevance
        assert len(results) > 0
        # First result should be mem1 (perfect match)
        assert results[0][0].id == mem1.id
        assert results[0][1] >= 0.99  # High relevance score

    async def test_search_memories_respects_limit(self, storage_backend, class_workspace_id):
        """Test search_memories() respects limit parameter."""
        workspace_id = class_workspace_id
        # Create 5 memories with embeddings
        for i in range(5):
            mem = await storage_backend.create_memory(
                workspace_id,
                RememberInput(content=f"Memory {i}", importance=0.5)
            )
            await storage_backend.update_memory(
                workspace_id, mem.id,
                embedding=[0.5] * EMBEDDING_DIM
            )

        # Search with limit=3
        query_embedding = [0.5] * EMBEDDING_DIM
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=3
        )

        assert len(results) == 3

    async def test_search_memories_respects_min_relevance(self, storage_backend, class_workspace_id):
        """Test search_memories() respects min_relevance threshold."""
        workspace_id = class_workspace_id
        # Create memories with different similarity levels
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="High relevance", importance=0.8)
        )
        await storage_backend.update_memory(
            workspace_id, mem1.id,
            embedding=[1.0] + [0.0] * (EMBEDDING_DIM - 1)  # Very similar
        )

        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Low relevance", importance=0.5)
        )
        await storage_backend.update_memory(
            workspace_id, mem2.id,
            embedding=[0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2)  # Very different
        )

        # Search with high min_relevance
        query_embedding = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=10,
            min_relevance=0.9  # High threshold
        )

        # Should only return high relevance memory
        assert len(results) >= 1
        # All results should meet min_relevance
        for memory, score in results:
            assert score >= 0.9

    async def test_search_memories_filters_by_types(self, storage_backend, class_workspace_id):
        """Test search_memories() filters by memory types."""
        workspace_id = class_workspace_id
        # Create memories of different types
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="Episodic memory",
                type=MemoryType.EPISODIC,
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem1.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="Semantic memory",
                type=MemoryType.SEMANTIC,
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem2.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        # Search filtered by type
        query_embedding = [0.5] * EMBEDDING_DIM
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=10,
            types=["episodic"]
        )

        # Should only return episodic memories
        assert len(results) == 1
        assert results[0][0].id == mem1.id
        assert results[0][0].type == MemoryType.EPISODIC

    async def test_search_memories_filters_by_subtypes(self, storage_backend, class_workspace_id):
        """Test search_memories() filters by memory subtypes."""
        workspace_id = class_workspace_id
        # Create memories with different subtypes
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="Preference memory",
                subtype=MemorySubtype.PREFERENCE,
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem1.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="Solution memory",
                subtype=MemorySubtype.SOLUTION,
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem2.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        # Search filtered by subtype
        query_embedding = [0.5] * EMBEDDING_DIM
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=10,
            subtypes=["preference"]
        )

        # Should only return preference memories
        assert len(results) == 1
        assert results[0][0].id == mem1.id
        assert results[0][0].subtype == MemorySubtype.PREFERENCE

    async def test_search_memories_filters_by_tags(self, storage_backend, class_workspace_id):
        """Test search_memories() filters by tags."""
        workspace_id = class_workspace_id
        # Create memories with different tags
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="Python memory",
                tags=["python", "programming"],
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem1.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(
                content="JavaScript memory",
                tags=["javascript", "programming"],
                importance=0.7
            )
        )
        await storage_backend.update_memory(
            workspace_id, mem2.id,
            embedding=[0.5] * EMBEDDING_DIM
        )

        # Search filtered by tag
        query_embedding = [0.5] * EMBEDDING_DIM
        results = await storage_backend.search_memories(
            workspace_id,
            query_embedding,
            limit=10,
            tags=["python"]
        )

        # Should only return python-tagged memories
        assert len(results) == 1
        assert results[0][0].id == mem1.id
        assert "python" in results[0][0].tags


# ============================================================================
# Full-Text Search Tests
# ============================================================================


@pytest.mark.asyncio
class TestFullTextSearch:
    """Test FTS5 full-text search functionality.

    Uses class_workspace_id for isolation from other test classes.
    """

    async def test_full_text_search_finds_content_matches(self, storage_backend, class_workspace_id):
        """Test full_text_search() finds content matches."""
        workspace_id = class_workspace_id
        # Create memories with searchable content
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Python is great for backend development", importance=0.7)
        )
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="JavaScript is used for frontend", importance=0.6)
        )
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Database design patterns", importance=0.5)
        )

        # Search for "Python"
        results = await storage_backend.full_text_search(workspace_id, "Python", limit=10)

        assert len(results) == 1
        assert "Python" in results[0].content

    async def test_full_text_search_respects_limit(self, storage_backend, class_workspace_id):
        """Test full_text_search() respects limit parameter."""
        workspace_id = class_workspace_id
        # Create multiple memories with common word
        for i in range(5):
            await storage_backend.create_memory(
                workspace_id,
                RememberInput(content=f"Programming language {i}", importance=0.5)
            )

        # Search with limit
        results = await storage_backend.full_text_search(workspace_id, "programming", limit=3)

        assert len(results) == 3

    async def test_full_text_search_case_insensitive(self, storage_backend, class_workspace_id):
        """Test full_text_search() is case insensitive."""
        workspace_id = class_workspace_id
        # Use a unique word to avoid collision with other tests in same class
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="UPPERCASE xyzuniqueterm LOWERCASE", importance=0.5)
        )

        # Search with different cases
        results_lower = await storage_backend.full_text_search(workspace_id, "xyzuniqueterm", limit=10)
        results_upper = await storage_backend.full_text_search(workspace_id, "XYZUNIQUETERM", limit=10)
        results_mixed = await storage_backend.full_text_search(workspace_id, "XyzUniqueTerm", limit=10)

        assert len(results_lower) == 1
        assert len(results_upper) == 1
        assert len(results_mixed) == 1

    async def test_full_text_search_excludes_deleted(self, storage_backend, class_workspace_id):
        """Test full_text_search() excludes soft-deleted memories."""
        workspace_id = class_workspace_id
        # Create and delete a memory
        mem = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="This will be deleted uniqueftsexcluded", importance=0.5)
        )
        await storage_backend.delete_memory(workspace_id, mem.id, hard=False)

        # Search should not find it
        results = await storage_backend.full_text_search(workspace_id, "uniqueftsexcluded", limit=10)
        assert len(results) == 0


# ============================================================================
# Association Storage Tests
# ============================================================================


@pytest.mark.asyncio
class TestAssociationStorage:
    """Test association (graph edge) storage and retrieval."""

    async def test_create_association_stores_correctly(self, storage_backend, workspace_id):
        """Test create_association() stores correctly."""
        # Create two memories to associate
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 1", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 2", importance=0.5)
        )

        # Create association
        assoc_input = AssociateInput(
            source_id=mem1.id,
            target_id=mem2.id,
            relationship="solves",
            strength=0.8,
            metadata={"reason": "test association"}
        )

        association = await storage_backend.create_association(workspace_id, assoc_input)

        assert association.id.startswith("assoc_")
        assert association.workspace_id == workspace_id
        assert association.source_id == mem1.id
        assert association.target_id == mem2.id
        assert association.relationship == "solves"
        assert association.strength == 0.8
        assert association.metadata == {"reason": "test association"}
        assert association.created_at is not None

    async def test_get_associations_retrieves_by_memory_id(self, storage_backend, workspace_id):
        """Test get_associations() retrieves by memory ID."""
        # Create memories
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Source memory", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Target memory 1", importance=0.5)
        )
        mem3 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Target memory 2", importance=0.5)
        )

        # Create associations
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="related_to",
                strength=0.7
            )
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem3.id,
                relationship="solves",
                strength=0.9
            )
        )

        # Get all associations for mem1
        associations = await storage_backend.get_associations(
            workspace_id,
            mem1.id,
            direction="both"
        )

        assert len(associations) == 2
        assert all(assoc.source_id == mem1.id for assoc in associations)

    async def test_get_associations_filters_by_direction_outgoing(self, storage_backend, workspace_id):
        """Test get_associations() filters by direction (outgoing)."""
        # Create memories and bidirectional associations
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 1", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 2", importance=0.5)
        )

        # mem1 -> mem2
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="leads_to",
                strength=0.8
            )
        )
        # mem2 -> mem1
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem2.id,
                target_id=mem1.id,
                relationship="builds_on",
                strength=0.6
            )
        )

        # Get outgoing associations from mem1
        outgoing = await storage_backend.get_associations(
            workspace_id,
            mem1.id,
            direction="outgoing"
        )

        assert len(outgoing) == 1
        assert outgoing[0].source_id == mem1.id
        assert outgoing[0].target_id == mem2.id

    async def test_get_associations_filters_by_direction_incoming(self, storage_backend, workspace_id):
        """Test get_associations() filters by direction (incoming)."""
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 1", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 2", importance=0.5)
        )

        # mem1 -> mem2
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="leads_to",
                strength=0.8
            )
        )
        # mem2 -> mem1
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem2.id,
                target_id=mem1.id,
                relationship="builds_on",
                strength=0.6
            )
        )

        # Get incoming associations to mem1
        incoming = await storage_backend.get_associations(
            workspace_id,
            mem1.id,
            direction="incoming"
        )

        assert len(incoming) == 1
        assert incoming[0].source_id == mem2.id
        assert incoming[0].target_id == mem1.id

    async def test_get_associations_filters_by_relationship_type(self, storage_backend, workspace_id):
        """Test get_associations() filters by relationship type."""
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Source", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Target 1", importance=0.5)
        )
        mem3 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Target 2", importance=0.5)
        )

        # Different relationship types
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="solves",
                strength=0.8
            )
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem3.id,
                relationship="related_to",
                strength=0.6
            )
        )

        # Filter by SOLVES relationship
        solves_assocs = await storage_backend.get_associations(
            workspace_id,
            mem1.id,
            relationships=["solves"]
        )

        assert len(solves_assocs) == 1
        assert solves_assocs[0].relationship == "solves"
        assert solves_assocs[0].target_id == mem2.id

    async def test_traverse_graph_multi_hop_queries(self, storage_backend, workspace_id):
        """Test traverse_graph() multi-hop queries."""
        # Create a chain: mem1 -> mem2 -> mem3 -> mem4
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 1", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 2", importance=0.5)
        )
        mem3 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 3", importance=0.5)
        )
        mem4 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Memory 4", importance=0.5)
        )

        # Create chain
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="leads_to",
                strength=0.8
            )
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem2.id,
                target_id=mem3.id,
                relationship="leads_to",
                strength=0.9
            )
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem3.id,
                target_id=mem4.id,
                relationship="leads_to",
                strength=0.7
            )
        )

        # Traverse from mem1 with max_depth=3
        result = await storage_backend.traverse_graph(
            workspace_id,
            mem1.id,
            max_depth=3,
            direction="outgoing"
        )

        # Should find 3 paths: mem1->mem2 (depth 1), mem1->mem2->mem3 (depth 2), mem1->mem2->mem3->mem4 (depth 3)
        assert result.total_paths == 3
        assert mem1.id in result.unique_nodes
        assert mem2.id in result.unique_nodes
        assert mem3.id in result.unique_nodes
        assert mem4.id in result.unique_nodes

        # Verify path depths
        depths = [path.depth for path in result.paths]
        assert 1 in depths  # Direct link to mem2
        assert 2 in depths  # Two hops to mem3
        assert 3 in depths  # Three hops to mem4

    async def test_traverse_graph_respects_max_depth(self, storage_backend, workspace_id):
        """Test traverse_graph() respects max_depth parameter."""
        # Create a long chain
        memories = []
        for i in range(5):
            mem = await storage_backend.create_memory(
                workspace_id,
                RememberInput(content=f"Memory {i}", importance=0.5)
            )
            memories.append(mem)

            if i > 0:
                await storage_backend.create_association(
                    workspace_id,
                    AssociateInput(
                        source_id=memories[i-1].id,
                        target_id=mem.id,
                        relationship="leads_to",
                        strength=0.8
                    )
                )

        # Traverse with max_depth=2
        result = await storage_backend.traverse_graph(
            workspace_id,
            memories[0].id,
            max_depth=2,
            direction="outgoing"
        )

        # Should only reach depth 2 (paths to memories[1] and memories[2])
        max_path_depth = max(path.depth for path in result.paths) if result.paths else 0
        assert max_path_depth <= 2
        assert result.total_paths == 2  # mem0->mem1 (depth 1), mem0->mem1->mem2 (depth 2)

        # Should reach memories[1] and memories[2]
        assert memories[1].id in result.unique_nodes
        assert memories[2].id in result.unique_nodes

        # Should not reach memories[3] or memories[4] (depth 3, 4)
        assert memories[3].id not in result.unique_nodes
        assert memories[4].id not in result.unique_nodes

    async def test_traverse_graph_filters_by_relationship(self, storage_backend, workspace_id):
        """Test traverse_graph() filters by relationship types."""
        # Create memories with different relationships
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Start", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Solves path", importance=0.5)
        )
        mem3 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Related path", importance=0.5)
        )

        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="solves",
                strength=0.8
            )
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem3.id,
                relationship="related_to",
                strength=0.6
            )
        )

        # Traverse with SOLVES filter
        result = await storage_backend.traverse_graph(
            workspace_id,
            mem1.id,
            max_depth=2,
            relationships=["solves"]
        )

        # Should only find mem2
        assert mem2.id in result.unique_nodes
        assert mem3.id not in result.unique_nodes


# ============================================================================
# Workspace Storage Tests
# ============================================================================


@pytest.mark.asyncio
class TestWorkspaceStorage:
    """Test workspace storage operations."""

    async def test_create_workspace_and_get_workspace(self, storage_backend):
        """Test create_workspace() and get_workspace()."""
        workspace = Workspace(
            id="test_workspace_123",
            tenant_id="tenant_abc",
            name="Test Workspace",
            settings={"auto_remember": True},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        # Create
        created = await storage_backend.create_workspace(workspace)
        assert created.id == workspace.id
        assert created.name == workspace.name

        # Retrieve
        retrieved = await storage_backend.get_workspace(workspace.id)
        assert retrieved is not None
        assert retrieved.id == workspace.id
        assert retrieved.tenant_id == workspace.tenant_id
        assert retrieved.name == workspace.name
        assert retrieved.settings == {"auto_remember": True}

    async def test_get_workspace_returns_none_for_nonexistent(self, storage_backend):
        """Test get_workspace() returns None for non-existent workspace."""
        result = await storage_backend.get_workspace("nonexistent_workspace")
        assert result is None




@pytest.mark.asyncio
class TestWorkspaceStats:
    """Test workspace statistics.

    Uses unique_workspace_id for each test method because these tests
    verify exact counts which require complete isolation.
    """

    async def test_get_workspace_stats(self, storage_backend, unique_workspace_id):
        """Test get_workspace_stats() returns correct counts."""
        workspace_id = unique_workspace_id
        # Create memories of different types
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Episodic 1", type=MemoryType.EPISODIC, importance=0.5)
        )
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Episodic 2", type=MemoryType.EPISODIC, importance=0.5)
        )
        await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Semantic 1", type=MemoryType.SEMANTIC, importance=0.5)
        )

        # Create associations
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Source", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Target", importance=0.5)
        )
        await storage_backend.create_association(
            workspace_id,
            AssociateInput(
                source_id=mem1.id,
                target_id=mem2.id,
                relationship="related_to",
                strength=0.8
            )
        )

        # Get stats
        stats = await storage_backend.get_workspace_stats(workspace_id)

        assert stats["total_memories"] == 5
        assert stats["memory_types"]["episodic"] == 2
        assert stats["memory_types"]["semantic"] == 3
        assert stats["total_associations"] == 1
        assert stats["total_categories"] == 0

    async def test_get_workspace_stats_excludes_deleted(self, storage_backend, unique_workspace_id):
        """Test get_workspace_stats() excludes soft-deleted memories."""
        workspace_id = unique_workspace_id
        # Create and delete a memory
        mem1 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Active memory stats", importance=0.5)
        )
        mem2 = await storage_backend.create_memory(
            workspace_id,
            RememberInput(content="Deleted memory stats", importance=0.5)
        )
        await storage_backend.delete_memory(workspace_id, mem2.id, hard=False)

        # Stats should only count active memory
        stats = await storage_backend.get_workspace_stats(workspace_id)
        assert stats["total_memories"] == 1


# ============================================================================
# Session Storage Tests
# ============================================================================


@pytest.mark.asyncio
class TestSessionStorage:
    """Test session persistence storage operations."""

    async def _ensure_workspace(self, sqlite_storage, workspace_id: str):
        """Helper to ensure workspace exists for foreign key constraint."""
        existing = await sqlite_storage.get_workspace(workspace_id)
        if not existing:
            workspace = Workspace(
                id=workspace_id,
                tenant_id="test_tenant",
                name=f"Test Workspace {workspace_id}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await sqlite_storage.create_workspace(workspace)

    async def test_create_session_stores_correctly(self, storage_backend, unique_workspace_id):
        """Test create_session() stores session correctly."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_test_create_123",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
            user_id="user_123",
            metadata={"client": "test-client", "version": "1.0"},
        )

        created = await storage_backend.create_session(workspace_id, session)

        assert created.id == "sess_test_create_123"
        assert created.workspace_id == workspace_id
        assert created.user_id == "user_123"
        assert created.metadata == {"client": "test-client", "version": "1.0"}
        assert created.expires_at is not None
        assert created.created_at is not None

    async def test_get_session_retrieves_by_id(self, storage_backend, unique_workspace_id):
        """Test get_session() retrieves session by ID."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_test_get_456",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        retrieved = await storage_backend.get_session(workspace_id, "sess_test_get_456")

        assert retrieved is not None
        assert retrieved.id == "sess_test_get_456"
        assert retrieved.workspace_id == workspace_id

    async def test_get_session_returns_none_for_nonexistent(self, storage_backend, unique_workspace_id):
        """Test get_session() returns None for non-existent session."""
        workspace_id = unique_workspace_id
        result = await storage_backend.get_session(workspace_id, "sess_nonexistent")
        assert result is None

    async def test_get_session_returns_expired_session(self, storage_backend, unique_workspace_id):
        """Test get_session() returns expired sessions (background task handles cleanup)."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        # Create an already-expired session
        session = Session.create_with_ttl(
            session_id="sess_expired_test",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=-3600,  # Expired 1 hour ago
        )
        await storage_backend.create_session(workspace_id, session)

        # get_session does NOT filter expired sessions; background cleanup handles that
        result = await storage_backend.get_session(workspace_id, "sess_expired_test")
        assert result is not None
        assert result.id == "sess_expired_test"
        assert result.is_expired

    async def test_delete_session_removes_session(self, storage_backend, unique_workspace_id):
        """Test delete_session() removes session."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_to_delete",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Delete
        result = await storage_backend.delete_session(workspace_id, "sess_to_delete")
        assert result is True

        # Verify deleted
        retrieved = await storage_backend.get_session(workspace_id, "sess_to_delete")
        assert retrieved is None

    async def test_delete_session_returns_false_for_nonexistent(self, storage_backend, unique_workspace_id):
        """Test delete_session() returns False for non-existent session."""
        workspace_id = unique_workspace_id
        result = await storage_backend.delete_session(workspace_id, "sess_does_not_exist")
        assert result is False


@pytest.mark.asyncio
class TestWorkingMemoryStorage:
    """Test working memory storage operations."""

    async def _ensure_workspace(self, sqlite_storage, workspace_id: str):
        """Helper to ensure workspace exists for foreign key constraint."""
        existing = await sqlite_storage.get_workspace(workspace_id)
        if not existing:
            workspace = Workspace(
                id=workspace_id,
                tenant_id="test_tenant",
                name=f"Test Workspace {workspace_id}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await sqlite_storage.create_workspace(workspace)

    async def test_set_working_memory_stores_correctly(self, storage_backend, unique_workspace_id):
        """Test set_working_memory() stores entry correctly."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        # Create session first
        session = Session.create_with_ttl(
            session_id="sess_ctx_test_1",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Set context
        ctx = await storage_backend.set_working_memory(
            workspace_id=workspace_id,
            session_id="sess_ctx_test_1",
            key="current_topic",
            value={"topic": "API design", "depth": 3},
            ttl_seconds=1800,
        )

        assert ctx.session_id == "sess_ctx_test_1"
        assert ctx.key == "current_topic"
        assert ctx.value == {"topic": "API design", "depth": 3}
        assert ctx.ttl_seconds == 1800
        assert ctx.created_at is not None
        assert ctx.updated_at is not None

    async def test_set_working_memory_upserts_existing(self, storage_backend, unique_workspace_id):
        """Test set_working_memory() updates existing key."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_upsert",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Set initial value
        await storage_backend.set_working_memory(
            workspace_id=workspace_id,
            session_id="sess_ctx_upsert",
            key="counter",
            value=1,
        )

        # Update value
        updated = await storage_backend.set_working_memory(
            workspace_id=workspace_id,
            session_id="sess_ctx_upsert",
            key="counter",
            value=42,
        )

        assert updated.value == 42

        # Verify only one entry exists
        all_ctx = await storage_backend.get_all_working_memory(workspace_id, "sess_ctx_upsert")
        assert len(all_ctx) == 1
        assert all_ctx[0].value == 42

    async def test_get_working_memory_retrieves_by_key(self, storage_backend, unique_workspace_id):
        """Test get_working_memory() retrieves specific key."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_get",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Set multiple contexts
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_get", "key1", "value1"
        )
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_get", "key2", "value2"
        )

        # Get specific key
        ctx = await storage_backend.get_working_memory(workspace_id, "sess_ctx_get", "key1")

        assert ctx is not None
        assert ctx.key == "key1"
        assert ctx.value == "value1"

    async def test_get_working_memory_returns_none_for_nonexistent_key(
        self, storage_backend, unique_workspace_id
    ):
        """Test get_working_memory() returns None for non-existent key."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_nokey",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        result = await storage_backend.get_working_memory(
            workspace_id, "sess_ctx_nokey", "nonexistent_key"
        )
        assert result is None

    async def test_get_all_working_memory_retrieves_all_keys(self, storage_backend, unique_workspace_id):
        """Test get_all_working_memory() retrieves all working memory entries."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_all",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Set multiple contexts
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_all", "topic", "Python"
        )
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_all", "mode", "learning"
        )
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_all", "preferences", {"dark_mode": True}
        )

        # Get all
        all_ctx = await storage_backend.get_all_working_memory(workspace_id, "sess_ctx_all")

        assert len(all_ctx) == 3
        keys = {ctx.key for ctx in all_ctx}
        assert keys == {"topic", "mode", "preferences"}

    async def test_get_all_working_memory_returns_empty_for_no_entries(
        self, storage_backend, unique_workspace_id
    ):
        """Test get_all_working_memory() returns empty list when no entries exist."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_empty",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        all_ctx = await storage_backend.get_all_working_memory(workspace_id, "sess_ctx_empty")
        assert all_ctx == []

    async def test_working_memory_stores_various_value_types(self, storage_backend, unique_workspace_id):
        """Test working memory stores various JSON-serializable value types."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        session = Session.create_with_ttl(
            session_id="sess_ctx_types",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # String
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "string_key", "string_value"
        )
        # Number
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "number_key", 42.5
        )
        # Boolean
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "bool_key", True
        )
        # List
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "list_key", [1, 2, 3, "four"]
        )
        # Dict
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "dict_key", {"nested": {"deep": "value"}}
        )
        # None/null
        await storage_backend.set_working_memory(
            workspace_id, "sess_ctx_types", "null_key", None
        )

        # Verify all values
        ctx_str = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "string_key")
        assert ctx_str.value == "string_value"

        ctx_num = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "number_key")
        assert ctx_num.value == 42.5

        ctx_bool = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "bool_key")
        assert ctx_bool.value is True

        ctx_list = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "list_key")
        assert ctx_list.value == [1, 2, 3, "four"]

        ctx_dict = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "dict_key")
        assert ctx_dict.value == {"nested": {"deep": "value"}}

        ctx_null = await storage_backend.get_working_memory(workspace_id, "sess_ctx_types", "null_key")
        assert ctx_null.value is None


@pytest.mark.asyncio
class TestSessionCleanup:
    """Test session cleanup operations."""

    async def _ensure_workspace(self, sqlite_storage, workspace_id: str):
        """Helper to ensure workspace exists for foreign key constraint."""
        existing = await sqlite_storage.get_workspace(workspace_id)
        if not existing:
            workspace = Workspace(
                id=workspace_id,
                tenant_id="test_tenant",
                name=f"Test Workspace {workspace_id}",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await sqlite_storage.create_workspace(workspace)

    async def test_cleanup_expired_sessions_removes_expired(self, storage_backend, unique_workspace_id):
        """Test cleanup_expired_sessions() removes expired sessions."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        # Create active session
        active_session = Session.create_with_ttl(
            session_id="sess_active_cleanup",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,  # Expires in 1 hour
        )
        await storage_backend.create_session(workspace_id, active_session)

        # Create expired sessions
        for i in range(3):
            expired_session = Session.create_with_ttl(
                session_id=f"sess_expired_cleanup_{i}",
                workspace_id=workspace_id,
                tenant_id="default_tenant",
                ttl_seconds=-3600,  # Expired 1 hour ago
            )
            await storage_backend.create_session(workspace_id, expired_session)

        # Cleanup
        count = await storage_backend.cleanup_expired_sessions(workspace_id)

        # Should have cleaned up 3 expired sessions
        assert count == 3

        # Active session should still exist
        active = await storage_backend.get_session(workspace_id, "sess_active_cleanup")
        assert active is not None

    async def test_cleanup_expired_sessions_returns_zero_when_none_expired(
        self, storage_backend, unique_workspace_id
    ):
        """Test cleanup_expired_sessions() returns 0 when no sessions are expired."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        # Create only active sessions
        for i in range(2):
            session = Session.create_with_ttl(
                session_id=f"sess_all_active_{i}",
                workspace_id=workspace_id,
                tenant_id="default_tenant",
                ttl_seconds=3600,
            )
            await storage_backend.create_session(workspace_id, session)

        count = await storage_backend.cleanup_expired_sessions(workspace_id)
        assert count == 0

    async def test_delete_session_cascades_to_context(self, storage_backend, unique_workspace_id):
        """Test delete_session() cascades to remove all session context."""
        workspace_id = unique_workspace_id
        await self._ensure_workspace(storage_backend, workspace_id)

        # Create session with context
        session = Session.create_with_ttl(
            session_id="sess_cascade_delete",
            workspace_id=workspace_id,
            tenant_id="default_tenant",
            ttl_seconds=3600,
        )
        await storage_backend.create_session(workspace_id, session)

        # Add context entries
        await storage_backend.set_working_memory(
            workspace_id, "sess_cascade_delete", "key1", "value1"
        )
        await storage_backend.set_working_memory(
            workspace_id, "sess_cascade_delete", "key2", "value2"
        )

        # Verify context exists
        ctx_before = await storage_backend.get_all_working_memory(workspace_id, "sess_cascade_delete")
        assert len(ctx_before) == 2

        # Delete session
        await storage_backend.delete_session(workspace_id, "sess_cascade_delete")

        # Context should be gone (CASCADE)
        ctx_after = await storage_backend.get_all_working_memory(workspace_id, "sess_cascade_delete")
        assert len(ctx_after) == 0
