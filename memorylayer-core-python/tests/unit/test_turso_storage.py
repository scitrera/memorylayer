"""
Unit tests for Turso/libSQL storage backend.

Tests storage layer functionality mirroring SQLite backend tests:
- Turso backend lifecycle (connect, disconnect, health check)
- Memory CRUD operations
- Native vector search (vector_distance_cos)
- Full-text search (Turso native FTS or LIKE fallback)
- Association storage and graph traversal (iterative BFS)
- Workspace and context operations
- Session and working memory operations
- Contradiction operations
"""

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

turso = pytest.importorskip("turso", reason="pyturso not installed")

from memorylayer_server.models.association import AssociateInput
from memorylayer_server.models.memory import MemorySubtype, MemoryType, RememberInput
from memorylayer_server.models.session import Session
from memorylayer_server.models.workspace import Context, Workspace
from memorylayer_server.services.storage.turso import TursoStorageBackend, TursoStorageBackendPlugin

EMBEDDING_DIM = 384


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def backend():
    """Create and connect an in-memory Turso backend for testing."""
    b = TursoStorageBackend(mode="local", db_path=":memory:")
    await b.connect()
    yield b
    await b.disconnect()


@pytest.fixture
async def workspace_id(backend) -> str:
    """Create a test workspace and return its ID."""
    ws_id = "test_workspace"
    now = datetime.now(UTC)
    try:
        await backend.create_workspace(
            Workspace(id=ws_id, tenant_id="_default", name="Test Workspace", settings={}, created_at=now, updated_at=now)
        )
    except Exception:
        pass  # May already exist
    return ws_id


# ============================================================================
# Lifecycle Tests
# ============================================================================


@pytest.mark.asyncio
class TestTursoLifecycle:
    """Test connection lifecycle and health checks."""

    async def test_connect_disconnect(self):
        backend = TursoStorageBackend(mode="local", db_path=":memory:")
        assert await backend.health_check() is False

        await backend.connect()
        assert backend._connection is not None
        assert await backend.health_check() is True

        await backend.disconnect()
        assert backend._connection is None

    async def test_health_check_when_connected(self, backend):
        assert await backend.health_check() is True

    async def test_reserved_entities_created(self, backend):
        workspaces = await backend.list_workspaces()
        ws_ids = [w.id for w in workspaces]
        assert "_default" in ws_ids
        assert "_global" in ws_ids

    async def test_invalid_mode_raises(self):
        backend = TursoStorageBackend(mode="invalid")
        with pytest.raises(ValueError, match="Invalid MEMORYLAYER_TURSO_MODE"):
            await backend.connect()

    async def test_remote_mode_requires_url(self):
        backend = TursoStorageBackend(mode="remote", url=None)
        with pytest.raises(ValueError, match="MEMORYLAYER_TURSO_URL is required"):
            await backend.connect()

    async def test_replica_mode_requires_url(self):
        backend = TursoStorageBackend(mode="replica", url=None)
        with pytest.raises(ValueError, match="MEMORYLAYER_TURSO_URL is required"):
            await backend.connect()


# ============================================================================
# Memory CRUD Tests
# ============================================================================


@pytest.mark.asyncio
class TestMemoryOperations:
    """Test core memory CRUD operations."""

    async def test_create_memory(self, backend, workspace_id):
        inp = RememberInput(
            content="Python is a programming language",
            type=MemoryType.SEMANTIC,
            subtype=MemorySubtype.PREFERENCE,
            importance=0.8,
            tags=["programming"],
            metadata={"source": "test"},
        )
        mem = await backend.create_memory(workspace_id, inp)

        assert mem.id.startswith("mem_")
        assert mem.workspace_id == workspace_id
        assert mem.content == "Python is a programming language"
        assert mem.type == MemoryType.SEMANTIC
        assert mem.subtype == MemorySubtype.PREFERENCE
        assert mem.importance == 0.8
        assert mem.content_hash == hashlib.sha256(inp.content.encode()).hexdigest()
        assert mem.access_count == 0
        assert mem.created_at is not None

    async def test_get_memory(self, backend, workspace_id):
        inp = RememberInput(content="Test get", type=MemoryType.EPISODIC)
        created = await backend.create_memory(workspace_id, inp)

        fetched = await backend.get_memory(workspace_id, created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.content == "Test get"

    async def test_get_memory_tracks_access(self, backend, workspace_id):
        inp = RememberInput(content="Track access", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        fetched = await backend.get_memory(workspace_id, created.id, track_access=True)
        assert fetched.access_count == 1
        assert fetched.last_accessed_at is not None

    async def test_get_memory_returns_none_for_missing(self, backend, workspace_id):
        result = await backend.get_memory(workspace_id, "nonexistent")
        assert result is None

    async def test_get_memory_by_id(self, backend, workspace_id):
        inp = RememberInput(content="Global lookup", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        fetched = await backend.get_memory_by_id(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    async def test_update_memory(self, backend, workspace_id):
        inp = RememberInput(content="Original", type=MemoryType.SEMANTIC, importance=0.5)
        created = await backend.create_memory(workspace_id, inp)

        updated = await backend.update_memory(workspace_id, created.id, importance=0.9, tags=["updated"])
        assert updated is not None
        assert updated.importance == 0.9
        assert updated.tags == ["updated"]

    async def test_update_memory_with_embedding(self, backend, workspace_id):
        inp = RememberInput(content="Embed me", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        embedding = [0.1] * EMBEDDING_DIM
        updated = await backend.update_memory(workspace_id, created.id, embedding=embedding)
        assert updated.embedding is not None
        assert len(updated.embedding) == EMBEDDING_DIM

    async def test_update_memory_invalid_fields(self, backend, workspace_id):
        inp = RememberInput(content="Test", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        with pytest.raises(ValueError, match="Invalid update fields"):
            await backend.update_memory(workspace_id, created.id, nonexistent_field="bad")

    async def test_soft_delete(self, backend, workspace_id):
        inp = RememberInput(content="Delete me", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        result = await backend.delete_memory(workspace_id, created.id)
        assert result is True

        fetched = await backend.get_memory(workspace_id, created.id)
        assert fetched is None

    async def test_hard_delete(self, backend, workspace_id):
        inp = RememberInput(content="Hard delete me", type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        result = await backend.delete_memory(workspace_id, created.id, hard=True)
        assert result is True

        fetched = await backend.get_memory(workspace_id, created.id)
        assert fetched is None

    async def test_get_memory_by_hash(self, backend, workspace_id):
        content = "Unique content for hash test"
        inp = RememberInput(content=content, type=MemoryType.SEMANTIC)
        created = await backend.create_memory(workspace_id, inp)

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        found = await backend.get_memory_by_hash(workspace_id, content_hash)
        assert found is not None
        assert found.id == created.id


# ============================================================================
# Vector Search Tests
# ============================================================================


@pytest.mark.asyncio
class TestVectorSearch:
    """Test native libSQL vector search."""

    async def test_vector_search_basic(self, backend, workspace_id):
        """Test basic vector similarity search."""
        inp = RememberInput(content="Vector test", type=MemoryType.SEMANTIC)
        mem = await backend.create_memory(workspace_id, inp)

        embedding = [0.5] * EMBEDDING_DIM
        await backend.update_memory(workspace_id, mem.id, embedding=embedding)

        results = await backend.search_memories(workspace_id, [0.5] * EMBEDDING_DIM, limit=5, min_relevance=0.0)
        assert len(results) > 0
        assert results[0][0].id == mem.id

    async def test_vector_search_identical_returns_perfect_score(self, backend, workspace_id):
        """Identical embeddings should yield relevance ~1.0."""
        inp = RememberInput(content="Perfect match", type=MemoryType.SEMANTIC)
        mem = await backend.create_memory(workspace_id, inp)

        embedding = [0.3] * EMBEDDING_DIM
        await backend.update_memory(workspace_id, mem.id, embedding=embedding)

        results = await backend.search_memories(workspace_id, [0.3] * EMBEDDING_DIM, limit=5, min_relevance=0.0)
        assert len(results) > 0
        _, relevance = results[0]
        assert relevance > 0.99

    async def test_vector_search_ordering(self, backend, workspace_id):
        """Test that more similar embeddings rank higher."""
        # Create two memories with different embeddings
        mem1 = await backend.create_memory(workspace_id, RememberInput(content="Close match", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="Far match", type=MemoryType.SEMANTIC))

        # mem1 gets an embedding close to query, mem2 gets a distant one
        close_emb = [1.0] * EMBEDDING_DIM
        far_emb = [0.0] * (EMBEDDING_DIM - 1) + [1.0]

        await backend.update_memory(workspace_id, mem1.id, embedding=close_emb)
        await backend.update_memory(workspace_id, mem2.id, embedding=far_emb)

        query = [1.0] * EMBEDDING_DIM
        results = await backend.search_memories(workspace_id, query, limit=10, min_relevance=0.0)

        # mem1 should rank before mem2
        result_ids = [r[0].id for r in results]
        assert mem1.id in result_ids
        idx1 = result_ids.index(mem1.id)
        if mem2.id in result_ids:
            idx2 = result_ids.index(mem2.id)
            assert idx1 < idx2

    async def test_vector_search_min_relevance_filter(self, backend, workspace_id):
        """Test min_relevance filtering."""
        inp = RememberInput(content="Filtered", type=MemoryType.SEMANTIC)
        mem = await backend.create_memory(workspace_id, inp)

        embedding = [1.0] * EMBEDDING_DIM
        await backend.update_memory(workspace_id, mem.id, embedding=embedding)

        # Orthogonal query should yield low relevance
        query = [0.0] * EMBEDDING_DIM
        query[0] = 1.0
        results = await backend.search_memories(workspace_id, query, limit=5, min_relevance=0.99)
        # Should be filtered out or only perfect matches
        for _, relevance in results:
            assert relevance >= 0.99

    async def test_vector_search_type_filter(self, backend, workspace_id):
        """Test filtering by memory type."""
        sem = await backend.create_memory(workspace_id, RememberInput(content="Semantic", type=MemoryType.SEMANTIC))
        epi = await backend.create_memory(workspace_id, RememberInput(content="Episodic", type=MemoryType.EPISODIC))

        emb = [0.5] * EMBEDDING_DIM
        await backend.update_memory(workspace_id, sem.id, embedding=emb)
        await backend.update_memory(workspace_id, epi.id, embedding=emb)

        results = await backend.search_memories(workspace_id, emb, limit=10, min_relevance=0.0, types=["semantic"])
        result_types = [r[0].type for r in results]
        assert all(t == MemoryType.SEMANTIC for t in result_types)

    async def test_vector_search_excludes_deleted(self, backend, workspace_id):
        """Soft-deleted memories should not appear in search results."""
        inp = RememberInput(content="Deleted memory", type=MemoryType.SEMANTIC)
        mem = await backend.create_memory(workspace_id, inp)

        emb = [0.7] * EMBEDDING_DIM
        await backend.update_memory(workspace_id, mem.id, embedding=emb)
        await backend.delete_memory(workspace_id, mem.id)

        results = await backend.search_memories(workspace_id, emb, limit=10, min_relevance=0.0)
        result_ids = [r[0].id for r in results]
        assert mem.id not in result_ids


# ============================================================================
# Full-Text Search Tests
# ============================================================================


@pytest.mark.asyncio
class TestFullTextSearch:
    """Test full-text search (Turso native FTS or LIKE fallback)."""

    async def test_basic_fts(self, backend, workspace_id):
        await backend.create_memory(workspace_id, RememberInput(content="The quick brown fox jumps", type=MemoryType.SEMANTIC))
        await backend.create_memory(workspace_id, RememberInput(content="A lazy dog sleeps", type=MemoryType.SEMANTIC))

        results = await backend.full_text_search(workspace_id, "fox")
        assert len(results) == 1
        assert "fox" in results[0].content

    async def test_fts_no_results(self, backend, workspace_id):
        await backend.create_memory(workspace_id, RememberInput(content="Nothing relevant", type=MemoryType.SEMANTIC))

        results = await backend.full_text_search(workspace_id, "nonexistent_xyz_term")
        assert len(results) == 0

    async def test_fts_excludes_deleted(self, backend, workspace_id):
        mem = await backend.create_memory(workspace_id, RememberInput(content="Deletable searchable content", type=MemoryType.SEMANTIC))
        await backend.delete_memory(workspace_id, mem.id)

        results = await backend.full_text_search(workspace_id, "Deletable")
        assert len(results) == 0

    async def test_fts_respects_workspace(self, backend):
        """FTS should only return results from the specified workspace."""
        ws1 = "fts_ws1"
        ws2 = "fts_ws2"
        now = datetime.now(UTC)
        for ws in [ws1, ws2]:
            try:
                await backend.create_workspace(Workspace(id=ws, tenant_id="_default", name=ws, settings={}, created_at=now, updated_at=now))
            except Exception:
                pass

        await backend.create_memory(ws1, RememberInput(content="Workspace one content", type=MemoryType.SEMANTIC))
        await backend.create_memory(ws2, RememberInput(content="Workspace two content", type=MemoryType.SEMANTIC))

        results = await backend.full_text_search(ws1, "Workspace")
        assert len(results) == 1
        assert results[0].workspace_id == ws1


# ============================================================================
# Association and Graph Tests
# ============================================================================


@pytest.mark.asyncio
class TestAssociations:
    """Test association CRUD and graph traversal."""

    async def test_create_association(self, backend, workspace_id):
        mem1 = await backend.create_memory(workspace_id, RememberInput(content="Source", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="Target", type=MemoryType.SEMANTIC))

        assoc = await backend.create_association(
            workspace_id, AssociateInput(source_id=mem1.id, target_id=mem2.id, relationship="related_to", strength=0.8)
        )
        assert assoc.id.startswith("assoc_")
        assert assoc.relationship == "related_to"
        assert assoc.strength == 0.8

    async def test_get_associations(self, backend, workspace_id):
        mem1 = await backend.create_memory(workspace_id, RememberInput(content="Node A", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="Node B", type=MemoryType.SEMANTIC))

        await backend.create_association(
            workspace_id, AssociateInput(source_id=mem1.id, target_id=mem2.id, relationship="causes", strength=0.9)
        )

        assocs = await backend.get_associations(workspace_id, mem1.id, direction="outgoing")
        assert len(assocs) == 1
        assert assocs[0].target_id == mem2.id

        assocs_in = await backend.get_associations(workspace_id, mem2.id, direction="incoming")
        assert len(assocs_in) == 1

        assocs_both = await backend.get_associations(workspace_id, mem1.id, direction="both")
        assert len(assocs_both) >= 1

    async def test_traverse_graph(self, backend, workspace_id):
        mem_a = await backend.create_memory(workspace_id, RememberInput(content="A", type=MemoryType.SEMANTIC))
        mem_b = await backend.create_memory(workspace_id, RememberInput(content="B", type=MemoryType.SEMANTIC))
        mem_c = await backend.create_memory(workspace_id, RememberInput(content="C", type=MemoryType.SEMANTIC))

        await backend.create_association(
            workspace_id, AssociateInput(source_id=mem_a.id, target_id=mem_b.id, relationship="related", strength=0.8)
        )
        await backend.create_association(
            workspace_id, AssociateInput(source_id=mem_b.id, target_id=mem_c.id, relationship="related", strength=0.7)
        )

        graph = await backend.traverse_graph(workspace_id, mem_a.id, max_depth=3)
        assert graph.total_paths >= 2
        assert len(graph.unique_nodes) >= 3

    async def test_traverse_graph_direction_outgoing(self, backend, workspace_id):
        mem_a = await backend.create_memory(workspace_id, RememberInput(content="Out A", type=MemoryType.SEMANTIC))
        mem_b = await backend.create_memory(workspace_id, RememberInput(content="Out B", type=MemoryType.SEMANTIC))

        await backend.create_association(
            workspace_id, AssociateInput(source_id=mem_a.id, target_id=mem_b.id, relationship="points_to", strength=0.9)
        )

        graph = await backend.traverse_graph(workspace_id, mem_a.id, max_depth=2, direction="outgoing")
        assert graph.total_paths >= 1

    async def test_hard_delete_cascades_associations(self, backend, workspace_id):
        mem1 = await backend.create_memory(workspace_id, RememberInput(content="Src", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="Tgt", type=MemoryType.SEMANTIC))

        await backend.create_association(
            workspace_id, AssociateInput(source_id=mem1.id, target_id=mem2.id, relationship="test", strength=0.5)
        )

        await backend.delete_memory(workspace_id, mem1.id, hard=True)
        assocs = await backend.get_associations(workspace_id, mem2.id)
        assert len(assocs) == 0


# ============================================================================
# Workspace and Context Tests
# ============================================================================


@pytest.mark.asyncio
class TestWorkspaceOperations:
    async def test_create_and_get_workspace(self, backend):
        now = datetime.now(UTC)
        ws = Workspace(id="ws_test", tenant_id="_default", name="Test WS", settings={"key": "val"}, created_at=now, updated_at=now)
        created = await backend.create_workspace(ws)
        assert created.id == "ws_test"

        fetched = await backend.get_workspace("ws_test")
        assert fetched is not None
        assert fetched.name == "Test WS"
        assert fetched.settings == {"key": "val"}

    async def test_list_workspaces(self, backend):
        workspaces = await backend.list_workspaces()
        assert len(workspaces) >= 2  # _default and _global

    async def test_get_nonexistent_workspace(self, backend):
        assert await backend.get_workspace("nonexistent") is None

    async def test_create_and_get_context(self, backend):
        now = datetime.now(UTC)
        ctx = Context(id="ctx_test", workspace_id="_default", name="Test Ctx", description="A test context", settings={}, created_at=now)
        created = await backend.create_context("_default", ctx)
        assert created.id == "ctx_test"

        fetched = await backend.get_context("_default", "ctx_test")
        assert fetched is not None
        assert fetched.name == "Test Ctx"

    async def test_list_contexts(self, backend):
        contexts = await backend.list_contexts("_default")
        assert len(contexts) >= 1  # _default context

    async def test_workspace_stats(self, backend, workspace_id):
        await backend.create_memory(workspace_id, RememberInput(content="Stats test", type=MemoryType.SEMANTIC))

        stats = await backend.get_workspace_stats(workspace_id)
        assert stats["total_memories"] >= 1
        assert "memory_types" in stats
        assert "total_associations" in stats

    async def test_list_all_workspace_ids(self, backend):
        ids = await backend.list_all_workspace_ids()
        assert "_default" in ids
        assert "_global" in ids


# ============================================================================
# Session and Working Memory Tests
# ============================================================================


@pytest.mark.asyncio
class TestSessionOperations:
    async def test_create_and_get_session(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_1",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user1",
            metadata={"key": "val"},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        created = await backend.create_session(workspace_id, session)
        assert created.id == "sess_1"

        fetched = await backend.get_session(workspace_id, "sess_1")
        assert fetched is not None
        assert fetched.user_id == "user1"

    async def test_get_session_by_id(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_global",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user2",
            metadata={},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        await backend.create_session(workspace_id, session)

        fetched = await backend.get_session_by_id("sess_global")
        assert fetched is not None
        assert fetched.id == "sess_global"

    async def test_delete_session(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_del",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user3",
            metadata={},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        await backend.create_session(workspace_id, session)

        result = await backend.delete_session(workspace_id, "sess_del")
        assert result is True

        fetched = await backend.get_session(workspace_id, "sess_del")
        assert fetched is None

    async def test_working_memory_crud(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_wm",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user4",
            metadata={},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        await backend.create_session(workspace_id, session)

        # Set
        wm = await backend.set_working_memory(workspace_id, "sess_wm", "counter", 42)
        assert wm.key == "counter"
        assert wm.value == 42

        # Get
        fetched = await backend.get_working_memory(workspace_id, "sess_wm", "counter")
        assert fetched is not None
        assert fetched.value == 42

        # Get all
        await backend.set_working_memory(workspace_id, "sess_wm", "name", "test")
        all_wm = await backend.get_all_working_memory(workspace_id, "sess_wm")
        assert len(all_wm) == 2

        # Upsert
        wm2 = await backend.set_working_memory(workspace_id, "sess_wm", "counter", 99)
        assert wm2.value == 99

    async def test_update_session(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_upd",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user5",
            metadata={},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        await backend.create_session(workspace_id, session)

        new_expires = now + timedelta(hours=2)
        updated = await backend.update_session(workspace_id, "sess_upd", expires_at=new_expires)
        assert updated is not None

    async def test_list_sessions(self, backend, workspace_id):
        now = datetime.now(UTC)
        session = Session(
            id="sess_list",
            tenant_id="_default",
            workspace_id=workspace_id,
            context_id="_default",
            user_id="user6",
            metadata={},
            auto_commit=True,
            expires_at=now + timedelta(hours=1),
            created_at=now,
        )
        await backend.create_session(workspace_id, session)

        sessions = await backend.list_sessions(workspace_id)
        assert len(sessions) >= 1


# ============================================================================
# Contradiction Tests
# ============================================================================


@pytest.mark.asyncio
class TestContradictions:
    async def test_create_and_get_contradiction(self, backend, workspace_id):
        from memorylayer_server.services.contradiction.base import ContradictionRecord
        from memorylayer_server.utils import generate_id

        mem1 = await backend.create_memory(workspace_id, RememberInput(content="Earth is round", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="Earth is flat", type=MemoryType.SEMANTIC))

        record = ContradictionRecord(
            id=generate_id("contra"),
            workspace_id=workspace_id,
            memory_a_id=mem1.id,
            memory_b_id=mem2.id,
            contradiction_type="factual",
            confidence=0.95,
            detection_method="semantic",
            detected_at=datetime.now(UTC),
        )
        created = await backend.create_contradiction(record)
        assert created.id == record.id

        fetched = await backend.get_contradiction(workspace_id, record.id)
        assert fetched is not None
        assert fetched.contradiction_type == "factual"

    async def test_get_unresolved_contradictions(self, backend, workspace_id):
        from memorylayer_server.services.contradiction.base import ContradictionRecord
        from memorylayer_server.utils import generate_id

        mem1 = await backend.create_memory(workspace_id, RememberInput(content="A1", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="A2", type=MemoryType.SEMANTIC))

        record = ContradictionRecord(
            id=generate_id("contra"),
            workspace_id=workspace_id,
            memory_a_id=mem1.id,
            memory_b_id=mem2.id,
            contradiction_type="factual",
            confidence=0.9,
            detection_method="test",
            detected_at=datetime.now(UTC),
        )
        await backend.create_contradiction(record)

        unresolved = await backend.get_unresolved_contradictions(workspace_id)
        assert len(unresolved) >= 1

    async def test_resolve_contradiction(self, backend, workspace_id):
        from memorylayer_server.services.contradiction.base import ContradictionRecord
        from memorylayer_server.utils import generate_id

        mem1 = await backend.create_memory(workspace_id, RememberInput(content="B1", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="B2", type=MemoryType.SEMANTIC))

        record = ContradictionRecord(
            id=generate_id("contra"),
            workspace_id=workspace_id,
            memory_a_id=mem1.id,
            memory_b_id=mem2.id,
            contradiction_type="factual",
            confidence=0.85,
            detection_method="test",
            detected_at=datetime.now(UTC),
        )
        await backend.create_contradiction(record)

        resolved = await backend.resolve_contradiction(workspace_id, record.id, "kept_first", "B1 is correct")
        assert resolved is not None
        assert resolved.resolution == "kept_first"
        assert resolved.merged_content == "B1 is correct"
        assert resolved.resolved_at is not None


# ============================================================================
# Plugin Tests
# ============================================================================


# ============================================================================
# Filtered Search Tests
# ============================================================================


@pytest.mark.asyncio
class TestSearchMemoriesByFilter:
    async def test_filter_by_subtype(self, backend, workspace_id):
        await backend.create_memory(
            workspace_id, RememberInput(content="Semantic one", type=MemoryType.SEMANTIC, subtype=MemorySubtype.PREFERENCE)
        )
        await backend.create_memory(workspace_id, RememberInput(content="Episodic one", type=MemoryType.EPISODIC))

        results = await backend.search_memories_by_filter(workspace_id, subtypes=["preference"])
        assert len(results) >= 1
        assert all(r.subtype == MemorySubtype.PREFERENCE for r in results)

    async def test_filter_by_tags(self, backend, workspace_id):
        await backend.create_memory(workspace_id, RememberInput(content="Tagged", type=MemoryType.SEMANTIC, tags=["rpg", "test"]))
        await backend.create_memory(workspace_id, RememberInput(content="Untagged", type=MemoryType.SEMANTIC))

        results = await backend.search_memories_by_filter(workspace_id, tags=["rpg"])
        assert len(results) >= 1
        assert all("rpg" in r.tags for r in results)

    async def test_filter_by_metadata(self, backend, workspace_id):
        await backend.create_memory(
            workspace_id, RememberInput(content="Node A", type=MemoryType.SEMANTIC, metadata={"rpg_node_id": "src/main.py"})
        )
        await backend.create_memory(
            workspace_id, RememberInput(content="Node B", type=MemoryType.SEMANTIC, metadata={"rpg_node_id": "src/utils.py"})
        )

        results = await backend.search_memories_by_filter(workspace_id, metadata_filter={"rpg_node_id": "src/main.py"})
        assert len(results) == 1
        assert results[0].metadata["rpg_node_id"] == "src/main.py"

    async def test_filter_combined(self, backend, workspace_id):
        await backend.create_memory(
            workspace_id,
            RememberInput(
                content="RPG file", type=MemoryType.SEMANTIC, subtype="rpg_file", tags=["rpg"], metadata={"rpg_node_id": "combined_test"}
            ),
        )
        await backend.create_memory(workspace_id, RememberInput(content="Not RPG", type=MemoryType.SEMANTIC, tags=["other"]))

        results = await backend.search_memories_by_filter(
            workspace_id, subtypes=["rpg_file"], tags=["rpg"], metadata_filter={"rpg_node_id": "combined_test"}
        )
        assert len(results) == 1
        assert results[0].content == "RPG file"

    async def test_filter_returns_empty_when_no_match(self, backend, workspace_id):
        results = await backend.search_memories_by_filter(workspace_id, metadata_filter={"nonexistent_key": "no_value"})
        assert results == []

    async def test_filter_excludes_deleted(self, backend, workspace_id):
        mem = await backend.create_memory(
            workspace_id, RememberInput(content="Deletable", type=MemoryType.SEMANTIC, subtype=MemorySubtype.PREFERENCE)
        )
        await backend.delete_memory(workspace_id, mem.id)

        results = await backend.search_memories_by_filter(workspace_id, subtypes=["preference"])
        assert all(r.id != mem.id for r in results)


# ============================================================================
# Delete Association Tests
# ============================================================================


@pytest.mark.asyncio
class TestDeleteAssociation:
    async def test_delete_association(self, backend, workspace_id):
        mem1 = await backend.create_memory(workspace_id, RememberInput(content="A", type=MemoryType.SEMANTIC))
        mem2 = await backend.create_memory(workspace_id, RememberInput(content="B", type=MemoryType.SEMANTIC))

        assoc = await backend.create_association(
            workspace_id, AssociateInput(source_id=mem1.id, target_id=mem2.id, relationship="test", strength=0.5)
        )

        result = await backend.delete_association(workspace_id, assoc.id)
        assert result is True

        # Verify it's gone
        assocs = await backend.get_associations(workspace_id, mem1.id)
        assert len(assocs) == 0

    async def test_delete_nonexistent_association(self, backend, workspace_id):
        result = await backend.delete_association(workspace_id, "nonexistent")
        assert result is False


# ============================================================================
# Update Workspace Tests
# ============================================================================


@pytest.mark.asyncio
class TestUpdateWorkspace:
    async def test_update_workspace_name(self, backend):
        now = datetime.now(UTC)
        await backend.create_workspace(
            Workspace(id="ws_upd", tenant_id="_default", name="Old Name", settings={}, created_at=now, updated_at=now)
        )

        updated = await backend.update_workspace("ws_upd", name="New Name")
        assert updated is not None
        assert updated.name == "New Name"

    async def test_update_workspace_settings(self, backend):
        now = datetime.now(UTC)
        await backend.create_workspace(
            Workspace(id="ws_settings", tenant_id="_default", name="Test", settings={"a": 1}, created_at=now, updated_at=now)
        )

        updated = await backend.update_workspace("ws_settings", settings={"a": 2, "b": 3})
        assert updated is not None
        assert updated.settings == {"a": 2, "b": 3}

    async def test_update_nonexistent_workspace(self, backend):
        result = await backend.update_workspace("nonexistent", name="Nope")
        assert result is None


# ============================================================================
# Plugin Tests
# ============================================================================


class TestTursoPlugin:
    def test_provider_name(self):
        assert TursoStorageBackendPlugin.PROVIDER_NAME == "turso"

    def test_plugin_name(self):
        plugin = TursoStorageBackendPlugin()
        assert "turso" in plugin.name()
