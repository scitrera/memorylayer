"""Turso/libSQL storage backend with native vector support."""

import hashlib
import json
import struct
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scitrera_app_framework import Variables as Variables

from ...config import (
    DEFAULT_CONTEXT_ID,
    DEFAULT_MEMORYLAYER_TURSO_DB_PATH,
    DEFAULT_MEMORYLAYER_TURSO_MODE,
    DEFAULT_MEMORYLAYER_TURSO_SYNC_INTERVAL,
    DEFAULT_MEMORYLAYER_TURSO_VECTOR_INDEX,
    DEFAULT_TENANT_ID,
    MEMORYLAYER_TURSO_AUTH_TOKEN,
    MEMORYLAYER_TURSO_DB_PATH,
    MEMORYLAYER_TURSO_MODE,
    MEMORYLAYER_TURSO_SYNC_INTERVAL,
    MEMORYLAYER_TURSO_URL,
    MEMORYLAYER_TURSO_VECTOR_INDEX,
)
from ...models.association import AssociateInput, Association, GraphPath, GraphQueryResult
from ...models.memory import Memory, MemoryStatus, MemoryType, RememberInput
from ...models.session import Session, WorkingMemory
from ...models.workspace import Context, Workspace
from ...utils import generate_id, parse_datetime_utc, utc_now_iso
from ..contradiction.base import ContradictionRecord
from .base import StorageBackend, StoragePluginBase

if TYPE_CHECKING:
    import turso

_UPDATABLE_MEMORY_COLUMNS = frozenset(
    {
        "content",
        "content_hash",
        "type",
        "subtype",
        "importance",
        "tags",
        "metadata",
        "embedding",
        "abstract",
        "overview",
        "pinned",
        "category",
        "decay_factor",
        "status",
        "archived_at",
        "observer_id",
        "subject_id",
        "access_count",
        "last_accessed_at",
        "created_at",
        "updated_at",
        "source_memory_id",
    }
)

_UPDATABLE_THREAD_COLUMNS = frozenset(
    {
        "title",
        "metadata",
        "model",
        "system_prompt",
        "max_messages",
        "ttl_seconds",
        "expires_at",
        "last_decomposed_index",
    }
)


class TursoStorageBackend(StorageBackend):
    """Turso/libSQL storage backend with native vector support.

    Supports three connection modes:
    - local: Pure local libSQL file (drop-in SQLite replacement with native vectors)
    - remote: Cloud-only Turso database via URL + auth token
    - replica: Embedded replica with local file syncing to remote Turso cloud
    """

    def __init__(
        self,
        mode: str = "local",
        db_path: str = "memorylayer.db",
        url: str | None = None,
        auth_token: str | None = None,
        sync_interval: int = 60,
        vector_index: bool = False,
        v: Variables = None,
    ):
        super().__init__(v)
        self.mode = mode
        self.db_path = db_path
        self.url = url
        self.auth_token = auth_token
        self.sync_interval = sync_interval
        self.vector_index = vector_index
        self._connection = None
        self._sync_connection = None  # For replica mode (sync connection wrapper)

    async def connect(self) -> None:
        """Initialize storage connection based on configured mode."""
        import turso
        import turso.aio
        import turso.sync

        if self.mode == "local":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.logger.info("Connecting to local Turso/libSQL database at %s", Path(self.db_path).absolute())
            self._connection = await turso.aio.connect(self.db_path, experimental_features="index_method")

        elif self.mode == "remote":
            if not self.url:
                raise ValueError("MEMORYLAYER_TURSO_URL is required for remote mode")
            self.logger.info("Connecting to remote Turso database at %s", self.url)
            self._connection = await turso.aio.connect(self.url, experimental_features="index_method")

        elif self.mode == "replica":
            if not self.url:
                raise ValueError("MEMORYLAYER_TURSO_URL is required for replica mode")
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(
                "Connecting to Turso embedded replica: local=%s, remote=%s",
                Path(self.db_path).absolute(),
                self.url,
            )
            # Sync connection is synchronous in pyturso; wrap for replica mode
            sync_conn = turso.sync.connect(
                self.db_path,
                remote_url=self.url,
                auth_token=self.auth_token,
            )
            # Pull latest state from remote
            sync_conn.pull()
            self.logger.info("Embedded replica synced from remote")
            self._sync_connection = sync_conn

            # Also open an async connection to the local file for query operations
            self._connection = await turso.aio.connect(self.db_path, experimental_features="index_method")

        else:
            raise ValueError(f"Invalid MEMORYLAYER_TURSO_MODE: {self.mode!r} (expected: local, remote, replica)")

        # Set row factory for dict-like access
        self._connection.row_factory = turso.Row

        # Enable WAL mode for better concurrent read performance (local/replica modes)
        if self.mode in ("local", "replica"):
            await self._connection.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Create tables
        await self._create_tables()

        # Ensure reserved entities exist
        await self._ensure_reserved_entities()

        self.logger.info("Connected to Turso/libSQL database (mode=%s)", self.mode)

    async def disconnect(self) -> None:
        """Close storage connection."""
        if self._sync_connection:
            try:
                # Push any pending changes before disconnecting in replica mode
                self._sync_connection.push()
                self.logger.info("Embedded replica pushed to remote before disconnect")
            except Exception as e:
                self.logger.warning("Failed to push replica changes on disconnect: %s", e)
            try:
                self._sync_connection.close()
            except Exception as e:
                self.logger.warning("Failed to close sync connection: %s", e)
            self._sync_connection = None

        if self._connection:
            await self._connection.close()
            self.logger.info("Disconnected from Turso/libSQL database")
            self._connection = None

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        try:
            if self._connection:
                await self._connection.execute("SELECT 1")
                return True
            return False
        except Exception as e:
            self.logger.error("Health check failed: %s", e)
            return False

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Workspaces
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Contexts
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces (id),
                name TEXT NOT NULL,
                description TEXT,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE (workspace_id, name)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_contexts_workspace ON contexts(workspace_id)")

        # Memories
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT '_default',
                workspace_id TEXT NOT NULL,
                context_id TEXT NOT NULL DEFAULT '_default',
                session_id TEXT,
                user_id TEXT,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                type TEXT NOT NULL CHECK (type IN ('episodic', 'semantic', 'procedural', 'working')),
                subtype TEXT,
                category TEXT,
                importance REAL DEFAULT 0.5,
                tags TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                embedding BLOB,
                abstract TEXT,
                overview TEXT,
                source_memory_id TEXT,
                source_document_id TEXT,
                source_page_id TEXT,
                source_dataset_id TEXT,
                source_thread_id TEXT,
                observer_id TEXT,
                subject_id TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                decay_factor REAL DEFAULT 1.0,
                status TEXT DEFAULT 'active',
                pinned INTEGER DEFAULT 0,
                deleted_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_workspace ON memories(workspace_id) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_workspace_type ON memories(workspace_id, type) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_common ON memories(workspace_id, type, created_at DESC) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(workspace_id, status) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_memory_id) WHERE source_memory_id IS NOT NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_observer ON memories(workspace_id, observer_id) WHERE observer_id IS NOT NULL AND deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_subject ON memories(workspace_id, subject_id) WHERE subject_id IS NOT NULL AND deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_source_dataset ON memories(source_dataset_id) WHERE source_dataset_id IS NOT NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_source_thread ON memories(source_thread_id) WHERE source_thread_id IS NOT NULL"
        )

        # Optional: DiskANN vector index for accelerated similarity search
        if self.vector_index:
            try:
                await self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memories_vec
                    ON memories(libsql_vector_idx(embedding))
                """)
                self.logger.info("DiskANN vector index created/verified")
            except Exception as e:
                self.logger.warning("Failed to create DiskANN vector index (may not be supported): %s", e)

        # Turso native FTS index (Tantivy-based, replaces SQLite FTS5).
        # Requires experimental index-method support in pyturso; falls back to LIKE search if unavailable.
        try:
            await self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_fts
                ON memories USING fts (content)
            """)
            self.logger.info("Turso native FTS index created/verified")
        except Exception as e:
            self.logger.info("Turso native FTS index not available (full-text search will use LIKE fallback): %s", e)

        # Memory Associations
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_associations (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                source_id TEXT NOT NULL REFERENCES memories (id),
                target_id TEXT NOT NULL REFERENCES memories (id),
                relationship TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (source_id, target_id, relationship)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_workspace ON memory_associations(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_source ON memory_associations(source_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_target ON memory_associations(target_id)")

        # Sessions table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT '_default',
                workspace_id TEXT NOT NULL,
                context_id TEXT NOT NULL,
                user_id TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                auto_commit INTEGER DEFAULT 1,
                expires_at TEXT NOT NULL,
                committed_at TEXT,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT,
                FOREIGN KEY (workspace_id) REFERENCES workspaces (id)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_context ON sessions(context_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

        # Working memory table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                ttl_seconds INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (session_id, key),
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_working_memory_session ON working_memory(session_id)")

        # Contradictions table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                memory_a_id TEXT NOT NULL,
                memory_b_id TEXT NOT NULL,
                contradiction_type TEXT,
                confidence REAL DEFAULT 0.0,
                detection_method TEXT,
                detected_at TEXT DEFAULT (datetime('now')),
                resolved_at TEXT,
                resolution TEXT,
                merged_content TEXT,
                FOREIGN KEY (memory_a_id) REFERENCES memories (id),
                FOREIGN KEY (memory_b_id) REFERENCES memories (id)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_workspace ON contradictions(workspace_id)")
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_unresolved ON contradictions(workspace_id) WHERE resolved_at IS NULL"
        )

        # Chat threads
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT '_default',
                user_id TEXT,
                context_id TEXT NOT NULL DEFAULT '_default',
                observer_id TEXT,
                subject_id TEXT,
                title TEXT,
                metadata TEXT DEFAULT '{}',
                message_count INTEGER DEFAULT 0,
                last_decomposed_at TEXT,
                last_decomposed_index INTEGER DEFAULT 0,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                scope TEXT,
                FOREIGN KEY (workspace_id) REFERENCES workspaces (id)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_workspace ON chat_threads(workspace_id)")
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(workspace_id, user_id) WHERE user_id IS NOT NULL"
        )
        # Migrate: add scope column to existing Turso databases (idempotent).
        try:
            await self._connection.execute(
                "ALTER TABLE chat_threads ADD COLUMN scope TEXT"
            )
            await self._connection.commit()
        except Exception:
            pass  # Column already exists — expected on databases created after the schema update

        # Chat messages
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id) REFERENCES chat_threads (id) ON DELETE CASCADE
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, message_index)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace ON chat_messages(workspace_id, thread_id)")

        await self._connection.commit()

    async def _ensure_reserved_entities(self) -> None:
        """Ensure reserved entities exist (_default workspace, _global workspace,
        _global_user workspace, _default contexts)."""
        from ...config import (
            DEFAULT_TENANT_ID,
            DEFAULT_WORKSPACE_ID,
            GLOBAL_USER_WORKSPACE_ID,
            GLOBAL_WORKSPACE_ID,
        )

        now = utc_now_iso()

        await self._connection.execute(
            "INSERT OR IGNORE INTO workspaces (id, tenant_id, name, settings, created_at, updated_at) VALUES (?, ?, 'Default Workspace', '{}', ?, ?)",
            (DEFAULT_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )
        await self._connection.execute(
            "INSERT OR IGNORE INTO workspaces (id, tenant_id, name, settings, created_at, updated_at) VALUES (?, ?, 'Global Workspace', '{}', ?, ?)",
            (GLOBAL_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )
        await self._connection.execute(
            "INSERT OR IGNORE INTO workspaces (id, tenant_id, name, settings, created_at, updated_at) VALUES (?, ?, 'Global User Workspace', '{}', ?, ?)",
            (GLOBAL_USER_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )

        cursor = await self._connection.execute("SELECT id FROM workspaces")
        workspaces = await cursor.fetchall()

        for workspace in workspaces:
            workspace_id = workspace["id"]
            await self._connection.execute(
                "INSERT OR IGNORE INTO contexts (id, workspace_id, name, description, settings, created_at, updated_at) VALUES ('_default', ?, '_default', 'Default context', '{}', ?, ?)",
                (workspace_id, now, now),
            )

        await self._connection.commit()
        self.logger.info("Reserved entities initialized (_default workspace, _global workspace, _default contexts)")

    # ============================================
    # Memory Operations
    # ============================================

    async def create_memory(self, workspace_id: str, input: RememberInput) -> Memory:
        """Store a new memory."""
        from ...config import DEFAULT_TENANT_ID

        content_hash = hashlib.sha256(input.content.encode()).hexdigest()
        memory_id = generate_id("mem")
        now = utc_now_iso()

        await self._connection.execute(
            """
            INSERT INTO memories (id, tenant_id, workspace_id, context_id, session_id, user_id,
                                  content, content_hash, type, subtype, category,
                                  importance, tags, metadata, abstract, overview,
                                  source_memory_id, status, pinned,
                                  observer_id, subject_id,
                                  source_document_id, source_page_id,
                                  source_dataset_id, source_thread_id,
                                  created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                getattr(input, "tenant_id", None) or DEFAULT_TENANT_ID,
                workspace_id,
                getattr(input, "context_id", None) or "_default",
                getattr(input, "session_id", None),
                input.user_id,
                input.content,
                content_hash,
                input.type.value if input.type else MemoryType.SEMANTIC.value,
                input.subtype if input.subtype else None,
                getattr(input, "category", None),
                input.importance,
                json.dumps(input.tags),
                json.dumps(input.metadata),
                getattr(input, "abstract", None),
                getattr(input, "overview", None),
                getattr(input, "source_memory_id", None),
                MemoryStatus.ACTIVE.value,
                0,
                getattr(input, "observer_id", None),
                getattr(input, "subject_id", None),
                getattr(input, "source_document_id", None),
                getattr(input, "source_page_id", None),
                getattr(input, "source_dataset_id", None),
                getattr(input, "source_thread_id", None),
                now,
                now,
            ),
        )

        await self._connection.commit()
        return await self.get_memory(workspace_id, memory_id, track_access=False)

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID within a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE id = ? AND workspace_id = ? AND deleted_at IS NULL",
            (memory_id, workspace_id),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access:
            await self._connection.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?",
                (memory_id,),
            )
            await self._connection.commit()
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.now(UTC)

        return memory

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID without workspace filter."""
        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE id = ? AND deleted_at IS NULL",
            (memory_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access:
            await self._connection.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?",
                (memory_id,),
            )
            await self._connection.commit()
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.now(UTC)

        return memory

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
        """Update memory fields."""
        invalid_keys = set(updates.keys()) - _UPDATABLE_MEMORY_COLUMNS
        if invalid_keys:
            raise ValueError(f"Invalid update fields: {invalid_keys}")

        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ("tags", "metadata"):
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == "embedding":
                set_parts.append(f"{key} = ?")
                values.append(self._serialize_embedding(value) if value else None)
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)

        if not set_parts:
            return await self.get_memory(workspace_id, memory_id, track_access=False)

        set_parts.append("updated_at = datetime('now')")
        values.extend([memory_id, workspace_id])

        query = f"""
            UPDATE memories
            SET {", ".join(set_parts)}
            WHERE id = ? AND workspace_id = ? AND deleted_at IS NULL
        """

        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None

        return await self.get_memory(workspace_id, memory_id, track_access=False)

    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        if hard:
            # Delete associations referencing this memory first (FK constraint)
            await self._connection.execute(
                "DELETE FROM memory_associations WHERE (source_id = ? OR target_id = ?) AND workspace_id = ?",
                (memory_id, memory_id, workspace_id),
            )
            cursor = await self._connection.execute(
                "DELETE FROM memories WHERE id = ? AND workspace_id = ?",
                (memory_id, workspace_id),
            )
        else:
            cursor = await self._connection.execute(
                "UPDATE memories SET deleted_at = datetime('now'), status = 'deleted' WHERE id = ? AND workspace_id = ?",
                (memory_id, workspace_id),
            )

        await self._connection.commit()
        return cursor.rowcount > 0

    async def get_memories_for_decay(
        self,
        workspace_id: str,
        min_age_days: int = 7,
        exclude_pinned: bool = True,
    ) -> list[Memory]:
        """Get memories eligible for importance decay."""
        where_parts = [
            "workspace_id = ?",
            "deleted_at IS NULL",
            "(status IS NULL OR status = 'active')",
            f"julianday('now') - julianday(created_at) >= {min_age_days}",
        ]
        params: list = [workspace_id]

        if exclude_pinned:
            where_parts.append("(pinned IS NULL OR pinned = 0)")

        query = f"SELECT * FROM memories WHERE {' AND '.join(where_parts)} ORDER BY importance DESC"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def get_archival_candidates(
        self,
        workspace_id: str,
        max_importance: float = 0.3,
        max_access_count: int = 5,
        older_than_days: int = 90,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories eligible for archival."""
        query = """
            SELECT * FROM memories
            WHERE workspace_id = ?
              AND deleted_at IS NULL
              AND (status IS NULL OR status = 'active')
              AND (pinned IS NULL OR pinned = 0)
              AND importance <= ?
              AND access_count <= ?
              AND julianday('now') - julianday(created_at) >= ?
            ORDER BY importance ASC
            LIMIT ?
        """
        cursor = await self._connection.execute(query, (workspace_id, max_importance, max_access_count, older_than_days, limit))
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs."""
        cursor = await self._connection.execute("SELECT id FROM workspaces")
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

    # ============================================
    # Filtered Memory Search (Non-Vector)
    # ============================================

    async def search_memories_by_filter(
        self,
        workspace_id: str,
        *,
        subtypes: list[str] | None = None,
        tags: list[str] | None = None,
        metadata_filter: dict[str, str] | None = None,
        status: str = "active",
        context_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by subtype, tags, and/or metadata without requiring embeddings."""
        where_parts = ["workspace_id = ?", "deleted_at IS NULL"]
        params: list = [workspace_id]

        if context_id is not None:
            where_parts.append("context_id = ?")
            params.append(context_id)

        if status:
            where_parts.append("(status IS NULL OR status = ?)")
            params.append(status)

        if subtypes:
            placeholders = ",".join("?" * len(subtypes))
            where_parts.append(f"subtype IN ({placeholders})")
            params.extend(subtypes)

        if tags:
            for tag in tags:
                where_parts.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        if metadata_filter:
            for key, value in metadata_filter.items():
                where_parts.append("json_extract(metadata, ?) = ?")
                params.append(f"$.{key}")
                params.append(value)

        where_clause = " AND ".join(where_parts)
        query = f"SELECT * FROM memories WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    # ============================================
    # Vector Search (Native libSQL)
    # ============================================

    async def search_memories(
        self,
        workspace_id: str,
        query_embedding: list[float],
        limit: int = 10,
        offset: int = 0,
        min_relevance: float = 0.5,
        types: list[str] | None = None,
        subtypes: list[str] | None = None,
        tags: list[str] | None = None,
        include_archived: bool = False,
        observer_id: str | None = None,
        subject_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        user_id: str | None = None,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search using native libSQL vector_distance_cos."""
        # Build WHERE clause
        where_parts = ["workspace_id = ?", "deleted_at IS NULL", "embedding IS NOT NULL"]
        if not include_archived:
            where_parts.append("(status IS NULL OR status = 'active')")
        params: list = [workspace_id]

        if types:
            placeholders = ",".join("?" * len(types))
            where_parts.append(f"type IN ({placeholders})")
            params.extend(types)

        if subtypes:
            placeholders = ",".join("?" * len(subtypes))
            where_parts.append(f"subtype IN ({placeholders})")
            params.extend(subtypes)

        if tags:
            for tag in tags:
                where_parts.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        if observer_id is not None:
            where_parts.append("observer_id = ?")
            params.append(observer_id)

        if subject_id is not None:
            where_parts.append("subject_id = ?")
            params.append(subject_id)

        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)

        if created_after is not None:
            where_parts.append("created_at >= ?")
            params.append(str(created_after))

        if created_before is not None:
            where_parts.append("created_at <= ?")
            params.append(str(created_before))

        where_clause = " AND ".join(where_parts)

        # Native libSQL vector distance: use vector32() to wrap the query embedding.
        # The embedding column stores struct-packed float BLOBs, and vector_distance_cos
        # can compute cosine distance directly on them.
        query_vec_blob = self._serialize_embedding(query_embedding)
        params_ordered = [query_vec_blob] + params + [limit, offset]

        query = f"""
            SELECT *, vector_distance_cos(embedding, ?) as distance
            FROM memories
            WHERE {where_clause}
            ORDER BY distance ASC
            LIMIT ? OFFSET ?
        """

        cursor = await self._connection.execute(query, params_ordered)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            distance = row["distance"]
            relevance = 1.0 - distance

            if relevance >= min_relevance:
                memory = self._row_to_memory(row)
                results.append((memory, relevance))

        return results

    # ============================================
    # Full-Text Search
    # ============================================

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Sanitize full-text search query for Turso native FTS."""
        # Remove characters that could break the MATCH expression
        return query.replace("'", "''")

    async def full_text_search(
        self,
        workspace_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search using Turso native FTS (Tantivy-based).

        Uses Turso's MATCH syntax on the content column with BM25 scoring.
        Falls back to LIKE-based search if FTS index is not available.
        """
        sanitized = self._sanitize_fts_query(query)
        ctx_clause = ""
        ctx_params: tuple = ()
        if context_id is not None:
            ctx_clause = " AND context_id = ?"
            ctx_params = (context_id,)

        try:
            # Turso native FTS: use (column) MATCH 'query' with fts_score for ranking
            cursor = await self._connection.execute(
                f"""
                SELECT *, fts_score(content, ?) as score
                FROM memories
                WHERE workspace_id = ?
                  AND (content) MATCH ?
                  AND deleted_at IS NULL{ctx_clause}
                ORDER BY score DESC
                LIMIT ? OFFSET ?
                """,
                (sanitized, workspace_id, sanitized, *ctx_params, limit, offset),
            )
            rows = await cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]
        except Exception:
            # Fallback to LIKE-based search if FTS is unavailable
            self.logger.debug("FTS MATCH failed, falling back to LIKE search")
            like_pattern = f"%{query}%"
            cursor = await self._connection.execute(
                f"""
                SELECT * FROM memories
                WHERE workspace_id = ?
                  AND content LIKE ?
                  AND deleted_at IS NULL{ctx_clause}
                LIMIT ? OFFSET ?
                """,
                (workspace_id, like_pattern, *ctx_params, limit, offset),
            )
            rows = await cursor.fetchall()
            return [self._row_to_memory(row) for row in rows]

    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Memory | None:
        """Get memory by content hash for deduplication."""
        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE workspace_id = ? AND content_hash = ? AND deleted_at IS NULL LIMIT 1",
            (workspace_id, content_hash),
        )
        row = await cursor.fetchone()
        return self._row_to_memory(row) if row else None

    async def get_recent_memories(
        self,
        workspace_id: str,
        created_after: datetime,
        limit: int = 10,
        detail_level: str = "abstract",
        offset: int = 0,
    ) -> list:
        """Get recent memories ordered by creation time (newest first)."""
        cursor = await self._connection.execute(
            """
            SELECT * FROM memories
            WHERE workspace_id = ?
              AND created_at > ?
              AND (status IS NULL OR status = 'active')
              AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (workspace_id, created_after.isoformat(), limit, offset),
        )
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            if detail_level == "abstract":
                results.append(
                    {
                        "id": row["id"],
                        "abstract": row["abstract"] if row["abstract"] else None,
                        "type": row["type"],
                        "subtype": row["subtype"] if row["subtype"] else None,
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "created_at": row["created_at"],
                    }
                )
            elif detail_level == "overview":
                results.append(
                    {
                        "id": row["id"],
                        "overview": row["overview"] if row["overview"] else None,
                        "type": row["type"],
                        "subtype": row["subtype"] if row["subtype"] else None,
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "created_at": row["created_at"],
                    }
                )
            else:
                memory = self._row_to_memory(row)
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "type": memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                        "subtype": memory.subtype.value
                        if memory.subtype and hasattr(memory.subtype, "value")
                        else str(memory.subtype)
                        if memory.subtype
                        else None,
                        "importance": memory.importance,
                        "tags": memory.tags,
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    }
                )

        return results

    # ============================================
    # Association Operations
    # ============================================

    async def create_association(self, workspace_id: str, input: AssociateInput) -> Association:
        """Create graph edge between memories."""
        association_id = generate_id("assoc")
        now = utc_now_iso()

        await self._connection.execute(
            """
            INSERT INTO memory_associations (id, workspace_id, source_id, target_id,
                                             relationship, strength, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                association_id,
                workspace_id,
                input.source_id,
                input.target_id,
                input.relationship,
                input.strength,
                json.dumps(input.metadata),
                now,
            ),
        )
        await self._connection.commit()

        cursor = await self._connection.execute(
            "SELECT * FROM memory_associations WHERE id = ?",
            (association_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_association(row)

    async def get_associations(
        self,
        workspace_id: str,
        memory_id: str,
        direction: str = "both",
        relationships: list[str] | None = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if direction == "outgoing":
            where_parts.append("source_id = ?")
            params.append(memory_id)
        elif direction == "incoming":
            where_parts.append("target_id = ?")
            params.append(memory_id)
        else:
            where_parts.append("(source_id = ? OR target_id = ?)")
            params.extend([memory_id, memory_id])

        if relationships:
            placeholders = ",".join("?" * len(relationships))
            where_parts.append(f"relationship IN ({placeholders})")
            params.extend(relationships)

        where_clause = " AND ".join(where_parts)

        cursor = await self._connection.execute(
            f"SELECT * FROM memory_associations WHERE {where_clause}",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_association(row) for row in rows]

    async def delete_association(self, workspace_id: str, association_id: str) -> bool:
        """Delete an association by ID."""
        cursor = await self._connection.execute(
            "DELETE FROM memory_associations WHERE id = ? AND workspace_id = ?",
            (association_id, workspace_id),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def traverse_graph(
        self,
        workspace_id: str,
        start_id: str,
        max_depth: int = 3,
        relationships: list[str] | None = None,
        direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop graph traversal using iterative BFS.

        Turso does not yet support recursive CTEs, so this uses iterative
        breadth-first traversal with per-depth queries instead.
        """
        # Build direction-specific WHERE clause
        rel_filter = ""
        rel_params: list = []
        if relationships:
            placeholders = ", ".join("?" * len(relationships))
            rel_filter = f"AND relationship IN ({placeholders})"
            rel_params = list(relationships)

        paths = []
        unique_nodes = set([start_id])
        visited_edges: set[str] = set()
        frontier = [start_id]

        for depth in range(1, max_depth + 1):
            if not frontier:
                break

            next_frontier = []
            for current_node in frontier:
                # Build direction condition
                if direction == "outgoing":
                    where = "source_id = ?"
                    params: list = [workspace_id, current_node]
                elif direction == "incoming":
                    where = "target_id = ?"
                    params = [workspace_id, current_node]
                else:
                    where = "(source_id = ? OR target_id = ?)"
                    params = [workspace_id, current_node, current_node]

                query = f"SELECT * FROM memory_associations WHERE workspace_id = ? AND {where} {rel_filter}"
                cursor = await self._connection.execute(query, params + rel_params)
                rows = await cursor.fetchall()

                for row in rows:
                    edge_id = row["id"]
                    if edge_id in visited_edges:
                        continue
                    visited_edges.add(edge_id)

                    source = row["source_id"]
                    target = row["target_id"]

                    # Determine next node
                    if direction == "outgoing":
                        next_node = target
                    elif direction == "incoming":
                        next_node = source
                    else:
                        next_node = target if source == current_node else source

                    # Skip if already visited (cycle prevention)
                    if next_node in unique_nodes and depth > 1:
                        # Still record the edge but don't expand further
                        pass
                    else:
                        next_frontier.append(next_node)

                    unique_nodes.add(source)
                    unique_nodes.add(target)

                    edge = Association(
                        id=edge_id,
                        workspace_id=workspace_id,
                        source_id=source,
                        target_id=target,
                        relationship=row["relationship"],
                        strength=row["strength"],
                        metadata=json.loads(row["metadata"]),
                        created_at=parse_datetime_utc(row["created_at"]),
                    )

                    path = GraphPath(
                        nodes=[source, target],
                        edges=[edge],
                        total_strength=row["strength"],
                        depth=depth,
                    )
                    paths.append(path)

            frontier = next_frontier

        return GraphQueryResult(
            paths=paths,
            total_paths=len(paths),
            unique_nodes=list(unique_nodes),
            query_latency_ms=0,
        )

    # ============================================
    # Workspace Operations
    # ============================================

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace."""
        await self._connection.execute(
            "INSERT INTO workspaces (id, tenant_id, name, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                workspace.id,
                workspace.tenant_id,
                workspace.name,
                json.dumps(workspace.settings),
                workspace.created_at.isoformat(),
                workspace.updated_at.isoformat(),
            ),
        )
        await self._connection.commit()
        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get workspace by ID."""
        cursor = await self._connection.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_workspace(row)

    async def list_workspaces(self) -> list[Workspace]:
        """List all workspaces."""
        cursor = await self._connection.execute("SELECT * FROM workspaces ORDER BY name")
        rows = await cursor.fetchall()
        return [self._row_to_workspace(row) for row in rows]

    async def update_workspace(self, workspace_id: str, **updates) -> Workspace | None:
        """Update workspace fields."""
        if not updates:
            return await self.get_workspace(workspace_id)

        set_parts = []
        values: list = []
        for key, value in updates.items():
            if key == "settings":
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)

        set_parts.append("updated_at = datetime('now')")
        values.append(workspace_id)

        query = f"UPDATE workspaces SET {', '.join(set_parts)} WHERE id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None

        return await self.get_workspace(workspace_id)

    # ============================================
    # Context Operations
    # ============================================

    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        await self._connection.execute(
            "INSERT INTO contexts (id, workspace_id, name, description, settings, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                context.id,
                workspace_id,
                context.name,
                context.description,
                json.dumps(context.settings),
                context.created_at.isoformat(),
                utc_now_iso(),
            ),
        )
        await self._connection.commit()
        return context

    async def get_context(self, workspace_id: str, context_id: str) -> Context | None:
        """Get context by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM contexts WHERE id = ? AND workspace_id = ?",
            (context_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_context(row)

    async def list_contexts(self, workspace_id: str) -> list[Context]:
        """List all contexts in a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM contexts WHERE workspace_id = ? ORDER BY created_at",
            (workspace_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_context(row) for row in rows]

    # ============================================
    # Statistics
    # ============================================

    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        cursor = await self._connection.execute(
            "SELECT type, COUNT(*) as count FROM memories WHERE workspace_id = ? AND deleted_at IS NULL GROUP BY type",
            (workspace_id,),
        )
        type_counts = {row["type"]: row["count"] for row in await cursor.fetchall()}

        cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM memory_associations WHERE workspace_id = ?",
            (workspace_id,),
        )
        assoc_count = (await cursor.fetchone())["count"]

        return {
            "total_memories": sum(type_counts.values()),
            "memory_types": type_counts,
            "total_associations": assoc_count,
            "total_categories": 0,
        }

    # ============================================
    # Session Operations
    # ============================================

    async def create_session(self, workspace_id: str, session: Session) -> Session:
        """Store a new session."""
        now = utc_now_iso()
        await self._connection.execute(
            "INSERT OR IGNORE INTO workspaces (id, tenant_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (workspace_id, "default", workspace_id, now, now),
        )

        await self._connection.execute(
            """
            INSERT INTO sessions (id, tenant_id, workspace_id, context_id, user_id, metadata, auto_commit, expires_at, committed_at,
                                  created_at, last_accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.tenant_id,
                workspace_id,
                session.context_id,
                session.user_id,
                json.dumps(session.metadata),
                1 if session.auto_commit else 0,
                session.expires_at.isoformat(),
                session.committed_at.isoformat() if session.committed_at else None,
                session.created_at.isoformat(),
                session.created_at.isoformat(),
            ),
        )
        await self._connection.commit()
        self.logger.info("Created persistent session: %s in workspace: %s", session.id, workspace_id)
        return session

    async def get_session(self, workspace_id: str, session_id: str) -> Session | None:
        """Get session by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ? AND workspace_id = ?",
            (session_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    async def get_session_by_id(self, session_id: str) -> Session | None:
        """Get session by ID without workspace filter."""
        cursor = await self._connection.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    async def delete_session(self, workspace_id: str, session_id: str) -> bool:
        """Delete session and all its context (CASCADE)."""
        cursor = await self._connection.execute(
            "DELETE FROM sessions WHERE id = ? AND workspace_id = ?",
            (session_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.info("Deleted session: %s", session_id)
        return deleted

    async def set_working_memory(
        self, workspace_id: str, session_id: str, key: str, value: Any, ttl_seconds: int | None = None
    ) -> WorkingMemory:
        """Set working memory key-value within session."""
        now_iso = utc_now_iso()
        now = datetime.now(UTC)

        await self._connection.execute(
            """
            INSERT INTO working_memory (session_id, key, value, ttl_seconds, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, key) DO UPDATE SET value       = excluded.value,
                                                       ttl_seconds = excluded.ttl_seconds,
                                                       updated_at  = excluded.updated_at
            """,
            (session_id, key, json.dumps(value), ttl_seconds, now_iso, now_iso),
        )
        await self._connection.commit()

        return WorkingMemory(
            session_id=session_id,
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
            created_at=now,
            updated_at=now,
        )

    async def get_working_memory(self, workspace_id: str, session_id: str, key: str) -> WorkingMemory | None:
        """Get specific working memory entry."""
        cursor = await self._connection.execute(
            "SELECT * FROM working_memory WHERE session_id = ? AND key = ?",
            (session_id, key),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_working_memory(row)

    async def get_all_working_memory(self, workspace_id: str, session_id: str) -> list[WorkingMemory]:
        """Get all working memory entries for session."""
        cursor = await self._connection.execute(
            "SELECT * FROM working_memory WHERE session_id = ?",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_working_memory(row) for row in rows]

    async def cleanup_expired_sessions(self, workspace_id: str) -> int:
        """Delete all expired sessions."""
        now = utc_now_iso()
        cursor = await self._connection.execute(
            "DELETE FROM sessions WHERE workspace_id = ? AND expires_at < ?",
            (workspace_id, now),
        )
        await self._connection.commit()
        return cursor.rowcount

    async def cleanup_all_expired_sessions(self) -> int:
        """Delete all expired sessions across all workspaces."""
        now = utc_now_iso()
        cursor = await self._connection.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        await self._connection.commit()
        return cursor.rowcount

    async def list_expired_sessions(self, limit: int = 100) -> list[Session]:
        """List expired sessions that need cleanup."""
        now = utc_now_iso()
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE expires_at < ? ORDER BY expires_at ASC LIMIT ?",
            (now, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_session(row) for row in rows]

    async def update_session(self, workspace_id: str, session_id: str, **updates) -> Session | None:
        """Update session fields."""
        if not updates:
            return await self.get_session(workspace_id, session_id)

        set_clauses = []
        values: list = []
        for field, value in updates.items():
            if field in ("committed_at", "expires_at") and isinstance(value, datetime):
                values.append(value.isoformat())
            elif field == "auto_commit":
                values.append(1 if value else 0)
            elif field == "metadata":
                values.append(json.dumps(value))
            else:
                values.append(value)
            set_clauses.append(f"{field} = ?")

        values.extend([session_id, workspace_id])

        query = f"UPDATE sessions SET {', '.join(set_clauses)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None

        return await self.get_session(workspace_id, session_id)

    async def list_sessions(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_expired: bool = False,
    ) -> list[Session]:
        """List sessions for a workspace."""
        conditions = ["workspace_id = ?"]
        values: list = [workspace_id]

        if context_id is not None:
            conditions.append("context_id = ?")
            values.append(context_id)

        if not include_expired:
            conditions.append("expires_at >= ?")
            values.append(utc_now_iso())

        query = f"SELECT * FROM sessions WHERE {' AND '.join(conditions)} ORDER BY created_at DESC"
        cursor = await self._connection.execute(query, values)
        rows = await cursor.fetchall()
        return [self._row_to_session(row) for row in rows]

    # ============================================
    # Contradiction Operations
    # ============================================

    async def create_contradiction(self, contradiction: ContradictionRecord) -> ContradictionRecord:
        """Store a contradiction record."""
        await self._connection.execute(
            """
            INSERT INTO contradictions (id, workspace_id, memory_a_id, memory_b_id,
                                        contradiction_type, confidence, detection_method,
                                        detected_at, resolved_at, resolution, merged_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contradiction.id,
                contradiction.workspace_id,
                contradiction.memory_a_id,
                contradiction.memory_b_id,
                contradiction.contradiction_type,
                contradiction.confidence,
                contradiction.detection_method,
                contradiction.detected_at.isoformat(),
                contradiction.resolved_at.isoformat() if contradiction.resolved_at else None,
                contradiction.resolution,
                contradiction.merged_content,
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created contradiction record: %s", contradiction.id)
        return contradiction

    async def get_contradiction(self, workspace_id: str, contradiction_id: str) -> ContradictionRecord | None:
        """Get a specific contradiction."""
        cursor = await self._connection.execute(
            "SELECT * FROM contradictions WHERE id = ? AND workspace_id = ?",
            (contradiction_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_contradiction(row)

    async def get_unresolved_contradictions(self, workspace_id: str, limit: int = 10) -> list[ContradictionRecord]:
        """Get unresolved contradictions for a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM contradictions WHERE workspace_id = ? AND resolved_at IS NULL ORDER BY detected_at DESC LIMIT ?",
            (workspace_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_contradiction(row) for row in rows]

    async def resolve_contradiction(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: str | None = None,
    ) -> ContradictionRecord | None:
        """Resolve a contradiction."""
        now = utc_now_iso()
        cursor = await self._connection.execute(
            "UPDATE contradictions SET resolved_at = ?, resolution = ?, merged_content = ? WHERE id = ? AND workspace_id = ?",
            (now, resolution, merged_content, contradiction_id, workspace_id),
        )
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_contradiction(workspace_id, contradiction_id)

    # ============================================
    # Chat History Operations
    # ============================================

    async def create_thread(self, thread: "ChatThread") -> "ChatThread":
        await self._connection.execute(
            """INSERT INTO chat_threads
               (id, workspace_id, tenant_id, user_id, context_id,
                observer_id, subject_id, title, metadata,
                message_count, last_decomposed_at, last_decomposed_index,
                expires_at, created_at, updated_at, scope)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread.id,
                thread.workspace_id,
                thread.tenant_id,
                thread.user_id,
                thread.context_id,
                thread.observer_id,
                thread.subject_id,
                thread.title,
                json.dumps(thread.metadata),
                thread.message_count,
                thread.last_decomposed_at.isoformat() if thread.last_decomposed_at else None,
                thread.last_decomposed_index,
                thread.expires_at.isoformat() if thread.expires_at else None,
                thread.created_at.isoformat(),
                thread.updated_at.isoformat(),
                thread.scope,
            ),
        )
        await self._connection.commit()
        return thread

    async def get_thread(self, workspace_id: str, thread_id: str) -> "ChatThread | None":
        cursor = await self._connection.execute(
            "SELECT * FROM chat_threads WHERE id = ? AND workspace_id = ?",
            (thread_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_chat_thread(row)

    async def list_threads(
        self,
        workspace_id: str,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        scope_filter: str | None = None,
    ) -> list:
        now = utc_now_iso()
        conditions = ["workspace_id = ?", "(expires_at IS NULL OR expires_at > ?)"]
        params: list = [workspace_id, now]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        # scope_filter="web"  → rows where scope='web' OR scope IS NULL (NULL ≡ web)
        # scope_filter="office" → rows where scope='office'
        # scope_filter=None   → no scope restriction (return all)
        if scope_filter == "web":
            conditions.append("(scope = ? OR scope IS NULL)")
            params.append("web")
        elif scope_filter is not None:
            conditions.append("scope = ?")
            params.append(scope_filter)

        where = " AND ".join(conditions)
        params.extend([limit, offset])
        cursor = await self._connection.execute(
            f"SELECT * FROM chat_threads WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def update_thread(self, workspace_id: str, thread_id: str, **updates) -> "ChatThread | None":
        if not updates:
            return await self.get_thread(workspace_id, thread_id)

        invalid_keys = set(updates.keys()) - _UPDATABLE_THREAD_COLUMNS
        if invalid_keys:
            raise ValueError(f"Invalid update fields: {invalid_keys}")

        set_clauses = []
        values: list = []
        for key, value in updates.items():
            if key == "metadata":
                value = json.dumps(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            set_clauses.append(f"{key} = ?")
            values.append(value)

        set_clauses.append("updated_at = ?")
        values.append(utc_now_iso())

        values.extend([thread_id, workspace_id])
        sql = f"UPDATE chat_threads SET {', '.join(set_clauses)} WHERE id = ? AND workspace_id = ?"
        await self._connection.execute(sql, values)
        await self._connection.commit()
        return await self.get_thread(workspace_id, thread_id)

    async def delete_thread(self, workspace_id: str, thread_id: str) -> bool:
        await self._connection.execute(
            "DELETE FROM chat_messages WHERE thread_id = ? AND workspace_id = ?",
            (thread_id, workspace_id),
        )
        cursor = await self._connection.execute(
            "DELETE FROM chat_threads WHERE id = ? AND workspace_id = ?",
            (thread_id, workspace_id),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def list_expired_threads(self, limit: int = 100) -> list["ChatThread"]:
        """List expired chat threads across all workspaces."""
        now = utc_now_iso()
        cursor = await self._connection.execute(
            "SELECT * FROM chat_threads WHERE expires_at IS NOT NULL AND expires_at < ? ORDER BY expires_at ASC LIMIT ?",
            (now, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        messages: list,
    ) -> list:
        from ...models.chat import ChatMessage

        cursor = await self._connection.execute(
            "SELECT message_count FROM chat_threads WHERE id = ? AND workspace_id = ?",
            (thread_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Thread {thread_id} not found in workspace {workspace_id}")

        current_count = row["message_count"]
        created_messages = []
        now = utc_now_iso()

        for i, msg_input in enumerate(messages):
            msg_id = msg_input.id or generate_id("msg")
            msg_index = current_count + i
            content = msg_input.content
            if not isinstance(content, str):
                content = json.dumps([block.model_dump() for block in content])

            await self._connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, workspace_id, message_index, role, content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, thread_id, workspace_id, msg_index, msg_input.role, content, json.dumps(msg_input.metadata or {}), now),
            )
            created_messages.append(
                ChatMessage(
                    id=msg_id,
                    thread_id=thread_id,
                    message_index=msg_index,
                    role=msg_input.role,
                    content=msg_input.content,
                    metadata=msg_input.metadata or {},
                    created_at=parse_datetime_utc(now),
                )
            )

        new_count = current_count + len(messages)
        await self._connection.execute(
            "UPDATE chat_threads SET message_count = ?, updated_at = ? WHERE id = ? AND workspace_id = ?",
            (new_count, now, thread_id, workspace_id),
        )
        await self._connection.commit()
        return created_messages

    async def get_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        after_index: int | None = None,
        order: str = "asc",
    ) -> list:
        order_clause = "ASC" if order.lower() == "asc" else "DESC"

        if after_index is not None:
            cursor = await self._connection.execute(
                f"""SELECT * FROM chat_messages
                    WHERE thread_id = ? AND workspace_id = ? AND message_index > ?
                    ORDER BY message_index {order_clause} LIMIT ? OFFSET ?""",
                (thread_id, workspace_id, after_index, limit, offset),
            )
        else:
            cursor = await self._connection.execute(
                f"""SELECT * FROM chat_messages
                    WHERE thread_id = ? AND workspace_id = ?
                    ORDER BY message_index {order_clause} LIMIT ? OFFSET ?""",
                (thread_id, workspace_id, limit, offset),
            )

        rows = await cursor.fetchall()
        return [self._row_to_chat_message(row) for row in rows]

    async def get_message_count(self, workspace_id: str, thread_id: str) -> int:
        cursor = await self._connection.execute(
            "SELECT message_count FROM chat_threads WHERE id = ? AND workspace_id = ?",
            (thread_id, workspace_id),
        )
        row = await cursor.fetchone()
        return row["message_count"] if row else 0

    async def delete_message(self, workspace_id: str, thread_id: str, message_id: str) -> bool:
        cursor = await self._connection.execute(
            "DELETE FROM chat_messages WHERE id = ? AND thread_id = ? AND workspace_id = ?",
            (message_id, thread_id, workspace_id),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    # ============================================
    # Row Conversion Helpers
    # ============================================

    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory domain model."""
        return Memory(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else DEFAULT_TENANT_ID,
            workspace_id=row["workspace_id"],
            context_id=row["context_id"] if "context_id" in row.keys() else DEFAULT_CONTEXT_ID,
            session_id=row["session_id"] if "session_id" in row.keys() and row["session_id"] else None,
            source_memory_id=row["source_memory_id"] if "source_memory_id" in row.keys() and row["source_memory_id"] else None,
            source_document_id=row["source_document_id"] if "source_document_id" in row.keys() and row["source_document_id"] else None,
            source_page_id=row["source_page_id"] if "source_page_id" in row.keys() and row["source_page_id"] else None,
            source_dataset_id=row["source_dataset_id"] if "source_dataset_id" in row.keys() and row["source_dataset_id"] else None,
            source_thread_id=row["source_thread_id"] if "source_thread_id" in row.keys() and row["source_thread_id"] else None,
            user_id=row["user_id"],
            observer_id=row["observer_id"] if "observer_id" in row.keys() and row["observer_id"] else None,
            subject_id=row["subject_id"] if "subject_id" in row.keys() and row["subject_id"] else None,
            content=row["content"],
            content_hash=row["content_hash"],
            type=MemoryType(row["type"]),
            subtype=row["subtype"] if row["subtype"] else None,
            category=row["category"] if "category" in row.keys() and row["category"] else None,
            importance=row["importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=self._deserialize_embedding(row["embedding"]) if row["embedding"] else None,
            abstract=row["abstract"] if "abstract" in row.keys() and row["abstract"] else None,
            overview=row["overview"] if "overview" in row.keys() and row["overview"] else None,
            access_count=row["access_count"],
            last_accessed_at=parse_datetime_utc(row["last_accessed_at"]),
            decay_factor=row["decay_factor"],
            status=MemoryStatus(row["status"]) if "status" in row.keys() and row["status"] else MemoryStatus.ACTIVE,
            pinned=bool(row["pinned"]) if "pinned" in row.keys() and row["pinned"] is not None else False,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _row_to_association(self, row) -> Association:
        """Convert database row to Association domain model."""
        return Association(
            id=row["id"],
            workspace_id=row["workspace_id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relationship=row["relationship"],
            strength=row["strength"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
        )

    def _row_to_workspace(self, row) -> Workspace:
        """Convert database row to Workspace domain model."""
        return Workspace(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _row_to_context(self, row) -> Context:
        """Convert database row to Context domain model."""
        return Context(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            description=row["description"],
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
        )

    def _row_to_session(self, row) -> Session:
        """Convert database row to Session domain model."""
        return Session(
            id=row["id"],
            tenant_id=row["tenant_id"] if "tenant_id" in row.keys() else DEFAULT_TENANT_ID,
            workspace_id=row["workspace_id"],
            context_id=row["context_id"] if "context_id" in row.keys() else DEFAULT_CONTEXT_ID,
            user_id=row["user_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            auto_commit=bool(row["auto_commit"]) if "auto_commit" in row.keys() else True,
            committed_at=parse_datetime_utc(row["committed_at"]) if "committed_at" in row.keys() and row["committed_at"] else None,
            expires_at=parse_datetime_utc(row["expires_at"]),
            created_at=parse_datetime_utc(row["created_at"]),
        )

    def _row_to_working_memory(self, row) -> WorkingMemory:
        """Convert database row to WorkingMemory domain model."""
        return WorkingMemory(
            session_id=row["session_id"],
            key=row["key"],
            value=json.loads(row["value"]) if row["value"] else None,
            ttl_seconds=row["ttl_seconds"],
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _row_to_contradiction(self, row) -> ContradictionRecord:
        """Convert database row to ContradictionRecord."""
        return ContradictionRecord(
            id=row["id"],
            workspace_id=row["workspace_id"],
            memory_a_id=row["memory_a_id"],
            memory_b_id=row["memory_b_id"],
            contradiction_type=row["contradiction_type"],
            confidence=row["confidence"] if row["confidence"] else 0.0,
            detection_method=row["detection_method"] if row["detection_method"] else "",
            detected_at=parse_datetime_utc(row["detected_at"]),
            resolved_at=parse_datetime_utc(row["resolved_at"]) if row["resolved_at"] else None,
            resolution=row["resolution"],
            merged_content=row["merged_content"],
        )

    def _row_to_chat_thread(self, row) -> "ChatThread":
        from ...models.chat import ChatThread

        # scope: column may not exist in very old databases (pre-migration); default to None.
        try:
            scope = row["scope"]
        except (IndexError, KeyError):
            scope = None

        return ChatThread(
            id=row["id"],
            workspace_id=row["workspace_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            context_id=row["context_id"],
            observer_id=row["observer_id"],
            subject_id=row["subject_id"],
            title=row["title"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            message_count=row["message_count"],
            last_decomposed_at=parse_datetime_utc(row["last_decomposed_at"]) if row["last_decomposed_at"] else None,
            last_decomposed_index=row["last_decomposed_index"],
            expires_at=parse_datetime_utc(row["expires_at"]) if row["expires_at"] else None,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
            scope=scope,
        )

    def _row_to_chat_message(self, row) -> "ChatMessage":
        from ...models.chat import ChatMessage, ChatMessageContent

        raw_content = row["content"]
        try:
            parsed = json.loads(raw_content)
            if isinstance(parsed, list):
                content = [ChatMessageContent(**block) for block in parsed]
            else:
                content = raw_content
        except (json.JSONDecodeError, TypeError):
            content = raw_content

        return ChatMessage(
            id=row["id"],
            thread_id=row["thread_id"],
            message_index=row["message_index"],
            role=row["role"],
            content=content,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
        )

    # ============================================
    # Embedding Serialization
    # ============================================

    @staticmethod
    def _serialize_embedding(embedding: list[float]) -> bytes:
        """Serialize embedding to binary format for storage."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> list[float]:
        """Deserialize embedding from binary format."""
        num_floats = len(blob) // 4
        return list(struct.unpack(f"{num_floats}f", blob))


# ============================================
# Plugin Registration
# ============================================


class TursoStorageBackendPlugin(StoragePluginBase):
    PROVIDER_NAME = "turso"

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        mode = v.environ(MEMORYLAYER_TURSO_MODE, default=DEFAULT_MEMORYLAYER_TURSO_MODE)
        db_path = v.environ(MEMORYLAYER_TURSO_DB_PATH, default=DEFAULT_MEMORYLAYER_TURSO_DB_PATH)
        url = v.environ(MEMORYLAYER_TURSO_URL, default=None)
        auth_token = v.environ(MEMORYLAYER_TURSO_AUTH_TOKEN, default=None)
        sync_interval = int(v.environ(MEMORYLAYER_TURSO_SYNC_INTERVAL, default=DEFAULT_MEMORYLAYER_TURSO_SYNC_INTERVAL))
        vector_index = v.environ(MEMORYLAYER_TURSO_VECTOR_INDEX, default=DEFAULT_MEMORYLAYER_TURSO_VECTOR_INDEX).lower() == "true"

        return TursoStorageBackend(
            mode=mode,
            db_path=db_path,
            url=url,
            auth_token=auth_token,
            sync_interval=sync_interval,
            vector_index=vector_index,
            v=v,
        )
