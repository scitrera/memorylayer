"""SQLite storage backend with sqlite-vec support."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Any, Optional

import aiosqlite

# Register datetime adapters/converters to fix Python 3.12 deprecation warning
# See: https://docs.python.org/3/library/sqlite3.html#default-adapters-and-converters-deprecated
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("datetime", lambda b: datetime.fromisoformat(b.decode()))

from scitrera_app_framework import Plugin, Variables as Variables

from ...models.memory import Memory, MemoryStatus, RememberInput, MemoryType, MemorySubtype
from ...models.association import Association, AssociateInput, GraphQueryResult, GraphPath
from ...models.workspace import Workspace, Context
from ...models.session import Session, WorkingMemory
from .base import StorageBackend, StoragePluginBase
from ...config import MEMORYLAYER_SQLITE_STORAGE_PATH, DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH
from ...utils import generate_id, utc_now_iso, parse_datetime_utc, cosine_similarity
from ...config import DEFAULT_TENANT_ID, DEFAULT_CONTEXT_ID
from ..contradiction.base import ContradictionRecord


_UPDATABLE_MEMORY_COLUMNS = frozenset({
    "content", "content_hash", "type", "subtype", "importance",
    "tags", "metadata", "embedding", "abstract", "overview",
    "pinned", "category", "decay_factor", "status", "archived_at",
    "observer_id", "subject_id", "source_scope",
    "access_count", "last_accessed_at", "created_at", "updated_at",
    "source_memory_id",
})

_UPDATABLE_THREAD_COLUMNS = frozenset({
    "title", "metadata", "model", "system_prompt",
    "max_messages", "ttl_seconds", "expires_at",
    "last_decomposed_index",
})


class SQLiteStorageBackend(StorageBackend):
    """SQLite storage backend with optional sqlite-vec support."""

    def __init__(self, db_path: str = "memorylayer.db", v: Variables = None):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to SQLite database file
            v: Variables for logging context
        """
        super().__init__(v)
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._has_vec_extension = False

    async def connect(self) -> None:
        """Initialize storage connection."""
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Connecting to SQLite database at %s", Path(self.db_path).absolute())

        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable WAL mode for better concurrent read performance
        await self._connection.execute("PRAGMA journal_mode=WAL")

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Try to load sqlite-vec extension
        try:
            await self._connection.enable_load_extension(True)
            from sqlite_vec import loadable_path
            lp = loadable_path()
            self.logger.debug("sqlite-vec extension path: %s", lp)
            await self._connection.load_extension(lp)
            self._has_vec_extension = True
            self.logger.info("sqlite-vec extension loaded successfully")
        except Exception as e:
            self.logger.warning("sqlite-vec extension not available, using fallback: %s", e)
            self._has_vec_extension = False
        finally:
            # disable extension loading regardless of success or failure
            await self._connection.enable_load_extension(False)

        # Create tables
        await self._create_tables()

        # Ensure reserved entities exist
        await self._ensure_reserved_entities()

        self.logger.info("Connected to SQLite database at %s", self.db_path)

    async def disconnect(self) -> None:
        """Close storage connection."""
        if self._connection:
            await self._connection.close()
            self.logger.info("Disconnected from SQLite database")

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
                                       CREATE TABLE IF NOT EXISTS workspaces
                                       (
                                           id
                                                      TEXT
                                               PRIMARY
                                                   KEY,
                                           tenant_id
                                                      TEXT
                                               NOT
                                                   NULL,
                                           name
                                                      TEXT
                                               NOT
                                                   NULL,
                                           settings
                                                      TEXT
                                                           DEFAULT
                                                               '{}',
                                           created_at
                                                      TEXT
                                                           DEFAULT (
                                                               datetime
                                                               (
                                                                       'now'
                                                               )),
                                           updated_at TEXT DEFAULT
                                                               (
                                                                   datetime
                                                                   (
                                                                           'now'
                                                                   ))
                                       )
                                       """)

        # Contexts (formerly memory_spaces)
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS contexts
                                       (
                                           id
                                                       TEXT
                                               PRIMARY
                                                   KEY,
                                           workspace_id
                                                       TEXT
                                                            NOT
                                                                NULL
                                               REFERENCES
                                                   workspaces
                                                       (
                                                        id
                                                           ),
                                           name        TEXT NOT NULL,
                                           description TEXT,
                                           settings    TEXT DEFAULT '{}',
                                           created_at  TEXT DEFAULT
                                                                (
                                                                    datetime
                                                                    (
                                                                            'now'
                                                                    )),
                                           updated_at  TEXT DEFAULT
                                                                (
                                                                    datetime
                                                                    (
                                                                            'now'
                                                                    )),
                                           UNIQUE
                                               (
                                                workspace_id,
                                                name
                                                   )
                                       )
                                       """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_contexts_workspace ON contexts(workspace_id)"
        )

        # Memories
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS memories
                                       (
                                           id
                                                            TEXT
                                               PRIMARY
                                                   KEY,
                                           tenant_id
                                                            TEXT
                                                                 NOT
                                                                     NULL
                                                                    DEFAULT
                                                                        '_default',
                                           workspace_id
                                                            TEXT
                                                                 NOT
                                                                     NULL,
                                           context_id
                                                            TEXT
                                                                 NOT
                                                                     NULL
                                                                    DEFAULT
                                                                        '_default',
                                           session_id       TEXT,
                                           user_id          TEXT,
                                           content          TEXT NOT NULL,
                                           content_hash     TEXT NOT NULL,
                                           type             TEXT NOT NULL CHECK
                                               (
                                               type
                                                   IN
                                               (
                                                'episodic',
                                                'semantic',
                                                'procedural',
                                                'working'
                                                   )),
                                           subtype          TEXT,
                                           category         TEXT,
                                           importance       REAL    DEFAULT 0.5,
                                           tags             TEXT    DEFAULT '[]',
                                           metadata         TEXT    DEFAULT '{}',
                                           embedding        BLOB,
                                           abstract         TEXT,
                                           overview         TEXT,
                                           source_document_id TEXT,
                                           source_page_id   TEXT,
                                           source_dataset_id TEXT,
                                           source_thread_id TEXT,
                                           access_count     INTEGER DEFAULT 0,
                                           last_accessed_at TEXT,
                                           decay_factor     REAL    DEFAULT 1.0,
                                           deleted_at       TEXT,
                                           created_at       TEXT    DEFAULT
                                                                        (
                                                                            datetime
                                                                            (
                                                                                    'now'
                                                                            )),
                                           updated_at       TEXT    DEFAULT
                                                                        (
                                                                            datetime
                                                                            (
                                                                                    'now'
                                                                            ))
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

        # Add status, pinned, and source_memory_id columns (migration for existing databases)
        for col_sql in [
            "ALTER TABLE memories ADD COLUMN status TEXT DEFAULT 'active'",
            "ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN source_memory_id TEXT",
            # v3: Entity attribution columns
            "ALTER TABLE memories ADD COLUMN observer_id TEXT",
            "ALTER TABLE memories ADD COLUMN subject_id TEXT",
            "ALTER TABLE memories ADD COLUMN source_document_id TEXT",
            "ALTER TABLE memories ADD COLUMN source_page_id TEXT",
            "ALTER TABLE memories ADD COLUMN source_dataset_id TEXT",
            "ALTER TABLE memories ADD COLUMN source_thread_id TEXT",
        ]:
            try:
                await self._connection.execute(col_sql)
            except Exception as e:
                # Column likely already exists (expected during migration)
                self.logger.debug("Column migration note for '%s': %s", col_sql, e)

        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(workspace_id, status) WHERE deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_memory_id) WHERE source_memory_id IS NOT NULL"
        )
        # v3: Entity attribution indexes
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

        # Create FTS5 virtual table for full-text search
        await self._connection.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                workspace_id UNINDEXED,
                content,
                tokenize='porter'
            )
        """)

        # Memory Associations
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS memory_associations
                                       (
                                           id
                                                        TEXT
                                               PRIMARY
                                                   KEY,
                                           workspace_id
                                                        TEXT
                                                             NOT
                                                                 NULL,
                                           source_id
                                                        TEXT
                                                             NOT
                                                                 NULL
                                               REFERENCES
                                                   memories
                                                       (
                                                        id
                                                           ),
                                           target_id    TEXT NOT NULL REFERENCES memories
                                               (
                                                id
                                                   ),
                                           relationship TEXT NOT NULL,
                                           strength     REAL DEFAULT 0.5,
                                           metadata     TEXT DEFAULT '{}',
                                           created_at   TEXT DEFAULT
                                                                 (
                                                                     datetime
                                                                     (
                                                                             'now'
                                                                     )),
                                           UNIQUE
                                               (
                                                source_id,
                                                target_id,
                                                relationship
                                                   )
                                       )
                                       """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_associations_workspace ON memory_associations(workspace_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_associations_source ON memory_associations(source_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_associations_target ON memory_associations(target_id)"
        )

        # Sessions table (for persistent session storage)
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS sessions
                                       (
                                           id
                                               TEXT
                                               PRIMARY
                                                   KEY,
                                           tenant_id
                                               TEXT
                                               NOT
                                                   NULL
                                               DEFAULT
                                                   '_default',
                                           workspace_id
                                               TEXT
                                               NOT
                                                   NULL,
                                           context_id
                                               TEXT
                                               NOT
                                                   NULL,
                                           user_id
                                               TEXT,
                                           metadata
                                               TEXT
                                               NOT
                                                   NULL
                                               DEFAULT
                                                   '{}',
                                           auto_commit
                                               INTEGER
                                               DEFAULT
                                                   1,
                                           expires_at
                                               TEXT
                                               NOT
                                                   NULL,
                                           committed_at
                                               TEXT,
                                           created_at
                                               TEXT
                                               NOT
                                                   NULL,
                                           last_accessed_at
                                               TEXT,
                                           FOREIGN
                                               KEY
                                               (
                                                workspace_id
                                                   ) REFERENCES workspaces
                                               (
                                                id
                                                   )
                                       )
                                       """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_context ON sessions(context_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)"
        )

        # Working memory table (formerly session_contexts)
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS working_memory
                                       (
                                           session_id
                                                      TEXT
                                               NOT
                                                   NULL,
                                           key
                                                      TEXT
                                               NOT
                                                   NULL,
                                           value
                                                      TEXT
                                               NOT
                                                   NULL,
                                           ttl_seconds
                                                      INTEGER,
                                           created_at
                                                      TEXT
                                                           DEFAULT (
                                                               datetime
                                                               (
                                                                       'now'
                                                               )),
                                           updated_at TEXT DEFAULT
                                                               (
                                                                   datetime
                                                                   (
                                                                           'now'
                                                                   )),
                                           PRIMARY KEY
                                               (
                                                session_id,
                                                key
                                                   ),
                                           FOREIGN KEY
                                               (
                                                session_id
                                                   ) REFERENCES sessions
                                               (
                                                id
                                                   ) ON DELETE CASCADE
                                       )
                                       """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_working_memory_session ON working_memory(session_id)"
        )

        # Contradictions table
        await self._connection.execute("""
                                       CREATE TABLE IF NOT EXISTS contradictions
                                       (
                                           id                 TEXT PRIMARY KEY,
                                           workspace_id       TEXT NOT NULL,
                                           memory_a_id        TEXT NOT NULL,
                                           memory_b_id        TEXT NOT NULL,
                                           contradiction_type TEXT,
                                           confidence         REAL DEFAULT 0.0,
                                           detection_method   TEXT,
                                           detected_at        TEXT DEFAULT (datetime('now')),
                                           resolved_at        TEXT,
                                           resolution         TEXT,
                                           merged_content     TEXT,
                                           FOREIGN KEY (memory_a_id) REFERENCES memories (id),
                                           FOREIGN KEY (memory_b_id) REFERENCES memories (id)
                                       )
                                       """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_workspace ON contradictions(workspace_id)"
        )
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
                FOREIGN KEY (workspace_id) REFERENCES workspaces (id)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_workspace ON chat_threads(workspace_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(workspace_id, user_id) WHERE user_id IS NOT NULL"
        )

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
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, message_index)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace ON chat_messages(workspace_id, thread_id)"
        )

        await self._connection.commit()

    async def _ensure_reserved_entities(self) -> None:
        """Ensure reserved entities exist (_default workspace, _global workspace, _default contexts)."""
        from ...config import DEFAULT_TENANT_ID, DEFAULT_WORKSPACE_ID, GLOBAL_WORKSPACE_ID

        now = utc_now_iso()

        # Create _default workspace (main default for auto-discovery)
        await self._connection.execute("""
                                       INSERT
                                           OR IGNORE
                                       INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
                                       VALUES (?, ?, 'Default Workspace', '{}', ?, ?)
                                       """, (DEFAULT_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now))

        # Create _global workspace (cross-workspace shared storage)
        await self._connection.execute("""
                                       INSERT
                                           OR IGNORE
                                       INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
                                       VALUES (?, ?, 'Global Workspace', '{}', ?, ?)
                                       """, (GLOBAL_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now))

        # Get all workspaces
        cursor = await self._connection.execute("SELECT id FROM workspaces")
        workspaces = await cursor.fetchall()

        # Create _default context for each workspace if not exists
        for workspace in workspaces:
            workspace_id = workspace["id"]
            await self._connection.execute("""
                                           INSERT
                                               OR IGNORE
                                           INTO contexts (id, workspace_id, name, description, settings, created_at, updated_at)
                                           VALUES ('_default', ?, '_default', 'Default context', '{}', ?, ?)
                                           """, (workspace_id, now, now))

        await self._connection.commit()
        self.logger.info("Reserved entities initialized (_default workspace, _global workspace, _default contexts)")

    # Memory operations
    async def create_memory(self, workspace_id: str, input: RememberInput) -> Memory:
        """Store a new memory."""
        from ...config import DEFAULT_TENANT_ID

        # Compute content hash
        content_hash = hashlib.sha256(input.content.encode()).hexdigest()

        memory_id = generate_id("mem")
        now = utc_now_iso()

        cursor = await self._connection.execute(
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
                getattr(input, 'tenant_id', None) or DEFAULT_TENANT_ID,
                workspace_id,
                getattr(input, 'context_id', None) or '_default',
                getattr(input, 'session_id', None),
                input.user_id,
                input.content,
                content_hash,
                input.type.value if input.type else MemoryType.SEMANTIC.value,
                input.subtype.value if input.subtype else None,
                getattr(input, 'category', None),
                input.importance,
                json.dumps(input.tags),
                json.dumps(input.metadata),
                getattr(input, 'abstract', None),
                getattr(input, 'overview', None),
                getattr(input, 'source_memory_id', None),
                MemoryStatus.ACTIVE.value,
                0,
                getattr(input, 'observer_id', None),
                getattr(input, 'subject_id', None),
                getattr(input, 'source_document_id', None),
                getattr(input, 'source_page_id', None),
                getattr(input, 'source_dataset_id', None),
                getattr(input, 'source_thread_id', None),
                now,
                now,
            ),
        )

        # Insert into FTS index
        await self._connection.execute(
            "INSERT INTO memories_fts (id, workspace_id, content) VALUES (?, ?, ?)",
            (memory_id, workspace_id, input.content),
        )

        await self._connection.commit()

        return await self.get_memory(workspace_id, memory_id, track_access=False)

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Optional[Memory]:
        """Get memory by ID within a workspace. Set track_access=False for internal reads that should not affect decay tracking."""
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memories
            WHERE id = ?
              AND workspace_id = ?
              AND deleted_at IS NULL
            """,
            (memory_id, workspace_id),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access:
            # Update access tracking
            await self._connection.execute(
                """
                UPDATE memories
                SET access_count     = access_count + 1,
                    last_accessed_at = datetime('now')
                WHERE id = ?
                """,
                (memory_id,),
            )
            await self._connection.commit()
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.now(timezone.utc)

        return memory

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Optional[Memory]:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memories
            WHERE id = ?
              AND deleted_at IS NULL
            """,
            (memory_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access:
            await self._connection.execute(
                """
                UPDATE memories
                SET access_count     = access_count + 1,
                    last_accessed_at = datetime('now')
                WHERE id = ?
                """,
                (memory_id,),
            )
            await self._connection.commit()
            # Reflect the increment in the returned object
            memory.access_count = (memory.access_count or 0) + 1
            memory.last_accessed_at = datetime.now(timezone.utc)

        return memory

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Optional[Memory]:
        """Update memory fields."""
        invalid_keys = set(updates.keys()) - _UPDATABLE_MEMORY_COLUMNS
        if invalid_keys:
            raise ValueError(f"Invalid update fields: {invalid_keys}")
        # Build SET clause
        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ("tags", "metadata"):
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            elif key == "embedding":
                # Embedding is stored as binary BLOB
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
            SET {', '.join(set_parts)}
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
            cursor = await self._connection.execute(
                "DELETE FROM memories WHERE id = ? AND workspace_id = ?",
                (memory_id, workspace_id),
            )
            # Also delete from FTS index
            await self._connection.execute(
                "DELETE FROM memories_fts WHERE id = ?",
                (memory_id,),
            )
        else:
            cursor = await self._connection.execute(
                """
                UPDATE memories
                SET deleted_at = datetime('now'),
                    status     = 'deleted'
                WHERE id = ?
                  AND workspace_id = ?
                """,
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
        params = [workspace_id]

        if exclude_pinned:
            where_parts.append("(pinned IS NULL OR pinned = 0)")

        query = f"""
            SELECT * FROM memories
            WHERE {' AND '.join(where_parts)}
            ORDER BY importance DESC
        """
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
                SELECT *
                FROM memories
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
        cursor = await self._connection.execute(
            query, (workspace_id, max_importance, max_access_count, older_than_days, limit)
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs."""
        cursor = await self._connection.execute("SELECT id FROM workspaces")
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

    async def search_memories(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            offset: int = 0,
            min_relevance: float = 0.5,
            types: Optional[list[str]] = None,
            subtypes: Optional[list[str]] = None,
            tags: Optional[list[str]] = None,
            include_archived: bool = False,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search using sqlite-vec or fallback."""
        if self._has_vec_extension:
            return await self._search_with_vec(
                workspace_id, query_embedding, limit, offset, min_relevance, types, subtypes, tags,
                include_archived=include_archived, observer_id=observer_id, subject_id=subject_id,
            )
        else:
            return await self._search_with_fallback(
                workspace_id, query_embedding, limit, offset, min_relevance, types, subtypes, tags,
                include_archived=include_archived, observer_id=observer_id, subject_id=subject_id,
            )

    async def _search_with_vec(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int,
            offset: int,
            min_relevance: float,
            types: Optional[list[str]],
            subtypes: Optional[list[str]],
            tags: Optional[list[str]],
            include_archived: bool = False,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Search using sqlite-vec extension."""
        # Build WHERE clause
        where_parts = ["workspace_id = ?", "deleted_at IS NULL", "embedding IS NOT NULL"]
        if not include_archived:
            where_parts.append("(status IS NULL OR status = 'active')")
        params = [workspace_id]

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

        where_clause = " AND ".join(where_parts)

        # Use sqlite-vec for similarity search
        # vec_distance_cosine(embedding, ?) appears first in SQL (SELECT clause),
        # so its parameter must come before the WHERE clause parameters.
        query_vec_blob = self._serialize_embedding(query_embedding)
        params_ordered = [query_vec_blob] + params + [limit, offset]

        query = f"""
            SELECT *, vec_distance_cosine(embedding, ?) as distance
            FROM memories
            WHERE {where_clause}
            ORDER BY distance ASC
            LIMIT ? OFFSET ?
        """

        cursor = await self._connection.execute(query, params_ordered)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            # Convert distance to relevance (1 - distance for cosine)
            distance = row["distance"]
            relevance = 1.0 - distance

            if relevance >= min_relevance:
                memory = self._row_to_memory(row)
                results.append((memory, relevance))

        return results

    async def _search_with_fallback(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int,
            offset: int,
            min_relevance: float,
            types: Optional[list[str]],
            subtypes: Optional[list[str]],
            tags: Optional[list[str]],
            include_archived: bool = False,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Fallback: compute cosine similarity in Python."""
        # Build WHERE clause
        where_parts = ["workspace_id = ?", "deleted_at IS NULL", "embedding IS NOT NULL"]
        if not include_archived:
            where_parts.append("(status IS NULL OR status = 'active')")
        params = [workspace_id]

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

        where_clause = " AND ".join(where_parts)

        query = f"SELECT * FROM memories WHERE {where_clause}"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        # Compute cosine similarity in Python
        results = []
        for row in rows:
            if row["embedding"]:
                embedding = self._deserialize_embedding(row["embedding"])
                relevance = cosine_similarity(query_embedding, embedding)

                if relevance >= min_relevance:
                    memory = self._row_to_memory(row)
                    results.append((memory, relevance))

        # Sort by relevance descending, apply offset and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[offset:offset + limit]

    async def full_text_search(
            self,
            workspace_id: str,
            query: str,
            limit: int = 10,
            offset: int = 0,
    ) -> list[Memory]:
        """Full-text search using SQLite FTS5."""
        cursor = await self._connection.execute(
            """
            SELECT m.*
            FROM memories m
                     INNER JOIN memories_fts fts ON m.id = fts.id
            WHERE fts.workspace_id = ?
              AND fts.content MATCH ?
              AND m.deleted_at IS NULL
            LIMIT ? OFFSET ?
            """,
            (workspace_id, query, limit, offset),
        )
        rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Optional[Memory]:
        """Get memory by content hash for deduplication."""
        cursor = await self._connection.execute(
            """
            SELECT *
            FROM memories
            WHERE workspace_id = ?
              AND content_hash = ?
              AND deleted_at IS NULL
            LIMIT 1
            """,
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
            SELECT *
            FROM memories
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

        # Convert rows to dicts based on detail_level
        results = []
        for row in rows:
            if detail_level == "abstract":
                # Return only id, abstract, type, subtype, importance, tags, created_at
                results.append({
                    "id": row["id"],
                    "abstract": row["abstract"] if row["abstract"] else None,
                    "type": row["type"],
                    "subtype": row["subtype"] if row["subtype"] else None,
                    "importance": row["importance"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "created_at": row["created_at"],
                })
            elif detail_level == "overview":
                # Add overview field (and exclude abstract field)
                results.append({
                    "id": row["id"],
                    # "abstract": row["abstract"] if row["abstract"] else None,
                    "overview": row["overview"] if row["overview"] else None,
                    "type": row["type"],
                    "subtype": row["subtype"] if row["subtype"] else None,
                    "importance": row["importance"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "created_at": row["created_at"],
                })
            else:  # "full" -- full detail will return the content and doesn't need to return the abstract and overview fields
                # Return everything as dict
                memory = self._row_to_memory(row)
                results.append({
                    "id": memory.id,
                    "content": memory.content,
                    # "abstract": memory.abstract,
                    # "overview": memory.overview,
                    "type": memory.type.value if hasattr(memory.type, 'value') else str(memory.type),
                    "subtype": memory.subtype.value if memory.subtype and hasattr(memory.subtype, 'value') else str(
                        memory.subtype) if memory.subtype else None,
                    "importance": memory.importance,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                })

        return results

    # Association operations
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
            relationships: Optional[list[str]] = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        # Build WHERE clause
        where_parts = ["workspace_id = ?"]
        params = [workspace_id]

        if direction == "outgoing":
            where_parts.append("source_id = ?")
            params.append(memory_id)
        elif direction == "incoming":
            where_parts.append("target_id = ?")
            params.append(memory_id)
        else:  # both
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

    async def traverse_graph(
            self,
            workspace_id: str,
            start_id: str,
            max_depth: int = 3,
            relationships: Optional[list[str]] = None,
            direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop graph traversal using recursive CTE."""
        # Build recursive CTE
        # Note: Use separate filters for base case (no table alias) and recursive case (with 'a.' prefix)
        base_rel_filter = ""
        recursive_rel_filter = ""
        rel_params: tuple = ()
        if relationships:
            placeholders = ", ".join("?" * len(relationships))
            base_rel_filter = f"AND relationship IN ({placeholders})"
            recursive_rel_filter = f"AND a.relationship IN ({placeholders})"
            rel_params = tuple(relationships)

        # Build direction condition for join and next node selection
        if direction == "outgoing":
            direction_condition = "a.source_id = gt.current_node"
            next_node = "a.target_id"
            # Base case: start from associations where source_id = start_id
            base_start_condition = "source_id = ?"
            base_current_node = "target_id"
        elif direction == "incoming":
            direction_condition = "a.target_id = gt.current_node"
            next_node = "a.source_id"
            # Base case: start from associations where target_id = start_id (finding who points to us)
            base_start_condition = "target_id = ?"
            base_current_node = "source_id"
        else:  # both
            direction_condition = "(a.source_id = gt.current_node OR a.target_id = gt.current_node)"
            next_node = "CASE WHEN a.source_id = gt.current_node THEN a.target_id ELSE a.source_id END"
            # Base case: start from associations where start_id is either source or target
            base_start_condition = "(source_id = ? OR target_id = ?)"
            base_current_node = "CASE WHEN source_id = ? THEN target_id ELSE source_id END"

        # Build params based on direction
        # For "both" direction, the CASE WHEN in SELECT needs start_id first
        # Params order: [SELECT CASE placeholder], WHERE workspace_id, WHERE condition placeholders
        if direction == "both":
            # CASE WHEN source_id = ? (start_id), workspace_id = ?, source_id = ? OR target_id = ?
            base_case_params = (start_id, workspace_id, start_id, start_id)
        else:
            # workspace_id = ?, start_condition = ?
            base_case_params = (workspace_id, start_id)

        query = f"""
        WITH RECURSIVE graph_traverse(
            id, source_id, target_id, relationship, strength, metadata, created_at,
            depth, current_node, path
        ) AS (
            -- Base case
            SELECT
                id, source_id, target_id, relationship, strength, metadata, created_at,
                1 as depth,
                {base_current_node} as current_node,
                json_array(source_id, target_id) as path
            FROM memory_associations
            WHERE workspace_id = ?
              AND {base_start_condition}
              {base_rel_filter}

            UNION

            -- Recursive case
            SELECT
                a.id, a.source_id, a.target_id, a.relationship, a.strength, a.metadata, a.created_at,
                gt.depth + 1,
                {next_node},
                json_insert(gt.path, '$[#]', {next_node})
            FROM memory_associations a
            INNER JOIN graph_traverse gt ON (
                {direction_condition}
                AND a.workspace_id = ?
                {recursive_rel_filter}
                AND gt.depth < ?
            )
            WHERE NOT EXISTS (
                SELECT 1 FROM json_each(gt.path)
                WHERE json_each.value = {next_node}
            )
        )
        SELECT * FROM graph_traverse;
        """

        # Build final parameters: base_case_params + rel_params (base filter) + recursive_case_params + rel_params (recursive filter)
        params = base_case_params + rel_params + (workspace_id,) + rel_params + (max_depth,)
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        # Build paths from results
        paths = []
        unique_nodes = set([start_id])

        for row in rows:
            path_nodes = json.loads(row["path"])
            unique_nodes.update(path_nodes)

            # Create association edge
            edge = Association(
                id=row["id"],
                workspace_id=workspace_id,
                source_id=row["source_id"],
                target_id=row["target_id"],
                relationship=row["relationship"],
                strength=row["strength"],
                metadata=json.loads(row["metadata"]),
                created_at=parse_datetime_utc(row["created_at"]),
            )

            path = GraphPath(
                nodes=path_nodes,
                edges=[edge],
                total_strength=row["strength"],
                depth=row["depth"],
            )
            paths.append(path)

        return GraphQueryResult(
            paths=paths,
            total_paths=len(paths),
            unique_nodes=list(unique_nodes),
            query_latency_ms=0,
        )

    # Workspace operations
    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace."""
        await self._connection.execute(
            """
            INSERT INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
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

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM workspaces WHERE id = ?",
            (workspace_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_workspace(row)

    async def list_workspaces(self) -> list[Workspace]:
        """List all workspaces."""
        cursor = await self._connection.execute(
            "SELECT * FROM workspaces ORDER BY name",
        )
        rows = await cursor.fetchall()
        return [self._row_to_workspace(row) for row in rows]

    # Context operations (formerly Memory Space)
    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        await self._connection.execute(
            """
            INSERT INTO contexts (id, workspace_id, name, description, settings, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
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

    async def get_context(self, workspace_id: str, context_id: str) -> Optional[Context]:
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

    # Statistics
    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        # Count memories by type
        cursor = await self._connection.execute(
            """
            SELECT type, COUNT(*) as count
            FROM memories
            WHERE workspace_id = ?
              AND deleted_at IS NULL
            GROUP BY type
            """,
            (workspace_id,),
        )
        type_counts = {row["type"]: row["count"] for row in await cursor.fetchall()}

        # Count associations
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

    # Session operations
    async def create_session(self, workspace_id: str, session: Session) -> Session:
        """Store a new session."""
        # Ensure workspace exists (auto-create for OSS local use)
        now = utc_now_iso()
        await self._connection.execute(
            """
            INSERT
                OR IGNORE
            INTO workspaces (id, tenant_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (workspace_id, "default", workspace_id, now, now)
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

    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """Get session by ID (returns None if not found or expired)."""
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ? AND workspace_id = ?",
            (session_id, workspace_id),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        session = self._row_to_session(row)

        # FIX: do not delete expired sessions on get/check BECAUSE we have background task
        #       for deleting expired sessions (that may trigger auto-commit but commit will fail
        #       if we deleted the session while looking it up here!)
        # # Check expiration
        # if session.is_expired:
        #     self.logger.info("Session expired: %s, cleaning up", session_id)
        #     await self.delete_session(workspace_id, session_id)
        #     return None

        return session

    async def get_session_by_id(self, session_id: str) -> Optional[Session]:
        """Get session by ID without workspace filter.

        This allows looking up a session when the workspace is not yet known,
        such as when resolving a session from the X-Session-ID header.
        """
        cursor = await self._connection.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        session = self._row_to_session(row)

        # FIX: do not delete expired sessions on get/check BECAUSE we have background task
        #       for deleting expired sessions (that may trigger auto-commit but commit will fail
        #       if we deleted the session while looking it up here!)

        return session

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
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """Set working memory key-value within session."""
        now_iso = utc_now_iso()
        now = datetime.now(timezone.utc)

        # Use INSERT OR REPLACE for upsert behavior
        await self._connection.execute(
            """
            INSERT INTO working_memory (session_id, key, value, ttl_seconds, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, key) DO UPDATE SET value       = excluded.value,
                                                       ttl_seconds = excluded.ttl_seconds,
                                                       updated_at  = excluded.updated_at
            """,
            (
                session_id,
                key,
                json.dumps(value),
                ttl_seconds,
                now_iso,
                now_iso,
            ),
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

    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional[WorkingMemory]:
        """Get specific working memory entry."""
        cursor = await self._connection.execute(
            "SELECT * FROM working_memory WHERE session_id = ? AND key = ?",
            (session_id, key),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_working_memory(row)

    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> list[WorkingMemory]:
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

        cursor = await self._connection.execute(
            "DELETE FROM sessions WHERE expires_at < ?",
            (now,),
        )
        await self._connection.commit()

        return cursor.rowcount

    async def list_expired_sessions(self, limit: int = 100) -> list[Session]:
        """List expired sessions that need cleanup.

        Used by the cleanup task to retrieve sessions before deletion,
        enabling auto-commit of working memory before cleanup.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of expired sessions
        """
        now = utc_now_iso()

        cursor = await self._connection.execute(
            """
            SELECT *
            FROM sessions
            WHERE expires_at < ?
            ORDER BY expires_at ASC
            LIMIT ?
            """,
            (now, limit),
        )
        rows = await cursor.fetchall()

        return [self._row_to_session(row) for row in rows]

    async def update_session(
            self,
            workspace_id: str,
            session_id: str,
            **updates
    ) -> Optional[Session]:
        """Update session fields.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to update
            **updates: Fields to update (e.g., committed_at, expires_at)

        Returns:
            Updated session or None if not found
        """
        if not updates:
            return await self.get_session(workspace_id, session_id)

        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        for field, value in updates.items():
            if field in ('committed_at', 'expires_at') and isinstance(value, datetime):
                values.append(value.isoformat())
            elif field == 'auto_commit':
                values.append(1 if value else 0)
            elif field == 'metadata':
                values.append(json.dumps(value))
            else:
                values.append(value)
            set_clauses.append(f"{field} = ?")

        values.extend([session_id, workspace_id])

        query = f"""
            UPDATE sessions
            SET {', '.join(set_clauses)}
            WHERE id = ? AND workspace_id = ?
        """

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

    # Contradiction operations
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

    async def get_contradiction(self, workspace_id: str, contradiction_id: str) -> Optional[ContradictionRecord]:
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
            """
            SELECT *
            FROM contradictions
            WHERE workspace_id = ?
              AND resolved_at IS NULL
            ORDER BY detected_at DESC
            LIMIT ?
            """,
            (workspace_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_contradiction(row) for row in rows]

    async def resolve_contradiction(
            self,
            workspace_id: str,
            contradiction_id: str,
            resolution: str,
            merged_content: Optional[str] = None,
    ) -> Optional[ContradictionRecord]:
        """Resolve a contradiction."""
        now = utc_now_iso()
        cursor = await self._connection.execute(
            """
            UPDATE contradictions
            SET resolved_at    = ?,
                resolution     = ?,
                merged_content = ?
            WHERE id = ?
              AND workspace_id = ?
            """,
            (now, resolution, merged_content, contradiction_id, workspace_id),
        )
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None

        return await self.get_contradiction(workspace_id, contradiction_id)

    def _row_to_contradiction(self, row: aiosqlite.Row) -> ContradictionRecord:
        """Convert database row to ContradictionRecord."""
        return ContradictionRecord(
            id=row["id"],
            workspace_id=row["workspace_id"],
            memory_a_id=row["memory_a_id"],
            memory_b_id=row["memory_b_id"],
            contradiction_type=row["contradiction_type"],
            confidence=row["confidence"] if row["confidence"] else 0.0,
            detection_method=row["detection_method"] if row["detection_method"] else '',
            detected_at=parse_datetime_utc(row["detected_at"]),
            resolved_at=parse_datetime_utc(row["resolved_at"]) if row["resolved_at"] else None,
            resolution=row["resolution"],
            merged_content=row["merged_content"],
        )

    # Helper methods
    def _row_to_memory(self, row: aiosqlite.Row) -> Memory:
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
            subtype=MemorySubtype(row["subtype"]) if row["subtype"] else None,
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

    def _row_to_association(self, row: aiosqlite.Row) -> Association:
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

    def _row_to_workspace(self, row: aiosqlite.Row) -> Workspace:
        """Convert database row to Workspace domain model."""
        return Workspace(
            id=row["id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _row_to_context(self, row: aiosqlite.Row) -> Context:
        """Convert database row to Context domain model."""
        return Context(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            description=row["description"],
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
        )

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
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

    def _row_to_working_memory(self, row: aiosqlite.Row) -> WorkingMemory:
        """Convert database row to WorkingMemory domain model."""
        return WorkingMemory(
            session_id=row["session_id"],
            key=row["key"],
            value=json.loads(row["value"]) if row["value"] else None,
            ttl_seconds=row["ttl_seconds"],
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _serialize_embedding(self, embedding: list[float]) -> bytes:
        """Serialize embedding to binary format for storage."""
        import struct
        return struct.pack(f'{len(embedding)}f', *embedding)

    def _deserialize_embedding(self, blob: bytes) -> list[float]:
        """Deserialize embedding from binary format."""
        import struct
        num_floats = len(blob) // 4
        return list(struct.unpack(f'{num_floats}f', blob))


    # ============================================
    # Chat History Operations
    # ============================================

    async def create_thread(self, thread: 'ChatThread') -> 'ChatThread':
        from ...models.chat import ChatThread as ChatThreadModel
        await self._connection.execute(
            """INSERT INTO chat_threads
               (id, workspace_id, tenant_id, user_id, context_id,
                observer_id, subject_id, title, metadata,
                message_count, last_decomposed_at, last_decomposed_index,
                expires_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread.id, thread.workspace_id, thread.tenant_id,
                thread.user_id, thread.context_id,
                thread.observer_id, thread.subject_id, thread.title,
                json.dumps(thread.metadata),
                thread.message_count,
                thread.last_decomposed_at.isoformat() if thread.last_decomposed_at else None,
                thread.last_decomposed_index,
                thread.expires_at.isoformat() if thread.expires_at else None,
                thread.created_at.isoformat(),
                thread.updated_at.isoformat(),
            ),
        )
        await self._connection.commit()
        return thread

    async def get_thread(self, workspace_id: str, thread_id: str) -> 'Optional[ChatThread]':
        from ...models.chat import ChatThread as ChatThreadModel
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
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list:
        now = utc_now_iso()
        if user_id:
            cursor = await self._connection.execute(
                """SELECT * FROM chat_threads
                   WHERE workspace_id = ? AND user_id = ?
                     AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (workspace_id, user_id, now, limit, offset),
            )
        else:
            cursor = await self._connection.execute(
                """SELECT * FROM chat_threads
                   WHERE workspace_id = ?
                     AND (expires_at IS NULL OR expires_at > ?)
                   ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
                (workspace_id, now, limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def update_thread(self, workspace_id: str, thread_id: str, **updates) -> 'Optional[ChatThread]':
        if not updates:
            return await self.get_thread(workspace_id, thread_id)

        invalid_keys = set(updates.keys()) - _UPDATABLE_THREAD_COLUMNS
        if invalid_keys:
            raise ValueError(f"Invalid update fields: {invalid_keys}")

        set_clauses = []
        values = []
        for key, value in updates.items():
            if key == "metadata":
                value = json.dumps(value)
            elif isinstance(value, datetime):
                value = value.isoformat()
            set_clauses.append(f"{key} = ?")
            values.append(value)

        # Always update updated_at
        set_clauses.append("updated_at = ?")
        values.append(utc_now_iso())

        values.extend([thread_id, workspace_id])
        sql = f"UPDATE chat_threads SET {', '.join(set_clauses)} WHERE id = ? AND workspace_id = ?"
        await self._connection.execute(sql, values)
        await self._connection.commit()
        return await self.get_thread(workspace_id, thread_id)

    async def delete_thread(self, workspace_id: str, thread_id: str) -> bool:
        # Delete messages first (FK cascade may handle this, but be explicit)
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

    async def list_expired_threads(self, limit: int = 100) -> list['ChatThread']:
        """List expired chat threads across all workspaces.

        Queries for threads where expires_at is set and in the past.

        Args:
            limit: Maximum number of threads to return

        Returns:
            List of expired ChatThread objects
        """
        now = utc_now_iso()

        cursor = await self._connection.execute(
            """
            SELECT *
            FROM chat_threads
            WHERE expires_at IS NOT NULL AND expires_at < ?
            ORDER BY expires_at ASC
            LIMIT ?
            """,
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

        # Get current message count for indexing
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
            msg_id = generate_id("msg")
            msg_index = current_count + i
            content = msg_input.content
            if not isinstance(content, str):
                # Structured content — serialize as JSON array
                content = json.dumps([block.model_dump() for block in content])

            await self._connection.execute(
                """INSERT INTO chat_messages
                   (id, thread_id, workspace_id, message_index, role, content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    msg_id, thread_id, workspace_id, msg_index,
                    msg_input.role, content,
                    json.dumps(msg_input.metadata or {}), now,
                ),
            )
            created_messages.append(ChatMessage(
                id=msg_id,
                thread_id=thread_id,
                message_index=msg_index,
                role=msg_input.role,
                content=msg_input.content,
                metadata=msg_input.metadata or {},
                created_at=parse_datetime_utc(now),
            ))

        # Update thread message count and updated_at
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
        after_index: Optional[int] = None,
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

    def _row_to_chat_thread(self, row: aiosqlite.Row) -> 'ChatThread':
        from ...models.chat import ChatThread
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
        )

    def _row_to_chat_message(self, row: aiosqlite.Row) -> 'ChatMessage':
        from ...models.chat import ChatMessage, ChatMessageContent
        raw_content = row["content"]
        # Try to parse as structured content (JSON array)
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


class SqliteStorageBackendPlugin(StoragePluginBase):
    PROVIDER_NAME = 'sqlite'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return SQLiteStorageBackend(
            db_path=v.environ(MEMORYLAYER_SQLITE_STORAGE_PATH, default=DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH),
            v=v
        )
