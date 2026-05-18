"""SQLite storage backend with sqlite-vec support."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from logging import Logger
from pathlib import Path
from typing import Any

import aiosqlite

# Register datetime adapters/converters to fix Python 3.12 deprecation warning
# See: https://docs.python.org/3/library/sqlite3.html#default-adapters-and-converters-deprecated
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("datetime", lambda b: datetime.fromisoformat(b.decode()))

from scitrera_app_framework import Variables as Variables

from ...config import DEFAULT_CONTEXT_ID, DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH, DEFAULT_TENANT_ID, MEMORYLAYER_SQLITE_STORAGE_PATH
from ...models.association import AssociateInput, Association, GraphPath, GraphQueryResult
from ...models.memory import Memory, MemoryStatus, MemoryType, RememberInput
from ...models.session import Session, WorkingMemory
from ...models.workspace import Context, Workspace
from ...utils import cosine_similarity, generate_id, parse_datetime_utc, utc_now_iso
from ..contradiction.base import ContradictionRecord
from .base import StorageBackend, StoragePluginBase

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
        self._connection: aiosqlite.Connection | None = None
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
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_contexts_workspace ON contexts(workspace_id)")

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
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_workspace ON memory_associations(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_source ON memory_associations(source_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_associations_target ON memory_associations(target_id)")

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
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_workspace ON sessions(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_context ON sessions(context_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")

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
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_working_memory_session ON working_memory(session_id)")

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
        # Migrate: add scope column to existing databases (idempotent — ALTER TABLE
        # fails silently if the column is already present in SQLite 3.37+; we catch
        # the OperationalError for older runtimes).
        try:
            await self._connection.execute("ALTER TABLE chat_threads ADD COLUMN scope TEXT")
            await self._connection.commit()
        except Exception:
            pass  # Column already exists — expected on databases created after the schema update
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_workspace ON chat_threads(workspace_id)")
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
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, message_index)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace ON chat_messages(workspace_id, thread_id)")

        # Documents
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                document_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mime_type TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                target_context_id TEXT NOT NULL DEFAULT '_default',
                extraction_options TEXT DEFAULT '{}',
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                memory_ids TEXT DEFAULT '[]',
                deduplicated_count INTEGER DEFAULT 0,
                error_message TEXT,
                metadata TEXT DEFAULT '{}',
                extracted_metadata TEXT DEFAULT '{}',
                raw_content BLOB,
                created_at TEXT NOT NULL,
                processing_started_at TEXT,
                processing_completed_at TEXT,
                UNIQUE (workspace_id, content_hash)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(workspace_id, status)")

        # Document pages
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(id),
                workspace_id TEXT NOT NULL,
                page_no INTEGER NOT NULL,
                transcript TEXT,
                multivector TEXT,
                transcript_model TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT,
                UNIQUE (document_id, page_no)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_document_pages_workspace ON document_pages(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_document_pages_document ON document_pages(document_id)")

        # Ingestion jobs
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_jobs (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                document_ids TEXT DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'queued',
                progress_percent INTEGER DEFAULT 0,
                documents_processed INTEGER DEFAULT 0,
                total_memories_created INTEGER DEFAULT 0,
                errors TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_workspace ON ingestion_jobs(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status ON ingestion_jobs(workspace_id, status)")

        # Data providers
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS data_providers (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                provider_type TEXT NOT NULL,
                description TEXT,
                enabled INTEGER NOT NULL DEFAULT 1,
                connection_args TEXT DEFAULT '{}',
                schedule TEXT,
                last_sync_at TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_data_providers_workspace ON data_providers(workspace_id)")

        # Skills
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT '',
                workspace_id TEXT NOT NULL,
                user_id TEXT,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                version TEXT NOT NULL DEFAULT '0.1.0',
                license TEXT,
                compatibility TEXT,
                allowed_tools TEXT,
                body TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                source_mode TEXT NOT NULL DEFAULT 'server',
                manifest_hash TEXT NOT NULL DEFAULT '',
                bundle_hash TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace ON skills(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace_name ON skills(workspace_id, name)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace_user ON skills(workspace_id, user_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")

        # Skill files
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS skill_files (
                id TEXT PRIMARY KEY,
                skill_id TEXT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                kind TEXT NOT NULL,
                content BLOB NOT NULL,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mime_type TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(skill_id, path)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skill_files_skill ON skill_files(skill_id)")

        # Knowledgebase articles
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS knowledgebase_articles (
                workspace_id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                article_type TEXT,
                title TEXT,
                content_md TEXT,
                metadata TEXT DEFAULT '{}',
                generated_at TEXT,
                PRIMARY KEY (workspace_id, article_id)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_kb_articles_workspace ON knowledgebase_articles(workspace_id)")

        # Graph analyses
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS graph_analyses (
                workspace_id TEXT PRIMARY KEY,
                analysis_json TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
        """)

        # MCP servers
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS mcp_servers (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL DEFAULT '_default',
                workspace_id TEXT NOT NULL,
                user_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                transport TEXT NOT NULL,
                command TEXT,
                args TEXT NOT NULL DEFAULT '[]',
                env TEXT NOT NULL DEFAULT '{}',
                url TEXT,
                headers TEXT NOT NULL DEFAULT '{}',
                metadata TEXT NOT NULL DEFAULT '{}',
                source_mode TEXT NOT NULL DEFAULT 'server',
                manifest_hash TEXT NOT NULL DEFAULT '',
                enabled INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_servers_ws_name ON mcp_servers(workspace_id, name) WHERE user_id IS NULL"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_mcp_servers_ws_user_name ON mcp_servers(workspace_id, user_id, name) WHERE user_id IS NOT NULL"
        )
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_mcp_servers_name ON mcp_servers(name)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_mcp_servers_tenant_ws ON mcp_servers(tenant_id, workspace_id)")

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

        # Create _default workspace (main default for auto-discovery)
        await self._connection.execute(
            """
                                       INSERT
                                           OR IGNORE
                                       INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
                                       VALUES (?, ?, 'Default Workspace', '{}', ?, ?)
                                       """,
            (DEFAULT_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )

        # Create _global workspace (cross-workspace shared storage)
        await self._connection.execute(
            """
                                       INSERT
                                           OR IGNORE
                                       INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
                                       VALUES (?, ?, 'Global Workspace', '{}', ?, ?)
                                       """,
            (GLOBAL_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )

        # Create _global_user workspace (per-user cross-workspace preferences:
        # stored with user_id filter, recalled via include_global_user=True)
        await self._connection.execute(
            """
                                       INSERT
                                           OR IGNORE
                                       INTO workspaces (id, tenant_id, name, settings, created_at, updated_at)
                                       VALUES (?, ?, 'Global User Workspace', '{}', ?, ?)
                                       """,
            (GLOBAL_USER_WORKSPACE_ID, DEFAULT_TENANT_ID, now, now),
        )

        # Get all workspaces
        cursor = await self._connection.execute("SELECT id FROM workspaces")
        workspaces = await cursor.fetchall()

        # Create _default context for each workspace if not exists
        for workspace in workspaces:
            workspace_id = workspace["id"]
            await self._connection.execute(
                """
                                           INSERT
                                               OR IGNORE
                                           INTO contexts (id, workspace_id, name, description, settings, created_at, updated_at)
                                           VALUES ('_default', ?, '_default', 'Default context', '{}', ?, ?)
                                           """,
                (workspace_id, now, now),
            )

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

        # Insert into FTS index
        await self._connection.execute(
            "INSERT INTO memories_fts (id, workspace_id, content) VALUES (?, ?, ?)",
            (memory_id, workspace_id, input.content),
        )

        await self._connection.commit()

        return await self.get_memory(workspace_id, memory_id, track_access=False)

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Memory | None:
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
            memory.last_accessed_at = datetime.now(UTC)

        return memory

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Memory | None:
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
            memory.last_accessed_at = datetime.now(UTC)

        return memory

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
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
            WHERE {" AND ".join(where_parts)}
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
        cursor = await self._connection.execute(query, (workspace_id, max_importance, max_access_count, older_than_days, limit))
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs."""
        cursor = await self._connection.execute("SELECT id FROM workspaces")
        rows = await cursor.fetchall()
        return [row["id"] for row in rows]

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
        """Vector similarity search using sqlite-vec or fallback."""
        if self._has_vec_extension:
            return await self._search_with_vec(
                workspace_id,
                query_embedding,
                limit,
                offset,
                min_relevance,
                types,
                subtypes,
                tags,
                include_archived=include_archived,
                observer_id=observer_id,
                subject_id=subject_id,
                created_after=created_after,
                created_before=created_before,
                user_id=user_id,
            )
        else:
            return await self._search_with_fallback(
                workspace_id,
                query_embedding,
                limit,
                offset,
                min_relevance,
                types,
                subtypes,
                tags,
                include_archived=include_archived,
                observer_id=observer_id,
                subject_id=subject_id,
                created_after=created_after,
                created_before=created_before,
                user_id=user_id,
            )

    async def _search_with_vec(
        self,
        workspace_id: str,
        query_embedding: list[float],
        limit: int,
        offset: int,
        min_relevance: float,
        types: list[str] | None,
        subtypes: list[str] | None,
        tags: list[str] | None,
        include_archived: bool = False,
        observer_id: str | None = None,
        subject_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        user_id: str | None = None,
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
        types: list[str] | None,
        subtypes: list[str] | None,
        tags: list[str] | None,
        include_archived: bool = False,
        observer_id: str | None = None,
        subject_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        user_id: str | None = None,
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

        if created_after is not None:
            where_parts.append("created_at >= ?")
            params.append(str(created_after))

        if created_before is not None:
            where_parts.append("created_at <= ?")
            params.append(str(created_before))

        if subject_id is not None:
            where_parts.append("subject_id = ?")
            params.append(subject_id)

        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)

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
        return results[offset : offset + limit]

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Escape FTS5 special syntax to prevent query injection."""
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    async def full_text_search(
        self,
        workspace_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search using SQLite FTS5."""
        sql = """
            SELECT m.*
            FROM memories m
                     INNER JOIN memories_fts fts ON m.id = fts.id
            WHERE fts.workspace_id = ?
              AND fts.content MATCH ?
              AND m.deleted_at IS NULL
        """
        params: list = [workspace_id, self._sanitize_fts5_query(query)]

        if context_id is not None:
            sql += "  AND m.context_id = ?\n"
            params.append(context_id)

        sql += "LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()

        return [self._row_to_memory(row) for row in rows]

    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Memory | None:
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
                # Add overview field (and exclude abstract field)
                results.append(
                    {
                        "id": row["id"],
                        # "abstract": row["abstract"] if row["abstract"] else None,
                        "overview": row["overview"] if row["overview"] else None,
                        "type": row["type"],
                        "subtype": row["subtype"] if row["subtype"] else None,
                        "importance": row["importance"],
                        "tags": json.loads(row["tags"]) if row["tags"] else [],
                        "created_at": row["created_at"],
                    }
                )
            else:  # "full" -- full detail will return the content and doesn't need to return the abstract and overview fields
                # Return everything as dict
                memory = self._row_to_memory(row)
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        # "abstract": memory.abstract,
                        # "overview": memory.overview,
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
        relationships: list[str] | None = None,
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

    async def get_associations_batch(
        self,
        workspace_id: str,
        memory_ids: list[str],
        direction: str = "outgoing",
        relationships: list[str] | None = None,
    ) -> list[Association]:
        """Get associations for multiple memories in a single query."""
        if not memory_ids:
            return []

        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        # Build direction filter with IN clause for batch
        placeholders = ",".join("?" * len(memory_ids))
        if direction == "outgoing":
            where_parts.append(f"source_id IN ({placeholders})")
            params.extend(memory_ids)
        elif direction == "incoming":
            where_parts.append(f"target_id IN ({placeholders})")
            params.extend(memory_ids)
        else:  # both
            where_parts.append(f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))")
            params.extend(memory_ids)
            params.extend(memory_ids)

        if relationships:
            rel_placeholders = ",".join("?" * len(relationships))
            where_parts.append(f"relationship IN ({rel_placeholders})")
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

    async def update_association(
        self,
        workspace_id: str,
        association_id: str,
        metadata: dict | None = None,
        strength: float | None = None,
    ) -> bool:
        """Update an existing association's metadata and/or strength."""
        set_parts = []
        values = []

        if metadata is not None:
            set_parts.append("metadata = ?")
            values.append(json.dumps(metadata))

        if strength is not None:
            set_parts.append("strength = ?")
            values.append(strength)

        if not set_parts:
            return False

        values.extend([association_id, workspace_id])
        query = f"UPDATE memory_associations SET {', '.join(set_parts)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
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

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
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

    async def update_workspace(self, workspace_id: str, **updates) -> Workspace | None:
        """Update workspace fields."""
        if not updates:
            return await self.get_workspace(workspace_id)

        set_parts = []
        values = []
        for key, value in updates.items():
            if key == "settings":
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(value))
            else:
                set_parts.append(f"{key} = ?")
                values.append(value)

        set_parts.append("updated_at = datetime('now')")
        values.extend([workspace_id])

        query = f"UPDATE workspaces SET {', '.join(set_parts)} WHERE id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None

        return await self.get_workspace(workspace_id)

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

    async def get_session_by_id(self, session_id: str) -> Session | None:
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
        self, workspace_id: str, session_id: str, key: str, value: Any, ttl_seconds: int | None = None
    ) -> WorkingMemory:
        """Set working memory key-value within session."""
        now_iso = utc_now_iso()
        now = datetime.now(UTC)

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

    async def update_session(self, workspace_id: str, session_id: str, **updates) -> Session | None:
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

        query = f"""
            UPDATE sessions
            SET {", ".join(set_clauses)}
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
        merged_content: str | None = None,
    ) -> ContradictionRecord | None:
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
            detection_method=row["detection_method"] if row["detection_method"] else "",
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

        return struct.pack(f"{len(embedding)}f", *embedding)

    def _deserialize_embedding(self, blob: bytes) -> list[float]:
        """Deserialize embedding from binary format."""
        import struct

        num_floats = len(blob) // 4
        return list(struct.unpack(f"{num_floats}f", blob))

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

    async def list_expired_threads(self, limit: int = 100) -> list["ChatThread"]:
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
            msg_id = msg_input.id or generate_id("msg")
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
                    msg_id,
                    thread_id,
                    workspace_id,
                    msg_index,
                    msg_input.role,
                    content,
                    json.dumps(msg_input.metadata or {}),
                    now,
                ),
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

    def _row_to_chat_thread(self, row: aiosqlite.Row) -> "ChatThread":
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

    def _row_to_chat_message(self, row: aiosqlite.Row) -> "ChatMessage":
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

    # ============================================
    # Document Operations
    # ============================================

    async def create_document(self, workspace_id: str, doc: "Document") -> "Document":
        """Store a new document record."""

        await self._connection.execute(
            """
            INSERT INTO documents (
                id, workspace_id, filename, document_type, content_hash, size_bytes,
                mime_type, status, target_context_id, extraction_options,
                page_count, chunk_count, memory_ids, deduplicated_count,
                error_message, metadata, extracted_metadata,
                created_at, processing_started_at, processing_completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc.id,
                workspace_id,
                doc.filename,
                doc.document_type.value if hasattr(doc.document_type, "value") else doc.document_type,
                doc.content_hash,
                doc.size_bytes,
                doc.mime_type,
                doc.status.value if hasattr(doc.status, "value") else doc.status,
                doc.target_context_id,
                json.dumps(doc.extraction_options.model_dump() if doc.extraction_options else {}),
                doc.page_count,
                doc.chunk_count,
                json.dumps(doc.memory_ids),
                doc.deduplicated_count,
                doc.error_message,
                json.dumps(doc.metadata),
                json.dumps(doc.extracted_metadata),
                doc.created_at.isoformat() if doc.created_at else utc_now_iso(),
                doc.processing_started_at.isoformat() if doc.processing_started_at else None,
                doc.processing_completed_at.isoformat() if doc.processing_completed_at else None,
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created document: %s in workspace: %s", doc.id, workspace_id)
        return await self.get_document(workspace_id, doc.id)

    async def get_document(self, workspace_id: str, doc_id: str) -> "Document | None":
        """Get document by ID within a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM documents WHERE id = ? AND workspace_id = ?",
            (doc_id, workspace_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    async def list_documents(
        self,
        workspace_id: str,
        status: str | None = None,
        document_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list["Document"], int]:
        """List documents in a workspace. Returns (documents, total_count)."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if status is not None:
            where_parts.append("status = ?")
            params.append(status)

        if document_type is not None:
            where_parts.append("document_type = ?")
            params.append(document_type)

        where_clause = " AND ".join(where_parts)

        # Get total count
        count_cursor = await self._connection.execute(f"SELECT COUNT(*) FROM documents WHERE {where_clause}", params)
        count_row = await count_cursor.fetchone()
        total = count_row[0] if count_row else 0

        # Get paginated results
        query = f"SELECT * FROM documents WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_document(row) for row in rows], total

    async def update_document(self, workspace_id: str, doc_id: str, **updates) -> "Document | None":
        """Update document fields."""
        if not updates:
            return await self.get_document(workspace_id, doc_id)

        set_parts = []
        values = []
        json_fields = {"extraction_options", "memory_ids", "metadata", "extracted_metadata"}
        datetime_fields = {"created_at", "processing_started_at", "processing_completed_at"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                if hasattr(value, "model_dump"):
                    values.append(json.dumps(value.model_dump()))
                else:
                    values.append(json.dumps(value))
            elif key in datetime_fields and isinstance(value, datetime):
                values.append(value.isoformat())
            elif key in ("status", "document_type") and hasattr(value, "value"):
                values.append(value.value)
            else:
                values.append(value)

        values.extend([doc_id, workspace_id])
        query = f"UPDATE documents SET {', '.join(set_parts)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_document(workspace_id, doc_id)

    async def delete_document(self, workspace_id: str, doc_id: str, delete_memories: bool = False) -> bool:
        """Delete a document and optionally cascade to memories."""
        if delete_memories:
            cursor = await self._connection.execute(
                "SELECT id FROM memories WHERE workspace_id = ? AND source_document_id = ? AND deleted_at IS NULL",
                (workspace_id, doc_id),
            )
            rows = await cursor.fetchall()
            for row in rows:
                await self._connection.execute(
                    "UPDATE memories SET deleted_at = datetime('now'), status = 'deleted' WHERE id = ?",
                    (row["id"],),
                )
                await self._connection.execute("DELETE FROM memories_fts WHERE id = ?", (row["id"],))
            self.logger.debug("Soft-deleted %d memories for document: %s", len(rows), doc_id)

        # Delete pages
        await self._connection.execute("DELETE FROM document_pages WHERE document_id = ?", (doc_id,))

        cursor = await self._connection.execute(
            "DELETE FROM documents WHERE id = ? AND workspace_id = ?",
            (doc_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.debug("Deleted document: %s", doc_id)
        return deleted

    async def find_document_by_hash(self, workspace_id: str, content_hash: str) -> "Document | None":
        """Find document by content hash for deduplication."""
        cursor = await self._connection.execute(
            "SELECT * FROM documents WHERE workspace_id = ? AND content_hash = ? LIMIT 1",
            (workspace_id, content_hash),
        )
        row = await cursor.fetchone()
        return self._row_to_document(row) if row else None

    async def get_document_memories(self, workspace_id: str, doc_id: str) -> list[Memory]:
        """Get all memories created from a document."""
        cursor = await self._connection.execute(
            """
            SELECT * FROM memories
            WHERE workspace_id = ? AND source_document_id = ? AND deleted_at IS NULL
            ORDER BY created_at ASC
            """,
            (workspace_id, doc_id),
        )
        rows = await cursor.fetchall()
        return [self._row_to_memory(row) for row in rows]

    def _row_to_document(self, row: aiosqlite.Row) -> "Document":
        """Convert database row to Document domain model."""
        from ...models.document import Document, DocumentExtractionOptions, DocumentStatus, DocumentType

        raw_opts = row["extraction_options"]
        try:
            opts_dict = json.loads(raw_opts) if raw_opts else {}
            extraction_options = DocumentExtractionOptions(**opts_dict)
        except Exception:
            extraction_options = DocumentExtractionOptions()

        return Document(
            id=row["id"],
            workspace_id=row["workspace_id"],
            filename=row["filename"],
            document_type=DocumentType(row["document_type"]),
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            mime_type=row["mime_type"],
            status=DocumentStatus(row["status"]) if row["status"] else DocumentStatus.PENDING,
            target_context_id=row["target_context_id"] or "_default",
            extraction_options=extraction_options,
            page_count=row["page_count"] or 0,
            chunk_count=row["chunk_count"] or 0,
            memory_ids=json.loads(row["memory_ids"]) if row["memory_ids"] else [],
            deduplicated_count=row["deduplicated_count"] or 0,
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            extracted_metadata=json.loads(row["extracted_metadata"]) if row["extracted_metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
            processing_started_at=parse_datetime_utc(row["processing_started_at"]) if row["processing_started_at"] else None,
            processing_completed_at=parse_datetime_utc(row["processing_completed_at"]) if row["processing_completed_at"] else None,
        )

    # ============================================
    # Document Page Operations
    # ============================================

    async def create_page(self, workspace_id: str, document_id: str, page: "DocumentPage") -> "DocumentPage":
        """Store a document page."""
        from ...utils import generate_id

        page_id = page.id or generate_id("page")
        now = utc_now_iso()

        await self._connection.execute(
            """
            INSERT INTO document_pages (id, document_id, workspace_id, page_no, transcript,
                multivector, transcript_model, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                document_id,
                workspace_id,
                page.page_no,
                page.transcript,
                json.dumps(page.multivector) if page.multivector is not None else None,
                page.transcript_model,
                json.dumps(page.metadata),
                page.created_at.isoformat() if page.created_at else now,
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created page %d for document: %s", page.page_no, document_id)
        return await self.get_page(page_id)

    async def get_pages(self, document_id: str, workspace_id: str | None = None) -> list["DocumentPage"]:
        """Get all pages for a document, ordered by page_no."""
        if workspace_id is not None:
            cursor = await self._connection.execute(
                "SELECT * FROM document_pages WHERE document_id = ? AND workspace_id = ? ORDER BY page_no ASC",
                (document_id, workspace_id),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM document_pages WHERE document_id = ? ORDER BY page_no ASC",
                (document_id,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_document_page(row) for row in rows]

    async def get_page(self, page_id: str) -> "DocumentPage | None":
        """Get a single page by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM document_pages WHERE id = ?",
            (page_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_document_page(row) if row else None

    async def update_page(self, page_id: str, **updates) -> "DocumentPage | None":
        """Update page fields (transcript, embedding, etc.)."""
        if not updates:
            return await self.get_page(page_id)

        set_parts = []
        values = []
        json_fields = {"multivector", "metadata"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value) if value is not None else None)
            else:
                values.append(value)

        values.append(page_id)
        query = f"UPDATE document_pages SET {', '.join(set_parts)} WHERE id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_page(page_id)

    def _row_to_document_page(self, row: aiosqlite.Row) -> "DocumentPage":
        """Convert database row to DocumentPage domain model."""
        from ...models.document import DocumentPage

        return DocumentPage(
            id=row["id"],
            document_id=row["document_id"],
            workspace_id=row["workspace_id"],
            page_no=row["page_no"],
            transcript=row["transcript"],
            multivector=json.loads(row["multivector"]) if row["multivector"] else None,
            transcript_model=row["transcript_model"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]) if row["created_at"] else None,
        )

    async def search_pages_by_maxsim(
        self,
        workspace_id: str,
        query_multivector: list[list[float]],
        limit: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple["DocumentPage", float]]:
        """Score pages in Python after a filtered fetch.

        SQLite has no native MaxSim, so candidate pages are loaded, then ``maxsim_score`` from
        ``services.embedding._maxsim`` is applied per page. Suitable for workspaces with tens to a few thousand multi-vector pages; larger
        deployments should use enterprise.
        """
        from ..embedding._maxsim import MultiVectorEmbedding, maxsim_score

        sql = "SELECT * FROM document_pages WHERE workspace_id = ? AND multivector IS NOT NULL"
        params: list[object] = [workspace_id]

        if doc_ids:
            placeholders = ",".join("?" for _ in doc_ids)
            sql += f" AND document_id IN ({placeholders})"
            params.extend(doc_ids)

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()

        if not rows:
            return []

        query_embedding = MultiVectorEmbedding(vectors=query_multivector)

        scored: list[tuple[DocumentPage, float]] = []
        for row in rows:
            page = self._row_to_document_page(row)
            if not page.multivector:
                continue
            try:
                score = maxsim_score(
                    query_embedding,
                    MultiVectorEmbedding(vectors=page.multivector),
                )
            except Exception:
                self.logger.debug(
                    "MaxSim scoring failed for page %s; skipping",
                    page.id,
                    exc_info=True,
                )
                continue
            scored.append((page, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    # ============================================
    # Ingestion Job Operations
    # ============================================

    async def create_job(self, job: "IngestionJob") -> "IngestionJob":
        """Store an ingestion job."""
        await self._connection.execute(
            """
            INSERT INTO ingestion_jobs (id, workspace_id, document_ids, status,
                progress_percent, documents_processed, total_memories_created,
                errors, metadata, created_at, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.workspace_id,
                json.dumps(job.document_ids),
                job.status.value if hasattr(job.status, "value") else job.status,
                job.progress_percent,
                job.documents_processed,
                job.total_memories_created,
                json.dumps(job.errors),
                json.dumps(job.metadata),
                job.created_at.isoformat() if job.created_at else utc_now_iso(),
                job.started_at.isoformat() if job.started_at else None,
                job.completed_at.isoformat() if job.completed_at else None,
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created ingestion job: %s in workspace: %s", job.id, job.workspace_id)
        return await self.get_job(job.id)

    async def get_job(self, job_id: str, workspace_id: str | None = None) -> "IngestionJob | None":
        """Get ingestion job by ID."""
        if workspace_id is not None:
            cursor = await self._connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ? AND workspace_id = ?",
                (job_id, workspace_id),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?",
                (job_id,),
            )
        row = await cursor.fetchone()
        return self._row_to_ingestion_job(row) if row else None

    async def list_jobs(
        self,
        workspace_id: str,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list["IngestionJob"]:
        """List ingestion jobs for a workspace."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if status is not None:
            where_parts.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(where_parts)
        query = f"SELECT * FROM ingestion_jobs WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_ingestion_job(row) for row in rows]

    async def update_job(self, job_id: str, **updates) -> "IngestionJob | None":
        """Update ingestion job fields (status, progress, etc.)."""
        if not updates:
            return await self.get_job(job_id)

        set_parts = []
        values = []
        json_fields = {"document_ids", "errors", "metadata"}
        datetime_fields = {"created_at", "started_at", "completed_at"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            elif key in datetime_fields and isinstance(value, datetime):
                values.append(value.isoformat())
            elif key == "status" and hasattr(value, "value"):
                values.append(value.value)
            else:
                values.append(value)

        values.append(job_id)
        query = f"UPDATE ingestion_jobs SET {', '.join(set_parts)} WHERE id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_job(job_id)

    def _row_to_ingestion_job(self, row: aiosqlite.Row) -> "IngestionJob":
        """Convert database row to IngestionJob domain model."""
        from ...models.document import IngestionJob, JobStatus

        return IngestionJob(
            id=row["id"],
            workspace_id=row["workspace_id"],
            document_ids=json.loads(row["document_ids"]) if row["document_ids"] else [],
            status=JobStatus(row["status"]) if row["status"] else JobStatus.QUEUED,
            progress_percent=row["progress_percent"] or 0,
            documents_processed=row["documents_processed"] or 0,
            total_memories_created=row["total_memories_created"] or 0,
            errors=json.loads(row["errors"]) if row["errors"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
            started_at=parse_datetime_utc(row["started_at"]) if row["started_at"] else None,
            completed_at=parse_datetime_utc(row["completed_at"]) if row["completed_at"] else None,
        )

    # ============================================
    # Data Provider Operations
    # ============================================

    async def create_data_provider(self, workspace_id: str, provider: "DataProvider") -> "DataProvider":
        """Store a data provider."""
        await self._connection.execute(
            """
            INSERT INTO data_providers (id, workspace_id, name, provider_type, description,
                enabled, connection_args, schedule, last_sync_at, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                provider.id,
                workspace_id,
                provider.name,
                provider.provider_type.value if hasattr(provider.provider_type, "value") else provider.provider_type,
                provider.description,
                1 if provider.enabled else 0,
                json.dumps(provider.connection_args),
                provider.schedule,
                provider.last_sync_at.isoformat() if provider.last_sync_at else None,
                json.dumps(provider.metadata),
                provider.created_at.isoformat() if provider.created_at else utc_now_iso(),
                provider.updated_at.isoformat() if provider.updated_at else utc_now_iso(),
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created data provider: %s in workspace: %s", provider.id, workspace_id)
        return await self.get_data_provider(workspace_id, provider.id)

    async def get_data_provider(self, workspace_id: str, provider_id: str) -> "DataProvider | None":
        """Get data provider by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM data_providers WHERE id = ? AND workspace_id = ?",
            (provider_id, workspace_id),
        )
        row = await cursor.fetchone()
        return self._row_to_data_provider(row) if row else None

    async def list_data_providers(
        self,
        workspace_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list["DataProvider"], int]:
        """List data providers for a workspace. Returns (providers, total_count)."""
        count_cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM data_providers WHERE workspace_id = ?",
            (workspace_id,),
        )
        count_row = await count_cursor.fetchone()
        total = count_row["count"] if count_row else 0

        cursor = await self._connection.execute(
            "SELECT * FROM data_providers WHERE workspace_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (workspace_id, limit, offset),
        )
        rows = await cursor.fetchall()
        return [self._row_to_data_provider(row) for row in rows], total

    async def update_data_provider(self, workspace_id: str, provider_id: str, **updates) -> "DataProvider | None":
        """Update data provider fields."""
        if not updates:
            return await self.get_data_provider(workspace_id, provider_id)

        set_parts = []
        values = []
        json_fields = {"connection_args", "metadata"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            elif key == "enabled":
                values.append(1 if value else 0)
            elif key in ("last_sync_at", "created_at", "updated_at") and isinstance(value, datetime):
                values.append(value.isoformat())
            elif key == "provider_type" and hasattr(value, "value"):
                values.append(value.value)
            else:
                values.append(value)

        set_parts.append("updated_at = ?")
        values.append(utc_now_iso())
        values.extend([provider_id, workspace_id])

        query = f"UPDATE data_providers SET {', '.join(set_parts)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_data_provider(workspace_id, provider_id)

    async def delete_data_provider(self, workspace_id: str, provider_id: str) -> bool:
        """Delete a data provider."""
        cursor = await self._connection.execute(
            "DELETE FROM data_providers WHERE id = ? AND workspace_id = ?",
            (provider_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.debug("Deleted data provider: %s", provider_id)
        return deleted

    def _row_to_data_provider(self, row: aiosqlite.Row) -> "DataProvider":
        """Convert database row to DataProvider domain model."""
        from ...models.data_provider import DataProvider

        return DataProvider(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            provider_type=row["provider_type"],
            description=row["description"],
            enabled=bool(row["enabled"]) if row["enabled"] is not None else True,
            connection_args=json.loads(row["connection_args"]) if row["connection_args"] else {},
            schedule=row["schedule"],
            last_sync_at=parse_datetime_utc(row["last_sync_at"]) if row["last_sync_at"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    # ============================================
    # Knowledgebase Article Operations
    # ============================================

    async def store_kb_article(
        self,
        workspace_id: str,
        article_id: str,
        article_type: str,
        title: str,
        content_md: str,
        metadata: dict | None = None,
    ) -> dict:
        """Store a knowledgebase article (upsert)."""
        now = utc_now_iso()
        await self._connection.execute(
            """
            INSERT INTO knowledgebase_articles
                (workspace_id, article_id, article_type, title, content_md, metadata, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, article_id) DO UPDATE SET
                article_type = excluded.article_type,
                title = excluded.title,
                content_md = excluded.content_md,
                metadata = excluded.metadata,
                generated_at = excluded.generated_at
            """,
            (
                workspace_id,
                article_id,
                article_type,
                title,
                content_md,
                json.dumps(metadata or {}),
                now,
            ),
        )
        await self._connection.commit()
        self.logger.debug("Stored KB article: %s in workspace: %s", article_id, workspace_id)
        return await self.get_kb_article(workspace_id, article_id)

    async def get_kb_article(self, workspace_id: str, article_id: str) -> dict | None:
        """Get a knowledgebase article by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM knowledgebase_articles WHERE workspace_id = ? AND article_id = ?",
            (workspace_id, article_id),
        )
        row = await cursor.fetchone()
        return self._row_to_kb_article(row) if row else None

    async def list_kb_articles(
        self,
        workspace_id: str,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List knowledgebase articles for a workspace."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if article_type is not None:
            where_parts.append("article_type = ?")
            params.append(article_type)

        where_clause = " AND ".join(where_parts)
        query = f"SELECT * FROM knowledgebase_articles WHERE {where_clause} ORDER BY generated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_kb_article(row) for row in rows]

    async def delete_kb_articles(self, workspace_id: str) -> int:
        """Delete all knowledgebase articles for a workspace (for regeneration)."""
        cursor = await self._connection.execute(
            "DELETE FROM knowledgebase_articles WHERE workspace_id = ?",
            (workspace_id,),
        )
        await self._connection.commit()
        count = cursor.rowcount
        self.logger.debug("Deleted %d KB articles for workspace: %s", count, workspace_id)
        return count

    def _row_to_kb_article(self, row: aiosqlite.Row) -> dict:
        """Convert database row to knowledgebase article dict."""
        return {
            "workspace_id": row["workspace_id"],
            "article_id": row["article_id"],
            "article_type": row["article_type"],
            "title": row["title"],
            "content_md": row["content_md"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            "generated_at": row["generated_at"],
        }

    # ============================================
    # Graph Analysis Operations
    # ============================================

    async def store_graph_analysis(self, workspace_id: str, analysis_json: dict) -> dict:
        """Cache a graph analysis result (upsert)."""
        now = utc_now_iso()
        await self._connection.execute(
            """
            INSERT INTO graph_analyses (workspace_id, analysis_json, generated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
                analysis_json = excluded.analysis_json,
                generated_at = excluded.generated_at
            """,
            (workspace_id, json.dumps(analysis_json), now),
        )
        await self._connection.commit()
        self.logger.debug("Stored graph analysis for workspace: %s", workspace_id)
        return {"workspace_id": workspace_id, "analysis_json": analysis_json, "generated_at": now}

    async def get_graph_analysis(self, workspace_id: str) -> dict | None:
        """Get cached graph analysis for a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM graph_analyses WHERE workspace_id = ?",
            (workspace_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        try:
            analysis_json = json.loads(row["analysis_json"]) if row["analysis_json"] else {}
        except (json.JSONDecodeError, TypeError):
            analysis_json = {}
        return {
            "workspace_id": row["workspace_id"],
            "analysis_json": analysis_json,
            "generated_at": row["generated_at"],
        }

    # ============================================
    # Skill Operations
    # ============================================

    async def create_skill(self, skill: "Skill") -> "Skill":
        """Store a new skill."""
        await self._connection.execute(
            """
            INSERT INTO skills (id, tenant_id, workspace_id, user_id, name, description, version,
                license, compatibility, allowed_tools, body, metadata, source_mode,
                manifest_hash, bundle_hash, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                skill.id,
                skill.tenant_id,
                skill.workspace_id,
                skill.user_id,
                skill.name,
                skill.description,
                skill.version,
                skill.license,
                skill.compatibility,
                skill.allowed_tools,
                skill.body,
                json.dumps(skill.metadata),
                skill.source_mode,
                skill.manifest_hash,
                skill.bundle_hash,
                1 if skill.enabled else 0,
                skill.created_at.isoformat() if skill.created_at else utc_now_iso(),
                skill.updated_at.isoformat() if skill.updated_at else utc_now_iso(),
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created skill: %s in workspace: %s", skill.id, skill.workspace_id)
        return await self.get_skill(skill.workspace_id, skill.id)

    async def get_skill(self, workspace_id: str, skill_id: str) -> "Skill | None":
        """Get skill by ID within a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM skills WHERE id = ? AND workspace_id = ?",
            (skill_id, workspace_id),
        )
        row = await cursor.fetchone()
        return self._row_to_skill(row) if row else None

    async def get_skill_by_name(self, workspace_id: str, name: str, user_id: str | None = None) -> "Skill | None":
        """Get skill by name within a workspace, optionally scoped to a user."""
        if user_id is not None:
            cursor = await self._connection.execute(
                "SELECT * FROM skills WHERE workspace_id = ? AND name = ? AND user_id = ?",
                (workspace_id, name, user_id),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM skills WHERE workspace_id = ? AND name = ? AND user_id IS NULL",
                (workspace_id, name),
            )
        row = await cursor.fetchone()
        return self._row_to_skill(row) if row else None

    async def list_skills(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["Skill"]:
        """List skills for a workspace with optional filters."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if name is not None:
            where_parts.append("name = ?")
            params.append(name)
        if enabled is not None:
            where_parts.append("enabled = ?")
            params.append(1 if enabled else 0)

        where_clause = " AND ".join(where_parts)
        query = f"SELECT * FROM skills WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def find_skills_by_name(self, name: str, scope_filters: list[dict]) -> list["Skill"]:
        """Find skills by name across multiple scope filters for resolution.

        Each filter dict has workspace_id and optional user_id keys.
        Returns all matches; the resolution service handles precedence ordering.
        """
        if not scope_filters:
            return []

        conditions = []
        params: list = [name]
        for sf in scope_filters:
            ws = sf.get("workspace_id")
            uid = sf.get("user_id")
            if uid is not None:
                conditions.append("(workspace_id = ? AND user_id = ?)")
                params.extend([ws, uid])
            else:
                conditions.append("(workspace_id = ? AND user_id IS NULL)")
                params.append(ws)

        where_clause = f"name = ? AND ({' OR '.join(conditions)})"
        query = f"SELECT * FROM skills WHERE {where_clause} ORDER BY updated_at DESC"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def update_skill(self, workspace_id: str, skill_id: str, updates: dict) -> "Skill | None":
        """Update skill fields."""
        if not updates:
            return await self.get_skill(workspace_id, skill_id)

        set_parts = []
        values = []
        json_fields = {"metadata"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            elif key == "enabled":
                values.append(1 if value else 0)
            elif key in ("created_at", "updated_at") and isinstance(value, datetime):
                values.append(value.isoformat())
            else:
                values.append(value)

        set_parts.append("updated_at = ?")
        values.append(utc_now_iso())
        values.extend([skill_id, workspace_id])

        query = f"UPDATE skills SET {', '.join(set_parts)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_skill(workspace_id, skill_id)

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Delete a skill and cascade to skill_files via FK constraint."""
        cursor = await self._connection.execute(
            "DELETE FROM skills WHERE id = ? AND workspace_id = ?",
            (skill_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.debug("Deleted skill: %s", skill_id)
        return deleted

    async def upsert_skill_file(self, skill_file: "SkillFile") -> "SkillFile":
        """Insert or update a skill file by (skill_id, path)."""
        now = utc_now_iso()
        await self._connection.execute(
            """
            INSERT INTO skill_files (id, skill_id, path, kind, content, content_hash,
                size_bytes, mime_type, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(skill_id, path) DO UPDATE SET
                id = excluded.id,
                kind = excluded.kind,
                content = excluded.content,
                content_hash = excluded.content_hash,
                size_bytes = excluded.size_bytes,
                mime_type = excluded.mime_type,
                updated_at = excluded.updated_at
            """,
            (
                skill_file.id,
                skill_file.skill_id,
                skill_file.path,
                skill_file.kind,
                skill_file.content,
                skill_file.content_hash,
                skill_file.size_bytes,
                skill_file.mime_type,
                skill_file.created_at.isoformat() if skill_file.created_at else now,
                skill_file.updated_at.isoformat() if skill_file.updated_at else now,
            ),
        )
        await self._connection.commit()
        return await self.get_skill_file(skill_file.skill_id, skill_file.path)

    async def get_skill_file(self, skill_id: str, path: str) -> "SkillFile | None":
        """Get a skill file by skill_id and path."""
        cursor = await self._connection.execute(
            "SELECT * FROM skill_files WHERE skill_id = ? AND path = ?",
            (skill_id, path),
        )
        row = await cursor.fetchone()
        return self._row_to_skill_file(row) if row else None

    async def list_skill_files(self, skill_id: str) -> list["SkillFile"]:
        """List all files in a skill bundle ordered by path."""
        cursor = await self._connection.execute(
            "SELECT * FROM skill_files WHERE skill_id = ? ORDER BY path",
            (skill_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_skill_file(row) for row in rows]

    async def delete_skill_file(self, skill_id: str, path: str) -> bool:
        """Delete a single skill file by path."""
        cursor = await self._connection.execute(
            "DELETE FROM skill_files WHERE skill_id = ? AND path = ?",
            (skill_id, path),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    def _row_to_skill(self, row: aiosqlite.Row) -> "Skill":
        """Convert database row to Skill domain model."""
        from ...models.skill import Skill

        return Skill(
            id=row["id"],
            tenant_id=row["tenant_id"] or "",
            workspace_id=row["workspace_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            version=row["version"],
            license=row["license"],
            compatibility=row["compatibility"],
            allowed_tools=row["allowed_tools"],
            body=row["body"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source_mode=row["source_mode"],
            manifest_hash=row["manifest_hash"] or "",
            bundle_hash=row["bundle_hash"] or "",
            enabled=bool(row["enabled"]) if row["enabled"] is not None else True,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    def _row_to_skill_file(self, row: aiosqlite.Row) -> "SkillFile":
        """Convert database row to SkillFile domain model."""
        from ...models.skill import SkillFile

        return SkillFile(
            id=row["id"],
            skill_id=row["skill_id"],
            path=row["path"],
            kind=row["kind"],
            content=bytes(row["content"]) if row["content"] is not None else b"",
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            mime_type=row["mime_type"],
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    # ============================================
    # MCP Server Operations
    # ============================================

    async def create_mcp_server(self, server: "McpServer") -> "McpServer":
        """Store a new MCP server record."""
        await self._connection.execute(
            """
            INSERT INTO mcp_servers (id, tenant_id, workspace_id, user_id, name, description,
                transport, command, args, env, url, headers, metadata, source_mode,
                manifest_hash, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                server.id,
                server.tenant_id,
                server.workspace_id,
                server.user_id,
                server.name,
                server.description,
                server.transport,
                server.command,
                json.dumps(server.args),
                json.dumps(server.env),
                server.url,
                json.dumps(server.headers),
                json.dumps(server.metadata),
                server.source_mode,
                server.manifest_hash,
                1 if server.enabled else 0,
                server.created_at.isoformat() if server.created_at else utc_now_iso(),
                server.updated_at.isoformat() if server.updated_at else utc_now_iso(),
            ),
        )
        await self._connection.commit()
        self.logger.debug("Created MCP server: %s in workspace: %s", server.id, server.workspace_id)
        return await self.get_mcp_server(server.workspace_id, server.id)

    async def get_mcp_server(self, workspace_id: str, server_id: str) -> "McpServer | None":
        """Get MCP server by ID within a workspace."""
        cursor = await self._connection.execute(
            "SELECT * FROM mcp_servers WHERE id = ? AND workspace_id = ?",
            (server_id, workspace_id),
        )
        row = await cursor.fetchone()
        return self._row_to_mcp_server(row) if row else None

    async def get_mcp_server_by_name(self, workspace_id: str, name: str, user_id: str | None = None) -> "McpServer | None":
        """Get MCP server by name within a workspace, optionally scoped to a user."""
        if user_id is not None:
            cursor = await self._connection.execute(
                "SELECT * FROM mcp_servers WHERE workspace_id = ? AND name = ? AND user_id = ?",
                (workspace_id, name, user_id),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM mcp_servers WHERE workspace_id = ? AND name = ? AND user_id IS NULL",
                (workspace_id, name),
            )
        row = await cursor.fetchone()
        return self._row_to_mcp_server(row) if row else None

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["McpServer"]:
        """List MCP servers for a workspace with optional filters."""
        where_parts = ["workspace_id = ?"]
        params: list = [workspace_id]

        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if name is not None:
            where_parts.append("name = ?")
            params.append(name)
        if transport is not None:
            where_parts.append("transport = ?")
            params.append(transport)
        if enabled is not None:
            where_parts.append("enabled = ?")
            params.append(1 if enabled else 0)

        where_clause = " AND ".join(where_parts)
        query = f"SELECT * FROM mcp_servers WHERE {where_clause} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_mcp_server(row) for row in rows]

    async def find_mcp_servers_by_name(self, name: str, scope_filters: list[dict]) -> list["McpServer"]:
        """Find MCP servers by name across multiple scope filters for resolution.

        Each filter dict has workspace_id and optional user_id keys.
        Returns all matches; the resolution service handles precedence ordering.
        """
        if not scope_filters:
            return []

        conditions = []
        params: list = [name]
        for sf in scope_filters:
            ws = sf.get("workspace_id")
            uid = sf.get("user_id")
            if uid is not None:
                conditions.append("(workspace_id = ? AND user_id = ?)")
                params.extend([ws, uid])
            else:
                conditions.append("(workspace_id = ? AND user_id IS NULL)")
                params.append(ws)

        where_clause = f"name = ? AND ({' OR '.join(conditions)})"
        query = f"SELECT * FROM mcp_servers WHERE {where_clause} ORDER BY updated_at DESC"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_mcp_server(row) for row in rows]

    async def update_mcp_server(self, workspace_id: str, server_id: str, updates: dict) -> "McpServer | None":
        """Update MCP server fields."""
        if not updates:
            return await self.get_mcp_server(workspace_id, server_id)

        set_parts = []
        values = []
        json_fields = {"args", "env", "headers", "metadata"}

        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            if key in json_fields:
                values.append(json.dumps(value))
            elif key == "enabled":
                values.append(1 if value else 0)
            elif key in ("created_at", "updated_at") and isinstance(value, datetime):
                values.append(value.isoformat())
            else:
                values.append(value)

        set_parts.append("updated_at = ?")
        values.append(utc_now_iso())
        values.extend([server_id, workspace_id])

        query = f"UPDATE mcp_servers SET {', '.join(set_parts)} WHERE id = ? AND workspace_id = ?"
        cursor = await self._connection.execute(query, values)
        await self._connection.commit()

        if cursor.rowcount == 0:
            return None
        return await self.get_mcp_server(workspace_id, server_id)

    async def delete_mcp_server(self, workspace_id: str, server_id: str) -> bool:
        """Delete an MCP server record."""
        cursor = await self._connection.execute(
            "DELETE FROM mcp_servers WHERE id = ? AND workspace_id = ?",
            (server_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.debug("Deleted MCP server: %s", server_id)
        return deleted

    def _row_to_mcp_server(self, row: aiosqlite.Row) -> "McpServer":
        """Convert database row to McpServer domain model."""
        from ...models.mcp_server import McpServer

        return McpServer(
            id=row["id"],
            tenant_id=row["tenant_id"] or "_default",
            workspace_id=row["workspace_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            transport=row["transport"],
            command=row["command"],
            args=json.loads(row["args"]) if row["args"] else [],
            env=json.loads(row["env"]) if row["env"] else {},
            url=row["url"],
            headers=json.loads(row["headers"]) if row["headers"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source_mode=row["source_mode"],
            manifest_hash=row["manifest_hash"] or "",
            enabled=bool(row["enabled"]) if row["enabled"] is not None else True,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )


class SqliteStorageBackendPlugin(StoragePluginBase):
    PROVIDER_NAME = "sqlite"

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return SQLiteStorageBackend(
            db_path=v.environ(MEMORYLAYER_SQLITE_STORAGE_PATH, default=DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH), v=v
        )
