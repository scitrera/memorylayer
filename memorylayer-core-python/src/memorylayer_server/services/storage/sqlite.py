"""SQLite storage backend with sqlite-vec support."""

import asyncio
import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from logging import Logger
from pathlib import Path
from typing import Any

import aiosqlite

# Register datetime adapters/converters to fix Python 3.12 deprecation warning
# See: https://docs.python.org/3/library/sqlite3.html#default-adapters-and-converters-deprecated
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("datetime", lambda b: datetime.fromisoformat(b.decode()))

from scitrera_app_framework import Variables as Variables

from ...config import (
    DEFAULT_CONTEXT_ID,
    DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
    DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH,
    DEFAULT_TENANT_ID,
    MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
    MEMORYLAYER_SQLITE_STORAGE_PATH,
)
from ...models.association import AssociateInput, Association, GraphPath, GraphQueryResult
from ...models.context_pack import (
    CheckpointCaptureStatus,
    CheckpointWorkStatus,
    ContextEventKind,
    SessionCheckpoint,
    SessionCheckpointInput,
    SessionContextEvent,
)
from ...models.entity_relation import EntityRelation, EntityRelationEvidence, EntityRelationPath
from ...models.memory import (
    Memory,
    MemoryMutation,
    MemoryMutationResult,
    MemoryRevision,
    MemoryStatus,
    MemoryType,
    RememberInput,
)
from ...models.session import Session, WorkingMemory
from ...models.skill import Skill, SkillMutation, SkillMutationResult, SkillRevision
from ...models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from ...models.workspace import Context, Workspace, normalize_tags
from ...utils import cosine_similarity, generate_id, parse_datetime_utc, to_utc_iso, utc_now_iso
from ..contradiction.base import ContradictionRecord
from ..memory.versioning import (
    SEMANTIC_MEMORY_FIELDS,
    memory_etag,
    memory_revision_snapshot,
    memory_semantic_state,
)
from ..skills.versioning import canonical_hash, manifest_etag, manifest_state
from .base import StorageBackend, StoragePluginBase
from .versioned_resources import RelationalVersionedResourceStore
from .workspace_purge import purge_workspace

_UPDATABLE_MEMORY_COLUMNS = frozenset(
    {
        "content",
        "content_hash",
        "type",
        "subtype",
        "importance",
        "tags",
        "metadata",
        "refinement_metadata",
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
        "event_time",
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
        "idle_action",
        "hidden_at",
        "last_decomposed_index",
        "last_decomposed_at",
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
        self._versioned_resource_store: RelationalVersionedResourceStore | None = None
        self._memory_mutation_lock = asyncio.Lock()
        self._skill_mutation_lock = asyncio.Lock()
        self._context_event_retention_days = (
            v.environ(
                MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
                default=DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
                type_fn=int,
            )
            if v is not None
            else DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS
        )

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
        self._versioned_resource_store = RelationalVersionedResourceStore(self._connection)
        await self._versioned_resource_store.create_tables()
        await self._adopt_legacy_memories()
        await self._adopt_legacy_skills()

        # Ensure reserved entities exist
        await self._ensure_reserved_entities()

        self.logger.info("Connected to SQLite database at %s", self.db_path)

    async def disconnect(self) -> None:
        """Close storage connection."""
        if self._connection:
            await self._connection.close()
            self._versioned_resource_store = None
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
                                           tags
                                                      TEXT
                                                           DEFAULT
                                                               '[]',
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

        # Add tags column (migration for existing databases; idempotent)
        try:
            await self._connection.execute("ALTER TABLE workspaces ADD COLUMN tags TEXT DEFAULT '[]'")
        except Exception as e:
            # Column likely already exists (expected during migration)
            self.logger.debug("Column migration note for workspaces.tags: %s", e)

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
                                           logical_key      TEXT,
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
                                           refinement_metadata TEXT NOT NULL DEFAULT '{}',
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
                                           revision         INTEGER NOT NULL DEFAULT 0,
                                           etag             TEXT NOT NULL DEFAULT '',
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
            # Temporal: when the memory's content is about (event time)
            "ALTER TABLE memories ADD COLUMN event_time TEXT",
            "ALTER TABLE memories ADD COLUMN logical_key TEXT",
            "ALTER TABLE memories ADD COLUMN refinement_metadata TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE memories ADD COLUMN revision INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE memories ADD COLUMN etag TEXT NOT NULL DEFAULT ''",
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
        # Timeline index: explicit event_time for temporal queries/ordering
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_event_time ON memories(workspace_id, event_time) WHERE event_time IS NOT NULL AND deleted_at IS NULL"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_workspace_key_global "
            "ON memories(workspace_id, logical_key) "
            "WHERE user_id IS NULL AND logical_key IS NOT NULL"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_workspace_user_key "
            "ON memories(workspace_id, user_id, logical_key) "
            "WHERE user_id IS NOT NULL AND logical_key IS NOT NULL"
        )
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                action TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                UNIQUE(tenant_id, workspace_id, memory_id, revision)
            )
        """)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS memory_operations (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY(tenant_id, workspace_id, operation_id)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_revisions_resource "
            "ON memory_revisions(tenant_id, workspace_id, memory_id, sequence DESC)"
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

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS session_checkpoints (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                raw_memory_id TEXT NOT NULL REFERENCES memories(id),
                source_kind TEXT NOT NULL,
                source_sequence INTEGER,
                source_boundary INTEGER,
                content_hash TEXT NOT NULL,
                byte_count INTEGER NOT NULL,
                capture_status TEXT NOT NULL,
                index_status TEXT NOT NULL,
                enrichment_status TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(session_id, idempotency_key)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_checkpoints_session ON session_checkpoints(workspace_id, session_id, created_at)"
        )
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS session_context_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                session_id TEXT,
                event_kind TEXT NOT NULL,
                subject_kind TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                event_time TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_events_scope ON session_context_events(workspace_id, session_id, sequence)"
        )

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS entity_relations (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                source_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                target_entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                relationship TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'outgoing',
                confidence REAL NOT NULL DEFAULT 1.0,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_relations_active "
            "ON entity_relations(workspace_id, source_entity_id, target_entity_id, relationship) "
            "WHERE active = 1"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_source "
            "ON entity_relations(workspace_id, source_entity_id, relationship) WHERE active = 1"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_relations_target "
            "ON entity_relations(workspace_id, target_entity_id, relationship) WHERE active = 1"
        )
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS entity_relation_evidence (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                relation_id TEXT NOT NULL REFERENCES entity_relations(id) ON DELETE CASCADE,
                source_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                evidence_kind TEXT NOT NULL,
                source_span_start INTEGER,
                source_span_end INTEGER,
                excerpt_hash TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                extraction_method TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                UNIQUE(relation_id, source_memory_id, evidence_kind, source_span_start, source_span_end, excerpt_hash)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_evidence_relation "
            "ON entity_relation_evidence(workspace_id, relation_id) WHERE active = 1"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_relation_evidence_memory "
            "ON entity_relation_evidence(workspace_id, source_memory_id) WHERE active = 1"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_relation_evidence_span "
            "ON entity_relation_evidence("
            "relation_id, source_memory_id, evidence_kind, "
            "COALESCE(source_span_start, -1), COALESCE(source_span_end, -1), excerpt_hash)"
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
                                           newer_memory_id    TEXT,
                                           FOREIGN KEY (memory_a_id) REFERENCES memories (id),
                                           FOREIGN KEY (memory_b_id) REFERENCES memories (id)
                                       )
                                       """)
        # Migration for databases created before supersession direction was persisted.
        # `newer_memory_id` says WHICH of the two memories is the current one — without it
        # a contradiction records that two memories conflict but not which is stale, so
        # recall-side supersession has nothing to act on.
        try:
            await self._connection.execute("ALTER TABLE contradictions ADD COLUMN newer_memory_id TEXT")
        except Exception as e:
            self.logger.debug("Column migration note for contradictions.newer_memory_id: %s", e)

        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_workspace ON contradictions(workspace_id)")
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_unresolved ON contradictions(workspace_id) WHERE resolved_at IS NULL"
        )
        # Supports the recall-side supersession lookup: "of these memory ids, which are
        # the stale side of an unresolved contradiction".
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_contradictions_superseded "
            "ON contradictions(workspace_id, newer_memory_id) WHERE resolved_at IS NULL"
        )

        # Chat threads
        #
        # Owner-scoped identity: ``row_id`` is an opaque storage-internal surrogate
        # PK (the only value chat_messages.thread_id references); the client-facing
        # ``id`` is stored verbatim and is unique PER OWNER via
        # UNIQUE(workspace_id, COALESCE(user_id,''), id) below. This replaces the
        # old global id-as-PK + ``::u::<hash>`` id materialization for _user_chat.
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_threads (
                row_id TEXT PRIMARY KEY,
                id TEXT NOT NULL,
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
                ownership TEXT NOT NULL DEFAULT 'user',
                idle_action TEXT,
                hidden_at TEXT,
                parent_thread TEXT,
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
        # Migrate: add ownership column to existing databases (idempotent). Existing
        # rows backfill to 'user' via the column default — matches the "leave legacy
        # per-workspace _default threads alone" decision.
        try:
            await self._connection.execute("ALTER TABLE chat_threads ADD COLUMN ownership TEXT NOT NULL DEFAULT 'user'")
            await self._connection.commit()
        except Exception:
            pass  # Column already exists
        # Migrate: idle policy columns (idempotent). idle_action: None|'hide'|'delete';
        # hidden_at: archive timestamp (None = visible). Existing rows default NULL.
        # Migrate: sub-thread parent column (idempotent). NULL = top-level.
        for _alter in (
            "ALTER TABLE chat_threads ADD COLUMN idle_action TEXT",
            "ALTER TABLE chat_threads ADD COLUMN hidden_at TEXT",
            "ALTER TABLE chat_threads ADD COLUMN parent_thread TEXT",
        ):
            try:
                await self._connection.execute(_alter)
                await self._connection.commit()
            except Exception:
                pass  # Column already exists
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_workspace ON chat_threads(workspace_id)")
        # Idle-policy scan index (delete/hide sweeps) and post-archive grace-purge index.
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_idle ON chat_threads(updated_at) WHERE idle_action IS NOT NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_hidden ON chat_threads(hidden_at) WHERE hidden_at IS NOT NULL"
        )
        # Sub-thread child lookup index.
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_parent ON chat_threads(parent_thread) WHERE parent_thread IS NOT NULL"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(workspace_id, user_id) WHERE user_id IS NOT NULL"
        )
        # Cross-workspace user thread index — keys on (tenant, user, ownership)
        # for the list_user_threads query path.
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_threads_tenant_user_ownership ON chat_threads(tenant_id, user_id, ownership) WHERE user_id IS NOT NULL"
        )
        # Owner-scoped identity: the client id is unique PER OWNER, not globally.
        # COALESCE(user_id,'') scopes user-owned threads (in the _user_chat
        # sentinel) by their OBO subject, while workspace-owned threads (user_id
        # NULL) are scoped by (workspace_id, id). This replaces the old global
        # UNIQUE(workspace_id, id) + the ``::u::<hash>`` id materialization.
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_threads_ws_user_id ON chat_threads(workspace_id, COALESCE(user_id, ''), id)"
        )

        # Chat messages
        #
        # thread_id references the thread's surrogate ``row_id`` (NOT its client
        # id), so a per-owner client id like "_default" is unambiguous.
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
                FOREIGN KEY (thread_id) REFERENCES chat_threads (row_id) ON DELETE CASCADE
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
                enrichment_status TEXT NOT NULL DEFAULT 'not_applicable',
                enrichment_memory_ids TEXT DEFAULT '[]',
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
        # Knowledge-phase columns for databases created before the split (must
        # run AFTER the CREATE TABLE above, or the ALTER hits a missing table on
        # a fresh database). Existing rows default to not_applicable: nothing
        # was recorded as scheduled for them, so claiming their enrichment
        # completed would be a fabrication.
        for col_sql in [
            "ALTER TABLE documents ADD COLUMN enrichment_status TEXT NOT NULL DEFAULT 'not_applicable'",
            "ALTER TABLE documents ADD COLUMN enrichment_memory_ids TEXT DEFAULT '[]'",
        ]:
            try:
                await self._connection.execute(col_sql)
            except Exception as e:
                # Column already exists (expected on every startup after the first).
                self.logger.debug("Column migration note for '%s': %s", col_sql, e)

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
                revision INTEGER NOT NULL DEFAULT 0,
                etag TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            )
        """)
        for col_sql in (
            "ALTER TABLE skills ADD COLUMN revision INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE skills ADD COLUMN etag TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE skills ADD COLUMN deleted_at TEXT",
        ):
            try:
                await self._connection.execute(col_sql)
            except Exception as e:
                self.logger.debug("Column migration note for '%s': %s", col_sql, e)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace ON skills(workspace_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace_name ON skills(workspace_id, name)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_workspace_user ON skills(workspace_id, user_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_workspace_name_global ON skills(workspace_id, name) WHERE user_id IS NULL"
        )
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_skills_workspace_user_name "
            "ON skills(workspace_id, user_id, name) WHERE user_id IS NOT NULL"
        )
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_skills_active ON skills(workspace_id, deleted_at)")

        # Skill manifests use the native ``skills`` row as their authoritative
        # head. These sidecars hold immutable snapshots and exact operation
        # results; they never act as a second mutable head.
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS skill_revisions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                snapshot TEXT NOT NULL,
                action TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                UNIQUE(tenant_id, workspace_id, skill_id, revision)
            )
        """)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS skill_operations (
                tenant_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                operation_id TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY(tenant_id, workspace_id, operation_id)
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_revisions_resource ON skill_revisions(tenant_id, workspace_id, skill_id, sequence DESC)"
        )

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

        # Entity registry: canonical entity nodes + aliases + members
        # (entity registry slice 1). Workspace-scoped; CASCADE on workspace and
        # memory delete. Mirrors the enterprise PG migration 022.
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL REFERENCES workspaces (id) ON DELETE CASCADE,
                entity_type TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                provenance TEXT NOT NULL DEFAULT '{}',
                representative_memory_id TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                merged_into TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Exact-match index: one active entity per (workspace, type, normalized_name).
        # Partial unique index (SQLite supports the WHERE clause) so merged
        # tombstones don't collide with live rows. Mirrors the PG partial index.
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entities_active_norm "
            "ON entities(workspace_id, entity_type, normalized_name) WHERE status = 'active'"
        )
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_entities_workspace ON entities(workspace_id)")

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS entity_aliases (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL
            )
        """)
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_entity_aliases_norm ON entity_aliases(workspace_id, normalized_alias)"
        )
        # One alias row per (entity, normalized_alias) so re-adding is a no-op.
        await self._connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_entity_aliases_entity_norm ON entity_aliases(entity_id, normalized_alias)"
        )

        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS entity_members (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
                role TEXT NOT NULL DEFAULT 'mention',
                confidence REAL NOT NULL DEFAULT 1.0,
                meta TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(entity_id, memory_id, role)
            )
        """)
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_entity_members_entity ON entity_members(workspace_id, entity_id)")
        await self._connection.execute("CREATE INDEX IF NOT EXISTS idx_entity_members_memory ON entity_members(workspace_id, memory_id)")

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

    async def _adopt_legacy_memories(self) -> None:
        """Give pre-versioning memory rows a deterministic revision-one snapshot."""

        cursor = await self._connection.execute("SELECT * FROM memories WHERE revision = 0 OR etag = ''")
        for row in await cursor.fetchall():
            memory = self._row_to_memory(row)
            final = memory.model_copy(update={"revision": 1})
            final = final.model_copy(update={"etag": memory_etag(1, final)})
            operation_id = f"legacy-adopt:{final.id}"
            request_hash = canonical_hash(
                {
                    "action": "create",
                    "state": memory_semantic_state(final),
                    "legacy_adoption": True,
                }
            )
            updated = await self._connection.execute(
                "UPDATE memories SET revision = ?, etag = ? WHERE id = ? AND workspace_id = ? AND revision = 0",
                (final.revision, final.etag, final.id, final.workspace_id),
            )
            if updated.rowcount == 0:
                continue
            await self._connection.execute(
                """
                INSERT INTO memory_revisions (
                    tenant_id, workspace_id, memory_id, revision, snapshot,
                    action, operation_id, request_hash
                ) VALUES (?, ?, ?, ?, ?, 'create', ?, ?)
                """,
                (
                    final.tenant_id,
                    final.workspace_id,
                    final.id,
                    final.revision,
                    _memory_snapshot_json(final),
                    operation_id,
                    request_hash,
                ),
            )
            await self._connection.execute(
                """
                INSERT INTO memory_operations (
                    tenant_id, workspace_id, operation_id, request_hash,
                    memory_id, revision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    final.tenant_id,
                    final.workspace_id,
                    operation_id,
                    request_hash,
                    final.id,
                    final.revision,
                ),
            )
        await self._connection.commit()

    async def _adopt_legacy_skills(self) -> None:
        """Give pre-versioning skill rows a deterministic revision-one snapshot."""

        cursor = await self._connection.execute("SELECT * FROM skills WHERE revision = 0 OR etag = ''")
        for row in await cursor.fetchall():
            skill = self._row_to_skill(row)
            final = skill.model_copy(update={"revision": 1})
            final = final.model_copy(update={"etag": manifest_etag(1, final)})
            operation_id = f"legacy-adopt:{final.id}"
            request_hash = canonical_hash({"action": "create", "state": manifest_state(final), "legacy_adoption": True})
            updated = await self._connection.execute(
                "UPDATE skills SET revision = ?, etag = ? WHERE id = ? AND workspace_id = ? AND revision = 0",
                (final.revision, final.etag, final.id, final.workspace_id),
            )
            if updated.rowcount == 0:
                continue
            await self._connection.execute(
                """
                INSERT INTO skill_revisions (
                    tenant_id, workspace_id, skill_id, revision, snapshot,
                    action, operation_id, request_hash
                ) VALUES (?, ?, ?, ?, ?, 'create', ?, ?)
                """,
                (
                    final.tenant_id,
                    final.workspace_id,
                    final.id,
                    final.revision,
                    _skill_snapshot_json(final),
                    operation_id,
                    request_hash,
                ),
            )
            await self._connection.execute(
                """
                INSERT INTO skill_operations (
                    tenant_id, workspace_id, operation_id, request_hash,
                    skill_id, revision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    final.tenant_id,
                    final.workspace_id,
                    operation_id,
                    request_hash,
                    final.id,
                    final.revision,
                ),
            )
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

        now = datetime.now(UTC)
        memory = Memory(
            id=generate_id("mem"),
            logical_key=input.logical_key,
            tenant_id=input.tenant_id or DEFAULT_TENANT_ID,
            workspace_id=workspace_id,
            context_id=input.context_id or "_default",
            session_id=getattr(input, "session_id", None),
            user_id=input.user_id,
            observer_id=input.observer_id,
            subject_id=input.subject_id,
            content=input.content,
            content_hash=hashlib.sha256(input.content.encode()).hexdigest(),
            type=input.type or MemoryType.SEMANTIC,
            subtype=input.subtype,
            importance=input.importance,
            tags=input.tags,
            metadata=input.metadata,
            refinement_metadata=input.refinement_metadata,
            abstract=getattr(input, "abstract", None),
            overview=getattr(input, "overview", None),
            source_memory_id=getattr(input, "source_memory_id", None),
            source_document_id=input.source_document_id,
            source_page_id=input.source_page_id,
            source_dataset_id=input.source_dataset_id,
            source_thread_id=input.source_thread_id,
            category=getattr(input, "category", None),
            status=MemoryStatus.ACTIVE,
            pinned=input.pinned,
            event_time=input.event_time,
            created_at=now,
            updated_at=now,
        )
        result = await self.mutate_memory(
            MemoryMutation(
                action="create",
                memory=memory,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "create", "state": memory_semantic_state(memory)}),
                expected_etag="*",
            )
        )
        return result.memory

    async def get_memory(
        self,
        workspace_id: str,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID within a workspace. Set track_access=False for internal reads that should not affect decay tracking."""
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"SELECT * FROM memories WHERE id = ? AND workspace_id = ?{deleted_clause}",
            (memory_id, workspace_id),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access and memory.deleted_at is None:
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

    async def get_memory_by_id(
        self,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"SELECT * FROM memories WHERE id = ?{deleted_clause}",
            (memory_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        memory = self._row_to_memory(row)

        if track_access and memory.deleted_at is None:
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

    async def get_memories_by_ids(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        include_deleted: bool = False,
    ) -> list[Memory]:
        if not memory_ids:
            return []
        unique_ids = list(dict.fromkeys(memory_ids))
        placeholders = ",".join("?" for _ in unique_ids)
        deleted = "" if include_deleted else " AND deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"SELECT * FROM memories WHERE workspace_id = ? AND id IN ({placeholders}){deleted}",
            (workspace_id, *unique_ids),
        )
        by_id = {row["id"]: self._row_to_memory(row, with_embedding=False) for row in await cursor.fetchall()}
        return [by_id[memory_id] for memory_id in unique_ids if memory_id in by_id]

    async def list_source_memories(
        self,
        workspace_id: str,
        source_memory_id: str,
    ) -> list[Memory]:
        cursor = await self._connection.execute(
            "SELECT * FROM memories WHERE workspace_id = ? AND source_memory_id = ? AND deleted_at IS NULL ORDER BY id",
            (workspace_id, source_memory_id),
        )
        return [self._row_to_memory(row) for row in await cursor.fetchall()]

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
        """Update memory fields."""
        invalid_keys = set(updates.keys()) - _UPDATABLE_MEMORY_COLUMNS
        if invalid_keys:
            raise ValueError(f"Invalid update fields: {invalid_keys}")
        if SEMANTIC_MEMORY_FIELDS.intersection(updates):
            current = await self.get_memory(workspace_id, memory_id, track_access=False)
            if current is None:
                return None
            semantic_updates = dict(updates)
            if "content" in semantic_updates and "content_hash" not in semantic_updates:
                semantic_updates["content_hash"] = hashlib.sha256(semantic_updates["content"].encode()).hexdigest()
            if isinstance(semantic_updates.get("type"), str):
                semantic_updates["type"] = MemoryType(semantic_updates["type"])
            if "pinned" in semantic_updates:
                semantic_updates["pinned"] = bool(semantic_updates["pinned"])
            desired = current.model_copy(update={**semantic_updates, "updated_at": datetime.now(UTC)})
            result = await self.mutate_memory(
                MemoryMutation(
                    action="replace",
                    memory=desired,
                    operation_id=generate_id("op"),
                    request_hash=canonical_hash(
                        {
                            "action": "replace",
                            "state": memory_semantic_state(desired),
                            "expected_etag": current.etag,
                        }
                    ),
                    expected_etag=current.etag,
                )
            )
            return result.memory
        # Build SET clause
        set_parts = []
        values = []
        for key, value in updates.items():
            if key in ("tags", "metadata", "refinement_metadata"):
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

        # Use microsecond-precision UTC ISO (utc_now_iso) instead of SQL
        # datetime('now') (second resolution). create_memory already writes
        # microsecond updated_at; a coarser update timestamp can be lexically
        # SMALLER than the create timestamp within the same second, which would
        # make a real content edit invisible to the dirty-watermark skip gate
        # (P2 strategy-C). Aligning the format keeps the watermark fail-safe.
        set_parts.append("updated_at = ?")
        values.append(utc_now_iso())
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
        current = await self.get_memory(workspace_id, memory_id, track_access=False, include_deleted=True)
        if current is None or (current.deleted_at is not None and not hard):
            return False
        if hard:
            await self._deactivate_relation_evidence_sql(workspace_id, memory_id)
            await self._connection.execute(
                "DELETE FROM memory_operations WHERE workspace_id = ? AND memory_id = ?",
                (workspace_id, memory_id),
            )
            await self._connection.execute(
                "DELETE FROM memory_revisions WHERE workspace_id = ? AND memory_id = ?",
                (workspace_id, memory_id),
            )
            cursor = await self._connection.execute(
                "DELETE FROM memories WHERE id = ? AND workspace_id = ?",
                (memory_id, workspace_id),
            )
            # Also delete from FTS index
            await self._connection.execute(
                "DELETE FROM memories_fts WHERE id = ?",
                (memory_id,),
            )
            await self._insert_context_event(
                workspace_id,
                current.session_id,
                ContextEventKind.MEMORY_DELETE.value,
                "memory",
                memory_id,
                {"hard": True},
            )
            await self._connection.commit()
            return cursor.rowcount > 0
        else:
            now = datetime.now(UTC)
            desired = current.model_copy(update={"deleted_at": now, "updated_at": now})
            await self.mutate_memory(
                MemoryMutation(
                    action="delete",
                    memory=desired,
                    operation_id=generate_id("op"),
                    request_hash=canonical_hash({"action": "delete", "id": memory_id, "expected_etag": current.etag}),
                    expected_etag=current.etag,
                )
            )
            return True

    async def mutate_memory(self, mutation: MemoryMutation) -> MemoryMutationResult:
        """Atomically mutate the native memory head and immutable sidecars."""

        desired = mutation.memory.model_copy(deep=True)
        async with self._memory_mutation_lock:
            replay = await self.get_memory_operation(
                desired.tenant_id,
                desired.workspace_id,
                mutation.operation_id,
                mutation.request_hash,
            )
            if replay is not None:
                return replay

            current = await self.get_memory(
                desired.workspace_id,
                desired.id,
                track_access=False,
                include_deleted=True,
            )
            if mutation.action == "create":
                if mutation.expected_etag != "*":
                    raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
                if current is not None:
                    raise VersionedResourceConflictError("memory id already exists")
                final = desired.model_copy(update={"revision": 1, "deleted_at": None})
            else:
                if current is None or current.tenant_id != desired.tenant_id:
                    raise VersionedResourceNotFoundError("memory not found")
                if mutation.expected_etag != current.etag:
                    raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
                if mutation.action == "restore":
                    if current.deleted_at is None:
                        raise VersionedResourceConflictError("memory is not deleted")
                elif current.deleted_at is not None:
                    raise VersionedResourceNotFoundError("memory not found")
                if (
                    desired.logical_key != current.logical_key
                    or desired.user_id != current.user_id
                    or desired.workspace_id != current.workspace_id
                ):
                    raise VersionedResourceConflictError("memory logical key and ownership are immutable")
                final = desired.model_copy(
                    update={
                        "created_at": current.created_at,
                        "revision": current.revision + 1,
                    }
                )
            final = final.model_copy(update={"etag": memory_etag(final.revision, final)})

            try:
                if mutation.action == "create":
                    await self._insert_memory_head(final)
                else:
                    deleted_predicate = "deleted_at IS NOT NULL" if mutation.action == "restore" else "deleted_at IS NULL"
                    cursor = await self._update_memory_head(final, current.etag, deleted_predicate)
                    if cursor.rowcount == 0:
                        await self._connection.rollback()
                        replay = await self.get_memory_operation(
                            desired.tenant_id,
                            desired.workspace_id,
                            mutation.operation_id,
                            mutation.request_hash,
                        )
                        if replay is not None:
                            return replay
                        raise VersionedResourcePreconditionFailedError("ETag does not match current revision")

                await self._connection.execute(
                    "DELETE FROM memories_fts WHERE id = ?",
                    (final.id,),
                )
                if final.deleted_at is None:
                    await self._connection.execute(
                        "INSERT INTO memories_fts (id, workspace_id, content) VALUES (?, ?, ?)",
                        (
                            final.id,
                            final.workspace_id,
                            self._build_fts_text(final.content, final.metadata),
                        ),
                    )
                if mutation.action in {"replace", "delete"}:
                    await self._deactivate_relation_evidence_sql(
                        final.workspace_id,
                        final.id,
                    )
                await self._connection.execute(
                    """
                    INSERT INTO memory_revisions (
                        tenant_id, workspace_id, memory_id, revision, snapshot,
                        action, operation_id, request_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final.tenant_id,
                        final.workspace_id,
                        final.id,
                        final.revision,
                        _memory_snapshot_json(final),
                        mutation.action,
                        mutation.operation_id,
                        mutation.request_hash,
                    ),
                )
                await self._connection.execute(
                    """
                    INSERT INTO memory_operations (
                        tenant_id, workspace_id, operation_id, request_hash,
                        memory_id, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final.tenant_id,
                        final.workspace_id,
                        mutation.operation_id,
                        mutation.request_hash,
                        final.id,
                        final.revision,
                    ),
                )
                await self._insert_context_event(
                    final.workspace_id,
                    final.session_id,
                    (ContextEventKind.MEMORY_DELETE.value if mutation.action == "delete" else ContextEventKind.MEMORY_UPSERT.value),
                    "memory",
                    final.id,
                    {"revision": final.revision, "action": mutation.action},
                )
                await self._connection.commit()
            except Exception as exc:
                await self._connection.rollback()
                replay = await self.get_memory_operation(
                    desired.tenant_id,
                    desired.workspace_id,
                    mutation.operation_id,
                    mutation.request_hash,
                )
                if replay is not None:
                    return replay
                latest = await self.get_memory(
                    desired.workspace_id,
                    desired.id,
                    track_access=False,
                    include_deleted=True,
                )
                if mutation.action == "create" and latest is not None:
                    raise VersionedResourceConflictError("memory id already exists")
                if mutation.action == "create" and isinstance(exc, (sqlite3.IntegrityError, aiosqlite.IntegrityError)):
                    raise VersionedResourceConflictError("memory id or scoped logical_key already exists") from exc
                if mutation.action != "create" and latest is not None and latest.etag != mutation.expected_etag:
                    raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
                raise
            return MemoryMutationResult(memory=final)

    async def get_memory_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> MemoryMutationResult | None:
        cursor = await self._connection.execute(
            """
            SELECT r.* FROM memory_operations o
            JOIN memory_revisions r
              ON r.tenant_id = o.tenant_id AND r.workspace_id = o.workspace_id
             AND r.memory_id = o.memory_id AND r.revision = o.revision
            WHERE o.tenant_id = ? AND o.workspace_id = ? AND o.operation_id = ?
            """,
            (tenant_id, workspace_id, operation_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise VersionedResourceConflictError("idempotency key was already used for a different request")
        return MemoryMutationResult(memory=_memory_snapshot(row["snapshot"]), replayed=True)

    async def list_memory_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[MemoryRevision]:
        clauses = ["tenant_id = ?", "workspace_id = ?", "memory_id = ?"]
        params: list[Any] = [tenant_id, workspace_id, memory_id]
        if before_sequence is not None:
            clauses.append("sequence < ?")
            params.append(before_sequence)
        params.append(limit)
        cursor = await self._connection.execute(
            f"SELECT * FROM memory_revisions WHERE {' AND '.join(clauses)} ORDER BY sequence DESC LIMIT ?",
            tuple(params),
        )
        return [_memory_revision_from_row(row) for row in await cursor.fetchall()]

    async def _insert_memory_head(self, memory: Memory) -> None:
        await self._connection.execute(
            """
            INSERT INTO memories (
                id, logical_key, tenant_id, workspace_id, context_id, session_id,
                user_id, content, content_hash, type, subtype, category,
                importance, tags, metadata, refinement_metadata, embedding,
                abstract, overview, source_memory_id, status, pinned,
                observer_id, subject_id, source_document_id, source_page_id,
                source_dataset_id, source_thread_id, access_count,
                last_accessed_at, decay_factor, revision, etag, deleted_at,
                event_time, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _memory_row_values(self, memory),
        )

    async def _update_memory_head(self, memory: Memory, current_etag: str, deleted_predicate: str):
        return await self._connection.execute(
            f"""
            UPDATE memories SET
                content = ?, content_hash = ?, type = ?, subtype = ?, tags = ?,
                refinement_metadata = ?, pinned = ?, embedding = ?, revision = ?,
                etag = ?, updated_at = ?, deleted_at = ?
            WHERE id = ? AND tenant_id = ? AND workspace_id = ? AND etag = ?
              AND {deleted_predicate}
            """,
            (
                memory.content,
                memory.content_hash,
                memory.type.value,
                memory.subtype,
                json.dumps(memory.tags, sort_keys=True, separators=(",", ":")),
                json.dumps(
                    memory.refinement_metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                1 if memory.pinned else 0,
                self._serialize_embedding(memory.embedding) if memory.embedding else None,
                memory.revision,
                memory.etag,
                to_utc_iso(memory.updated_at),
                to_utc_iso(memory.deleted_at),
                memory.id,
                memory.tenant_id,
                memory.workspace_id,
                current_etag,
            ),
        )

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
        return [self._row_to_memory(row, with_embedding=False) for row in rows]

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
        return [self._row_to_memory(row, with_embedding=False) for row in rows]

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
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by subtype, tags, and/or metadata without requiring embeddings."""
        where_parts = ["workspace_id = ?", "deleted_at IS NULL"]
        params: list = [workspace_id]

        if context_id is not None:
            where_parts.append("context_id = ?")
            params.append(context_id)

        # Forced user-scope partition filter (the user-global read boundary).
        if user_id is not None:
            where_parts.append("user_id = ?")
            params.append(user_id)

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
        return [self._row_to_memory(row, with_embedding=False) for row in rows]

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

    async def search_memories_by_entities(
        self,
        workspace_id: str,
        query_embedding: list[float],
        entities: list[str],
        limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Entity-anchored vector search (see StorageBackend.search_memories_by_entities).

        Restrict to active, embedded memories whose ``metadata['speaker']`` is one
        of ``entities`` (the speaker is the discriminating, validated signal — the
        broader ``metadata['entities']`` mention set dilutes recall; see
        MemoryService._fuse_entity_results), then rank that set by cosine
        similarity in Python. SQLite stores metadata as JSON text, so the speaker
        prefilter is applied in Python after pulling the workspace's embedded rows
        (LoCoMo-scale corpora are tractable; this is an eval/parity path, not the
        hot production OSS write path).
        """
        if not entities:
            return []

        entity_set = set(entities)
        query = (
            "SELECT * FROM memories "
            "WHERE workspace_id = ? AND deleted_at IS NULL AND embedding IS NOT NULL "
            "AND (status IS NULL OR status = 'active')"
        )
        cursor = await self._connection.execute(query, [workspace_id])
        rows = await cursor.fetchall()

        results: list[tuple[Memory, float]] = []
        for row in rows:
            if not row["embedding"]:
                continue
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except (TypeError, ValueError):
                meta = {}
            speaker = meta.get("speaker")
            if speaker is None or speaker not in entity_set:
                continue
            embedding = self._deserialize_embedding(row["embedding"])
            relevance = cosine_similarity(query_embedding, embedding)
            results.append((self._row_to_memory(row, with_embedding=False), relevance))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:limit]

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

        # Use sqlite-vec for similarity search.
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

            # Post-filter after LIMIT is correct: qualifying rows (relevance >=
            # min_relevance, i.e. distance <= max_distance) form a contiguous
            # low-distance prefix under ORDER BY distance ASC, so they always
            # occupy the first slots of the LIMIT window and cannot be displaced
            # by sub-threshold rows.
            if relevance >= min_relevance:
                memory = self._row_to_memory(row, with_embedding=False)
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
                    memory = self._row_to_memory(row, with_embedding=False)
                    results.append((memory, relevance))

        # Sort by relevance descending, apply offset and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[offset : offset + limit]

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Escape FTS5 special syntax to prevent query injection."""
        escaped = query.replace('"', '""')
        return f'"{escaped}"'

    @staticmethod
    def _build_fts_text(content: str, metadata: dict | None) -> str:
        """Build the text indexed into memories_fts: content plus any aliases.

        Folding metadata['aliases'] into the indexed text lets alias-only queries
        retrieve the canonical memory while leaving memory.content unchanged.
        """
        aliases = (metadata or {}).get("aliases")
        if isinstance(aliases, list) and aliases:
            return content + " " + " ".join(str(a) for a in aliases)
        return content

    async def reindex_memory(self, workspace_id: str, memory_id: str) -> None:
        """Rebuild the FTS5 row for a memory after its content/aliases changed.

        Used for write-behind reconciliation (the update path does not touch the
        FTS index inline). Deletes and re-inserts the memory's index row from its
        current content + aliases; if the memory is gone, just drops the row.
        """
        memory = await self.get_memory(workspace_id, memory_id, track_access=False)
        await self._connection.execute("DELETE FROM memories_fts WHERE id = ?", (memory_id,))
        if memory is not None:
            await self._connection.execute(
                "INSERT INTO memories_fts (id, workspace_id, content) VALUES (?, ?, ?)",
                (memory_id, workspace_id, self._build_fts_text(memory.content, memory.metadata)),
            )
        await self._connection.commit()

    @staticmethod
    def _build_fts5_match_query(query: str) -> str | None:
        """Build a term-based FTS5 MATCH expression from a free-text query.

        The query is tokenized into alphanumeric terms, each escaped as a quoted
        FTS5 token, and joined with ``OR`` so that natural-language queries match
        documents sharing *any* term (BM25-ranked), rather than requiring the
        exact phrase (the prior behaviour, which made multi-word queries match
        almost nothing). Returns None when the query has no usable terms.
        """
        terms = re.findall(r"[0-9A-Za-z]+", query)
        if not terms:
            return None
        return " OR ".join(f'"{term}"' for term in terms)

    async def full_text_search(
        self,
        workspace_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search using SQLite FTS5, ranked by BM25 relevance."""
        match_query = self._build_fts5_match_query(query)
        if match_query is None:
            return []

        sql = """
            SELECT m.*
            FROM memories_fts
                     INNER JOIN memories m ON m.id = memories_fts.id
            WHERE memories_fts.workspace_id = ?
              AND memories_fts.content MATCH ?
              AND m.deleted_at IS NULL
        """
        params: list = [workspace_id, match_query]

        if context_id is not None:
            sql += "  AND m.context_id = ?\n"
            params.append(context_id)

        # bm25() returns more-negative scores for better matches, so ascending
        # order surfaces the most relevant rows first.
        sql += "ORDER BY bm25(memories_fts)\n"
        sql += "LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()

        return [self._row_to_memory(row, with_embedding=False) for row in rows]

    async def get_timeline(
        self,
        workspace_id: str,
        event_after: str | None = None,
        event_before: str | None = None,
        ascending: bool = True,
        limit: int = 50,
        offset: int = 0,
        types: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Return memories ordered by effective event time (event_time or created_at).

        Powers explicit timeline browsing and temporal-neighbour lookups.
        ``event_after``/``event_before`` are canonical UTC ISO strings bounding the
        window against the effective time. Memories without an explicit event_time
        fall back to created_at so undated memories still appear on the timeline.
        """
        effective = "COALESCE(event_time, created_at)"
        where = ["workspace_id = ?", "deleted_at IS NULL"]
        params: list = [workspace_id]
        if not include_archived:
            where.append("(status IS NULL OR status = 'active')")
        if types:
            placeholders = ",".join("?" * len(types))
            where.append(f"type IN ({placeholders})")
            params.extend(types)
        if event_after is not None:
            where.append(f"{effective} >= ?")
            params.append(str(event_after))
        if event_before is not None:
            where.append(f"{effective} <= ?")
            params.append(str(event_before))

        order = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM memories WHERE {' AND '.join(where)} ORDER BY {effective} {order} LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_memory(row, with_embedding=False) for row in rows]

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
        """Create graph edge between memories (idempotent).

        ``ON CONFLICT ... DO NOTHING`` on the ``uq_association`` key
        (source_id, target_id, relationship) makes a re-run (reprocess / re-enrich)
        or a race between concurrent ``auto_enrich`` tasks a harmless no-op rather
        than a duplicate-key IntegrityError — which on Postgres/asyncpg aborts the
        surrounding transaction and takes the rest of the batch down with it.
        Returns the existing edge on conflict.
        """
        association_id = generate_id("assoc")
        now = utc_now_iso()

        await self._connection.execute(
            """
            INSERT INTO memory_associations (id, workspace_id, source_id, target_id,
                                             relationship, strength, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, relationship) DO NOTHING
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

        # A fresh insert has our new id; a conflict (skipped) does not, so fall
        # back to the pre-existing edge by its unique key.
        cursor = await self._connection.execute(
            "SELECT * FROM memory_associations WHERE id = ?",
            (association_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            cursor = await self._connection.execute(
                """
                SELECT * FROM memory_associations
                WHERE source_id = ? AND target_id = ? AND relationship = ?
                """,
                (input.source_id, input.target_id, input.relationship),
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
        """Get associations for multiple memories, chunked to respect SQLite's
        host-parameter limit (default 999).

        SQLite raises "too many SQL variables" when an IN clause has more than
        ~999 placeholders.  For direction="both" each id appears twice, so the
        effective limit is halved (~499).  The graph-analysis node cap is 10000,
        making this easily reachable.

        Strategy: split memory_ids into chunks of at most _SQLITE_BATCH_CHUNK
        ids, run the existing query per chunk, concatenate, and dedup by
        assoc.id to prevent duplicates at chunk boundaries (direction="both"
        can return the same edge from two adjacent chunks if a node id lands in
        both the source and target set of different chunks).
        """
        if not memory_ids:
            return []

        # Conservative chunk size: leaves room for workspace_id placeholder and
        # any relationship-filter placeholders.  direction="both" uses each id
        # twice, so 450 ids -> 900 placeholders, still safely under 999.
        _CHUNK = 450

        seen_ids: set[str] = set()
        results: list[Association] = []

        for chunk_start in range(0, len(memory_ids), _CHUNK):
            chunk = memory_ids[chunk_start : chunk_start + _CHUNK]

            where_parts = ["workspace_id = ?"]
            params: list = [workspace_id]

            # Build direction filter with IN clause for this chunk
            placeholders = ",".join("?" * len(chunk))
            if direction == "outgoing":
                where_parts.append(f"source_id IN ({placeholders})")
                params.extend(chunk)
            elif direction == "incoming":
                where_parts.append(f"target_id IN ({placeholders})")
                params.extend(chunk)
            else:  # both
                where_parts.append(f"(source_id IN ({placeholders}) OR target_id IN ({placeholders}))")
                params.extend(chunk)
                params.extend(chunk)

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

            for row in rows:
                assoc = self._row_to_association(row)
                # Dedup: the same edge can appear in multiple chunks when
                # direction="both" and both endpoints fall in different chunks.
                if assoc.id not in seen_ids:
                    seen_ids.add(assoc.id)
                    results.append(assoc)

        return results

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
        """Update an existing association's strength and/or metadata.

        Watermark-visibility contract: after updating the edge row we bump
        ``updated_at`` on both the source and target memories.  This advances
        ``max_memory_updated_at`` in ``get_workspace_change_watermark``, making
        the edge change visible to the KB skip-generate gate (Gate A) and the
        AGE watermark-gated materialize gate (Gate B).  Phase 1 fix 1.6 already
        decoupled the recency boost from ``updated_at``, so the bump does NOT
        produce a recall feedback loop.
        """
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
        if cursor.rowcount == 0:
            await self._connection.commit()
            return False

        # Fetch the association endpoints so we can bump the memory timestamps.
        # This makes the edge change visible to get_workspace_change_watermark via
        # max_memory_updated_at (the associations table has no updated_at column).
        endpoint_cursor = await self._connection.execute(
            "SELECT source_id, target_id FROM memory_associations WHERE id = ? AND workspace_id = ?",
            (association_id, workspace_id),
        )
        row = await endpoint_cursor.fetchone()
        if row:
            now = utc_now_iso()
            source_id = row["source_id"]
            target_id = row["target_id"]
            await self._connection.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ? AND workspace_id = ? AND deleted_at IS NULL",
                (now, source_id, workspace_id),
            )
            if target_id != source_id:
                await self._connection.execute(
                    "UPDATE memories SET updated_at = ? WHERE id = ? AND workspace_id = ? AND deleted_at IS NULL",
                    (now, target_id, workspace_id),
                )

        await self._connection.commit()
        return True

    async def count_associations_by_relationship(self, workspace_id: str) -> dict[str, int]:
        """Per-relationship edge counts via a single GROUP BY aggregate.

        Avoids loading all edges into Python for ``relationship_rollup``.
        """
        cursor = await self._connection.execute(
            "SELECT relationship, COUNT(*) AS cnt FROM memory_associations WHERE workspace_id = ? GROUP BY relationship",
            (workspace_id,),
        )
        rows = await cursor.fetchall()
        return {row["relationship"]: row["cnt"] for row in rows}

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
            INSERT INTO workspaces (id, tenant_id, name, settings, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace.id,
                workspace.tenant_id,
                workspace.name,
                json.dumps(workspace.settings),
                json.dumps(normalize_tags(workspace.tags)),
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

    async def list_workspaces(
        self,
        *,
        tags: list[str] | None = None,
        match: str = "all",
    ) -> list[Workspace]:
        """List workspaces, optionally filtered by tag.

        Workspace counts are small, so tag filtering is applied in Python after load.
        """
        cursor = await self._connection.execute(
            "SELECT * FROM workspaces ORDER BY name",
        )
        rows = await cursor.fetchall()
        workspaces = [self._row_to_workspace(row) for row in rows]

        query_tags = normalize_tags(tags)
        if not query_tags:
            return workspaces

        wanted = set(query_tags)
        if match == "any":
            return [w for w in workspaces if wanted & set(w.tags)]
        # default: 'all' — workspace must carry every requested tag
        return [w for w in workspaces if wanted <= set(w.tags)]

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Atomically delete a workspace and all of its persisted resources."""
        deleted = await purge_workspace(self._connection, workspace_id)
        if deleted:
            self.logger.info("Deleted workspace and associated data: %s", workspace_id)
        return deleted

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
            elif key == "tags":
                set_parts.append(f"{key} = ?")
                values.append(json.dumps(normalize_tags(value)))
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

    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Hard-delete a context within a workspace.

        Contexts are hard-deleted (no soft-delete column). Memories keep their
        context_id; deleting a context does not remove its memories. Returns
        True if a row was deleted, False if the context did not exist in this
        workspace.
        """
        cursor = await self._connection.execute(
            "DELETE FROM contexts WHERE id = ? AND workspace_id = ?",
            (context_id, workspace_id),
        )
        await self._connection.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            self.logger.debug("Deleted context: %s", context_id)
        return deleted

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

    async def get_workspace_change_watermark(self, workspace_id: str) -> tuple[str, str, int, int] | None:
        """Deterministic dirty-watermark over active memories + all associations.

        One aggregate query per table (no full scan into Python). See the ABC
        docstring for the staleness rationale (association_count catches edge
        deletes that advance no timestamp). Returns None on any error so callers
        fall back to doing the full work (fail-safe).
        """
        try:
            mem_cursor = await self._connection.execute(
                """
                SELECT COALESCE(MAX(updated_at), '') AS max_updated, COUNT(*) AS cnt
                FROM memories
                WHERE workspace_id = ? AND deleted_at IS NULL
                """,
                (workspace_id,),
            )
            mem_row = await mem_cursor.fetchone()

            assoc_cursor = await self._connection.execute(
                """
                SELECT COALESCE(MAX(created_at), '') AS max_created, COUNT(*) AS cnt
                FROM memory_associations
                WHERE workspace_id = ?
                """,
                (workspace_id,),
            )
            assoc_row = await assoc_cursor.fetchone()
        except Exception as e:  # fail-safe: ambiguous watermark -> do the work
            self.logger.debug("Could not compute change watermark for %s: %s", workspace_id, e)
            return None

        return (
            mem_row["max_updated"] or "",
            assoc_row["max_created"] or "",
            int(mem_row["cnt"] or 0),
            int(assoc_row["cnt"] or 0),
        )

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
        if await self.get_session(workspace_id, session_id) is None:
            raise ValueError(f"Session {session_id} not found in workspace {workspace_id}")
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
        await self._insert_context_event(
            workspace_id,
            session_id,
            ContextEventKind.WORKING_UPSERT.value,
            "working_memory",
            key,
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
            "SELECT w.* FROM working_memory w JOIN sessions s ON s.id = w.session_id "
            "WHERE w.session_id = ? AND w.key = ? AND s.workspace_id = ?",
            (session_id, key, workspace_id),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_working_memory(row)

    async def get_all_working_memory(self, workspace_id: str, session_id: str) -> list[WorkingMemory]:
        """Get all working memory entries for session."""
        cursor = await self._connection.execute(
            "SELECT w.* FROM working_memory w JOIN sessions s ON s.id = w.session_id WHERE w.session_id = ? AND s.workspace_id = ?",
            (session_id, workspace_id),
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
        if cursor.rowcount and updates.get("committed_at") is not None:
            await self._insert_context_event(
                workspace_id,
                session_id,
                ContextEventKind.COMMIT.value,
                "session",
                session_id,
            )
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

    def supports_capability(self, capability: str) -> bool:
        return capability in {
            "session_checkpoints",
            "session_context_events",
            "entity_relations",
        }

    async def _insert_context_event(
        self,
        workspace_id: str,
        session_id: str | None,
        event_kind: str,
        subject_kind: str,
        subject_id: str,
        metadata: dict | None = None,
    ) -> SessionContextEvent:
        event_time = utc_now_iso()
        cursor = await self._connection.execute(
            """
            INSERT INTO session_context_events (
                workspace_id, session_id, event_kind, subject_kind,
                subject_id, event_time, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                session_id,
                event_kind,
                subject_kind,
                subject_id,
                event_time,
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )
        if self._context_event_retention_days > 0:
            cutoff = to_utc_iso(datetime.now(UTC) - timedelta(days=self._context_event_retention_days))
            await self._connection.execute(
                "DELETE FROM session_context_events WHERE event_time < ?",
                (cutoff,),
            )
        return SessionContextEvent(
            sequence=cursor.lastrowid,
            workspace_id=workspace_id,
            session_id=session_id,
            event_kind=ContextEventKind(event_kind),
            subject_kind=subject_kind,
            subject_id=subject_id,
            event_time=parse_datetime_utc(event_time),
            metadata=metadata or {},
        )

    async def append_context_event(
        self,
        workspace_id: str,
        session_id: str | None,
        event_kind: str,
        subject_kind: str,
        subject_id: str,
        metadata: dict | None = None,
    ) -> SessionContextEvent:
        event = await self._insert_context_event(
            workspace_id,
            session_id,
            event_kind,
            subject_kind,
            subject_id,
            metadata,
        )
        await self._connection.commit()
        return event

    async def list_context_events(
        self,
        workspace_id: str,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
    ) -> list[SessionContextEvent]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM session_context_events
            WHERE workspace_id = ? AND (session_id = ? OR session_id IS NULL)
              AND sequence > ?
            ORDER BY sequence ASC
            LIMIT ?
            """,
            (workspace_id, session_id, after_sequence, limit),
        )
        return [self._row_to_context_event(row) for row in await cursor.fetchall()]

    async def get_context_event_bounds(self, workspace_id: str, session_id: str) -> tuple[int, int]:
        cursor = await self._connection.execute(
            """
            SELECT COALESCE(MIN(sequence), 0) AS minimum,
                   COALESCE(MAX(sequence), 0) AS maximum
            FROM session_context_events
            WHERE workspace_id = ? AND (session_id = ? OR session_id IS NULL)
            """,
            (workspace_id, session_id),
        )
        row = await cursor.fetchone()
        return int(row["minimum"]), int(row["maximum"])

    def _row_to_context_event(self, row: aiosqlite.Row) -> SessionContextEvent:
        return SessionContextEvent(
            sequence=row["sequence"],
            workspace_id=row["workspace_id"],
            session_id=row["session_id"],
            event_kind=ContextEventKind(row["event_kind"]),
            subject_kind=row["subject_kind"],
            subject_id=row["subject_id"],
            event_time=parse_datetime_utc(row["event_time"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    async def create_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        input: SessionCheckpointInput,
    ) -> tuple[SessionCheckpoint, bool]:
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found in workspace {workspace_id}")
        digest = hashlib.sha256(f"{workspace_id}\0{session_id}\0{input.idempotency_key}".encode()).hexdigest()
        checkpoint_id = f"chk_{digest[:32]}"
        raw_memory_id = f"mem_chk_{digest[:32]}"
        existing = await self.get_session_checkpoint(workspace_id, session_id, checkpoint_id)
        if existing is not None:
            if existing.content_hash != input.content_hash:
                raise ValueError("idempotency key was already used for different checkpoint content")
            return existing, True

        raw = Memory(
            id=raw_memory_id,
            tenant_id=session.tenant_id,
            workspace_id=workspace_id,
            context_id=session.context_id,
            session_id=session_id,
            content=input.transcript_segment,
            content_hash=input.content_hash,
            type=MemoryType.EPISODIC,
            subtype="checkpoint",
            metadata={
                "source": "session_checkpoint",
                "source_kind": input.source_kind,
                "source_sequence": input.source_sequence,
                "source_boundary": input.source_boundary,
            },
        )
        # Memory's ordinary content validator trims surrounding whitespace;
        # checkpoint sources must retain the exact UTF-8 segment and hash.
        raw = raw.model_copy(update={"content": input.transcript_segment})
        await self.mutate_memory(
            MemoryMutation(
                action="create",
                memory=raw,
                operation_id=f"checkpoint-raw:{digest}",
                request_hash=canonical_hash({"action": "checkpoint_raw", "content_hash": input.content_hash}),
                expected_etag="*",
            )
        )
        now = utc_now_iso()
        try:
            await self._connection.execute(
                """
                INSERT INTO session_checkpoints (
                    id, workspace_id, session_id, raw_memory_id, source_kind,
                    source_sequence, source_boundary, content_hash, byte_count,
                    capture_status, index_status, enrichment_status,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    workspace_id,
                    session_id,
                    raw_memory_id,
                    input.source_kind,
                    input.source_sequence,
                    input.source_boundary,
                    input.content_hash,
                    len(input.transcript_segment.encode("utf-8")),
                    CheckpointCaptureStatus.DURABLE.value,
                    CheckpointWorkStatus.PENDING.value,
                    CheckpointWorkStatus.SKIPPED.value,
                    input.idempotency_key,
                    now,
                    now,
                ),
            )
            await self._insert_context_event(
                workspace_id,
                session_id,
                ContextEventKind.CHECKPOINT.value,
                "checkpoint",
                checkpoint_id,
                {"raw_memory_id": raw_memory_id},
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            existing = await self.get_session_checkpoint(workspace_id, session_id, checkpoint_id)
            if existing is not None and existing.content_hash == input.content_hash:
                return existing, True
            raise
        checkpoint = await self.get_session_checkpoint(workspace_id, session_id, checkpoint_id)
        return checkpoint, False

    async def get_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
    ) -> SessionCheckpoint | None:
        cursor = await self._connection.execute(
            "SELECT * FROM session_checkpoints WHERE id = ? AND workspace_id = ? AND session_id = ?",
            (checkpoint_id, workspace_id, session_id),
        )
        row = await cursor.fetchone()
        return self._row_to_checkpoint(row) if row else None

    async def list_session_checkpoints(
        self,
        workspace_id: str,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[SessionCheckpoint]:
        cursor = await self._connection.execute(
            "SELECT * FROM session_checkpoints WHERE workspace_id = ? AND session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
            (workspace_id, session_id, limit),
        )
        return [self._row_to_checkpoint(row) for row in await cursor.fetchall()]

    async def update_session_checkpoint_status(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
        *,
        index_status: str | None = None,
        enrichment_status: str | None = None,
    ) -> SessionCheckpoint | None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [utc_now_iso()]
        if index_status is not None:
            assignments.append("index_status = ?")
            values.append(index_status)
        if enrichment_status is not None:
            assignments.append("enrichment_status = ?")
            values.append(enrichment_status)
        values.extend([checkpoint_id, workspace_id, session_id])
        await self._connection.execute(
            f"UPDATE session_checkpoints SET {', '.join(assignments)} WHERE id = ? AND workspace_id = ? AND session_id = ?",
            values,
        )
        await self._connection.commit()
        return await self.get_session_checkpoint(workspace_id, session_id, checkpoint_id)

    @staticmethod
    def _row_to_checkpoint(row: aiosqlite.Row) -> SessionCheckpoint:
        return SessionCheckpoint(
            id=row["id"],
            workspace_id=row["workspace_id"],
            session_id=row["session_id"],
            raw_memory_id=row["raw_memory_id"],
            source_kind=row["source_kind"],
            source_sequence=row["source_sequence"],
            source_boundary=row["source_boundary"],
            content_hash=row["content_hash"],
            byte_count=row["byte_count"],
            capture_status=CheckpointCaptureStatus(row["capture_status"]),
            index_status=CheckpointWorkStatus(row["index_status"]),
            enrichment_status=CheckpointWorkStatus(row["enrichment_status"]),
            idempotency_key=row["idempotency_key"],
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    async def upsert_entity_relation(
        self,
        relation: EntityRelation,
        evidence: EntityRelationEvidence,
    ) -> tuple[EntityRelation, bool]:
        if evidence.workspace_id != relation.workspace_id:
            raise ValueError("relation evidence must use the edge workspace")
        cursor = await self._connection.execute(
            "SELECT id FROM entities WHERE workspace_id = ? AND id IN (?, ?) AND status = 'active'",
            (relation.workspace_id, relation.source_entity_id, relation.target_entity_id),
        )
        if len(await cursor.fetchall()) != 2:
            raise ValueError("relation endpoints must be active entities in the same workspace")
        cursor = await self._connection.execute(
            "SELECT 1 FROM memories WHERE workspace_id = ? AND id = ? AND deleted_at IS NULL",
            (relation.workspace_id, evidence.source_memory_id),
        )
        if await cursor.fetchone() is None:
            raise ValueError("relation evidence memory must be active in the same workspace")
        cursor = await self._connection.execute(
            """
            SELECT * FROM entity_relations
            WHERE workspace_id = ? AND source_entity_id = ? AND target_entity_id = ?
              AND relationship = ? AND active = 1
            """,
            (
                relation.workspace_id,
                relation.source_entity_id,
                relation.target_entity_id,
                relation.relationship,
            ),
        )
        row = await cursor.fetchone()
        if row:
            relation = self._row_to_entity_relation(row)
        else:
            await self._connection.execute(
                """
                INSERT INTO entity_relations (
                    id, workspace_id, source_entity_id, target_entity_id,
                    relationship, direction, confidence, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    relation.id,
                    relation.workspace_id,
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation.relationship,
                    relation.direction,
                    relation.confidence,
                    to_utc_iso(relation.created_at),
                    to_utc_iso(relation.updated_at),
                ),
            )
        evidence = evidence.model_copy(update={"relation_id": relation.id})
        inserted = await self._connection.execute(
            """
            INSERT OR IGNORE INTO entity_relation_evidence (
                id, workspace_id, relation_id, source_memory_id, evidence_kind,
                source_span_start, source_span_end, excerpt_hash, confidence,
                extraction_method, active, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                evidence.id,
                evidence.workspace_id,
                evidence.relation_id,
                evidence.source_memory_id,
                evidence.evidence_kind.value,
                evidence.source_span_start,
                evidence.source_span_end,
                evidence.excerpt_hash,
                evidence.confidence,
                evidence.extraction_method,
                to_utc_iso(evidence.created_at),
            ),
        )
        await self._connection.execute(
            """
            UPDATE entity_relations
            SET confidence = COALESCE((
                    SELECT AVG(confidence) FROM entity_relation_evidence
                    WHERE relation_id = ? AND active = 1
                ), confidence),
                updated_at = ?
            WHERE id = ? AND workspace_id = ?
            """,
            (relation.id, utc_now_iso(), relation.id, relation.workspace_id),
        )
        await self._connection.commit()
        cursor = await self._connection.execute(
            "SELECT * FROM entity_relations WHERE id = ? AND workspace_id = ?",
            (relation.id, relation.workspace_id),
        )
        return self._row_to_entity_relation(await cursor.fetchone()), inserted.rowcount == 0

    async def traverse_entity_relations(
        self,
        workspace_id: str,
        seed_entity_ids: list[str],
        *,
        relationships: list[str] | None,
        direction: str,
        max_hops: int,
        max_edges: int,
    ) -> list[EntityRelationPath]:
        if not seed_entity_ids or max_edges <= 0:
            return []
        frontier = list(dict.fromkeys(seed_entity_ids))
        paths: list[tuple[str, list[str], list[EntityRelation]]] = [(seed, [seed], []) for seed in frontier]
        completed: list[tuple[str, list[str], list[EntityRelation]]] = []
        remaining = max_edges
        for _hop in range(min(max_hops, 2)):
            if not frontier or remaining <= 0:
                break
            placeholders = ",".join("?" for _ in frontier)
            conditions = []
            params: list[Any] = [workspace_id]
            if direction in ("outgoing", "both"):
                conditions.append(f"source_entity_id IN ({placeholders})")
                params.extend(frontier)
            if direction in ("incoming", "both"):
                conditions.append(f"target_entity_id IN ({placeholders})")
                params.extend(frontier)
            rel_clause = ""
            if relationships:
                rel_placeholders = ",".join("?" for _ in relationships)
                rel_clause = f" AND relationship IN ({rel_placeholders})"
                params.extend(relationships)
            params.append(remaining)
            cursor = await self._connection.execute(
                f"SELECT * FROM entity_relations WHERE workspace_id = ? AND active = 1 "
                f"AND ({' OR '.join(conditions)}){rel_clause} ORDER BY id LIMIT ?",
                params,
            )
            edges = [self._row_to_entity_relation(row) for row in await cursor.fetchall()]
            remaining -= len(edges)
            next_paths: list[tuple[str, list[str], list[EntityRelation]]] = []
            next_frontier: list[str] = []
            for seed, nodes, path_edges in paths:
                current = nodes[-1]
                for edge in edges:
                    if edge.source_entity_id == current:
                        target = edge.target_entity_id
                    elif edge.target_entity_id == current and direction in ("incoming", "both"):
                        target = edge.source_entity_id
                    else:
                        continue
                    if target in nodes:
                        continue
                    candidate = (seed, [*nodes, target], [*path_edges, edge])
                    completed.append(candidate)
                    next_paths.append(candidate)
                    next_frontier.append(target)
            paths = next_paths
            frontier = list(dict.fromkeys(next_frontier))
        relation_ids = [edge.id for _seed, _nodes, edges in completed for edge in edges]
        evidence_by_relation: dict[str, list[tuple[str, str]]] = {}
        if relation_ids:
            placeholders = ",".join("?" for _ in relation_ids)
            cursor = await self._connection.execute(
                f"SELECT id, relation_id, source_memory_id FROM entity_relation_evidence "
                f"WHERE workspace_id = ? AND active = 1 AND relation_id IN ({placeholders}) "
                "ORDER BY relation_id, id",
                (workspace_id, *relation_ids),
            )
            for row in await cursor.fetchall():
                evidence_by_relation.setdefault(row["relation_id"], []).append((row["id"], row["source_memory_id"]))
        results: list[EntityRelationPath] = []
        for seed, nodes, edges in completed:
            evidence = [item for edge in edges for item in evidence_by_relation.get(edge.id, [])]
            if not evidence:
                continue
            results.append(
                EntityRelationPath(
                    seed_entity_id=seed,
                    entity_ids=nodes,
                    relations=edges,
                    evidence_ids=list(dict.fromkeys(item[0] for item in evidence)),
                    evidence_memory_ids=list(dict.fromkeys(item[1] for item in evidence)),
                )
            )
        return results

    async def _deactivate_relation_evidence_sql(self, workspace_id: str, memory_id: str) -> int:
        cursor = await self._connection.execute(
            "UPDATE entity_relation_evidence SET active = 0 WHERE workspace_id = ? AND source_memory_id = ? AND active = 1",
            (workspace_id, memory_id),
        )
        await self._connection.execute(
            """
            UPDATE entity_relations
            SET active = CASE WHEN EXISTS (
                    SELECT 1 FROM entity_relation_evidence e
                    WHERE e.relation_id = entity_relations.id AND e.active = 1
                ) THEN 1 ELSE 0 END,
                confidence = COALESCE((
                    SELECT AVG(e.confidence) FROM entity_relation_evidence e
                    WHERE e.relation_id = entity_relations.id AND e.active = 1
                ), confidence),
                updated_at = ?
            WHERE workspace_id = ?
            """,
            (utc_now_iso(), workspace_id),
        )
        return cursor.rowcount

    async def deactivate_relation_evidence_for_memory(self, workspace_id: str, memory_id: str) -> int:
        count = await self._deactivate_relation_evidence_sql(workspace_id, memory_id)
        await self._connection.commit()
        return count

    async def reconcile_entity_relations_after_merge(
        self,
        workspace_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> int:
        cursor = await self._connection.execute(
            "SELECT * FROM entity_relations WHERE workspace_id = ? AND active = 1 "
            "AND (source_entity_id = ? OR target_entity_id = ?) ORDER BY id",
            (workspace_id, source_entity_id, source_entity_id),
        )
        rows = await cursor.fetchall()
        changed = 0
        try:
            for row in rows:
                new_source = target_entity_id if row["source_entity_id"] == source_entity_id else row["source_entity_id"]
                new_target = target_entity_id if row["target_entity_id"] == source_entity_id else row["target_entity_id"]
                if new_source == new_target:
                    await self._connection.execute(
                        "UPDATE entity_relation_evidence SET active = 0 WHERE relation_id = ?",
                        (row["id"],),
                    )
                    await self._connection.execute(
                        "UPDATE entity_relations SET active = 0, updated_at = ? WHERE id = ?",
                        (utc_now_iso(), row["id"]),
                    )
                    changed += 1
                    continue
                collision_cursor = await self._connection.execute(
                    "SELECT id FROM entity_relations WHERE workspace_id = ? "
                    "AND source_entity_id = ? AND target_entity_id = ? "
                    "AND relationship = ? AND active = 1 AND id != ?",
                    (workspace_id, new_source, new_target, row["relationship"], row["id"]),
                )
                collision = await collision_cursor.fetchone()
                if collision:
                    await self._connection.execute(
                        "UPDATE OR IGNORE entity_relation_evidence SET relation_id = ? WHERE relation_id = ?",
                        (collision["id"], row["id"]),
                    )
                    await self._connection.execute(
                        "DELETE FROM entity_relations WHERE id = ? AND workspace_id = ?",
                        (row["id"], workspace_id),
                    )
                    relation_id = collision["id"]
                else:
                    await self._connection.execute(
                        "UPDATE entity_relations SET source_entity_id = ?, target_entity_id = ?, updated_at = ? "
                        "WHERE id = ? AND workspace_id = ?",
                        (new_source, new_target, utc_now_iso(), row["id"], workspace_id),
                    )
                    relation_id = row["id"]
                await self._connection.execute(
                    "UPDATE entity_relations SET confidence = COALESCE(("
                    "SELECT AVG(confidence) FROM entity_relation_evidence "
                    "WHERE relation_id = ? AND active = 1), confidence), updated_at = ? "
                    "WHERE id = ? AND workspace_id = ?",
                    (relation_id, utc_now_iso(), relation_id, workspace_id),
                )
                changed += 1
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return changed

    @staticmethod
    def _row_to_entity_relation(row: aiosqlite.Row) -> EntityRelation:
        return EntityRelation(
            id=row["id"],
            workspace_id=row["workspace_id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            relationship=row["relationship"],
            direction=row["direction"],
            confidence=row["confidence"],
            active=bool(row["active"]),
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
        )

    # Contradiction operations
    async def create_contradiction(self, contradiction: ContradictionRecord) -> ContradictionRecord:
        """Store a contradiction record."""
        await self._connection.execute(
            """
            INSERT INTO contradictions (id, workspace_id, memory_a_id, memory_b_id,
                                        contradiction_type, confidence, detection_method,
                                        detected_at, resolved_at, resolution, merged_content,
                                        newer_memory_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                contradiction.newer_memory_id,
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

    async def get_superseded_memory_ids(self, workspace_id: str, memory_ids: list[str]) -> set[str]:
        """Indexed form of the base scan — see ``StorageBackend.get_superseded_memory_ids``."""
        if not memory_ids:
            return set()
        placeholders = ",".join("?" * len(memory_ids))
        cursor = await self._connection.execute(
            f"""
            SELECT memory_a_id, memory_b_id, newer_memory_id
              FROM contradictions
             WHERE workspace_id = ?
               AND resolved_at IS NULL
               AND newer_memory_id IS NOT NULL
               AND (memory_a_id IN ({placeholders}) OR memory_b_id IN ({placeholders}))
            """,
            (workspace_id, *memory_ids, *memory_ids),
        )
        rows = await cursor.fetchall()
        wanted = set(memory_ids)
        superseded: set[str] = set()
        for row in rows:
            newer = row["newer_memory_id"]
            for candidate in (row["memory_a_id"], row["memory_b_id"]):
                if candidate in wanted and candidate != newer:
                    superseded.add(candidate)
        return superseded

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
            # Guarded: a database created before this column existed still round-trips,
            # it just reports no supersession direction.
            newer_memory_id=row["newer_memory_id"] if "newer_memory_id" in row.keys() else None,
        )

    # Helper methods
    def _row_to_memory(self, row: aiosqlite.Row, *, with_embedding: bool = True) -> Memory:
        """Convert database row to Memory domain model.

        Args:
            row: Result row from a ``memories`` query.
            with_embedding: Whether to decode the stored vector. Bulk read
                paths (search, timeline, decay, document listing) pass
                ``False``: ranking has already happened in SQL and the API
                strips ``embedding`` before serializing, so decoding it only
                burns memory. The blob is ~7.7 KB at dim 1920 but becomes
                ~61 KB as a Python ``list[float]`` — a float object is 24
                bytes plus 8 for the list slot — so a large candidate pool
                cost megabytes per request for a field nobody read. Paths
                whose callers *do* read ``Memory.embedding`` (``get_memory``,
                ``get_recent_memories`` at ``detail_level="full"``) keep the
                default.
        """
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        memory = Memory(
            id=row["id"],
            logical_key=row["logical_key"] if "logical_key" in row.keys() else None,
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
            metadata=metadata,
            refinement_metadata=(
                json.loads(row["refinement_metadata"]) if "refinement_metadata" in row.keys() and row["refinement_metadata"] else {}
            ),
            embedding=(self._deserialize_embedding(row["embedding"]) if with_embedding and row["embedding"] else None),
            abstract=row["abstract"] if "abstract" in row.keys() and row["abstract"] else None,
            overview=row["overview"] if "overview" in row.keys() and row["overview"] else None,
            access_count=row["access_count"],
            last_accessed_at=parse_datetime_utc(row["last_accessed_at"]),
            decay_factor=row["decay_factor"],
            status=MemoryStatus(row["status"]) if "status" in row.keys() and row["status"] else MemoryStatus.ACTIVE,
            pinned=bool(row["pinned"]) if "pinned" in row.keys() and row["pinned"] is not None else False,
            revision=row["revision"] if "revision" in row.keys() else 0,
            etag=row["etag"] if "etag" in row.keys() else "",
            event_time=parse_datetime_utc(row["event_time"]) if "event_time" in row.keys() and row["event_time"] else None,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
            deleted_at=(parse_datetime_utc(row["deleted_at"]) if "deleted_at" in row.keys() and row["deleted_at"] else None),
        )
        if metadata.get("source") == "session_checkpoint":
            memory = memory.model_copy(update={"content": row["content"]})
        return memory

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
            tags=json.loads(row["tags"]) if ("tags" in row.keys() and row["tags"]) else [],
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

    async def _resolve_thread_row_id(self, workspace_id: str, client_id: str, user_id: str | None) -> str | None:
        """Resolve the opaque surrogate ``row_id`` for an owner-scoped thread.

        SECURITY: threads are identified by (workspace_id, COALESCE(user_id,''),
        id), so a shared client id like "_default" in the _user_chat sentinel
        resolves to the correct owner's thread. Returns None when no such thread
        exists. row_id is INTERNAL — the only value chat_messages.thread_id references.
        """
        cursor = await self._connection.execute(
            "SELECT row_id FROM chat_threads WHERE workspace_id = ? AND id = ? AND COALESCE(user_id, '') = ?",
            (workspace_id, client_id, user_id or ""),
        )
        row = await cursor.fetchone()
        return row["row_id"] if row else None

    async def create_thread(self, thread: "ChatThread") -> "ChatThread":
        row_id = generate_id("cthr")
        await self._connection.execute(
            """INSERT INTO chat_threads
               (row_id, id, workspace_id, tenant_id, user_id, context_id,
                observer_id, subject_id, title, metadata,
                message_count, last_decomposed_at, last_decomposed_index,
                expires_at, created_at, updated_at, scope, ownership,
                idle_action, hidden_at, parent_thread)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row_id,
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
                thread.ownership,
                thread.idle_action,
                thread.hidden_at.isoformat() if thread.hidden_at else None,
                thread.parent_thread,
            ),
        )
        await self._connection.commit()
        return thread

    async def get_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> "ChatThread | None":
        cursor = await self._connection.execute(
            "SELECT * FROM chat_threads WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (thread_id, workspace_id, user_id or ""),
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
        ownership_filter: str | None = None,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list:
        now = utc_now_iso()
        conditions = ["workspace_id = ?", "(expires_at IS NULL OR expires_at > ?)"]
        params: list = [workspace_id, now]

        if not include_hidden:
            conditions.append("hidden_at IS NULL")

        # Default: top-level only (parent_thread IS NULL). A value lists children.
        if parent_thread is None:
            conditions.append("parent_thread IS NULL")
        else:
            conditions.append("parent_thread = ?")
            params.append(parent_thread)

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

        if ownership_filter is not None:
            conditions.append("ownership = ?")
            params.append(ownership_filter)

        where = " AND ".join(conditions)
        params.extend([limit, offset])
        cursor = await self._connection.execute(
            f"SELECT * FROM chat_threads WHERE {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def list_user_threads(
        self,
        tenant_id: str,
        user_id: str,
        ownership: str = "user",
        scope_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_hidden: bool = False,
        parent_thread: str | None = None,
    ) -> list:
        """List threads owned by a user across all workspaces.

        Filters by (tenant_id, user_id, ownership) — does NOT constrain workspace_id.
        Used for the user-session-scoped right rail.
        """
        now = utc_now_iso()
        conditions = [
            "tenant_id = ?",
            "user_id = ?",
            "ownership = ?",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        params: list = [tenant_id, user_id, ownership, now]

        if not include_hidden:
            conditions.append("hidden_at IS NULL")

        if parent_thread is None:
            conditions.append("parent_thread IS NULL")
        else:
            conditions.append("parent_thread = ?")
            params.append(parent_thread)

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

    async def update_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None, **updates) -> "ChatThread | None":
        if not updates:
            return await self.get_thread(workspace_id, thread_id, user_id=user_id)

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

        values.extend([thread_id, workspace_id, user_id or ""])
        sql = f"UPDATE chat_threads SET {', '.join(set_clauses)} WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?"
        await self._connection.execute(sql, values)
        await self._connection.commit()
        return await self.get_thread(workspace_id, thread_id, user_id=user_id)

    async def delete_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> bool:
        # Resolve the owner-scoped thread to its surrogate row_id; messages key on
        # row_id, and children share the same owner.
        row_id = await self._resolve_thread_row_id(workspace_id, thread_id, user_id)
        if row_id is None:
            return False

        # Cascade: delete sub-threads (children of the same owner) first so
        # deleting a parent never leaves orphaned children. Recurses for depth.
        child_cursor = await self._connection.execute(
            "SELECT id FROM chat_threads WHERE parent_thread = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (thread_id, workspace_id, user_id or ""),
        )
        child_ids = [r["id"] for r in await child_cursor.fetchall()]
        for child_id in child_ids:
            await self.delete_thread(workspace_id, child_id, user_id=user_id)

        # Delete messages first (FK cascade may handle this, but be explicit).
        # Messages reference the surrogate row_id.
        await self._connection.execute(
            "DELETE FROM chat_messages WHERE thread_id = ? AND workspace_id = ?",
            (row_id, workspace_id),
        )
        cursor = await self._connection.execute(
            "DELETE FROM chat_threads WHERE row_id = ?",
            (row_id,),
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

    async def list_idle_threads(
        self,
        updated_before: datetime,
        *,
        idle_action: str,
        only_hidden: bool | None = None,
        limit: int = 100,
    ) -> list["ChatThread"]:
        conditions = ["idle_action = ?", "updated_at < ?"]
        params: list = [idle_action, updated_before.isoformat()]
        if only_hidden is True:
            conditions.append("hidden_at IS NOT NULL")
        elif only_hidden is False:
            conditions.append("hidden_at IS NULL")
        where = " AND ".join(conditions)
        params.append(limit)
        cursor = await self._connection.execute(
            f"SELECT * FROM chat_threads WHERE {where} ORDER BY updated_at ASC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def list_hidden_threads(self, hidden_before: datetime, limit: int = 100) -> list["ChatThread"]:
        cursor = await self._connection.execute(
            """
            SELECT * FROM chat_threads
            WHERE hidden_at IS NOT NULL AND hidden_at < ?
            ORDER BY hidden_at ASC
            LIMIT ?
            """,
            (hidden_before.isoformat(), limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_chat_thread(row) for row in rows]

    async def hide_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> "ChatThread | None":
        # Set hidden_at WITHOUT bumping updated_at (hiding is not activity).
        await self._connection.execute(
            "UPDATE chat_threads SET hidden_at = ? WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (utc_now_iso(), thread_id, workspace_id, user_id or ""),
        )
        await self._connection.commit()
        return await self.get_thread(workspace_id, thread_id, user_id=user_id)

    async def unhide_thread(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> "ChatThread | None":
        await self._connection.execute(
            "UPDATE chat_threads SET hidden_at = NULL WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (thread_id, workspace_id, user_id or ""),
        )
        await self._connection.commit()
        return await self.get_thread(workspace_id, thread_id, user_id=user_id)

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        messages: list,
        user_id: str | None = None,
    ) -> list:
        from ...models.chat import ChatMessage

        # Resolve the owner-scoped thread to its surrogate row_id + current count.
        # Messages reference row_id, and the response thread_id echoes the client id.
        cursor = await self._connection.execute(
            "SELECT row_id, message_count FROM chat_threads WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (thread_id, workspace_id, user_id or ""),
        )
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Thread {thread_id} not found in workspace {workspace_id}")

        row_id = row["row_id"]
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
                    row_id,
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

        # Update thread message count and updated_at; clear hidden_at so a new
        # message revives (un-archives) a hidden thread.
        new_count = current_count + len(messages)
        await self._connection.execute(
            "UPDATE chat_threads SET message_count = ?, updated_at = ?, hidden_at = NULL WHERE row_id = ?",
            (new_count, now, row_id),
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
        user_id: str | None = None,
    ) -> list:
        order_clause = "ASC" if order.lower() == "asc" else "DESC"

        # Resolve the owner-scoped thread to its surrogate row_id (messages key on it).
        row_id = await self._resolve_thread_row_id(workspace_id, thread_id, user_id)
        if row_id is None:
            return []

        if after_index is not None:
            cursor = await self._connection.execute(
                f"""SELECT * FROM chat_messages
                    WHERE thread_id = ? AND workspace_id = ? AND message_index > ?
                    ORDER BY message_index {order_clause} LIMIT ? OFFSET ?""",
                (row_id, workspace_id, after_index, limit, offset),
            )
        else:
            cursor = await self._connection.execute(
                f"""SELECT * FROM chat_messages
                    WHERE thread_id = ? AND workspace_id = ?
                    ORDER BY message_index {order_clause} LIMIT ? OFFSET ?""",
                (row_id, workspace_id, limit, offset),
            )

        rows = await cursor.fetchall()
        return [self._row_to_chat_message(row, thread_id) for row in rows]

    async def get_message_count(self, workspace_id: str, thread_id: str, user_id: str | None = None) -> int:
        cursor = await self._connection.execute(
            "SELECT message_count FROM chat_threads WHERE id = ? AND workspace_id = ? AND COALESCE(user_id, '') = ?",
            (thread_id, workspace_id, user_id or ""),
        )
        row = await cursor.fetchone()
        return row["message_count"] if row else 0

    async def delete_message(self, workspace_id: str, thread_id: str, message_id: str, user_id: str | None = None) -> bool:
        row_id = await self._resolve_thread_row_id(workspace_id, thread_id, user_id)
        if row_id is None:
            return False
        cursor = await self._connection.execute(
            "DELETE FROM chat_messages WHERE id = ? AND thread_id = ? AND workspace_id = ?",
            (message_id, row_id, workspace_id),
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

        # ownership: column may not exist in very old databases (pre-migration); default to 'user'.
        try:
            ownership = row["ownership"] or "user"
        except (IndexError, KeyError):
            ownership = "user"

        # idle_action / hidden_at: columns may not exist in very old databases.
        try:
            idle_action = row["idle_action"]
        except (IndexError, KeyError):
            idle_action = None
        try:
            hidden_raw = row["hidden_at"]
        except (IndexError, KeyError):
            hidden_raw = None
        try:
            parent_thread = row["parent_thread"]
        except (IndexError, KeyError):
            parent_thread = None

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
            idle_action=idle_action,
            hidden_at=parse_datetime_utc(hidden_raw) if hidden_raw else None,
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
            scope=scope,
            ownership=ownership,
            parent_thread=parent_thread,
        )

    def _row_to_chat_message(self, row: aiosqlite.Row, client_thread_id: str | None = None) -> "ChatMessage":
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

        # The stored thread_id is the surrogate row_id (internal); present the
        # client id when known so the surrogate never leaves storage.
        return ChatMessage(
            id=row["id"],
            thread_id=client_thread_id if client_thread_id is not None else row["thread_id"],
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
                page_count, chunk_count, memory_ids,
                enrichment_status, enrichment_memory_ids, deduplicated_count,
                error_message, metadata, extracted_metadata,
                created_at, processing_started_at, processing_completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                doc.enrichment_status.value if hasattr(doc.enrichment_status, "value") else doc.enrichment_status,
                json.dumps(doc.enrichment_memory_ids),
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
        json_fields = {
            "extraction_options",
            "memory_ids",
            "metadata",
            "extracted_metadata",
            "enrichment_memory_ids",
        }
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
            elif key in ("status", "document_type", "enrichment_status") and hasattr(value, "value"):
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
        return [self._row_to_memory(row, with_embedding=False) for row in rows]

    async def try_claim_document(
        self,
        document_id: str,
        workspace_id: str,
        ttl_seconds: int,
    ) -> bool:
        """Atomically claim a document for (re)processing via a conditional UPDATE.

        Flips the row to ``processing`` (stamping ``processing_started_at=now``)
        only when it is not already owned by a fresh in-flight worker (status not
        ``processing``, or no ``processing_started_at``, or that timestamp older
        than ``ttl_seconds``). Returns ``True`` iff exactly one row was updated
        (the caller won the claim). Timestamps are stored as ISO-8601 UTC strings,
        so the ``< cutoff`` comparison is a well-ordered lexicographic compare.
        """
        from datetime import timedelta

        now = datetime.now(UTC)
        cutoff = (now - timedelta(seconds=ttl_seconds)).isoformat()
        cursor = await self._connection.execute(
            """
            UPDATE documents
            SET status = 'processing', processing_started_at = ?
            WHERE id = ? AND workspace_id = ?
              AND (status != 'processing'
                   OR processing_started_at IS NULL
                   OR processing_started_at < ?)
            """,
            (now.isoformat(), document_id, workspace_id, cutoff),
        )
        await self._connection.commit()
        return cursor.rowcount == 1

    def _row_to_document(self, row: aiosqlite.Row) -> "Document":
        """Convert database row to Document domain model."""
        from ...models.document import (
            Document,
            DocumentEnrichmentStatus,
            DocumentExtractionOptions,
            DocumentStatus,
            DocumentType,
        )

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
            # Guarded by key presence: a row read through a connection opened
            # before the ALTER ran has no such column.
            enrichment_status=(
                DocumentEnrichmentStatus(row["enrichment_status"])
                if "enrichment_status" in row.keys() and row["enrichment_status"]
                else DocumentEnrichmentStatus.NOT_APPLICABLE
            ),
            enrichment_memory_ids=(
                json.loads(row["enrichment_memory_ids"]) if "enrichment_memory_ids" in row.keys() and row["enrichment_memory_ids"] else []
            ),
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

    async def delete_pages(self, document_id: str, workspace_id: str | None = None) -> int:
        """Delete all pages for a document, returning the row count."""
        if workspace_id is not None:
            cursor = await self._connection.execute(
                "DELETE FROM document_pages WHERE document_id = ? AND workspace_id = ?",
                (document_id, workspace_id),
            )
        else:
            cursor = await self._connection.execute(
                "DELETE FROM document_pages WHERE document_id = ?",
                (document_id,),
            )
        await self._connection.commit()
        return cursor.rowcount

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

    async def list_active_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] = ("queued", "running"),
    ) -> list["IngestionJob"]:
        """Return non-terminal jobs overlapping ``document_ids``."""
        if not statuses:
            return []
        return await self.list_jobs_for_documents(document_ids, statuses=statuses)

    async def list_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list["IngestionJob"]:
        """Return jobs overlapping ``document_ids``, newest first.

        ``statuses=None`` means ANY status — what an operator inspecting a single
        document wants, since the failed and superseded attempts are usually the
        informative ones.

        SQLite stores ``document_ids`` as a JSON array in a TEXT column, so the
        overlap is evaluated in Python after narrowing by status. ``limit`` is
        therefore applied AFTER the overlap filter, not in SQL: a LIMIT on the
        status-narrowed query would count non-overlapping jobs toward it and
        silently return fewer than asked for.
        """
        if not document_ids:
            return []
        wanted = set(document_ids)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql = f"SELECT * FROM ingestion_jobs WHERE status IN ({placeholders}) ORDER BY created_at DESC"
            params = tuple(statuses)
        else:
            sql = "SELECT * FROM ingestion_jobs ORDER BY created_at DESC"
            params = ()
        cursor = await self._connection.execute(sql, params)
        rows = await cursor.fetchall()
        matches: list[IngestionJob] = []
        for row in rows:
            job_doc_ids = json.loads(row["document_ids"]) if row["document_ids"] else []
            if wanted.intersection(job_doc_ids):
                matches.append(self._row_to_ingestion_job(row))
                if limit is not None and len(matches) >= limit:
                    break
        return matches

    async def cancel_orphaned_ingestion_jobs(self) -> int:
        """Cancel queued/running jobs whose referenced documents are all completed.

        Mirrors the PostgreSQL single-statement sweep: a job is orphaned iff every
        document it references exists and is ``completed``. A job referencing a
        missing document is left untouched (its liveness is undeterminable).
        """
        cursor = await self._connection.execute("SELECT id, document_ids FROM ingestion_jobs WHERE status IN ('queued', 'running')")
        rows = await cursor.fetchall()

        orphan_ids: list[str] = []
        for row in rows:
            job_doc_ids = json.loads(row["document_ids"]) if row["document_ids"] else []
            # A job with no document_ids has undeterminable liveness — never treat
            # the empty set as "all docs completed"; leave it untouched (parity with
            # the PostgreSQL `cardinality(document_ids) > 0` guard).
            if not job_doc_ids:
                continue
            all_completed = True
            for doc_id in job_doc_ids:
                dcur = await self._connection.execute(
                    "SELECT status FROM documents WHERE id = ?",
                    (doc_id,),
                )
                drow = await dcur.fetchone()
                if drow is None or drow["status"] != "completed":
                    all_completed = False
                    break
            if all_completed:
                orphan_ids.append(row["id"])

        if not orphan_ids:
            return 0

        now = utc_now_iso()
        placeholders = ",".join("?" for _ in orphan_ids)
        await self._connection.execute(
            f"UPDATE ingestion_jobs SET status = 'cancelled', completed_at = ? WHERE id IN ({placeholders})",
            (now, *orphan_ids),
        )
        await self._connection.commit()
        return len(orphan_ids)

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

    async def delete_kb_article(self, workspace_id: str, article_id: str) -> bool:
        """Delete a single knowledgebase article by id (for stale-article GC)."""
        cursor = await self._connection.execute(
            "DELETE FROM knowledgebase_articles WHERE workspace_id = ? AND article_id = ?",
            (workspace_id, article_id),
        )
        await self._connection.commit()
        deleted = (cursor.rowcount or 0) > 0
        self.logger.debug("Deleted KB article %s in workspace %s: %s", article_id, workspace_id, deleted)
        return deleted

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
    # Entity Registry Operations
    # ============================================

    def _row_to_entity(self, row: aiosqlite.Row, aliases: list[str]) -> dict:
        """Convert an entities row + its aliases into the entity dict contract."""
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "entity_type": row["entity_type"],
            "canonical_name": row["canonical_name"],
            "normalized_name": row["normalized_name"],
            "aliases": aliases,
            "confidence": row["confidence"],
            "provenance": json.loads(row["provenance"]) if row["provenance"] else {},
            "representative_memory_id": row["representative_memory_id"],
            "status": row["status"],
            "merged_into": row["merged_into"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def _entity_aliases(self, workspace_id: str, entity_id: str) -> list[str]:
        """Return an entity's alias surface forms (deterministic order)."""
        cursor = await self._connection.execute(
            "SELECT alias FROM entity_aliases WHERE workspace_id = ? AND entity_id = ? ORDER BY alias",
            (workspace_id, entity_id),
        )
        rows = await cursor.fetchall()
        return [r["alias"] for r in rows]

    async def store_entity(self, entity: dict) -> dict:
        """Insert a canonical entity row (and its initial aliases).

        Handles the create-race: two concurrent callers that both miss the
        exact-match check can both attempt the INSERT. The partial unique index
        ``uq_entities_active_norm`` on ``(workspace_id, entity_type,
        normalized_name) WHERE status='active'`` will reject the loser with an
        IntegrityError. On collision we re-fetch and return the winner's row so
        the caller (and any subsequent ``add_member``) see a valid entity.
        Provenance follows first-writer-wins; the race loser's provenance is
        silently discarded.
        """
        now = utc_now_iso()
        entity_id = entity.get("id") or generate_id("ent")
        workspace_id = entity["workspace_id"]
        entity_type = entity["entity_type"]
        try:
            await self._connection.execute(
                """
                INSERT INTO entities
                    (id, workspace_id, entity_type, canonical_name, normalized_name,
                     confidence, provenance, representative_memory_id, status, merged_into,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    workspace_id,
                    entity_type,
                    entity["canonical_name"],
                    entity["normalized_name"],
                    entity.get("confidence", 1.0),
                    json.dumps(entity.get("provenance") or {}),
                    entity.get("representative_memory_id"),
                    entity.get("status", "active"),
                    entity.get("merged_into"),
                    entity.get("created_at") or now,
                    entity.get("updated_at") or now,
                ),
            )
            for alias in entity.get("aliases") or []:
                await self.add_entity_alias(workspace_id, entity_id, alias, self._normalize_alias(alias), source="initial")
            await self._connection.commit()
            self.logger.debug("Stored entity %s in workspace %s", entity_id, workspace_id)
            return await self.get_entity(workspace_id, entity_id)
        except aiosqlite.IntegrityError:
            # Partial-unique-index collision: a concurrent writer won the race.
            # Re-fetch the active winner row so the caller gets a valid entity.
            self.logger.debug(
                "store_entity race on (%s, %s, %s) — re-fetching winner",
                workspace_id,
                entity_type,
                entity["normalized_name"],
            )
            existing = await self.find_entity_by_normalized_name(workspace_id, entity_type, entity["normalized_name"])
            if existing is not None:
                return existing
            # Extremely unlikely (winner was merged/deleted between our INSERT
            # failure and this re-fetch). Surface the original error.
            raise

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        # Local import keeps the storage layer free of a service-package import
        # at module load time; the normalize function is the parity contract.
        from ..entity_registry import normalize_entity_name

        return normalize_entity_name(alias)

    async def get_entity(self, workspace_id: str, entity_id: str) -> dict | None:
        """Get a canonical entity by id (with aliases folded in)."""
        cursor = await self._connection.execute(
            "SELECT * FROM entities WHERE workspace_id = ? AND id = ?",
            (workspace_id, entity_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        aliases = await self._entity_aliases(workspace_id, entity_id)
        return self._row_to_entity(row, aliases)

    async def find_entity_by_normalized_name(
        self,
        workspace_id: str,
        entity_type: str,
        normalized_name: str,
    ) -> dict | None:
        """Find the active entity matching (workspace, type, normalized_name)."""
        cursor = await self._connection.execute(
            "SELECT * FROM entities WHERE workspace_id = ? AND entity_type = ? AND normalized_name = ? AND status = 'active'",
            (workspace_id, entity_type, normalized_name),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        aliases = await self._entity_aliases(workspace_id, row["id"])
        return self._row_to_entity(row, aliases)

    async def find_entities_by_normalized_name_any_type(
        self,
        workspace_id: str,
        normalized_name: str,
    ) -> list[dict]:
        """Find ALL active entities matching (workspace, normalized_name), any type.

        Ordered PERSON-first (so the accretion name-first resolver can prefer an
        existing PERSON node), then by entity id for a deterministic tie-break.
        """
        cursor = await self._connection.execute(
            "SELECT * FROM entities WHERE workspace_id = ? AND normalized_name = ? "
            "AND status = 'active' "
            "ORDER BY CASE WHEN entity_type = 'person' THEN 0 ELSE 1 END, id",
            (workspace_id, normalized_name),
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            aliases = await self._entity_aliases(workspace_id, row["id"])
            result.append(self._row_to_entity(row, aliases))
        return result

    async def find_entities_by_normalized_alias(
        self,
        workspace_id: str,
        normalized_alias: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Find active entities whose aliases include normalized_alias."""
        params: list = [workspace_id, normalized_alias]
        query = (
            "SELECT DISTINCT e.* FROM entities e "
            "JOIN entity_aliases a ON a.entity_id = e.id AND a.workspace_id = e.workspace_id "
            "WHERE e.workspace_id = ? AND a.normalized_alias = ? AND e.status = 'active'"
        )
        if entity_type is not None:
            query += " AND e.entity_type = ?"
            params.append(entity_type)
        query += " ORDER BY e.id"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            aliases = await self._entity_aliases(workspace_id, row["id"])
            result.append(self._row_to_entity(row, aliases))
        return result

    async def add_entity_alias(
        self,
        workspace_id: str,
        entity_id: str,
        alias: str,
        normalized_alias: str,
        source: str = "manual",
    ) -> None:
        """Add an alias to an entity (idempotent on (entity_id, normalized_alias))."""
        await self._connection.execute(
            """
            INSERT INTO entity_aliases
                (id, workspace_id, entity_id, alias, normalized_alias, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id, normalized_alias) DO NOTHING
            """,
            (
                generate_id("ealias"),
                workspace_id,
                entity_id,
                alias,
                normalized_alias,
                source,
                utc_now_iso(),
            ),
        )
        await self._connection.commit()

    async def add_entity_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        role: str = "mention",
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> dict:
        """Attach a memory to an entity as a member (idempotent on (entity_id, memory_id, role))."""
        await self._connection.execute(
            """
            INSERT INTO entity_members
                (id, workspace_id, entity_id, memory_id, role, confidence, meta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(entity_id, memory_id, role) DO NOTHING
            """,
            (
                generate_id("emem"),
                workspace_id,
                entity_id,
                memory_id,
                role,
                confidence,
                json.dumps(meta or {}),
                utc_now_iso(),
            ),
        )
        await self._connection.commit()
        return {
            "entity_id": entity_id,
            "memory_id": memory_id,
            "role": role,
            "confidence": confidence,
        }

    async def list_entity_members(
        self,
        workspace_id: str,
        entity_id: str,
        role: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List member rows for an entity, optionally filtered by role."""
        params: list = [workspace_id, entity_id]
        query = "SELECT * FROM entity_members WHERE workspace_id = ? AND entity_id = ?"
        if role is not None:
            query += " AND role = ?"
            params.append(role)
        query += " ORDER BY created_at, memory_id LIMIT ?"
        params.append(limit)
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "memory_id": r["memory_id"],
                "role": r["role"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    async def list_workspace_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 10000,
    ) -> list[dict]:
        """List ALL canonical entities for a workspace, ordered by id."""
        params: list = [workspace_id]
        query = "SELECT * FROM entities WHERE workspace_id = ?"
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        out = []
        for row in rows:
            aliases = await self._entity_aliases(workspace_id, row["id"])
            out.append(self._row_to_entity(row, aliases))
        return out

    async def list_workspace_entity_members(
        self,
        workspace_id: str,
        *,
        role: str | None = None,
        limit: int = 100000,
    ) -> list[dict]:
        """List ALL member edges for a workspace, ordered by (entity_id, memory_id, role)."""
        params: list = [workspace_id]
        query = "SELECT * FROM entity_members WHERE workspace_id = ?"
        if role is not None:
            query += " AND role = ?"
            params.append(role)
        query += " ORDER BY entity_id, memory_id, role LIMIT ?"
        params.append(limit)
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [
            {
                "entity_id": r["entity_id"],
                "memory_id": r["memory_id"],
                "role": r["role"],
                "confidence": r["confidence"],
            }
            for r in rows
        ]

    async def reassign_entity_members(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> int:
        """Reassign all members from source_id to target_id (for merge).

        Members already present on the target for the same (memory_id, role)
        are dropped from the source first to honor the UNIQUE constraint.
        Returns the number of member rows moved.
        """
        # Drop source members that would collide with an existing target member.
        await self._connection.execute(
            """
            DELETE FROM entity_members
            WHERE workspace_id = ? AND entity_id = ?
              AND EXISTS (
                SELECT 1 FROM entity_members t
                WHERE t.workspace_id = entity_members.workspace_id
                  AND t.entity_id = ?
                  AND t.memory_id = entity_members.memory_id
                  AND t.role = entity_members.role
              )
            """,
            (workspace_id, source_id, target_id),
        )
        cursor = await self._connection.execute(
            "UPDATE entity_members SET entity_id = ? WHERE workspace_id = ? AND entity_id = ?",
            (target_id, workspace_id, source_id),
        )
        await self._connection.commit()
        return cursor.rowcount or 0

    async def update_entity(
        self,
        workspace_id: str,
        entity_id: str,
        **updates,
    ) -> dict | None:
        """Update mutable entity fields (status/merged_into/confidence/...)."""
        allowed = {
            "entity_type",
            "canonical_name",
            "normalized_name",
            "confidence",
            "provenance",
            "representative_memory_id",
            "status",
            "merged_into",
        }
        set_parts = []
        params: list = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            set_parts.append(f"{key} = ?")
            params.append(json.dumps(value) if key == "provenance" else value)
        if not set_parts:
            return await self.get_entity(workspace_id, entity_id)
        set_parts.append("updated_at = ?")
        params.append(utc_now_iso())
        params.extend([workspace_id, entity_id])
        await self._connection.execute(
            f"UPDATE entities SET {', '.join(set_parts)} WHERE workspace_id = ? AND id = ?",
            params,
        )
        await self._connection.commit()
        return await self.get_entity(workspace_id, entity_id)

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
        result = await self.mutate_skill(
            SkillMutation(
                action="create",
                skill=skill,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "create", "state": manifest_state(skill)}),
                expected_etag="*",
            )
        )
        self.logger.debug("Created skill: %s in workspace: %s", skill.id, skill.workspace_id)
        return result.skill

    async def get_skill(
        self,
        workspace_id: str,
        skill_id: str,
        *,
        include_deleted: bool = False,
    ) -> "Skill | None":
        """Get skill by ID within a workspace."""
        deleted_clause = "" if include_deleted else " AND deleted_at IS NULL"
        cursor = await self._connection.execute(
            f"SELECT * FROM skills WHERE id = ? AND workspace_id = ?{deleted_clause}",
            (skill_id, workspace_id),
        )
        row = await cursor.fetchone()
        return self._row_to_skill(row) if row else None

    async def get_skill_by_name(self, workspace_id: str, name: str, user_id: str | None = None) -> "Skill | None":
        """Get skill by name within a workspace, optionally scoped to a user."""
        if user_id is not None:
            cursor = await self._connection.execute(
                "SELECT * FROM skills WHERE workspace_id = ? AND name = ? AND user_id = ? AND deleted_at IS NULL",
                (workspace_id, name, user_id),
            )
        else:
            cursor = await self._connection.execute(
                "SELECT * FROM skills WHERE workspace_id = ? AND name = ? AND user_id IS NULL AND deleted_at IS NULL",
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
        include_global: bool = False,
    ) -> list["Skill"]:
        """List skills for a workspace with optional filters.

        When ``include_global`` is set, tenant-shared ``_global`` skills
        (``user_id IS NULL``) are unioned into the result.
        """
        from ...config import GLOBAL_WORKSPACE_ID

        where_parts: list[str] = ["deleted_at IS NULL"]
        params: list = []

        # Workspace scope: the current workspace, plus optionally the shared
        # _global workspace (global skills are never user-scoped).
        if include_global and user_id is None and workspace_id != GLOBAL_WORKSPACE_ID:
            where_parts.append("(workspace_id = ? OR (workspace_id = ? AND user_id IS NULL))")
            params.extend([workspace_id, GLOBAL_WORKSPACE_ID])
        else:
            where_parts.append("workspace_id = ?")
            params.append(workspace_id)

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

        where_clause = f"name = ? AND deleted_at IS NULL AND ({' OR '.join(conditions)})"
        query = f"SELECT * FROM skills WHERE {where_clause} ORDER BY updated_at DESC"
        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_skill(row) for row in rows]

    async def update_skill(self, workspace_id: str, skill_id: str, updates: dict) -> "Skill | None":
        """Update skill fields."""
        if not updates:
            return await self.get_skill(workspace_id, skill_id)
        current = await self.get_skill(workspace_id, skill_id)
        if current is None:
            return None
        desired = current.model_copy(update={key: value for key, value in updates.items() if hasattr(current, key)})
        desired = desired.model_copy(update={"updated_at": datetime.now(UTC)})
        semantic_updates = set(updates) - {"bundle_hash", "updated_at"}
        if not semantic_updates:
            await self._connection.execute(
                "UPDATE skills SET bundle_hash = ?, updated_at = ? WHERE id = ? AND workspace_id = ? AND deleted_at IS NULL",
                (desired.bundle_hash, to_utc_iso(desired.updated_at), skill_id, workspace_id),
            )
            await self._connection.commit()
            return await self.get_skill(workspace_id, skill_id)
        result = await self.mutate_skill(
            SkillMutation(
                action="replace",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "replace", "state": manifest_state(desired), "expected_etag": current.etag}),
                expected_etag=current.etag,
            )
        )
        return result.skill

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Write a durable tombstone; bundle files remain available for restore."""
        current = await self.get_skill(workspace_id, skill_id)
        if current is None:
            return False
        now = datetime.now(UTC)
        desired = current.model_copy(update={"updated_at": now, "deleted_at": now})
        await self.mutate_skill(
            SkillMutation(
                action="delete",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "delete", "id": skill_id, "expected_etag": current.etag}),
                expected_etag=current.etag,
            )
        )
        self.logger.debug("Deleted skill: %s", skill_id)
        return True

    async def mutate_skill(self, mutation: SkillMutation) -> SkillMutationResult:
        """Atomically mutate the native skill head and immutable sidecars."""

        desired = mutation.skill.model_copy(deep=True)
        async with self._skill_mutation_lock:
            replay = await self.get_skill_operation(
                desired.tenant_id,
                desired.workspace_id,
                mutation.operation_id,
                mutation.request_hash,
            )
            if replay is not None:
                return replay

            current = await self.get_skill(
                desired.workspace_id,
                desired.id,
                include_deleted=True,
            )
            if mutation.action == "create":
                if mutation.expected_etag != "*":
                    raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
                if current is not None:
                    raise VersionedResourceConflictError("skill id already exists")
                final = desired.model_copy(update={"revision": 1, "deleted_at": None})
            else:
                if current is None or current.tenant_id != desired.tenant_id:
                    raise VersionedResourceNotFoundError("skill not found")
                if mutation.expected_etag != current.etag:
                    raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
                if mutation.action == "restore":
                    if current.deleted_at is None:
                        raise VersionedResourceConflictError("skill is not deleted")
                elif current.deleted_at is not None:
                    raise VersionedResourceNotFoundError("skill not found")
                if desired.name != current.name or desired.user_id != current.user_id:
                    raise VersionedResourceConflictError("skill name and ownership are immutable")
                final = desired.model_copy(update={"created_at": current.created_at, "revision": current.revision + 1})
            final = final.model_copy(update={"etag": manifest_etag(final.revision, final)})

            try:
                if mutation.action == "create":
                    await self._insert_skill_head(final)
                else:
                    deleted_predicate = "deleted_at IS NOT NULL" if mutation.action == "restore" else "deleted_at IS NULL"
                    cursor = await self._update_skill_head(
                        final,
                        current.etag,
                        deleted_predicate,
                    )
                    if cursor.rowcount == 0:
                        await self._connection.rollback()
                        replay = await self.get_skill_operation(
                            desired.tenant_id,
                            desired.workspace_id,
                            mutation.operation_id,
                            mutation.request_hash,
                        )
                        if replay is not None:
                            return replay
                        raise VersionedResourcePreconditionFailedError("ETag does not match current revision")

                await self._connection.execute(
                    """
                    INSERT INTO skill_revisions (
                        tenant_id, workspace_id, skill_id, revision, snapshot,
                        action, operation_id, request_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final.tenant_id,
                        final.workspace_id,
                        final.id,
                        final.revision,
                        _skill_snapshot_json(final),
                        mutation.action,
                        mutation.operation_id,
                        mutation.request_hash,
                    ),
                )
                await self._connection.execute(
                    """
                    INSERT INTO skill_operations (
                        tenant_id, workspace_id, operation_id, request_hash,
                        skill_id, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        final.tenant_id,
                        final.workspace_id,
                        mutation.operation_id,
                        mutation.request_hash,
                        final.id,
                        final.revision,
                    ),
                )
                await self._connection.commit()
            except Exception as exc:
                await self._connection.rollback()
                replay = await self.get_skill_operation(
                    desired.tenant_id,
                    desired.workspace_id,
                    mutation.operation_id,
                    mutation.request_hash,
                )
                if replay is not None:
                    return replay
                latest = await self.get_skill(
                    desired.workspace_id,
                    desired.id,
                    include_deleted=True,
                )
                if mutation.action == "create" and latest is not None:
                    raise VersionedResourceConflictError("skill id already exists")
                if mutation.action == "create" and isinstance(exc, (sqlite3.IntegrityError, aiosqlite.IntegrityError)):
                    raise VersionedResourceConflictError("skill id or scoped name already exists") from exc
                if mutation.action != "create" and latest is not None and latest.etag != mutation.expected_etag:
                    raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
                raise
            return SkillMutationResult(skill=final)

    async def get_skill_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> SkillMutationResult | None:
        cursor = await self._connection.execute(
            """
            SELECT r.* FROM skill_operations o
            JOIN skill_revisions r
              ON r.tenant_id = o.tenant_id AND r.workspace_id = o.workspace_id
             AND r.skill_id = o.skill_id AND r.revision = o.revision
            WHERE o.tenant_id = ? AND o.workspace_id = ? AND o.operation_id = ?
            """,
            (tenant_id, workspace_id, operation_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise VersionedResourceConflictError("idempotency key was already used for a different request")
        return SkillMutationResult(skill=_skill_snapshot(row["snapshot"]), replayed=True)

    async def list_skill_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        skill_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[SkillRevision]:
        clauses = ["tenant_id = ?", "workspace_id = ?", "skill_id = ?"]
        params: list[Any] = [tenant_id, workspace_id, skill_id]
        if before_sequence is not None:
            clauses.append("sequence < ?")
            params.append(before_sequence)
        params.append(limit)
        cursor = await self._connection.execute(
            f"SELECT * FROM skill_revisions WHERE {' AND '.join(clauses)} ORDER BY sequence DESC LIMIT ?",
            tuple(params),
        )
        return [_skill_revision_from_row(row) for row in await cursor.fetchall()]

    async def _insert_skill_head(self, skill: Skill) -> None:
        await self._connection.execute(
            """
            INSERT INTO skills (id, tenant_id, workspace_id, user_id, name, description, version,
                license, compatibility, allowed_tools, body, metadata, source_mode,
                manifest_hash, bundle_hash, enabled, revision, etag, created_at, updated_at, deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _skill_row_values(skill),
        )

    async def _update_skill_head(self, skill: Skill, current_etag: str, deleted_predicate: str):
        return await self._connection.execute(
            f"""
            UPDATE skills SET
                description = ?, version = ?, license = ?, compatibility = ?,
                allowed_tools = ?, body = ?, metadata = ?, source_mode = ?,
                manifest_hash = ?, bundle_hash = ?, enabled = ?, revision = ?,
                etag = ?, updated_at = ?, deleted_at = ?
            WHERE id = ? AND tenant_id = ? AND workspace_id = ? AND etag = ?
              AND {deleted_predicate}
            """,
            (
                skill.description,
                skill.version,
                skill.license,
                skill.compatibility,
                skill.allowed_tools,
                skill.body,
                json.dumps(skill.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                skill.source_mode,
                skill.manifest_hash,
                skill.bundle_hash,
                1 if skill.enabled else 0,
                skill.revision,
                skill.etag,
                to_utc_iso(skill.updated_at),
                to_utc_iso(skill.deleted_at),
                skill.id,
                skill.tenant_id,
                skill.workspace_id,
                current_etag,
            ),
        )

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
            revision=row["revision"] if "revision" in row.keys() else 0,
            etag=row["etag"] if "etag" in row.keys() else "",
            created_at=parse_datetime_utc(row["created_at"]),
            updated_at=parse_datetime_utc(row["updated_at"]),
            deleted_at=parse_datetime_utc(row["deleted_at"]) if "deleted_at" in row.keys() else None,
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

    # ============================================
    # Internal Versioned Resource Operations
    # ============================================

    def _versioned_store(self) -> RelationalVersionedResourceStore:
        if self._versioned_resource_store is None:
            raise RuntimeError("SQLite storage is not connected")
        return self._versioned_resource_store

    async def mutate_versioned_resource(
        self,
        mutation: VersionedResourceMutation,
    ) -> VersionedResourceMutationResult:
        return await self._versioned_store().mutate(mutation)

    async def get_versioned_resource_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        return await self._versioned_store().get_operation_result(tenant_id, workspace_id, namespace, operation_id, request_hash)

    async def get_versioned_resource(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        return await self._versioned_store().get(tenant_id, workspace_id, namespace, resource_id, include_deleted=include_deleted)

    async def get_versioned_resource_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        return await self._versioned_store().get_by_key(tenant_id, workspace_id, namespace, resource_key, include_deleted=include_deleted)

    async def list_versioned_resources(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        before_sequence: int | None = None,
        include_deleted: bool = False,
    ) -> list[VersionedResource]:
        return await self._versioned_store().list(
            tenant_id,
            workspace_id,
            namespace,
            limit=limit,
            before_sequence=before_sequence,
            include_deleted=include_deleted,
        )

    async def list_versioned_resource_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[VersionedResourceRevision]:
        return await self._versioned_store().list_revisions(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            limit=limit,
            before_sequence=before_sequence,
        )


def _memory_snapshot_json(memory: Memory) -> str:
    return json.dumps(
        memory_revision_snapshot(memory).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _memory_snapshot(payload: str) -> Memory:
    return Memory.model_validate(json.loads(payload))


def _memory_revision_from_row(row: aiosqlite.Row) -> MemoryRevision:
    return MemoryRevision(
        memory=_memory_snapshot(row["snapshot"]),
        sequence=row["sequence"],
        action=row["action"],
        operation_id=row["operation_id"],
        request_hash=row["request_hash"],
    )


def _memory_row_values(backend: SQLiteStorageBackend, memory: Memory) -> tuple:
    return (
        memory.id,
        memory.logical_key,
        memory.tenant_id,
        memory.workspace_id,
        memory.context_id,
        memory.session_id,
        memory.user_id,
        memory.content,
        memory.content_hash,
        memory.type.value,
        memory.subtype,
        memory.category,
        memory.importance,
        json.dumps(memory.tags, sort_keys=True, separators=(",", ":")),
        json.dumps(memory.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        json.dumps(
            memory.refinement_metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        backend._serialize_embedding(memory.embedding) if memory.embedding else None,
        memory.abstract,
        memory.overview,
        memory.source_memory_id,
        memory.status.value,
        1 if memory.pinned else 0,
        memory.observer_id,
        memory.subject_id,
        memory.source_document_id,
        memory.source_page_id,
        memory.source_dataset_id,
        memory.source_thread_id,
        memory.access_count,
        to_utc_iso(memory.last_accessed_at),
        memory.decay_factor,
        memory.revision,
        memory.etag,
        to_utc_iso(memory.deleted_at),
        to_utc_iso(memory.event_time),
        to_utc_iso(memory.created_at),
        to_utc_iso(memory.updated_at),
    )


def _skill_snapshot_json(skill: Skill) -> str:
    return json.dumps(
        skill.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _skill_snapshot(payload: str) -> Skill:
    return Skill.model_validate(json.loads(payload))


def _skill_revision_from_row(row: aiosqlite.Row) -> SkillRevision:
    return SkillRevision(
        skill=_skill_snapshot(row["snapshot"]),
        sequence=row["sequence"],
        action=row["action"],
        operation_id=row["operation_id"],
        request_hash=row["request_hash"],
    )


def _skill_row_values(skill: Skill) -> tuple:
    return (
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
        json.dumps(skill.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        skill.source_mode,
        skill.manifest_hash,
        skill.bundle_hash,
        1 if skill.enabled else 0,
        skill.revision,
        skill.etag,
        to_utc_iso(skill.created_at),
        to_utc_iso(skill.updated_at),
        to_utc_iso(skill.deleted_at),
    )


class SqliteStorageBackendPlugin(StoragePluginBase):
    PROVIDER_NAME = "sqlite"

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        return SQLiteStorageBackend(
            db_path=v.environ(MEMORYLAYER_SQLITE_STORAGE_PATH, default=DEFAULT_MEMORYLAYER_SQLITE_STORAGE_PATH), v=v
        )
