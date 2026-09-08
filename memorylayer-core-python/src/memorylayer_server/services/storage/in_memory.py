"""
In-memory storage backend for testing.

Provides a complete storage implementation that stores all data in memory.
Data is lost on service restart - use only for testing.
"""

import asyncio
from datetime import UTC, datetime
from logging import Logger
from typing import Any

from scitrera_app_framework import Variables

from ...config import DEFAULT_CONTEXT_ID, DEFAULT_TENANT_ID
from ...models.association import AssociateInput, Association, GraphPath, GraphQueryResult
from ...models.mcp_server import McpServer
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
from ...models.skill import Skill, SkillFile, SkillMutation, SkillMutationResult, SkillRevision
from ...models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from ...models.workspace import Context, Workspace
from ...utils import compute_content_hash, cosine_similarity, generate_id, parse_datetime_utc, utc_now_iso
from ..memory.versioning import (
    SEMANTIC_MEMORY_FIELDS,
    memory_etag,
    memory_revision_snapshot,
    memory_semantic_state,
)
from ..skills.versioning import canonical_hash, manifest_etag, manifest_state
from .base import StorageBackend, StoragePluginBase


class MemoryStorageBackend(StorageBackend):
    """
    In-memory storage backend for testing.

    All data is stored in dictionaries and lost on restart.
    Supports vector similarity search using cosine similarity.
    """

    def __init__(self, v: Variables = None):
        super().__init__(v)
        # Storage containers
        self._workspaces: dict[str, Workspace] = {}
        self._contexts: dict[str, dict[str, Context]] = {}  # workspace_id -> {context_id -> Context}
        self._memories: dict[str, dict[str, Memory]] = {}  # workspace_id -> {memory_id -> Memory}
        self._deleted_memories: set[str] = set()  # Track deleted memory IDs (soft delete)
        self._memory_revisions: dict[tuple[str, str, str], list[MemoryRevision]] = {}
        self._memory_operations: dict[tuple[str, str, str], MemoryRevision] = {}
        self._memory_sequence = 0
        self._memory_mutation_lock = asyncio.Lock()
        self._associations: dict[str, dict[str, Association]] = {}  # workspace_id -> {assoc_id -> Association}
        self._sessions: dict[str, dict[str, Session]] = {}  # workspace_id -> {session_id -> Session}
        self._working_memory: dict[str, dict[str, dict[str, WorkingMemory]]] = {}  # ws -> {sess -> {key -> WM}}
        self._skills: dict[str, dict[str, Skill]] = {}  # workspace_id -> {skill_id -> Skill}
        self._skill_files: dict[str, dict[str, SkillFile]] = {}  # skill_id -> {path -> SkillFile}
        self._skill_revisions: dict[tuple[str, str, str], list[SkillRevision]] = {}
        self._skill_operations: dict[tuple[str, str, str], SkillRevision] = {}
        self._skill_sequence = 0
        self._skill_mutation_lock = asyncio.Lock()
        self._mcp_servers: dict[str, dict[str, McpServer]] = {}  # workspace_id -> {server_id -> McpServer}
        self._versioned_resource_heads: dict[tuple[str, str, str, str], VersionedResource] = {}
        self._versioned_resource_keys: dict[tuple[str, str, str, str], str] = {}
        self._versioned_resource_revisions: dict[tuple[str, str, str, str], list[VersionedResourceRevision]] = {}
        self._versioned_resource_operations: dict[tuple[str, str, str, str], VersionedResourceRevision] = {}
        self._versioned_resource_sequence = 0
        # Entity registry (entity registry slice 1): canonical entity dicts +
        # alias rows + member rows, keyed by workspace.
        self._entities: dict[str, dict[str, dict]] = {}  # workspace_id -> {entity_id -> entity dict}
        self._entity_aliases: dict[str, list[dict]] = {}  # workspace_id -> [alias row dicts]
        self._entity_members: dict[str, list[dict]] = {}  # workspace_id -> [member row dicts]
        self.logger.info("Initialized MemoryStorageBackend")

    async def connect(self) -> None:
        """Initialize storage (no-op for in-memory)."""
        self.logger.info("In-memory storage connected")

    async def disconnect(self) -> None:
        """Close storage (no-op for in-memory)."""
        self.logger.info("In-memory storage disconnected")

    async def health_check(self) -> bool:
        """Always healthy."""
        return True

    # ========== Memory Operations ==========

    async def create_memory(self, workspace_id: str, input: RememberInput) -> Memory:
        """Store a new memory."""

        if workspace_id not in self._memories:
            self._memories[workspace_id] = {}

        now = utc_now_iso()
        content_hash = compute_content_hash(input.content)

        memory = Memory(
            id=generate_id("mem"),
            logical_key=input.logical_key,
            workspace_id=workspace_id,
            tenant_id=getattr(input, "tenant_id", None) or DEFAULT_TENANT_ID,
            context_id=getattr(input, "context_id", None) or DEFAULT_CONTEXT_ID,
            user_id=input.user_id,
            observer_id=getattr(input, "observer_id", None),
            subject_id=getattr(input, "subject_id", None),
            source_document_id=getattr(input, "source_document_id", None),
            source_page_id=getattr(input, "source_page_id", None),
            source_dataset_id=getattr(input, "source_dataset_id", None),
            source_thread_id=getattr(input, "source_thread_id", None),
            content=input.content,
            content_hash=content_hash,
            type=input.type or MemoryType.SEMANTIC,
            subtype=input.subtype,
            importance=input.importance if input.importance is not None else 0.5,
            tags=input.tags or [],
            metadata=input.metadata or {},
            refinement_metadata=input.refinement_metadata or {},
            embedding=None,  # Added via update_memory() later
            access_count=0,
            pinned=input.pinned,
            event_time=getattr(input, "event_time", None),
            created_at=now,
            updated_at=now,
        )
        result = await self.mutate_memory(
            MemoryMutation(
                action="create",
                memory=memory,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "create", "state": memory_semantic_state(memory)}
                ),
                expected_etag="*",
            )
        )
        self.logger.debug("Created memory: %s in workspace: %s", memory.id, workspace_id)
        return result.memory

    async def get_memory(
        self,
        workspace_id: str,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID within a workspace."""
        ws_memories = self._memories.get(workspace_id, {})
        memory = ws_memories.get(memory_id)
        if memory and memory_id in self._deleted_memories and not include_deleted:
            return None
        return memory.model_copy(deep=True) if memory else None

    async def get_memory_by_id(
        self,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        if memory_id in self._deleted_memories and not include_deleted:
            return None
        for ws_memories in self._memories.values():
            if memory_id in ws_memories:
                return ws_memories[memory_id].model_copy(deep=True)
        return None

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
        """Update memory fields."""
        memory = await self.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            return None

        valid_updates = {key: value for key, value in updates.items() if hasattr(memory, key)}
        desired = memory.model_copy(
            update={**valid_updates, "updated_at": datetime.now(UTC)}
        )
        if SEMANTIC_MEMORY_FIELDS.intersection(valid_updates):
            result = await self.mutate_memory(
                MemoryMutation(
                    action="replace",
                    memory=desired,
                    operation_id=generate_id("op"),
                    request_hash=canonical_hash(
                        {
                            "action": "replace",
                            "state": memory_semantic_state(desired),
                            "expected_etag": memory.etag,
                        }
                    ),
                    expected_etag=memory.etag,
                )
            )
            return result.memory

        self._memories[workspace_id][memory_id] = desired.model_copy(deep=True)
        return desired

    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        memory = await self.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            return False

        if hard:
            del self._memories[workspace_id][memory_id]
            self._deleted_memories.discard(memory_id)
            revision_key = (memory.tenant_id, workspace_id, memory_id)
            self._memory_revisions.pop(revision_key, None)
            for operation_key, revision in list(self._memory_operations.items()):
                if revision.memory.id == memory_id:
                    del self._memory_operations[operation_key]
        else:
            now = datetime.now(UTC)
            desired = memory.model_copy(update={"deleted_at": now, "updated_at": now})
            await self.mutate_memory(
                MemoryMutation(
                    action="delete",
                    memory=desired,
                    operation_id=generate_id("op"),
                    request_hash=canonical_hash(
                        {"action": "delete", "id": memory_id, "expected_etag": memory.etag}
                    ),
                    expected_etag=memory.etag,
                )
            )
        return True

    async def mutate_memory(self, mutation: MemoryMutation) -> MemoryMutationResult:
        """Atomically mutate a native memory head and its immutable sidecars."""

        desired = mutation.memory.model_copy(deep=True)
        operation_key = (desired.tenant_id, desired.workspace_id, mutation.operation_id)
        async with self._memory_mutation_lock:
            if replay := self._memory_operations.get(operation_key):
                if replay.request_hash != mutation.request_hash:
                    raise VersionedResourceConflictError(
                        "idempotency key was already used for a different request"
                    )
                return MemoryMutationResult(
                    memory=replay.memory.model_copy(deep=True), replayed=True
                )

            current = self._memories.get(desired.workspace_id, {}).get(desired.id)
            if mutation.action == "create":
                if mutation.expected_etag != "*":
                    raise VersionedResourcePreconditionFailedError(
                        "create requires If-None-Match: *"
                    )
                if current is not None:
                    raise VersionedResourceConflictError("memory id already exists")
                if desired.logical_key and any(
                    memory.tenant_id == desired.tenant_id
                    and memory.user_id == desired.user_id
                    and memory.logical_key == desired.logical_key
                    for memory in self._memories.get(desired.workspace_id, {}).values()
                ):
                    raise VersionedResourceConflictError(
                        "scoped memory logical_key already exists"
                    )
                final = desired.model_copy(
                    update={"revision": 1, "deleted_at": None}
                )
            else:
                if current is None or current.tenant_id != desired.tenant_id:
                    raise VersionedResourceNotFoundError("memory not found")
                if mutation.expected_etag != current.etag:
                    raise VersionedResourcePreconditionFailedError(
                        "ETag does not match current revision"
                    )
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
                    raise VersionedResourceConflictError(
                        "memory logical key and ownership are immutable"
                    )
                final = desired.model_copy(
                    update={
                        "created_at": current.created_at,
                        "revision": current.revision + 1,
                    }
                )

            final = final.model_copy(update={"etag": memory_etag(final.revision, final)})
            self._memory_sequence += 1
            snapshot = MemoryRevision(
                memory=memory_revision_snapshot(final),
                sequence=self._memory_sequence,
                action=mutation.action,
                operation_id=mutation.operation_id,
                request_hash=mutation.request_hash,
            )
            self._memories.setdefault(final.workspace_id, {})[final.id] = final.model_copy(
                deep=True
            )
            if final.deleted_at is None:
                self._deleted_memories.discard(final.id)
            else:
                self._deleted_memories.add(final.id)
            revision_key = (final.tenant_id, final.workspace_id, final.id)
            self._memory_revisions.setdefault(revision_key, []).append(
                snapshot.model_copy(deep=True)
            )
            self._memory_operations[operation_key] = snapshot.model_copy(deep=True)
            return MemoryMutationResult(memory=final.model_copy(deep=True))

    async def get_memory_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> MemoryMutationResult | None:
        revision = self._memory_operations.get((tenant_id, workspace_id, operation_id))
        if revision is None:
            return None
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError(
                "idempotency key was already used for a different request"
            )
        return MemoryMutationResult(memory=revision.memory.model_copy(deep=True), replayed=True)

    async def list_memory_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[MemoryRevision]:
        revisions = self._memory_revisions.get((tenant_id, workspace_id, memory_id), [])
        if before_sequence is not None:
            revisions = [item for item in revisions if item.sequence < before_sequence]
        return [item.model_copy(deep=True) for item in reversed(revisions[-limit:])]

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
        """Vector similarity search using cosine similarity."""
        ws_memories = self._memories.get(workspace_id, {})
        results = []

        for memory in ws_memories.values():
            # Skip deleted memories
            if memory.id in self._deleted_memories:
                continue

            # Filter by type
            if types and memory.memory_type.value not in types:
                continue

            # Filter by subtype
            if subtypes and (not memory.memory_subtype or memory.memory_subtype.value not in subtypes):
                continue

            # Filter by tags
            if tags:
                memory_tags = memory.tags or []
                if not any(t in memory_tags for t in tags):
                    continue

            # Filter by entity attribution
            if observer_id is not None and getattr(memory, "observer_id", None) != observer_id:
                continue
            if subject_id is not None and getattr(memory, "subject_id", None) != subject_id:
                continue
            if user_id is not None and getattr(memory, "user_id", None) != user_id:
                continue

            # Calculate cosine similarity
            if memory.embedding:
                similarity = cosine_similarity(query_embedding, memory.embedding)
                if similarity >= min_relevance:
                    results.append((memory, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Apply offset and limit
        return results[offset : offset + limit]

    async def full_text_search(
        self,
        workspace_id: str,
        query: str,
        limit: int = 10,
        offset: int = 0,
        context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search on memory content."""
        ws_memories = self._memories.get(workspace_id, {})
        query_lower = query.lower()
        results = []

        for memory in ws_memories.values():
            if memory.id in self._deleted_memories:
                continue
            if context_id is not None and getattr(memory, "context_id", None) != context_id:
                continue
            if query_lower in memory.content.lower():
                results.append(memory)

        return results[offset : offset + limit]

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
        """Return memories ordered by effective event time (event_time or created_at)."""

        def effective(memory: Memory) -> datetime:
            et = memory.event_time or memory.created_at
            return et.replace(tzinfo=UTC) if et.tzinfo is None else et

        after_dt = parse_datetime_utc(event_after)
        before_dt = parse_datetime_utc(event_before)

        ws_memories = self._memories.get(workspace_id, {})
        results: list[Memory] = []
        for memory in ws_memories.values():
            if memory.id in self._deleted_memories:
                continue
            if not include_archived and memory.status != MemoryStatus.ACTIVE:
                continue
            if types and memory.type.value not in types:
                continue
            eff = effective(memory)
            if after_dt is not None and eff < after_dt:
                continue
            if before_dt is not None and eff > before_dt:
                continue
            results.append(memory)

        results.sort(key=effective, reverse=not ascending)
        return results[offset : offset + limit]

    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Memory | None:
        """Get memory by content hash for deduplication."""
        ws_memories = self._memories.get(workspace_id, {})
        for memory in ws_memories.values():
            if memory.content_hash == content_hash and memory.id not in self._deleted_memories:
                return memory
        return None

    async def get_recent_memories(
        self,
        workspace_id: str,
        created_after: datetime,
        limit: int = 10,
        detail_level: str = "abstract",
        offset: int = 0,
    ) -> list:
        """Get recent memories ordered by creation time (newest first)."""
        ws_memories = self._memories.get(workspace_id, {})
        candidates = []

        for memory in ws_memories.values():
            # Filter: workspace_id matches, created_at > created_after, not deleted, status active
            if memory.id in self._deleted_memories:
                continue
            # Check status
            status = getattr(memory, "status", None)
            if status and str(status) != "active":
                continue
            if memory.created_at > created_after:
                candidates.append(memory)

        # Sort by created_at descending (newest first)
        candidates.sort(key=lambda m: m.created_at, reverse=True)

        # Apply offset and limit
        if limit > 0:
            candidates = candidates[offset : offset + limit]
        else:
            candidates = candidates[offset:]

        # Convert to dicts based on detail_level
        results = []
        for memory in candidates:
            if detail_level == "abstract":
                # Return only id, abstract, type, subtype, importance, tags, created_at
                results.append(
                    {
                        "id": memory.id,
                        "abstract": getattr(memory, "abstract", None),
                        "type": memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                        "subtype": memory.subtype.value
                        if memory.subtype and hasattr(memory.subtype, "value")
                        else str(memory.subtype)
                        if memory.subtype
                        else None,
                        "importance": memory.importance,
                        "tags": memory.tags if memory.tags else [],
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    }
                )
            elif detail_level == "overview":
                # Add overview field
                results.append(
                    {
                        "id": memory.id,
                        "abstract": getattr(memory, "abstract", None),
                        "overview": getattr(memory, "overview", None),
                        "type": memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                        "subtype": memory.subtype.value
                        if memory.subtype and hasattr(memory.subtype, "value")
                        else str(memory.subtype)
                        if memory.subtype
                        else None,
                        "importance": memory.importance,
                        "tags": memory.tags if memory.tags else [],
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    }
                )
            else:  # "full"
                # Return everything
                results.append(
                    {
                        "id": memory.id,
                        "content": memory.content,
                        "abstract": getattr(memory, "abstract", None),
                        "overview": getattr(memory, "overview", None),
                        "type": memory.type.value if hasattr(memory.type, "value") else str(memory.type),
                        "subtype": memory.subtype.value
                        if memory.subtype and hasattr(memory.subtype, "value")
                        else str(memory.subtype)
                        if memory.subtype
                        else None,
                        "importance": memory.importance,
                        "tags": memory.tags if memory.tags else [],
                        "created_at": memory.created_at.isoformat() if memory.created_at else None,
                    }
                )

        return results

    # ========== Association Operations ==========

    async def create_association(self, workspace_id: str, input: AssociateInput) -> Association:
        """Create graph edge between memories."""
        if workspace_id not in self._associations:
            self._associations[workspace_id] = {}

        now = utc_now_iso()
        assoc = Association(
            id=generate_id("assoc"),
            source_id=input.source_id,
            target_id=input.target_id,
            relationship=input.relationship,
            strength=input.strength or 1.0,
            metadata=input.metadata or {},
            created_at=now,
        )
        self._associations[workspace_id][assoc.id] = assoc
        return assoc

    async def get_associations(
        self,
        workspace_id: str,
        memory_id: str,
        direction: str = "both",
        relationships: list[str] | None = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        ws_assocs = self._associations.get(workspace_id, {})
        results = []

        for assoc in ws_assocs.values():
            # Check direction
            if direction == "outgoing" and assoc.source_id != memory_id:
                continue
            if direction == "incoming" and assoc.target_id != memory_id:
                continue
            if direction == "both" and assoc.source_id != memory_id and assoc.target_id != memory_id:
                continue

            # Check relationship type
            if relationships:
                if assoc.relationship not in relationships:
                    continue

            results.append(assoc)

        return results

    async def count_associations_by_relationship(self, workspace_id: str) -> dict[str, int]:
        """Per-relationship edge counts (in-memory GROUP BY equivalent)."""
        counts: dict[str, int] = {}
        for assoc in self._associations.get(workspace_id, {}).values():
            counts[assoc.relationship] = counts.get(assoc.relationship, 0) + 1
        return counts

    async def traverse_graph(
        self,
        workspace_id: str,
        start_id: str,
        max_depth: int = 3,
        relationships: list[str] | None = None,
        direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop graph traversal."""
        visited = set()
        paths = []
        unique_nodes = set()

        async def traverse(current_id: str, path: list[str], depth: int):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)
            unique_nodes.add(current_id)
            current_path = path + [current_id]

            if len(current_path) > 1:
                paths.append(GraphPath(nodes=current_path, depth=depth))

            if depth < max_depth:
                associations = await self.get_associations(workspace_id, current_id, direction, relationships)
                for assoc in associations:
                    next_id = assoc.target_id if assoc.source_id == current_id else assoc.source_id
                    await traverse(next_id, current_path, depth + 1)

        await traverse(start_id, [], 0)

        return GraphQueryResult(
            paths=paths,
            unique_nodes=list(unique_nodes),
            total_paths=len(paths),
        )

    # ========== Workspace Operations ==========

    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace."""
        self._workspaces[workspace.id] = workspace
        return workspace

    async def get_workspace(self, workspace_id: str) -> Workspace | None:
        """Get workspace by ID."""
        return self._workspaces.get(workspace_id)

    async def list_workspaces(
        self,
        *,
        tags: list[str] | None = None,
        match: str = "all",
    ) -> list[Workspace]:
        """List workspaces, optionally filtered by tag.

        Filtering is applied in Python after load (workspace counts are small).
        """
        from ...models.workspace import normalize_tags

        workspaces = list(self._workspaces.values())

        query_tags = normalize_tags(tags)
        if not query_tags:
            return workspaces

        wanted = set(query_tags)
        if match == "any":
            return [w for w in workspaces if wanted & set(w.tags)]
        # default: 'all' — workspace must carry every requested tag
        return [w for w in workspaces if wanted <= set(w.tags)]

    async def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and every in-memory resource it owns."""
        if workspace_id not in self._workspaces:
            return False

        memory_ids = set(self._memories.get(workspace_id, {}))
        skill_ids = set(self._skills.get(workspace_id, {}))

        for container in (
            self._contexts,
            self._memories,
            self._associations,
            self._sessions,
            self._working_memory,
            self._skills,
            self._mcp_servers,
            self._entities,
            self._entity_aliases,
            self._entity_members,
        ):
            container.pop(workspace_id, None)

        self._deleted_memories.difference_update(memory_ids)
        for skill_id in skill_ids:
            self._skill_files.pop(skill_id, None)

        for container in (
            self._memory_revisions,
            self._memory_operations,
            self._skill_revisions,
            self._skill_operations,
            self._versioned_resource_heads,
            self._versioned_resource_keys,
            self._versioned_resource_revisions,
            self._versioned_resource_operations,
        ):
            for key in [key for key in container if key[1] == workspace_id]:
                del container[key]

        del self._workspaces[workspace_id]
        return True

    # ========== Context Operations ==========

    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        if workspace_id not in self._contexts:
            self._contexts[workspace_id] = {}
        self._contexts[workspace_id][context.id] = context
        return context

    async def get_context(self, workspace_id: str, context_id: str) -> Context | None:
        """Get context by ID."""
        ws_contexts = self._contexts.get(workspace_id, {})
        return ws_contexts.get(context_id)

    async def list_contexts(self, workspace_id: str) -> list[Context]:
        """List all contexts in a workspace."""
        ws_contexts = self._contexts.get(workspace_id, {})
        return list(ws_contexts.values())

    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Delete a context within a workspace.

        Returns True if the context existed and was removed, False otherwise.
        Memories keep their context_id; deleting a context does not remove its
        memories.
        """
        ws_contexts = self._contexts.get(workspace_id)
        if not ws_contexts or context_id not in ws_contexts:
            return False
        del ws_contexts[context_id]
        return True

    # ========== Statistics ==========

    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        ws_memories = self._memories.get(workspace_id, {})
        active_memories = [m for m in ws_memories.values() if m.id not in self._deleted_memories]

        return {
            "total_memories": len(active_memories),
            "total_associations": len(self._associations.get(workspace_id, {})),
            "total_categories": 0,
        }

    async def get_workspace_change_watermark(
        self, workspace_id: str
    ) -> tuple[str, str, int, int] | None:
        """Deterministic dirty-watermark over active memories + all associations.

        Matches the SQLite/PG semantics: max memory ``updated_at`` and max
        association ``created_at`` (ISO strings here) plus both counts so an
        association delete (which advances no timestamp) is still detected. The
        in-memory store cannot raise, but the contract returns None on any error
        so callers fall back to doing the work (fail-safe).
        """
        try:
            ws_memories = self._memories.get(workspace_id, {})
            active = [m for m in ws_memories.values() if m.id not in self._deleted_memories]
            max_updated = max((m.updated_at or "" for m in active), default="")

            assocs = list(self._associations.get(workspace_id, {}).values())
            max_created = max((a.created_at or "" for a in assocs), default="")

            return (str(max_updated), str(max_created), len(active), len(assocs))
        except Exception as e:  # pragma: no cover - fail-safe guard
            self.logger.debug("Could not compute change watermark for %s: %s", workspace_id, e)
            return None

    # ========== Session Operations ==========

    async def create_session(self, workspace_id: str, session: Session) -> Session:
        """Store a new session."""
        if workspace_id not in self._sessions:
            self._sessions[workspace_id] = {}
        self._sessions[workspace_id][session.id] = session
        return session

    async def get_session(self, workspace_id: str, session_id: str) -> Session | None:
        """Get session by ID."""
        ws_sessions = self._sessions.get(workspace_id, {})
        session = ws_sessions.get(session_id)
        if session and session.is_expired:
            return None
        return session

    async def get_session_by_id(self, session_id: str) -> Session | None:
        """Get session by ID without workspace filter.

        Searches all workspaces. Within a tenant's storage backend,
        session IDs are globally unique.
        """
        for ws_sessions in self._sessions.values():
            session = ws_sessions.get(session_id)
            if session:
                if session.is_expired:
                    return None
                return session
        return None

    async def delete_session(self, workspace_id: str, session_id: str) -> bool:
        """Delete session and all its context."""
        ws_sessions = self._sessions.get(workspace_id, {})
        if session_id in ws_sessions:
            del ws_sessions[session_id]
            # Also delete working memory
            ws_wm = self._working_memory.get(workspace_id, {})
            if session_id in ws_wm:
                del ws_wm[session_id]
            return True
        return False

    async def set_working_memory(
        self, workspace_id: str, session_id: str, key: str, value: Any, ttl_seconds: int | None = None
    ) -> WorkingMemory:
        """Set working memory key-value within session."""
        if workspace_id not in self._working_memory:
            self._working_memory[workspace_id] = {}
        if session_id not in self._working_memory[workspace_id]:
            self._working_memory[workspace_id][session_id] = {}

        now = datetime.now(UTC)
        existing = self._working_memory[workspace_id][session_id].get(key)

        wm = WorkingMemory(
            session_id=session_id,
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
        self._working_memory[workspace_id][session_id][key] = wm
        return wm

    async def get_working_memory(self, workspace_id: str, session_id: str, key: str) -> WorkingMemory | None:
        """Get specific working memory entry."""
        ws_wm = self._working_memory.get(workspace_id, {})
        sess_wm = ws_wm.get(session_id, {})
        return sess_wm.get(key)

    async def get_all_working_memory(self, workspace_id: str, session_id: str) -> list[WorkingMemory]:
        """Get all working memory entries for session."""
        ws_wm = self._working_memory.get(workspace_id, {})
        sess_wm = ws_wm.get(session_id, {})
        return list(sess_wm.values())

    async def cleanup_expired_sessions(self, workspace_id: str) -> int:
        """Delete all expired sessions."""
        ws_sessions = self._sessions.get(workspace_id, {})
        expired = [sid for sid, s in ws_sessions.items() if s.is_expired]
        for sid in expired:
            await self.delete_session(workspace_id, sid)
        return len(expired)

    # ========== Skill Operations ==========

    async def create_skill(self, skill: Skill) -> Skill:
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
    ) -> Skill | None:
        """Get skill by ID within a workspace."""
        skill = self._skills.get(workspace_id, {}).get(skill_id)
        if skill is None or (skill.deleted_at is not None and not include_deleted):
            return None
        return skill.model_copy(deep=True)

    async def get_skill_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: str | None = None,
    ) -> Skill | None:
        """Get skill by name within a workspace, optionally filtering by user scope."""
        for skill in self._skills.get(workspace_id, {}).values():
            if skill.deleted_at is None and skill.name == name:
                if user_id is None or skill.user_id == user_id:
                    return skill
        return None

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
    ) -> list[Skill]:
        """List skills in a workspace with optional filters.

        When ``include_global`` is set, tenant-shared ``_global`` skills
        (``user_id IS NULL``) are unioned into the result.
        """
        from ...config import GLOBAL_WORKSPACE_ID

        source_ids = [workspace_id]
        if include_global and user_id is None and workspace_id != GLOBAL_WORKSPACE_ID:
            source_ids.append(GLOBAL_WORKSPACE_ID)

        results = []
        for source_id in source_ids:
            for skill in self._skills.get(source_id, {}).values():
                if skill.deleted_at is not None:
                    continue
                if user_id is not None and skill.user_id != user_id:
                    continue
                # Global skills are tenant-shared, never user-scoped.
                if source_id == GLOBAL_WORKSPACE_ID and skill.user_id is not None:
                    continue
                if name is not None and skill.name != name:
                    continue
                if enabled is not None and skill.enabled != enabled:
                    continue
                if tags:
                    skill_tags = skill.metadata.get("tags", [])
                    if not any(t in skill_tags for t in tags):
                        continue
                results.append(skill)
        return results[offset : offset + limit]

    async def find_skills_by_name(
        self,
        name: str,
        scope_filters: list[dict],
    ) -> list[Skill]:
        """Find skills by name across multiple scopes for precedence resolution."""
        results = []
        for scope in scope_filters:
            ws_id = scope.get("workspace_id")
            uid = scope.get("user_id")
            if not ws_id:
                continue
            for skill in self._skills.get(ws_id, {}).values():
                if skill.deleted_at is not None or skill.name != name:
                    continue
                if uid is not None and skill.user_id != uid:
                    continue
                results.append(skill)
        return results

    async def update_skill(
        self,
        workspace_id: str,
        skill_id: str,
        updates: dict,
    ) -> Skill | None:
        """Update skill fields."""
        skill = await self.get_skill(workspace_id, skill_id)
        if not skill:
            return None
        desired = skill.model_copy(update={key: value for key, value in updates.items() if hasattr(skill, key)})
        desired = desired.model_copy(update={"updated_at": datetime.now(UTC)})
        semantic_updates = set(updates) - {"bundle_hash", "updated_at"}
        if not semantic_updates:
            self._skills[workspace_id][skill_id] = desired
            return desired.model_copy(deep=True)
        result = await self.mutate_skill(
            SkillMutation(
                action="replace",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "replace", "state": manifest_state(desired), "expected_etag": skill.etag}
                ),
                expected_etag=skill.etag,
            )
        )
        return result.skill

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Write a durable skill tombstone; bundle files remain available for restore."""
        skill = await self.get_skill(workspace_id, skill_id)
        if skill is None:
            return False
        desired = skill.model_copy(update={"deleted_at": datetime.now(UTC), "updated_at": datetime.now(UTC)})
        await self.mutate_skill(
            SkillMutation(
                action="delete",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "delete", "id": skill_id, "expected_etag": skill.etag}),
                expected_etag=skill.etag,
            )
        )
        return True

    async def mutate_skill(self, mutation: SkillMutation) -> SkillMutationResult:
        """Atomically mutate a native skill manifest and its immutable audit rows."""
        desired = mutation.skill.model_copy(deep=True)
        operation_key = (desired.tenant_id, desired.workspace_id, mutation.operation_id)
        async with self._skill_mutation_lock:
            if replay := self._skill_operations.get(operation_key):
                if replay.request_hash != mutation.request_hash:
                    raise VersionedResourceConflictError(
                        "idempotency key was already used for a different request"
                    )
                return SkillMutationResult(skill=replay.skill.model_copy(deep=True), replayed=True)

            current = self._skills.get(desired.workspace_id, {}).get(desired.id)
            if mutation.action == "create":
                if mutation.expected_etag != "*":
                    raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
                if current is not None:
                    raise VersionedResourceConflictError("skill id already exists")
                if any(
                    skill.tenant_id == desired.tenant_id
                    and skill.name == desired.name
                    and skill.user_id == desired.user_id
                    for skill in self._skills.get(desired.workspace_id, {}).values()
                ):
                    raise VersionedResourceConflictError("scoped skill name already exists")
                final = desired.model_copy(
                    update={"revision": 1, "deleted_at": None}
                )
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
                final = desired.model_copy(
                    update={
                        "created_at": current.created_at,
                        "revision": current.revision + 1,
                    }
                )

            final = final.model_copy(update={"etag": manifest_etag(final.revision, final)})
            self._skill_sequence += 1
            snapshot = SkillRevision(
                skill=final.model_copy(deep=True),
                sequence=self._skill_sequence,
                action=mutation.action,
                operation_id=mutation.operation_id,
                request_hash=mutation.request_hash,
            )
            self._skills.setdefault(final.workspace_id, {})[final.id] = final.model_copy(deep=True)
            revision_key = (final.tenant_id, final.workspace_id, final.id)
            self._skill_revisions.setdefault(revision_key, []).append(snapshot.model_copy(deep=True))
            self._skill_operations[operation_key] = snapshot.model_copy(deep=True)
            return SkillMutationResult(skill=final.model_copy(deep=True))

    async def get_skill_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> SkillMutationResult | None:
        revision = self._skill_operations.get((tenant_id, workspace_id, operation_id))
        if revision is None:
            return None
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError("idempotency key was already used for a different request")
        return SkillMutationResult(skill=revision.skill.model_copy(deep=True), replayed=True)

    async def list_skill_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        skill_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[SkillRevision]:
        revisions = self._skill_revisions.get((tenant_id, workspace_id, skill_id), [])
        selected = [
            revision
            for revision in reversed(revisions)
            if before_sequence is None or revision.sequence < before_sequence
        ]
        return [revision.model_copy(deep=True) for revision in selected[:limit]]

    async def upsert_skill_file(self, skill_file: SkillFile) -> SkillFile:
        """Insert or update a file within a skill bundle."""
        if skill_file.skill_id not in self._skill_files:
            self._skill_files[skill_file.skill_id] = {}
        self._skill_files[skill_file.skill_id][skill_file.path] = skill_file
        return skill_file

    async def get_skill_file(self, skill_id: str, path: str) -> SkillFile | None:
        """Get a single skill file by skill ID and relative path."""
        return self._skill_files.get(skill_id, {}).get(path)

    async def list_skill_files(self, skill_id: str) -> list[SkillFile]:
        """List all files belonging to a skill bundle."""
        return list(self._skill_files.get(skill_id, {}).values())

    async def delete_skill_file(self, skill_id: str, path: str) -> bool:
        """Delete a single skill file by path."""
        skill_files = self._skill_files.get(skill_id, {})
        if path not in skill_files:
            return False
        del skill_files[path]
        return True

    # ========== MCP Server Operations ==========

    async def create_mcp_server(self, server: McpServer) -> McpServer:
        """Store a new MCP server record."""
        if server.workspace_id not in self._mcp_servers:
            self._mcp_servers[server.workspace_id] = {}
        self._mcp_servers[server.workspace_id][server.id] = server
        self.logger.debug("Created mcp_server: %s in workspace: %s", server.id, server.workspace_id)
        return server

    async def get_mcp_server(self, workspace_id: str, server_id: str) -> McpServer | None:
        """Get MCP server by ID within a workspace."""
        return self._mcp_servers.get(workspace_id, {}).get(server_id)

    async def get_mcp_server_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: str | None = None,
    ) -> McpServer | None:
        """Get MCP server by name within a workspace, optionally filtering by user scope."""
        for server in self._mcp_servers.get(workspace_id, {}).values():
            if server.name == name:
                if user_id is None or server.user_id == user_id:
                    return server
        return None

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[McpServer]:
        """List MCP servers in a workspace with optional filters."""
        results = []
        for server in self._mcp_servers.get(workspace_id, {}).values():
            if user_id is not None and server.user_id != user_id:
                continue
            if name is not None and server.name != name:
                continue
            if transport is not None and server.transport != transport:
                continue
            if enabled is not None and server.enabled != enabled:
                continue
            results.append(server)
        return results[offset : offset + limit]

    async def find_mcp_servers_by_name(
        self,
        name: str,
        scope_filters: list[dict],
    ) -> list[McpServer]:
        """Find MCP servers by name across multiple scopes for precedence resolution."""
        results = []
        for scope in scope_filters:
            ws_id = scope.get("workspace_id")
            uid = scope.get("user_id")
            if not ws_id:
                continue
            for server in self._mcp_servers.get(ws_id, {}).values():
                if server.name != name:
                    continue
                if uid is not None and server.user_id != uid:
                    continue
                results.append(server)
        return results

    async def update_mcp_server(
        self,
        workspace_id: str,
        server_id: str,
        updates: dict,
    ) -> McpServer | None:
        """Update MCP server fields."""
        server = await self.get_mcp_server(workspace_id, server_id)
        if not server:
            return None
        for key, value in updates.items():
            if hasattr(server, key):
                setattr(server, key, value)
        server.updated_at = datetime.now(UTC)
        return server

    async def delete_mcp_server(self, workspace_id: str, server_id: str) -> bool:
        """Delete an MCP server record."""
        ws_servers = self._mcp_servers.get(workspace_id, {})
        if server_id not in ws_servers:
            return False
        del ws_servers[server_id]
        return True

    # ========== Entity Registry Operations ==========

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        from ..entity_registry import normalize_entity_name

        return normalize_entity_name(alias)

    def _entity_alias_surface(self, workspace_id: str, entity_id: str) -> list[str]:
        rows = [a for a in self._entity_aliases.get(workspace_id, []) if a["entity_id"] == entity_id]
        return sorted(a["alias"] for a in rows)

    def _hydrate_entity(self, workspace_id: str, entity: dict) -> dict:
        out = dict(entity)
        out["aliases"] = self._entity_alias_surface(workspace_id, entity["id"])
        return out

    async def store_entity(self, entity: dict) -> dict:
        """Insert a canonical entity row (and its initial aliases)."""
        workspace_id = entity["workspace_id"]
        now = utc_now_iso()
        entity_id = entity.get("id") or generate_id("ent")
        stored = {
            "id": entity_id,
            "workspace_id": workspace_id,
            "entity_type": entity["entity_type"],
            "canonical_name": entity["canonical_name"],
            "normalized_name": entity["normalized_name"],
            "confidence": entity.get("confidence", 1.0),
            "provenance": dict(entity.get("provenance") or {}),
            "representative_memory_id": entity.get("representative_memory_id"),
            "status": entity.get("status", "active"),
            "merged_into": entity.get("merged_into"),
            "created_at": entity.get("created_at") or now,
            "updated_at": entity.get("updated_at") or now,
        }
        self._entities.setdefault(workspace_id, {})[entity_id] = stored
        for alias in entity.get("aliases") or []:
            await self.add_entity_alias(
                workspace_id, entity_id, alias, self._normalize_alias(alias), source="initial"
            )
        return self._hydrate_entity(workspace_id, stored)

    async def get_entity(self, workspace_id: str, entity_id: str) -> dict | None:
        entity = self._entities.get(workspace_id, {}).get(entity_id)
        return self._hydrate_entity(workspace_id, entity) if entity else None

    async def find_entity_by_normalized_name(
        self,
        workspace_id: str,
        entity_type: str,
        normalized_name: str,
    ) -> dict | None:
        for entity in self._entities.get(workspace_id, {}).values():
            if (
                entity["status"] == "active"
                and entity["entity_type"] == entity_type
                and entity["normalized_name"] == normalized_name
            ):
                return self._hydrate_entity(workspace_id, entity)
        return None

    async def find_entities_by_normalized_name_any_type(
        self,
        workspace_id: str,
        normalized_name: str,
    ) -> list[dict]:
        matches = [
            self._hydrate_entity(workspace_id, entity)
            for entity in self._entities.get(workspace_id, {}).values()
            if entity["status"] == "active" and entity["normalized_name"] == normalized_name
        ]
        # PERSON-first, then deterministic by id (mirrors the SQLite ordering).
        matches.sort(key=lambda e: (0 if e["entity_type"] == "person" else 1, e["id"]))
        return matches

    async def find_entities_by_normalized_alias(
        self,
        workspace_id: str,
        normalized_alias: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        matched_ids = {
            a["entity_id"]
            for a in self._entity_aliases.get(workspace_id, [])
            if a["normalized_alias"] == normalized_alias
        }
        result = []
        for entity_id in matched_ids:
            entity = self._entities.get(workspace_id, {}).get(entity_id)
            if not entity or entity["status"] != "active":
                continue
            if entity_type is not None and entity["entity_type"] != entity_type:
                continue
            result.append(self._hydrate_entity(workspace_id, entity))
        result.sort(key=lambda e: e["id"])
        return result

    async def add_entity_alias(
        self,
        workspace_id: str,
        entity_id: str,
        alias: str,
        normalized_alias: str,
        source: str = "manual",
    ) -> None:
        rows = self._entity_aliases.setdefault(workspace_id, [])
        for r in rows:
            if r["entity_id"] == entity_id and r["normalized_alias"] == normalized_alias:
                return  # idempotent
        rows.append(
            {
                "id": generate_id("ealias"),
                "workspace_id": workspace_id,
                "entity_id": entity_id,
                "alias": alias,
                "normalized_alias": normalized_alias,
                "source": source,
                "created_at": utc_now_iso(),
            }
        )

    async def add_entity_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        role: str = "mention",
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> dict:
        rows = self._entity_members.setdefault(workspace_id, [])
        for r in rows:
            if r["entity_id"] == entity_id and r["memory_id"] == memory_id and r["role"] == role:
                return {
                    "entity_id": entity_id,
                    "memory_id": memory_id,
                    "role": role,
                    "confidence": r["confidence"],
                }
        rows.append(
            {
                "id": generate_id("emem"),
                "workspace_id": workspace_id,
                "entity_id": entity_id,
                "memory_id": memory_id,
                "role": role,
                "confidence": confidence,
                "meta": dict(meta or {}),
                "created_at": utc_now_iso(),
            }
        )
        return {"entity_id": entity_id, "memory_id": memory_id, "role": role, "confidence": confidence}

    async def list_entity_members(
        self,
        workspace_id: str,
        entity_id: str,
        role: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        rows = [r for r in self._entity_members.get(workspace_id, []) if r["entity_id"] == entity_id]
        if role is not None:
            rows = [r for r in rows if r["role"] == role]
        rows.sort(key=lambda r: (r["created_at"], r["memory_id"]))
        return [
            {
                "entity_id": r["entity_id"],
                "memory_id": r["memory_id"],
                "role": r["role"],
                "confidence": r["confidence"],
            }
            for r in rows[:limit]
        ]

    async def list_workspace_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 10000,
    ) -> list[dict]:
        out = [
            self._hydrate_entity(workspace_id, entity)
            for entity in self._entities.get(workspace_id, {}).values()
            if status is None or entity["status"] == status
        ]
        out.sort(key=lambda e: e["id"])
        return out[:limit]

    async def list_workspace_entity_members(
        self,
        workspace_id: str,
        *,
        role: str | None = None,
        limit: int = 100000,
    ) -> list[dict]:
        rows = [
            r for r in self._entity_members.get(workspace_id, [])
            if role is None or r["role"] == role
        ]
        rows.sort(key=lambda r: (r["entity_id"], r["memory_id"], r["role"]))
        return [
            {
                "entity_id": r["entity_id"],
                "memory_id": r["memory_id"],
                "role": r["role"],
                "confidence": r["confidence"],
            }
            for r in rows[:limit]
        ]

    async def reassign_entity_members(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> int:
        rows = self._entity_members.get(workspace_id, [])
        target_keys = {(r["memory_id"], r["role"]) for r in rows if r["entity_id"] == target_id}
        moved = 0
        kept = []
        for r in rows:
            if r["entity_id"] != source_id:
                kept.append(r)
                continue
            if (r["memory_id"], r["role"]) in target_keys:
                continue  # collision: drop source row in favor of existing target row
            r["entity_id"] = target_id
            kept.append(r)
            moved += 1
        self._entity_members[workspace_id] = kept
        return moved

    async def update_entity(
        self,
        workspace_id: str,
        entity_id: str,
        **updates,
    ) -> dict | None:
        entity = self._entities.get(workspace_id, {}).get(entity_id)
        if not entity:
            return None
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
        changed = False
        for key, value in updates.items():
            if key not in allowed:
                continue
            entity[key] = dict(value) if key == "provenance" and value else value
            changed = True
        if changed:
            entity["updated_at"] = utc_now_iso()
        return self._hydrate_entity(workspace_id, entity)

    # ========== Internal Versioned Resource Operations ==========

    async def mutate_versioned_resource(
        self,
        mutation: VersionedResourceMutation,
    ) -> VersionedResourceMutationResult:
        desired = mutation.resource
        scope = (desired.tenant_id, desired.workspace_id, desired.namespace)
        operation_key = (*scope, mutation.operation_id)
        if replay := self._versioned_resource_operations.get(operation_key):
            if replay.request_hash != mutation.request_hash:
                raise VersionedResourceConflictError("idempotency key was already used for a different request")
            return VersionedResourceMutationResult(resource=replay.as_resource(), replayed=True)

        head_key = (*scope, desired.id)
        logical_key = (*scope, desired.resource_key)
        current = self._versioned_resource_heads.get(head_key)
        if mutation.action == "create":
            if mutation.expected_etag != "*":
                raise VersionedResourcePreconditionFailedError("create requires If-None-Match: *")
            if current is not None or logical_key in self._versioned_resource_keys:
                raise VersionedResourceConflictError("resource id or key already exists")
            revision = 1
            final = desired.model_copy(update={"revision": revision})
        else:
            if current is None:
                raise VersionedResourceNotFoundError("resource not found")
            if mutation.expected_etag != current.etag:
                raise VersionedResourcePreconditionFailedError("ETag does not match current revision")
            if mutation.action == "restore":
                if not current.is_deleted:
                    raise VersionedResourceConflictError("resource is not deleted")
            elif current.is_deleted:
                raise VersionedResourceNotFoundError("resource not found")
            if desired.resource_key != current.resource_key:
                raise VersionedResourceConflictError("resource key is immutable")
            revision = current.revision + 1
            final = desired.model_copy(
                update={
                    "created_at": current.created_at,
                    "created_by": current.created_by,
                    "revision": revision,
                }
            )

        self._versioned_resource_sequence += 1
        final = final.model_copy(
            update={
                "sequence": self._versioned_resource_sequence,
                "etag": f'"vr-{revision}-{final.state_hash}"',
            }
        )
        snapshot = VersionedResourceRevision(
            **final.model_dump(),
            action=mutation.action,
            operation_id=mutation.operation_id,
            request_hash=mutation.request_hash,
        )
        self._versioned_resource_heads[head_key] = final.model_copy(deep=True)
        self._versioned_resource_keys[logical_key] = final.id
        self._versioned_resource_revisions.setdefault(head_key, []).append(snapshot.model_copy(deep=True))
        self._versioned_resource_operations[operation_key] = snapshot.model_copy(deep=True)
        return VersionedResourceMutationResult(resource=final.model_copy(deep=True))

    async def get_versioned_resource_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        revision = self._versioned_resource_operations.get(
            (tenant_id, workspace_id, namespace, operation_id)
        )
        if revision is None:
            return None
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError("idempotency key was already used for a different request")
        return VersionedResourceMutationResult(resource=revision.as_resource(), replayed=True)

    async def get_versioned_resource(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        resource = self._versioned_resource_heads.get((tenant_id, workspace_id, namespace, resource_id))
        if resource is None or (resource.is_deleted and not include_deleted):
            return None
        return resource.model_copy(deep=True)

    async def get_versioned_resource_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        scope = (tenant_id, workspace_id, namespace)
        resource_id = self._versioned_resource_keys.get((*scope, resource_key))
        if resource_id is None:
            return None
        return await self.get_versioned_resource(
            *scope,
            resource_id,
            include_deleted=include_deleted,
        )

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
        scope = (tenant_id, workspace_id, namespace)
        resources = [
            resource
            for key, resource in self._versioned_resource_heads.items()
            if key[:3] == scope
            and (include_deleted or not resource.is_deleted)
            and (before_sequence is None or resource.sequence < before_sequence)
        ]
        resources.sort(key=lambda resource: resource.sequence, reverse=True)
        return [resource.model_copy(deep=True) for resource in resources[:limit]]

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
        revisions = self._versioned_resource_revisions.get(
            (tenant_id, workspace_id, namespace, resource_id),
            [],
        )
        result = [
            revision
            for revision in reversed(revisions)
            if before_sequence is None or revision.sequence < before_sequence
        ]
        return [revision.model_copy(deep=True) for revision in result[:limit]]


class MemoryStoragePlugin(StoragePluginBase):
    """Plugin for in-memory storage backend."""

    PROVIDER_NAME = "memory"

    def initialize(self, v: Variables, logger: Logger) -> MemoryStorageBackend:
        return MemoryStorageBackend(v=v)
