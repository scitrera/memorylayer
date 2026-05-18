"""
In-memory storage backend for testing.

Provides a complete storage implementation that stores all data in memory.
Data is lost on service restart - use only for testing.
"""

from datetime import UTC, datetime
from logging import Logger
from typing import Any

from scitrera_app_framework import Variables

from ...config import DEFAULT_CONTEXT_ID, DEFAULT_TENANT_ID
from ...models.association import AssociateInput, Association, GraphPath, GraphQueryResult
from ...models.mcp_server import McpServer
from ...models.memory import Memory, MemoryType, RememberInput
from ...models.session import Session, WorkingMemory
from ...models.skill import Skill, SkillFile
from ...models.workspace import Context, Workspace
from ...utils import compute_content_hash, cosine_similarity, generate_id, utc_now_iso
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
        self._associations: dict[str, dict[str, Association]] = {}  # workspace_id -> {assoc_id -> Association}
        self._sessions: dict[str, dict[str, Session]] = {}  # workspace_id -> {session_id -> Session}
        self._working_memory: dict[str, dict[str, dict[str, WorkingMemory]]] = {}  # ws -> {sess -> {key -> WM}}
        self._skills: dict[str, dict[str, Skill]] = {}  # workspace_id -> {skill_id -> Skill}
        self._skill_files: dict[str, dict[str, SkillFile]] = {}  # skill_id -> {path -> SkillFile}
        self._mcp_servers: dict[str, dict[str, McpServer]] = {}  # workspace_id -> {server_id -> McpServer}
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
            embedding=None,  # Added via update_memory() later
            access_count=0,
            created_at=now,
            updated_at=now,
        )
        self._memories[workspace_id][memory.id] = memory
        self.logger.debug("Created memory: %s in workspace: %s", memory.id, workspace_id)
        return memory

    async def get_memory(self, workspace_id: str, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID within a workspace."""
        ws_memories = self._memories.get(workspace_id, {})
        memory = ws_memories.get(memory_id)
        if memory and memory_id in self._deleted_memories:
            return None
        return memory

    async def get_memory_by_id(self, memory_id: str, track_access: bool = True) -> Memory | None:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        if memory_id in self._deleted_memories:
            return None
        for ws_memories in self._memories.values():
            if memory_id in ws_memories:
                return ws_memories[memory_id]
        return None

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Memory | None:
        """Update memory fields."""
        memory = await self.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            return None

        for key, value in updates.items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        memory.updated_at = utc_now_iso()
        return memory

    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        memory = await self.get_memory(workspace_id, memory_id, track_access=False)
        if not memory:
            return False

        if hard:
            del self._memories[workspace_id][memory_id]
            self._deleted_memories.discard(memory_id)
        else:
            self._deleted_memories.add(memory_id)
        return True

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

    async def list_workspaces(self) -> list[Workspace]:
        """List all workspaces."""
        return list(self._workspaces.values())

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
        if skill.workspace_id not in self._skills:
            self._skills[skill.workspace_id] = {}
        self._skills[skill.workspace_id][skill.id] = skill
        self.logger.debug("Created skill: %s in workspace: %s", skill.id, skill.workspace_id)
        return skill

    async def get_skill(self, workspace_id: str, skill_id: str) -> Skill | None:
        """Get skill by ID within a workspace."""
        return self._skills.get(workspace_id, {}).get(skill_id)

    async def get_skill_by_name(
        self,
        workspace_id: str,
        name: str,
        user_id: str | None = None,
    ) -> Skill | None:
        """Get skill by name within a workspace, optionally filtering by user scope."""
        for skill in self._skills.get(workspace_id, {}).values():
            if skill.name == name:
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
    ) -> list[Skill]:
        """List skills in a workspace with optional filters."""
        results = []
        for skill in self._skills.get(workspace_id, {}).values():
            if user_id is not None and skill.user_id != user_id:
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
                if skill.name != name:
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
        for key, value in updates.items():
            if hasattr(skill, key):
                setattr(skill, key, value)
        skill.updated_at = datetime.now(UTC)
        return skill

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Delete a skill and cascade to its files."""
        ws_skills = self._skills.get(workspace_id, {})
        if skill_id not in ws_skills:
            return False
        del ws_skills[skill_id]
        # Cascade: remove all files for this skill
        self._skill_files.pop(skill_id, None)
        return True

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


class MemoryStoragePlugin(StoragePluginBase):
    """Plugin for in-memory storage backend."""

    PROVIDER_NAME = "memory"

    def initialize(self, v: Variables, logger: Logger) -> MemoryStorageBackend:
        return MemoryStorageBackend(v=v)
