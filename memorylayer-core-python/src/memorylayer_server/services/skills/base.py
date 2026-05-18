"""SkillsService: CRUD business logic for agent skills."""
from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, Optional

from ...models.memory import MemoryType, RememberInput
from ...models.skill import Skill, SkillCreateInput, SkillFile, SkillUpdateInput
from ...utils import generate_id
from ..storage import StorageBackend
from .frontmatter import render_skill_md

if TYPE_CHECKING:
    from ..memory import MemoryService

logger = logging.getLogger(__name__)


def _compute_manifest_hash(skill: Skill) -> str:
    """Compute SHA-256 over the canonical SKILL.md text for conflict detection."""
    frontmatter: dict[str, Any] = {
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
    }
    if skill.license:
        frontmatter["license"] = skill.license
    if skill.compatibility:
        frontmatter["compatibility"] = skill.compatibility
    if skill.allowed_tools:
        frontmatter["allowed-tools"] = skill.allowed_tools
    if skill.metadata:
        frontmatter["metadata"] = str(skill.metadata)
    text = render_skill_md(frontmatter, skill.body)
    return hashlib.sha256(text.encode()).hexdigest()


def _compute_bundle_hash(files: list[SkillFile]) -> str:
    """Compute SHA-256 over sorted (path, content_hash) pairs."""
    pairs = sorted((f.path, f.content_hash) for f in files)
    payload = "\n".join(f"{p}:{h}" for p, h in pairs)
    return hashlib.sha256(payload.encode()).hexdigest()


class SkillsService:
    """Service for managing agent skills stored in MemoryLayer.

    Wraps StorageBackend with ID generation, hash computation, and an
    optional memory_service for auto-mirroring skills as procedural memories
    (Phase 2). Also accepts a legacy memory_indexer callable hook for tests.
    """

    def __init__(
        self,
        storage: StorageBackend,
        memory_service: "Optional[MemoryService]" = None,
        memory_indexer: Optional[Callable[[Skill], Any]] = None,
    ) -> None:
        self._storage = storage
        self._memory_service = memory_service
        self._memory_indexer = memory_indexer

    # ------------------------------------------------------------------
    # Skill CRUD
    # ------------------------------------------------------------------

    async def create_skill(
        self,
        input: SkillCreateInput,
        workspace_id: str,
        tenant_id: str = "",
        user_id: Optional[str] = None,
    ) -> Skill:
        """Create a new skill, computing manifest hash."""
        now = datetime.now(UTC)
        skill = Skill(
            id=generate_id("skl"),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            user_id=user_id if user_id else input.user_id,
            name=input.name,
            description=input.description,
            version=input.version,
            license=input.license,
            compatibility=input.compatibility,
            allowed_tools=input.allowed_tools,
            body=input.body,
            metadata=input.metadata,
            source_mode=input.source_mode,
            manifest_hash="",
            bundle_hash="",
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        skill = skill.model_copy(update={"manifest_hash": _compute_manifest_hash(skill)})
        result = await self._storage.create_skill(skill)
        await self._maybe_index(result)
        return result

    async def get_skill(
        self,
        workspace_id: str,
        skill_id: str,
    ) -> Optional[Skill]:
        """Get a skill by ID."""
        return await self._storage.get_skill(workspace_id, skill_id)

    async def list_skills(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Skill]:
        """List skills with optional filters."""
        return await self._storage.list_skills(
            workspace_id=workspace_id,
            user_id=user_id,
            name=name,
            enabled=enabled,
            limit=limit,
            offset=offset,
        )

    async def update_skill(
        self,
        workspace_id: str,
        skill_id: str,
        input: SkillUpdateInput,
    ) -> Optional[Skill]:
        """Apply partial updates to a skill, recomputing manifest_hash if content changed."""
        updates: dict[str, Any] = {
            k: v for k, v in input.model_dump(exclude_none=True).items()
        }
        updates["updated_at"] = datetime.now(UTC)

        result = await self._storage.update_skill(workspace_id, skill_id, updates)
        if result is None:
            return None

        # Recompute manifest hash after any content change
        new_hash = _compute_manifest_hash(result)
        if new_hash != result.manifest_hash:
            result = await self._storage.update_skill(
                workspace_id, skill_id, {"manifest_hash": new_hash}
            )

        await self._maybe_index(result)
        return result

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Delete a skill, its files, and its memory mirror."""
        result = await self._storage.delete_skill(workspace_id, skill_id)
        if result and self._memory_service:
            await self._delete_mirror_memory(workspace_id, skill_id)
        return result

    # ------------------------------------------------------------------
    # Skill file operations
    # ------------------------------------------------------------------

    async def upsert_file(
        self,
        skill_id: str,
        path: str,
        content: bytes,
        mime_type: Optional[str] = None,
        workspace_id: Optional[str] = None,
        index_references: bool = True,
    ) -> SkillFile:
        """Insert or update a file in a skill bundle, updating bundle_hash."""
        content_hash = hashlib.sha256(content).hexdigest()
        now = datetime.now(UTC)

        # Check if file exists to decide create vs update
        existing = await self._storage.get_skill_file(skill_id, path)
        kind = _infer_kind(path)

        if existing:
            file = existing.model_copy(
                update={
                    "content": content,
                    "content_hash": content_hash,
                    "size_bytes": len(content),
                    "mime_type": mime_type,
                    "updated_at": now,
                }
            )
        else:
            file = SkillFile(
                id=generate_id("sklf"),
                skill_id=skill_id,
                path=path,
                kind=kind,
                content=content,
                content_hash=content_hash,
                size_bytes=len(content),
                mime_type=mime_type,
                created_at=now,
                updated_at=now,
            )

        result = await self._storage.upsert_skill_file(file)

        # Recompute bundle_hash after change
        if workspace_id:
            all_files = await self._storage.list_skill_files(skill_id)
            new_bundle_hash = _compute_bundle_hash(all_files)
            skill_rec = await self._storage.update_skill(workspace_id, skill_id, {"bundle_hash": new_bundle_hash})

            # Auto-index reference files as skill_reference memories
            if index_references and kind == "reference" and self._memory_service and skill_rec:
                await self._upsert_reference_memory(skill_rec, result)

        return result

    async def get_file(self, skill_id: str, path: str) -> Optional[SkillFile]:
        """Get a single skill file by path."""
        return await self._storage.get_skill_file(skill_id, path)

    async def list_files(self, skill_id: str) -> list[SkillFile]:
        """List all files in a skill bundle."""
        return await self._storage.list_skill_files(skill_id)

    async def delete_file(
        self,
        skill_id: str,
        path: str,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """Delete a file from a skill bundle, updating bundle_hash and cleaning up reference memories."""
        result = await self._storage.delete_skill_file(skill_id, path)
        if result and workspace_id:
            all_files = await self._storage.list_skill_files(skill_id)
            new_bundle_hash = _compute_bundle_hash(all_files)
            await self._storage.update_skill(workspace_id, skill_id, {"bundle_hash": new_bundle_hash})
            if self._memory_service:
                await self._delete_reference_memory(workspace_id, skill_id, path)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_index(self, skill: Optional[Skill]) -> None:
        """Mirror skill as a procedural memory (upsert) and call legacy hook."""
        if not skill:
            return

        if self._memory_service:
            await self._upsert_mirror_memory(skill)

        if self._memory_indexer:
            try:
                result = self._memory_indexer(skill)
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                pass  # indexer errors must not break skill operations

    async def _upsert_mirror_memory(self, skill: Skill) -> None:
        """Create or replace the procedural memory mirror for a skill.

        Uses search_memories_by_filter to find an existing mirror by skill_id
        metadata key, then deletes it before creating a fresh one (upsert
        semantics without requiring a dedicated storage upsert method).
        """
        try:
            await self._delete_mirror_memory(skill.workspace_id, skill.id)

            content = f"{skill.name}: {skill.description}"
            if skill.body:
                content += f"\n\n{skill.body[:4000]}"

            tags = ["skill", f"skill:{skill.name}"]
            extra_tags = skill.metadata.get("tags", [])
            if isinstance(extra_tags, list):
                tags += [str(t) for t in extra_tags]

            await self._memory_service.remember(
                workspace_id=skill.workspace_id,
                input=RememberInput(
                    content=content,
                    type=MemoryType.PROCEDURAL,
                    subtype="skill",
                    tags=tags,
                    metadata={
                        "skill_id": skill.id,
                        "skill_name": skill.name,
                        "skill_version": skill.version,
                    },
                    user_id=skill.user_id,
                ),
                user_id=skill.user_id,
                inline=True,
            )
        except Exception:
            logger.debug("Failed to upsert memory mirror for skill %s", skill.id, exc_info=True)

    async def _delete_mirror_memory(self, workspace_id: str, skill_id: str) -> None:
        """Delete any existing memory mirror for the given skill_id."""
        try:
            existing = await self._storage.search_memories_by_filter(
                workspace_id=workspace_id,
                subtypes=["skill"],
                metadata_filter={"skill_id": skill_id},
                limit=10,
            )
            for mem in existing:
                await self._storage.delete_memory(workspace_id, mem.id, hard=True)
        except Exception:
            logger.debug("Failed to delete mirror memory for skill %s", skill_id, exc_info=True)

    async def _upsert_reference_memory(self, skill: Skill, skill_file: SkillFile) -> None:
        """Create or replace a skill_reference memory for a reference file."""
        try:
            await self._delete_reference_memory(skill.workspace_id, skill.id, skill_file.path)

            text_content = skill_file.content.decode("utf-8", errors="replace") if isinstance(skill_file.content, bytes) else str(skill_file.content)
            content = f"[{skill.name}/{skill_file.path}]\n{text_content[:8000]}"

            await self._memory_service.remember(
                workspace_id=skill.workspace_id,
                input=RememberInput(
                    content=content,
                    type=MemoryType.PROCEDURAL,
                    subtype="skill_reference",
                    tags=["skill", f"skill:{skill.name}", "skill_reference", skill_file.path],
                    metadata={
                        "skill_id": skill.id,
                        "skill_name": skill.name,
                        "ref_path": skill_file.path,
                    },
                    user_id=skill.user_id,
                ),
                user_id=skill.user_id,
                inline=True,
            )
        except Exception:
            logger.debug("Failed to upsert reference memory for %s/%s", skill.id, skill_file.path, exc_info=True)

    async def _delete_reference_memory(self, workspace_id: str, skill_id: str, path: str) -> None:
        """Delete an existing skill_reference memory for a specific file path."""
        try:
            existing = await self._storage.search_memories_by_filter(
                workspace_id=workspace_id,
                subtypes=["skill_reference"],
                metadata_filter={"skill_id": skill_id, "ref_path": path},
                limit=5,
            )
            for mem in existing:
                await self._storage.delete_memory(workspace_id, mem.id, hard=True)
        except Exception:
            logger.debug("Failed to delete reference memory for %s/%s", skill_id, path, exc_info=True)


def _infer_kind(path: str) -> str:
    """Derive SkillFile.kind from the top-level directory of the path."""
    top = path.split("/")[0] if "/" in path else ""
    if top == "scripts":
        return "script"
    if top == "references":
        return "reference"
    if top == "assets":
        return "asset"
    return "other"
