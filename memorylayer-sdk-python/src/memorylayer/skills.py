"""Skills namespace for MemoryLayer.ai Python SDK.

Provides CRUD, resolve, pull/push/materialize, and parse_skill_folder.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .client import MemoryLayerClient
    from .models import AuthorityContext

# Agentskills spec: dirs that map to known kinds
_KIND_MAP = {
    "scripts": "script",
    "references": "reference",
    "assets": "asset",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md into frontmatter dict and body string."""
    import json

    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    fm_raw, body = m.group(1), m.group(2)
    # Try pyyaml if available, fall back to JSON, then simple key:value parser
    try:
        import yaml  # type: ignore[import-untyped]

        fm = yaml.safe_load(fm_raw) or {}
    except ImportError:
        try:
            fm = json.loads(fm_raw)
        except Exception:
            fm = _parse_simple_yaml(fm_raw)
    except Exception:
        try:
            fm = json.loads(fm_raw)
        except Exception:
            fm = {}

    return fm, body


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse simple key: value YAML without external dependencies."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            # Strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            result[key] = val
    return result


def parse_skill_folder(path: Path) -> tuple[dict[str, Any], list[tuple[str, bytes]]]:
    """Parse a skill directory into (manifest_dict, [(rel_path, content), ...]).

    Reads SKILL.md for frontmatter (name, description, version, etc.) + body.
    Walks scripts/, references/, assets/ for bundle files.
    """
    path = Path(path)
    skill_md = path / "SKILL.md"
    if not skill_md.exists():
        raise FileNotFoundError(f"SKILL.md not found in {path}")

    text = skill_md.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text)

    manifest: dict[str, Any] = {
        "name": fm.get("name", path.name),
        "description": fm.get("description", ""),
        "version": fm.get("version", "0.1.0"),
        "body": body,
    }
    for key in ("license", "compatibility", "allowed_tools"):
        if key in fm:
            manifest[key] = fm[key]
    # Remaining frontmatter keys go to metadata
    known = {"name", "description", "version", "license", "compatibility", "allowed_tools"}
    extras = {k: v for k, v in fm.items() if k not in known}
    if extras:
        manifest["metadata"] = extras

    # Collect bundle files
    files: list[tuple[str, bytes]] = []
    for kind_dir, kind in _KIND_MAP.items():
        dir_path = path / kind_dir
        if not dir_path.is_dir():
            continue
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file():
                rel = str(file_path.relative_to(path))
                files.append((rel, file_path.read_bytes()))

    # Also pick up any other files at root (not SKILL.md, not known subdirs)
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.name != "SKILL.md":
            files.append((file_path.name, file_path.read_bytes()))

    return manifest, files


class SkillModel(BaseModel):
    """Pydantic model mirroring the server Skill."""

    model_config = ConfigDict(extra="allow")

    id: str
    workspace_id: str
    tenant_id: str = ""
    user_id: str | None = None
    name: str
    description: str
    version: str = "0.1.0"
    license: str | None = None
    compatibility: str | None = None
    allowed_tools: str | None = None
    body: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_mode: str = "server"
    manifest_hash: str = ""
    bundle_hash: str = ""
    enabled: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class SkillsAPI:
    """Skills namespace — access via client.skills.<method>."""

    def __init__(self, client: "MemoryLayerClient") -> None:
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def list(
        self,
        scope: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        include_shadowed: bool = False,
        workspace_id: str | None = None,
        authority: "AuthorityContext | None" = None,
    ) -> list[SkillModel]:
        """List visible skills."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if scope:
            params["scope"] = scope
        if name:
            params["name"] = name
        if tags:
            params["tags"] = ",".join(tags)
        if enabled is not None:
            params["enabled"] = str(enabled).lower()
        if include_shadowed:
            params["include_shadowed"] = "true"

        data = await self._client._request("GET", "/skills", params=params, authority=authority)
        return [SkillModel(**s) for s in data.get("skills", [])]

    async def get(self, skill_id: str, authority: "AuthorityContext | None" = None) -> SkillModel:
        """Get a skill by ID."""
        data = await self._client._request("GET", f"/skills/{skill_id}", authority=authority)
        return SkillModel(**data["skill"])

    async def get_manifest(self, skill_id: str) -> str:
        """Get the rendered SKILL.md text for a skill."""
        transport = self._client._ensure_transport()
        response = await transport.request("GET", f"/skills/{skill_id}/manifest")
        response.raise_for_status()
        return response.text

    async def get_file(self, skill_id: str, path: str) -> bytes:
        """Get a single file from the skill bundle."""
        transport = self._client._ensure_transport()
        response = await transport.request("GET", f"/skills/{skill_id}/files/{path}")
        response.raise_for_status()
        return response.content

    async def list_files(
        self, skill_id: str, authority: "AuthorityContext | None" = None,
    ) -> list[dict[str, Any]]:
        """List files in a skill bundle."""
        data = await self._client._request(
            "GET", f"/skills/{skill_id}/files", authority=authority,
        )
        return data.get("files", [])

    async def save(
        self,
        name: str,
        description: str,
        body: str = "",
        files: list[tuple[str, bytes]] | None = None,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
        user_id: str | None = None,
        authority: "AuthorityContext | None" = None,
        **manifest_extras: Any,
    ) -> SkillModel:
        """Create or update a skill by name."""
        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "body": body,
            "source_mode": source_mode,
        }
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if user_id:
            payload["user_id"] = user_id
        payload.update(manifest_extras)

        if files:
            payload["files"] = [
                {
                    "path": rel_path,
                    "content": content.decode("utf-8", errors="replace"),
                }
                for rel_path, content in files
            ]

        data = await self._client._request("POST", "/skills", json=payload, authority=authority)
        return SkillModel(**data["skill"])

    async def delete(self, skill_id: str, authority: "AuthorityContext | None" = None) -> None:
        """Delete a skill and its files."""
        await self._client._request("DELETE", f"/skills/{skill_id}", authority=authority)

    async def resolve(
        self,
        name: str | None = None,
        query: str | None = None,
        workspace_id: str | None = None,
        authority: "AuthorityContext | None" = None,
    ) -> "SkillModel | list[SkillModel]":
        """Resolve a skill by name (precedence-winner) or query (vector recall)."""
        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        if query:
            payload["query"] = query
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id

        data = await self._client._request(
            "POST", "/skills/resolve", json=payload, authority=authority,
        )
        # Server returns {"skill": null} when a name lookup misses, vs
        # {"skills": []} for query-mode empty results. The key may be
        # present with a None value, so check the value, not just the key.
        skill = data.get("skill")
        if skill is not None:
            return SkillModel(**skill)
        return [SkillModel(**s) for s in (data.get("skills") or [])]

    # ------------------------------------------------------------------
    # Local filesystem helpers
    # ------------------------------------------------------------------

    async def pull(self, name: str, out_dir: Path, workspace_id: str | None = None) -> Path:
        """Materialize a named skill to out_dir/{name}/."""
        ws_id = self._ws(workspace_id)
        params: dict[str, Any] = {"name": name}
        if ws_id:
            params["workspace_id"] = ws_id

        # Resolve the skill
        skill = await self.resolve(name=name, workspace_id=ws_id)
        if isinstance(skill, list):
            if not skill:
                raise FileNotFoundError(f"Skill '{name}' not found")
            skill = skill[0]

        skill_dir = Path(out_dir) / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Write SKILL.md
        manifest_text = await self.get_manifest(skill.id)
        (skill_dir / "SKILL.md").write_text(manifest_text, encoding="utf-8")

        # Write bundle files
        files = await self.list_files(skill.id)
        for file_info in files:
            file_path = skill_dir / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = await self.get_file(skill.id, file_info["path"])
            file_path.write_bytes(content)

        return skill_dir

    async def push(
        self,
        skill_dir: Path,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
    ) -> SkillModel:
        """Parse a skill directory and upload to MemoryLayer."""
        manifest, files = parse_skill_folder(Path(skill_dir))
        return await self.save(
            scope=scope,
            source_mode=source_mode,
            workspace_id=workspace_id,
            files=files,
            **manifest,
        )

    async def materialize(
        self,
        target_dir: Path,
        scope: str | None = None,
        workspace_id: str | None = None,
    ) -> list[Path]:
        """Bulk pull all visible skills into target_dir. Idempotent via bundle_hash."""
        target_dir = Path(target_dir)
        skills = await self.list(scope=scope, workspace_id=workspace_id)
        materialized: list[Path] = []

        for skill in skills:
            skill_dir = target_dir / skill.name
            hash_file = skill_dir / ".bundle_hash"

            # Skip if already up to date
            if skill_dir.exists() and hash_file.exists():
                existing_hash = hash_file.read_text().strip()
                if existing_hash == skill.bundle_hash and skill.bundle_hash:
                    materialized.append(skill_dir)
                    continue

            skill_dir.mkdir(parents=True, exist_ok=True)

            # Write SKILL.md
            manifest_text = await self.get_manifest(skill.id)
            (skill_dir / "SKILL.md").write_text(manifest_text, encoding="utf-8")

            # Write bundle files
            files = await self.list_files(skill.id)
            for file_info in files:
                file_path = skill_dir / file_info["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                content = await self.get_file(skill.id, file_info["path"])
                file_path.write_bytes(content)

            # Record bundle_hash for idempotency
            if skill.bundle_hash:
                hash_file.write_text(skill.bundle_hash)

            materialized.append(skill_dir)

        return materialized


class SyncSkillsAPI:
    """Synchronous Skills namespace — access via sync_client.skills.<method>."""

    def __init__(self, client: "SyncMemoryLayerClient") -> None:  # type: ignore[name-defined]
        self._client = client

    def _ws(self, workspace_id: str | None) -> str | None:
        return workspace_id or self._client.workspace_id

    def list(
        self,
        scope: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        include_shadowed: bool = False,
        workspace_id: str | None = None,
    ) -> list[SkillModel]:
        """List visible skills."""
        params: dict[str, Any] = {}
        ws_id = self._ws(workspace_id)
        if ws_id:
            params["workspace_id"] = ws_id
        if scope:
            params["scope"] = scope
        if name:
            params["name"] = name
        if tags:
            params["tags"] = ",".join(tags)
        if enabled is not None:
            params["enabled"] = str(enabled).lower()
        if include_shadowed:
            params["include_shadowed"] = "true"

        data = self._client._request("GET", "/skills", params=params)
        return [SkillModel(**s) for s in data.get("skills", [])]

    def get(self, skill_id: str) -> SkillModel:
        """Get a skill by ID."""
        data = self._client._request("GET", f"/skills/{skill_id}")
        return SkillModel(**data["skill"])

    def get_manifest(self, skill_id: str) -> str:
        """Get the rendered SKILL.md text for a skill."""
        client = self._client._ensure_client()
        response = client.get(f"/skills/{skill_id}/manifest")
        response.raise_for_status()
        return response.text

    def get_file(self, skill_id: str, path: str) -> bytes:
        """Get a single file from the skill bundle."""
        client = self._client._ensure_client()
        response = client.get(f"/skills/{skill_id}/files/{path}")
        response.raise_for_status()
        return response.content

    def list_files(self, skill_id: str) -> list[dict[str, Any]]:
        """List files in a skill bundle."""
        data = self._client._request("GET", f"/skills/{skill_id}/files")
        return data.get("files", [])

    def save(
        self,
        name: str,
        description: str,
        body: str = "",
        files: list[tuple[str, bytes]] | None = None,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
        user_id: str | None = None,
        **manifest_extras: Any,
    ) -> SkillModel:
        """Create or update a skill by name."""
        payload: dict[str, Any] = {
            "name": name,
            "description": description,
            "body": body,
            "source_mode": source_mode,
        }
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id
        if user_id:
            payload["user_id"] = user_id
        payload.update(manifest_extras)

        if files:
            payload["files"] = [
                {
                    "path": rel_path,
                    "content": content.decode("utf-8", errors="replace"),
                }
                for rel_path, content in files
            ]

        data = self._client._request("POST", "/skills", json=payload)
        return SkillModel(**data["skill"])

    def delete(self, skill_id: str) -> None:
        """Delete a skill and its files."""
        self._client._request("DELETE", f"/skills/{skill_id}")

    def resolve(
        self,
        name: str | None = None,
        query: str | None = None,
        workspace_id: str | None = None,
    ) -> "SkillModel | list[SkillModel]":
        """Resolve a skill by name (precedence-winner) or query (vector recall)."""
        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        if query:
            payload["query"] = query
        ws_id = self._ws(workspace_id)
        if ws_id:
            payload["workspace_id"] = ws_id

        data = self._client._request("POST", "/skills/resolve", json=payload)
        # Server returns {"skill": null} on name-mode misses; check value.
        skill = data.get("skill")
        if skill is not None:
            return SkillModel(**skill)
        return [SkillModel(**s) for s in (data.get("skills") or [])]

    def pull(self, name: str, out_dir: Path, workspace_id: str | None = None) -> Path:
        """Materialize a named skill to out_dir/{name}/."""
        ws_id = self._ws(workspace_id)
        skill = self.resolve(name=name, workspace_id=ws_id)
        if isinstance(skill, list):
            if not skill:
                raise FileNotFoundError(f"Skill '{name}' not found")
            skill = skill[0]

        skill_dir = Path(out_dir) / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        manifest_text = self.get_manifest(skill.id)
        (skill_dir / "SKILL.md").write_text(manifest_text, encoding="utf-8")

        files = self.list_files(skill.id)
        for file_info in files:
            file_path = skill_dir / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            content = self.get_file(skill.id, file_info["path"])
            file_path.write_bytes(content)

        return skill_dir

    def push(
        self,
        skill_dir: Path,
        scope: str = "workspace",
        source_mode: str = "server",
        workspace_id: str | None = None,
    ) -> SkillModel:
        """Parse a skill directory and upload to MemoryLayer."""
        manifest, files = parse_skill_folder(Path(skill_dir))
        return self.save(
            scope=scope,
            source_mode=source_mode,
            workspace_id=workspace_id,
            files=files,
            **manifest,
        )

    def materialize(
        self,
        target_dir: Path,
        scope: str | None = None,
        workspace_id: str | None = None,
    ) -> list[Path]:
        """Bulk pull all visible skills into target_dir. Idempotent via bundle_hash."""
        target_dir = Path(target_dir)
        skills = self.list(scope=scope, workspace_id=workspace_id)
        materialized: list[Path] = []

        for skill in skills:
            skill_dir = target_dir / skill.name
            hash_file = skill_dir / ".bundle_hash"

            if skill_dir.exists() and hash_file.exists():
                existing_hash = hash_file.read_text().strip()
                if existing_hash == skill.bundle_hash and skill.bundle_hash:
                    materialized.append(skill_dir)
                    continue

            skill_dir.mkdir(parents=True, exist_ok=True)

            manifest_text = self.get_manifest(skill.id)
            (skill_dir / "SKILL.md").write_text(manifest_text, encoding="utf-8")

            files = self.list_files(skill.id)
            for file_info in files:
                file_path = skill_dir / file_info["path"]
                file_path.parent.mkdir(parents=True, exist_ok=True)
                content = self.get_file(skill.id, file_info["path"])
                file_path.write_bytes(content)

            if skill.bundle_hash:
                hash_file.write_text(skill.bundle_hash)

            materialized.append(skill_dir)

        return materialized
