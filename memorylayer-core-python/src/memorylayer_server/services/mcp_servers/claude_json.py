"""Surgical reader/writer for ~/.claude.json MCP server blocks.

Reads and writes ONLY the relevant mcpServers block — never touches other keys.
Atomic write via temp-file + rename.

Supported scopes:
- "user"  → top-level mcpServers key
- "local" → projects[project_path].mcpServers key
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

_DEFAULT_CLAUDE_JSON = Path("~/.claude.json").expanduser()

McpServerEntry = dict[str, Any]


def _default_target() -> Path:
    return _DEFAULT_CLAUDE_JSON


def read_claude_json_servers(
    scope: Literal["local", "user"],
    project_path: Optional[str] = None,
    target_path: Optional[Path] = None,
) -> dict[str, McpServerEntry]:
    """Read mcpServers from the given scope in the Claude JSON file.

    Returns an empty dict if the file or key doesn't exist.
    """
    path = target_path or _default_target()
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if scope == "user":
        return dict(data.get("mcpServers") or {})

    # local scope: projects[project_path].mcpServers
    if not project_path:
        return {}
    projects = data.get("projects") or {}
    project_block = projects.get(project_path) or {}
    return dict(project_block.get("mcpServers") or {})


def write_claude_json_servers(
    scope: Literal["local", "user"],
    servers: dict[str, McpServerEntry],
    project_path: Optional[str] = None,
    target_path: Optional[Path] = None,
) -> None:
    """Write mcpServers into the given scope, preserving all other keys.

    Uses atomic write (temp file + rename) to avoid partial writes.
    """
    path = target_path or _default_target()

    # Load existing data (create empty dict if file missing)
    if path.exists():
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    else:
        data = {}

    if scope == "user":
        data["mcpServers"] = servers
    else:
        # local scope: projects[project_path].mcpServers
        if not project_path:
            raise ValueError("project_path is required for scope='local'")
        projects = data.setdefault("projects", {})
        project_block = projects.setdefault(project_path, {})
        project_block["mcpServers"] = servers

    _atomic_write_json(path, data)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON to path atomically via a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
