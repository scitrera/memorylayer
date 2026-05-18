"""Unit tests for surgical ~/.claude.json reader/writer.

Verifies that writes preserve unrelated keys, test against a fixture
~/.claude.json with theme + claude.user_id + multiple projects.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memorylayer_server.services.mcp_servers.claude_json import (
    read_claude_json_servers,
    write_claude_json_servers,
)

_FIXTURE_DATA = {
    "theme": "dark",
    "userID": "user_fixture_001",
    "autoUpdaterStatus": "enabled",
    "projects": {
        "/home/user/project-a": {
            "allowedTools": ["bash", "read"],
            "mcpServers": {
                "existing-server": {
                    "command": "npx",
                    "args": ["-y", "existing-mcp"],
                }
            },
        },
        "/home/user/project-b": {
            "allowedTools": ["write"],
        },
    },
    "mcpServers": {
        "global-server": {
            "command": "npx",
            "args": ["-y", "global-mcp"],
        }
    },
}


@pytest.fixture()
def claude_json_file(tmp_path: Path) -> Path:
    path = tmp_path / ".claude.json"
    path.write_text(json.dumps(_FIXTURE_DATA, indent=2), encoding="utf-8")
    return path


class TestReadClaudeJsonServers:
    def test_read_user_scope(self, claude_json_file: Path):
        servers = read_claude_json_servers("user", target_path=claude_json_file)
        assert "global-server" in servers
        assert servers["global-server"]["command"] == "npx"

    def test_read_local_scope(self, claude_json_file: Path):
        servers = read_claude_json_servers(
            "local", project_path="/home/user/project-a", target_path=claude_json_file
        )
        assert "existing-server" in servers

    def test_read_local_scope_missing_project(self, claude_json_file: Path):
        servers = read_claude_json_servers(
            "local", project_path="/home/user/no-such-project", target_path=claude_json_file
        )
        assert servers == {}

    def test_read_missing_file(self, tmp_path: Path):
        servers = read_claude_json_servers("user", target_path=tmp_path / "nonexistent.json")
        assert servers == {}

    def test_read_local_scope_no_project_path(self, claude_json_file: Path):
        servers = read_claude_json_servers("local", target_path=claude_json_file)
        assert servers == {}


class TestWriteClaudeJsonServers:
    def test_write_user_scope_preserves_unrelated_keys(self, claude_json_file: Path):
        new_servers = {
            "new-server": {"command": "node", "args": ["server.js"]},
        }
        write_claude_json_servers("user", new_servers, target_path=claude_json_file)

        data = json.loads(claude_json_file.read_text(encoding="utf-8"))
        # Unrelated keys preserved
        assert data["theme"] == "dark"
        assert data["userID"] == "user_fixture_001"
        assert data["autoUpdaterStatus"] == "enabled"
        # Projects preserved
        assert "/home/user/project-a" in data["projects"]
        assert "/home/user/project-b" in data["projects"]
        # New mcpServers written
        assert "new-server" in data["mcpServers"]
        assert "global-server" not in data["mcpServers"]

    def test_write_local_scope_preserves_unrelated_keys(self, claude_json_file: Path):
        new_servers = {
            "project-mcp": {"command": "npx", "args": ["-y", "project-tool"]},
        }
        write_claude_json_servers(
            "local", new_servers,
            project_path="/home/user/project-a",
            target_path=claude_json_file,
        )

        data = json.loads(claude_json_file.read_text(encoding="utf-8"))
        # Top-level user mcpServers untouched
        assert "global-server" in data["mcpServers"]
        # project-b allowedTools preserved
        assert data["projects"]["/home/user/project-b"]["allowedTools"] == ["write"]
        # project-a allowedTools preserved
        assert data["projects"]["/home/user/project-a"]["allowedTools"] == ["bash", "read"]
        # new mcp servers written
        assert "project-mcp" in data["projects"]["/home/user/project-a"]["mcpServers"]
        assert "existing-server" not in data["projects"]["/home/user/project-a"]["mcpServers"]

    def test_write_local_scope_creates_new_project(self, claude_json_file: Path):
        write_claude_json_servers(
            "local",
            {"brand-new": {"command": "python", "args": ["server.py"]}},
            project_path="/home/user/brand-new-project",
            target_path=claude_json_file,
        )

        data = json.loads(claude_json_file.read_text(encoding="utf-8"))
        assert "/home/user/brand-new-project" in data["projects"]
        assert "brand-new" in data["projects"]["/home/user/brand-new-project"]["mcpServers"]

    def test_write_creates_file_if_missing(self, tmp_path: Path):
        path = tmp_path / "new_claude.json"
        assert not path.exists()
        write_claude_json_servers("user", {"srv": {"command": "cat"}}, target_path=path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "srv" in data["mcpServers"]

    def test_write_local_scope_raises_without_project_path(self, claude_json_file: Path):
        with pytest.raises(ValueError, match="project_path"):
            write_claude_json_servers("local", {}, target_path=claude_json_file)

    def test_write_user_scope_overwrites_existing_mcpservers(self, claude_json_file: Path):
        write_claude_json_servers("user", {}, target_path=claude_json_file)
        data = json.loads(claude_json_file.read_text(encoding="utf-8"))
        assert data["mcpServers"] == {}
        # Other keys still intact
        assert data["theme"] == "dark"

    def test_atomic_write_produces_valid_json(self, claude_json_file: Path):
        write_claude_json_servers(
            "user",
            {"srv": {"command": "echo", "args": ["hello"]}},
            target_path=claude_json_file,
        )
        # Should parse cleanly
        data = json.loads(claude_json_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
