"""Unit tests for SQLite skill storage implementation."""
import hashlib
import tempfile
from pathlib import Path

import pytest

from memorylayer_server.models.skill import Skill, SkillFile
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend
from memorylayer_server.utils import generate_id, utc_now_iso


def _make_skill(**overrides) -> Skill:
    defaults = dict(
        id=generate_id("skl"),
        workspace_id="ws_test",
        name="pdf-processing",
        description="Extract text and tables from PDF files",
        version="0.1.0",
        body="## Usage\nUse this skill to extract tables.",
        source_mode="server",
    )
    defaults.update(overrides)
    return Skill(**defaults)


def _make_skill_file(skill_id: str, path: str = "scripts/extract.py", content: bytes = b"print('hi')") -> SkillFile:
    content_hash = hashlib.sha256(content).hexdigest()
    return SkillFile(
        id=generate_id("sklf"),
        skill_id=skill_id,
        path=path,
        kind="script",
        content=content,
        content_hash=content_hash,
        size_bytes=len(content),
        mime_type="text/x-python",
    )


@pytest.fixture
async def sqlite_backend(tmp_path):
    db_path = str(tmp_path / "test_skills.db")
    backend = SQLiteStorageBackend(db_path)
    await backend.connect()
    yield backend
    await backend.disconnect()


@pytest.mark.asyncio
class TestSkillCRUD:
    async def test_create_and_get_skill(self, sqlite_backend):
        skill = _make_skill()
        created = await sqlite_backend.create_skill(skill)
        assert created.id == skill.id
        assert created.name == "pdf-processing"
        assert created.workspace_id == "ws_test"

        fetched = await sqlite_backend.get_skill("ws_test", skill.id)
        assert fetched is not None
        assert fetched.id == skill.id
        assert fetched.description == skill.description

    async def test_get_skill_wrong_workspace(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)
        result = await sqlite_backend.get_skill("other_ws", skill.id)
        assert result is None

    async def test_get_skill_by_name(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)
        fetched = await sqlite_backend.get_skill_by_name("ws_test", "pdf-processing")
        assert fetched is not None
        assert fetched.id == skill.id

    async def test_get_skill_by_name_user_scoped(self, sqlite_backend):
        user_skill = _make_skill(id=generate_id("skl"), user_id="user_a")
        ws_skill = _make_skill(id=generate_id("skl"), user_id=None)
        await sqlite_backend.create_skill(user_skill)
        await sqlite_backend.create_skill(ws_skill)

        # user-scoped lookup
        found = await sqlite_backend.get_skill_by_name("ws_test", "pdf-processing", user_id="user_a")
        assert found is not None
        assert found.user_id == "user_a"

        # workspace-scoped lookup (no user_id)
        found_ws = await sqlite_backend.get_skill_by_name("ws_test", "pdf-processing")
        assert found_ws is not None
        assert found_ws.user_id is None

    async def test_list_skills(self, sqlite_backend):
        skill_a = _make_skill(id=generate_id("skl"), name="skill-a", description="First skill")
        skill_b = _make_skill(id=generate_id("skl"), name="skill-b", description="Second skill")
        await sqlite_backend.create_skill(skill_a)
        await sqlite_backend.create_skill(skill_b)

        skills = await sqlite_backend.list_skills("ws_test")
        names = {s.name for s in skills}
        assert "skill-a" in names
        assert "skill-b" in names

    async def test_list_skills_filter_enabled(self, sqlite_backend):
        enabled = _make_skill(id=generate_id("skl"), name="active-skill", description="Active")
        disabled = _make_skill(id=generate_id("skl"), name="disabled-skill", description="Disabled", enabled=False)
        await sqlite_backend.create_skill(enabled)
        await sqlite_backend.create_skill(disabled)

        active = await sqlite_backend.list_skills("ws_test", enabled=True)
        active_names = {s.name for s in active}
        assert "active-skill" in active_names
        assert "disabled-skill" not in active_names

    async def test_update_skill(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)

        updated = await sqlite_backend.update_skill("ws_test", skill.id, {"description": "Updated description"})
        assert updated is not None
        assert updated.description == "Updated description"

    async def test_update_skill_not_found(self, sqlite_backend):
        result = await sqlite_backend.update_skill("ws_test", "nonexistent_id", {"description": "x"})
        assert result is None

    async def test_delete_skill(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)

        deleted = await sqlite_backend.delete_skill("ws_test", skill.id)
        assert deleted is True

        fetched = await sqlite_backend.get_skill("ws_test", skill.id)
        assert fetched is None

    async def test_delete_skill_not_found(self, sqlite_backend):
        result = await sqlite_backend.delete_skill("ws_test", "nonexistent_id")
        assert result is False


@pytest.mark.asyncio
class TestSkillFiles:
    async def test_upsert_and_get_skill_file(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)

        sf = _make_skill_file(skill.id)
        upserted = await sqlite_backend.upsert_skill_file(sf)
        assert upserted.path == "scripts/extract.py"
        assert upserted.skill_id == skill.id
        assert upserted.content == b"print('hi')"

        fetched = await sqlite_backend.get_skill_file(skill.id, "scripts/extract.py")
        assert fetched is not None
        assert fetched.content_hash == sf.content_hash

    async def test_upsert_skill_file_updates_on_conflict(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)

        sf = _make_skill_file(skill.id, content=b"v1")
        await sqlite_backend.upsert_skill_file(sf)

        new_content = b"v2"
        sf2 = _make_skill_file(skill.id, content=new_content)
        sf2 = sf2.model_copy(update={"id": generate_id("sklf")})
        await sqlite_backend.upsert_skill_file(sf2)

        fetched = await sqlite_backend.get_skill_file(skill.id, "scripts/extract.py")
        assert fetched.content == b"v2"

    async def test_list_skill_files(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)

        sf1 = _make_skill_file(skill.id, "scripts/a.py", b"a")
        sf2 = _make_skill_file(skill.id, "references/ref.md", b"ref")
        sf2 = sf2.model_copy(update={"kind": "reference"})
        await sqlite_backend.upsert_skill_file(sf1)
        await sqlite_backend.upsert_skill_file(sf2)

        files = await sqlite_backend.list_skill_files(skill.id)
        paths = {f.path for f in files}
        assert "scripts/a.py" in paths
        assert "references/ref.md" in paths

    async def test_delete_skill_file(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)
        sf = _make_skill_file(skill.id)
        await sqlite_backend.upsert_skill_file(sf)

        deleted = await sqlite_backend.delete_skill_file(skill.id, "scripts/extract.py")
        assert deleted is True

        fetched = await sqlite_backend.get_skill_file(skill.id, "scripts/extract.py")
        assert fetched is None

    async def test_delete_skill_cascades_to_files(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)
        sf = _make_skill_file(skill.id)
        await sqlite_backend.upsert_skill_file(sf)

        await sqlite_backend.delete_skill("ws_test", skill.id)
        # FK cascade should have removed the file
        fetched = await sqlite_backend.get_skill_file(skill.id, "scripts/extract.py")
        assert fetched is None


@pytest.mark.asyncio
class TestFindSkillsByName:
    async def test_find_across_scopes(self, sqlite_backend):
        user_skill = _make_skill(id=generate_id("skl"), workspace_id="ws1", user_id="user_a")
        ws_skill = _make_skill(id=generate_id("skl"), workspace_id="ws1", user_id=None)
        await sqlite_backend.create_skill(user_skill)
        await sqlite_backend.create_skill(ws_skill)

        results = await sqlite_backend.find_skills_by_name(
            "pdf-processing",
            [
                {"workspace_id": "ws1", "user_id": "user_a"},
                {"workspace_id": "ws1"},
            ],
        )
        ids = {r.id for r in results}
        assert user_skill.id in ids
        assert ws_skill.id in ids

    async def test_find_skills_empty_scope_filters(self, sqlite_backend):
        skill = _make_skill()
        await sqlite_backend.create_skill(skill)
        results = await sqlite_backend.find_skills_by_name("pdf-processing", [])
        assert results == []

    async def test_find_skills_no_match(self, sqlite_backend):
        results = await sqlite_backend.find_skills_by_name(
            "nonexistent-skill", [{"workspace_id": "ws_test"}]
        )
        assert results == []
