"""
Unit tests for in-memory skill storage backend.

Tests skill CRUD + file operations against MemoryStorageBackend directly,
without requiring the full service stack.
"""

import hashlib
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.skill import Skill, SkillFile
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend

WORKSPACE_ID = "ws_skill_test"
USER_ID = "user_abc"


def _make_skill(
    name: str = "pdf-extractor",
    workspace_id: str = WORKSPACE_ID,
    user_id: str | None = None,
    enabled: bool = True,
) -> Skill:
    return Skill(
        id=f"skl_{name.replace('-', '')}001",
        workspace_id=workspace_id,
        user_id=user_id,
        name=name,
        description=f"Skill: {name}",
        body=f"## {name}\nDoes something useful.",
        enabled=enabled,
    )


def _make_skill_file(skill_id: str, path: str = "scripts/run.py", content: bytes = b"print('hello')") -> SkillFile:
    return SkillFile(
        id=f"sklf_{skill_id[-6:]}_{path.replace('/', '_')}",
        skill_id=skill_id,
        path=path,
        kind="script",
        content=content,
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
    )


@pytest.fixture
def backend():
    return MemoryStorageBackend()


@pytest.mark.asyncio
class TestSkillCRUD:
    """Test create / get / update / delete for skills."""

    async def test_create_and_get_skill(self, backend):
        skill = _make_skill()
        created = await backend.create_skill(skill)
        assert created.id == skill.id
        assert created.name == "pdf-extractor"

        fetched = await backend.get_skill(WORKSPACE_ID, skill.id)
        assert fetched is not None
        assert fetched.id == skill.id

    async def test_get_skill_returns_none_for_missing(self, backend):
        result = await backend.get_skill(WORKSPACE_ID, "skl_nonexistent")
        assert result is None

    async def test_get_skill_wrong_workspace(self, backend):
        skill = _make_skill()
        await backend.create_skill(skill)
        result = await backend.get_skill("other_ws", skill.id)
        assert result is None

    async def test_get_skill_by_name(self, backend):
        skill = _make_skill(name="table-parser")
        await backend.create_skill(skill)

        found = await backend.get_skill_by_name(WORKSPACE_ID, "table-parser")
        assert found is not None
        assert found.id == skill.id

    async def test_get_skill_by_name_with_user_filter(self, backend):
        skill = _make_skill(name="my-skill", user_id=USER_ID)
        await backend.create_skill(skill)

        # Match user_id
        found = await backend.get_skill_by_name(WORKSPACE_ID, "my-skill", user_id=USER_ID)
        assert found is not None

        # Wrong user_id returns None
        not_found = await backend.get_skill_by_name(WORKSPACE_ID, "my-skill", user_id="other_user")
        assert not_found is None

    async def test_get_skill_by_name_returns_none_for_missing(self, backend):
        result = await backend.get_skill_by_name(WORKSPACE_ID, "nonexistent-skill")
        assert result is None

    async def test_update_skill_fields(self, backend):
        skill = _make_skill(name="update-me")
        await backend.create_skill(skill)

        updated = await backend.update_skill(WORKSPACE_ID, skill.id, {"description": "Updated desc", "enabled": False})
        assert updated is not None
        assert updated.description == "Updated desc"
        assert updated.enabled is False

    async def test_update_skill_returns_none_for_missing(self, backend):
        result = await backend.update_skill(WORKSPACE_ID, "skl_missing", {"enabled": False})
        assert result is None

    async def test_update_skill_sets_updated_at(self, backend):
        skill = _make_skill(name="timestamp-test")
        original_time = skill.updated_at
        await backend.create_skill(skill)

        updated = await backend.update_skill(WORKSPACE_ID, skill.id, {"description": "New desc"})
        assert updated.updated_at >= original_time

    async def test_delete_skill(self, backend):
        skill = _make_skill(name="delete-me")
        await backend.create_skill(skill)

        result = await backend.delete_skill(WORKSPACE_ID, skill.id)
        assert result is True

        fetched = await backend.get_skill(WORKSPACE_ID, skill.id)
        assert fetched is None

    async def test_delete_skill_returns_false_for_missing(self, backend):
        result = await backend.delete_skill(WORKSPACE_ID, "skl_gone")
        assert result is False

    async def test_delete_skill_cascades_to_files(self, backend):
        skill = _make_skill(name="cascade-test")
        await backend.create_skill(skill)
        sf = _make_skill_file(skill.id)
        await backend.upsert_skill_file(sf)

        # Verify file exists before delete
        assert await backend.get_skill_file(skill.id, sf.path) is not None

        await backend.delete_skill(WORKSPACE_ID, skill.id)

        # File should be gone after skill delete
        assert await backend.get_skill_file(skill.id, sf.path) is None
        assert await backend.list_skill_files(skill.id) == []


@pytest.mark.asyncio
class TestListSkills:
    """Test list_skills filtering and pagination."""

    async def test_list_skills_basic(self, backend):
        ws = "ws_list_test"
        for name in ["alpha-skill", "beta-skill", "gamma-skill"]:
            await backend.create_skill(_make_skill(name=name, workspace_id=ws))

        results = await backend.list_skills(ws)
        assert len(results) == 3

    async def test_list_skills_filter_by_name(self, backend):
        ws = "ws_list_name"
        await backend.create_skill(_make_skill(name="named-skill", workspace_id=ws))
        await backend.create_skill(_make_skill(name="other-skill", workspace_id=ws))

        results = await backend.list_skills(ws, name="named-skill")
        assert len(results) == 1
        assert results[0].name == "named-skill"

    async def test_list_skills_filter_by_enabled(self, backend):
        ws = "ws_list_enabled"
        await backend.create_skill(_make_skill(name="active-skill", workspace_id=ws, enabled=True))
        await backend.create_skill(_make_skill(name="inactive-skill", workspace_id=ws, enabled=False))

        active = await backend.list_skills(ws, enabled=True)
        assert len(active) == 1
        assert active[0].name == "active-skill"

        inactive = await backend.list_skills(ws, enabled=False)
        assert len(inactive) == 1
        assert inactive[0].name == "inactive-skill"

    async def test_list_skills_filter_by_user_id(self, backend):
        ws = "ws_list_user"
        await backend.create_skill(_make_skill(name="user-skill", workspace_id=ws, user_id=USER_ID))
        await backend.create_skill(_make_skill(name="ws-skill", workspace_id=ws, user_id=None))

        user_skills = await backend.list_skills(ws, user_id=USER_ID)
        assert len(user_skills) == 1
        assert user_skills[0].name == "user-skill"

    async def test_list_skills_pagination(self, backend):
        ws = "ws_list_page"
        for i in range(5):
            await backend.create_skill(_make_skill(name=f"skill-{i:02d}", workspace_id=ws))

        page1 = await backend.list_skills(ws, limit=3, offset=0)
        page2 = await backend.list_skills(ws, limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2

    async def test_list_skills_empty_workspace(self, backend):
        results = await backend.list_skills("ws_empty_skills")
        assert results == []


@pytest.mark.asyncio
class TestFindSkillsByName:
    """Test find_skills_by_name for precedence resolution."""

    async def test_find_skills_across_scopes(self, backend):
        ws_global = "_global"
        ws_project = "ws_proj"

        global_skill = Skill(
            id="skl_global001",
            workspace_id=ws_global,
            name="shared-skill",
            description="Global version",
        )
        project_skill = Skill(
            id="skl_proj001",
            workspace_id=ws_project,
            name="shared-skill",
            description="Project version",
        )
        await backend.create_skill(global_skill)
        await backend.create_skill(project_skill)

        scope_filters = [
            {"workspace_id": ws_project},
            {"workspace_id": ws_global},
        ]
        results = await backend.find_skills_by_name("shared-skill", scope_filters)
        assert len(results) == 2
        ids = {s.id for s in results}
        assert "skl_global001" in ids
        assert "skl_proj001" in ids

    async def test_find_skills_by_name_no_match(self, backend):
        results = await backend.find_skills_by_name("nonexistent", [{"workspace_id": WORKSPACE_ID}])
        assert results == []

    async def test_find_skills_by_name_user_scope_filter(self, backend):
        ws = "ws_scope"
        user_skill = Skill(
            id="skl_user001",
            workspace_id=ws,
            user_id=USER_ID,
            name="scoped-skill",
            description="User scope",
        )
        ws_skill = Skill(
            id="skl_ws001",
            workspace_id=ws,
            user_id=None,
            name="scoped-skill",
            description="Workspace scope",
        )
        await backend.create_skill(user_skill)
        await backend.create_skill(ws_skill)

        # Filter for user scope only
        results = await backend.find_skills_by_name("scoped-skill", [{"workspace_id": ws, "user_id": USER_ID}])
        assert len(results) == 1
        assert results[0].id == "skl_user001"


@pytest.mark.asyncio
class TestSkillFiles:
    """Test skill file upsert / get / list / delete."""

    async def test_upsert_and_get_skill_file(self, backend):
        sf = _make_skill_file("skl_file001", "scripts/run.py")
        await backend.upsert_skill_file(sf)

        fetched = await backend.get_skill_file("skl_file001", "scripts/run.py")
        assert fetched is not None
        assert fetched.content == b"print('hello')"

    async def test_upsert_skill_file_overwrites(self, backend):
        original = _make_skill_file("skl_over001", "README.md", b"# Old")
        await backend.upsert_skill_file(original)

        updated_content = b"# New"
        updated = _make_skill_file("skl_over001", "README.md", updated_content)
        await backend.upsert_skill_file(updated)

        fetched = await backend.get_skill_file("skl_over001", "README.md")
        assert fetched.content == b"# New"

    async def test_list_skill_files(self, backend):
        skill_id = "skl_list_files"
        for path in ["scripts/a.py", "references/ref.md", "assets/logo.png"]:
            sf = _make_skill_file(skill_id, path, b"content")
            await backend.upsert_skill_file(sf)

        files = await backend.list_skill_files(skill_id)
        assert len(files) == 3
        paths = {f.path for f in files}
        assert paths == {"scripts/a.py", "references/ref.md", "assets/logo.png"}

    async def test_list_skill_files_empty(self, backend):
        files = await backend.list_skill_files("skl_no_files")
        assert files == []

    async def test_get_skill_file_returns_none_for_missing(self, backend):
        result = await backend.get_skill_file("skl_x", "nonexistent.py")
        assert result is None

    async def test_delete_skill_file(self, backend):
        sf = _make_skill_file("skl_del_file", "scripts/del.py")
        await backend.upsert_skill_file(sf)

        result = await backend.delete_skill_file("skl_del_file", "scripts/del.py")
        assert result is True

        assert await backend.get_skill_file("skl_del_file", "scripts/del.py") is None

    async def test_delete_skill_file_returns_false_for_missing(self, backend):
        result = await backend.delete_skill_file("skl_x", "nope.py")
        assert result is False

    async def test_skill_file_isolation_by_skill_id(self, backend):
        sf_a = _make_skill_file("skl_iso_a", "scripts/main.py", b"a")
        sf_b = _make_skill_file("skl_iso_b", "scripts/main.py", b"b")
        await backend.upsert_skill_file(sf_a)
        await backend.upsert_skill_file(sf_b)

        fetched_a = await backend.get_skill_file("skl_iso_a", "scripts/main.py")
        fetched_b = await backend.get_skill_file("skl_iso_b", "scripts/main.py")
        assert fetched_a.content == b"a"
        assert fetched_b.content == b"b"
