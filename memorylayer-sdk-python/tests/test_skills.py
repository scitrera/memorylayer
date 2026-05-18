"""Tests for MemoryLayer skills namespace (async and sync)."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from memorylayer import MemoryLayerClient
from memorylayer.skills import SkillModel, parse_skill_folder
from memorylayer.sync_client import SyncMemoryLayerClient

BASE_URL = "http://test.memorylayer.ai"

_SKILL_PAYLOAD = {
    "id": "skl_abc",
    "workspace_id": "ws_test",
    "tenant_id": "",
    "name": "my-skill",
    "description": "A test skill",
    "version": "0.1.0",
    "body": "# instructions",
    "metadata": {},
    "source_mode": "server",
    "manifest_hash": "",
    "bundle_hash": "h123",
    "enabled": True,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}


@pytest.fixture
def async_client() -> MemoryLayerClient:
    return MemoryLayerClient(
        base_url=BASE_URL,
        api_key="test_key",
        workspace_id="ws_test",
    )


@pytest.fixture
def sync_client_obj() -> SyncMemoryLayerClient:
    return SyncMemoryLayerClient(
        base_url=BASE_URL,
        api_key="test_key",
        workspace_id="ws_test",
    )


# ---------------------------------------------------------------------------
# parse_skill_folder
# ---------------------------------------------------------------------------


def test_parse_skill_folder_basic(tmp_path):
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Test\nversion: 1.0.0\n---\n# Body\n",
        encoding="utf-8",
    )
    manifest, files = parse_skill_folder(skill_dir)
    assert manifest["name"] == "my-skill"
    assert manifest["description"] == "Test"
    assert manifest["version"] == "1.0.0"
    assert "# Body" in manifest["body"]
    assert files == []


def test_parse_skill_folder_with_scripts(tmp_path):
    skill_dir = tmp_path / "s"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\n", encoding="utf-8")
    scripts = skill_dir / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_bytes(b"print('hi')")
    manifest, files = parse_skill_folder(skill_dir)
    paths = [p for p, _ in files]
    assert "scripts/run.py" in paths


def test_parse_skill_folder_missing_skill_md(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_skill_folder(tmp_path / "no-such-dir")


# ---------------------------------------------------------------------------
# Async SkillsAPI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_skills_list(async_client: MemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/skills").mock(return_value=Response(200, json={"skills": [_SKILL_PAYLOAD]}))
    async with async_client:
        skills = await async_client.skills.list()
    assert len(skills) == 1
    assert skills[0].name == "my-skill"
    assert isinstance(skills[0], SkillModel)


@pytest.mark.asyncio
@respx.mock
async def test_skills_get(async_client: MemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/skills/skl_abc").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    async with async_client:
        skill = await async_client.skills.get("skl_abc")
    assert skill.id == "skl_abc"


@pytest.mark.asyncio
@respx.mock
async def test_skills_save(async_client: MemoryLayerClient) -> None:
    respx.post(f"{BASE_URL}/v1/skills").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    async with async_client:
        skill = await async_client.skills.save(
            name="my-skill",
            description="A test skill",
            body="# instructions",
        )
    assert skill.name == "my-skill"


@pytest.mark.asyncio
@respx.mock
async def test_skills_delete(async_client: MemoryLayerClient) -> None:
    respx.delete(f"{BASE_URL}/v1/skills/skl_abc").mock(return_value=Response(204))
    async with async_client:
        await async_client.skills.delete("skl_abc")  # should not raise


@pytest.mark.asyncio
@respx.mock
async def test_skills_resolve_single(async_client: MemoryLayerClient) -> None:
    respx.post(f"{BASE_URL}/v1/skills/resolve").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    async with async_client:
        result = await async_client.skills.resolve(name="my-skill")
    assert isinstance(result, SkillModel)
    assert result.name == "my-skill"


@pytest.mark.asyncio
@respx.mock
async def test_skills_pull(async_client: MemoryLayerClient, tmp_path) -> None:
    respx.post(f"{BASE_URL}/v1/skills/resolve").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    respx.get(f"{BASE_URL}/v1/skills/skl_abc/manifest").mock(return_value=Response(200, text="---\nname: my-skill\n---\n# Body\n"))
    respx.get(f"{BASE_URL}/v1/skills/skl_abc/files").mock(return_value=Response(200, json={"files": []}))
    async with async_client:
        skill_dir = await async_client.skills.pull("my-skill", tmp_path)
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()


# ---------------------------------------------------------------------------
# Sync SyncSkillsAPI
# ---------------------------------------------------------------------------


@respx.mock
def test_sync_skills_list(sync_client_obj: SyncMemoryLayerClient) -> None:
    respx.get(f"{BASE_URL}/v1/skills").mock(return_value=Response(200, json={"skills": [_SKILL_PAYLOAD]}))
    with sync_client_obj:
        skills = sync_client_obj.skills.list()
    assert len(skills) == 1
    assert skills[0].name == "my-skill"


@respx.mock
def test_sync_skills_save(sync_client_obj: SyncMemoryLayerClient) -> None:
    respx.post(f"{BASE_URL}/v1/skills").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    with sync_client_obj:
        skill = sync_client_obj.skills.save(name="my-skill", description="A test skill")
    assert skill.name == "my-skill"


@respx.mock
def test_sync_skills_pull(sync_client_obj: SyncMemoryLayerClient, tmp_path) -> None:
    respx.post(f"{BASE_URL}/v1/skills/resolve").mock(return_value=Response(200, json={"skill": _SKILL_PAYLOAD}))
    respx.get(f"{BASE_URL}/v1/skills/skl_abc/manifest").mock(return_value=Response(200, text="---\nname: my-skill\n---\n# Body\n"))
    respx.get(f"{BASE_URL}/v1/skills/skl_abc/files").mock(return_value=Response(200, json={"files": []}))
    with sync_client_obj:
        skill_dir = sync_client_obj.skills.pull("my-skill", tmp_path)
    assert skill_dir.exists()
    assert (skill_dir / "SKILL.md").exists()
