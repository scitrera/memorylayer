"""Unit tests for SkillsService and frontmatter utilities."""
import pytest

from memorylayer_server.models.skill import Skill, SkillCreateInput, SkillFile, SkillUpdateInput
from memorylayer_server.services.skills import SkillsService
from memorylayer_server.services.skills.frontmatter import parse_skill_md, render_skill_md


# ---------------------------------------------------------------------------
# Minimal in-memory storage stub for SkillsService tests
# ---------------------------------------------------------------------------

class _SkillStore:
    """Minimal skills-capable storage stub (no full StorageBackend required)."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._files: dict[str, SkillFile] = {}  # key: f"{skill_id}:{path}"

    async def connect(self): pass
    async def disconnect(self): pass
    async def health_check(self): return True

    # Stub mandatory abstract methods (not needed for skills tests)
    async def create_memory(self, *a, **kw): raise NotImplementedError
    async def get_memory(self, *a, **kw): return None
    async def update_memory(self, *a, **kw): return None
    async def delete_memory(self, *a, **kw): return False
    async def search_memories(self, *a, **kw): return []
    async def full_text_search(self, *a, **kw): return []
    async def get_memory_by_hash(self, *a, **kw): return None
    async def get_recent_memories(self, *a, **kw): return []
    async def create_association(self, *a, **kw): raise NotImplementedError
    async def get_associations(self, *a, **kw): return []
    async def traverse_graph(self, *a, **kw): raise NotImplementedError
    async def create_workspace(self, *a, **kw): raise NotImplementedError
    async def get_workspace(self, *a, **kw): return None
    async def create_context(self, *a, **kw): raise NotImplementedError
    async def get_context(self, *a, **kw): return None
    async def list_contexts(self, *a, **kw): return []
    async def list_workspaces(self, *a, **kw): return []
    async def get_workspace_stats(self, *a, **kw): return {}
    async def create_session(self, *a, **kw): raise NotImplementedError
    async def get_session(self, *a, **kw): return None
    async def get_session_by_id(self, *a, **kw): return None
    async def delete_session(self, *a, **kw): return False
    async def set_working_memory(self, *a, **kw): raise NotImplementedError
    async def get_working_memory(self, *a, **kw): return None
    async def get_all_working_memory(self, *a, **kw): return []
    async def cleanup_expired_sessions(self, *a, **kw): return 0

    # Skill operations
    async def create_skill(self, skill: Skill) -> Skill:
        self._skills[skill.id] = skill
        return skill

    async def get_skill(self, workspace_id: str, skill_id: str):
        s = self._skills.get(skill_id)
        return s if s and s.workspace_id == workspace_id else None

    async def get_skill_by_name(self, workspace_id, name, user_id=None):
        for s in self._skills.values():
            if s.workspace_id == workspace_id and s.name == name:
                if user_id is None or s.user_id == user_id:
                    return s
        return None

    async def list_skills(self, workspace_id, user_id=None, name=None, tags=None, enabled=None, limit=100, offset=0):
        results = [s for s in self._skills.values() if s.workspace_id == workspace_id]
        if user_id is not None:
            results = [s for s in results if s.user_id == user_id]
        if name is not None:
            results = [s for s in results if s.name == name]
        if enabled is not None:
            results = [s for s in results if s.enabled == enabled]
        return results[offset: offset + limit]

    async def find_skills_by_name(self, name, scope_filters):
        results = []
        for sf in scope_filters:
            ws = sf["workspace_id"]
            uid = sf.get("user_id")
            for s in self._skills.values():
                if s.name == name and s.workspace_id == ws:
                    if uid is None or s.user_id == uid:
                        results.append(s)
        return results

    async def update_skill(self, workspace_id, skill_id, updates):
        s = self._skills.get(skill_id)
        if s is None or s.workspace_id != workspace_id:
            return None
        updated = s.model_copy(update=updates)
        self._skills[skill_id] = updated
        return updated

    async def delete_skill(self, workspace_id, skill_id):
        s = self._skills.get(skill_id)
        if s and s.workspace_id == workspace_id:
            del self._skills[skill_id]
            # cascade files
            to_del = [k for k in self._files if k.startswith(f"{skill_id}:")]
            for k in to_del:
                del self._files[k]
            return True
        return False

    async def upsert_skill_file(self, skill_file: SkillFile) -> SkillFile:
        key = f"{skill_file.skill_id}:{skill_file.path}"
        self._files[key] = skill_file
        return skill_file

    async def get_skill_file(self, skill_id, path):
        return self._files.get(f"{skill_id}:{path}")

    async def list_skill_files(self, skill_id):
        return [f for k, f in self._files.items() if k.startswith(f"{skill_id}:")]

    async def delete_skill_file(self, skill_id, path):
        key = f"{skill_id}:{path}"
        if key in self._files:
            del self._files[key]
            return True
        return False


def make_service():
    storage = _SkillStore()
    return SkillsService(storage=storage), storage


# ---------------------------------------------------------------------------
# SkillsService CRUD tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_and_get_skill():
    svc, _ = make_service()
    inp = SkillCreateInput(name="pdf-processing", description="Extract PDFs")
    skill = await svc.create_skill(inp, workspace_id="ws1")
    assert skill.id.startswith("skl_")
    assert skill.name == "pdf-processing"
    assert skill.manifest_hash != ""

    fetched = await svc.get_skill("ws1", skill.id)
    assert fetched is not None
    assert fetched.id == skill.id


@pytest.mark.asyncio
async def test_create_skill_manifest_hash_computed():
    svc, _ = make_service()
    inp = SkillCreateInput(name="my-skill", description="Does things", body="# Instructions\nDo the thing.")
    skill = await svc.create_skill(inp, workspace_id="ws1")
    assert len(skill.manifest_hash) == 64  # SHA-256 hex


@pytest.mark.asyncio
async def test_list_skills():
    svc, _ = make_service()
    await svc.create_skill(SkillCreateInput(name="skill-a", description="A"), workspace_id="ws1")
    await svc.create_skill(SkillCreateInput(name="skill-b", description="B"), workspace_id="ws1")
    await svc.create_skill(SkillCreateInput(name="skill-c", description="C"), workspace_id="ws2")

    ws1_skills = await svc.list_skills("ws1")
    assert len(ws1_skills) == 2
    names = {s.name for s in ws1_skills}
    assert names == {"skill-a", "skill-b"}


@pytest.mark.asyncio
async def test_update_skill():
    svc, _ = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="my-skill", description="Original"), workspace_id="ws1"
    )
    old_hash = skill.manifest_hash

    updated = await svc.update_skill(
        "ws1", skill.id, SkillUpdateInput(description="Updated description")
    )
    assert updated is not None
    assert updated.description == "Updated description"
    assert updated.manifest_hash != old_hash


@pytest.mark.asyncio
async def test_delete_skill():
    svc, storage = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="delete-me", description="Temp"), workspace_id="ws1"
    )
    assert await svc.delete_skill("ws1", skill.id) is True
    assert await svc.get_skill("ws1", skill.id) is None


@pytest.mark.asyncio
async def test_delete_skill_not_found():
    svc, _ = make_service()
    assert await svc.delete_skill("ws1", "skl_nonexistent") is False


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_and_get_file():
    svc, _ = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="file-skill", description="Has files"), workspace_id="ws1"
    )
    content = b"print('hello')"
    sf = await svc.upsert_file(skill.id, "scripts/hello.py", content, workspace_id="ws1")
    assert sf.kind == "script"
    assert sf.size_bytes == len(content)
    assert len(sf.content_hash) == 64

    fetched = await svc.get_file(skill.id, "scripts/hello.py")
    assert fetched is not None
    assert fetched.content == content


@pytest.mark.asyncio
async def test_bundle_hash_updates_on_upsert():
    svc, storage = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="bundle-skill", description="Bundle"), workspace_id="ws1"
    )
    assert skill.bundle_hash == ""

    await svc.upsert_file(skill.id, "scripts/a.py", b"a", workspace_id="ws1")
    updated = await storage.get_skill("ws1", skill.id)
    hash1 = updated.bundle_hash
    assert hash1 != ""

    await svc.upsert_file(skill.id, "scripts/b.py", b"b", workspace_id="ws1")
    updated2 = await storage.get_skill("ws1", skill.id)
    assert updated2.bundle_hash != hash1


@pytest.mark.asyncio
async def test_file_kind_inference():
    svc, _ = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="kind-skill", description="Kinds"), workspace_id="ws1"
    )
    cases = [
        ("scripts/run.py", "script"),
        ("references/GUIDE.md", "reference"),
        ("assets/logo.png", "asset"),
        ("README.md", "other"),
    ]
    for path, expected_kind in cases:
        sf = await svc.upsert_file(skill.id, path, b"x")
        assert sf.kind == expected_kind, f"{path} -> expected {expected_kind}, got {sf.kind}"


@pytest.mark.asyncio
async def test_delete_file():
    svc, _ = make_service()
    skill = await svc.create_skill(
        SkillCreateInput(name="rm-skill", description="Remove"), workspace_id="ws1"
    )
    await svc.upsert_file(skill.id, "scripts/x.py", b"x")
    assert await svc.delete_file(skill.id, "scripts/x.py") is True
    assert await svc.get_file(skill.id, "scripts/x.py") is None


# ---------------------------------------------------------------------------
# memory_indexer hook
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_indexer_called_on_create():
    indexed = []

    async def indexer(skill: Skill):
        indexed.append(skill.id)

    storage = _SkillStore()
    svc = SkillsService(storage=storage, memory_indexer=indexer)
    skill = await svc.create_skill(
        SkillCreateInput(name="indexed-skill", description="Will be indexed"), workspace_id="ws1"
    )
    assert skill.id in indexed


@pytest.mark.asyncio
async def test_memory_indexer_not_required():
    svc, _ = make_service()  # no indexer
    skill = await svc.create_skill(
        SkillCreateInput(name="no-index", description="No indexer"), workspace_id="ws1"
    )
    assert skill is not None


# ---------------------------------------------------------------------------
# Frontmatter parse / render round-trip
# ---------------------------------------------------------------------------

def test_frontmatter_roundtrip_basic():
    fm = {"name": "pdf-processing", "description": "Extract PDFs", "version": "1.0.0"}
    body = "# Instructions\n\nDo the thing."
    rendered = render_skill_md(fm, body)
    parsed_fm, parsed_body = parse_skill_md(rendered)
    assert parsed_fm["name"] == "pdf-processing"
    assert parsed_fm["version"] == "1.0.0"
    assert "Do the thing." in parsed_body


def test_frontmatter_no_frontmatter():
    text = "Just a body with no frontmatter."
    fm, body = parse_skill_md(text)
    assert fm == {}
    assert body == text


def test_frontmatter_empty_body():
    fm = {"name": "x", "description": "y"}
    rendered = render_skill_md(fm, "")
    parsed_fm, parsed_body = parse_skill_md(rendered)
    assert parsed_fm["name"] == "x"


def test_frontmatter_stable_key_order():
    fm = {"version": "1.0", "name": "b-skill", "description": "desc"}
    rendered = render_skill_md(fm, "")
    lines = rendered.splitlines()
    # name should come before description, description before version
    name_idx = next(i for i, l in enumerate(lines) if l.startswith("name:"))
    desc_idx = next(i for i, l in enumerate(lines) if l.startswith("description:"))
    ver_idx = next(i for i, l in enumerate(lines) if l.startswith("version:"))
    assert name_idx < desc_idx < ver_idx


def test_frontmatter_roundtrip_multiple_passes():
    fm = {"name": "my-skill", "description": "A skill", "version": "0.2.0", "license": "MIT"}
    body = "## Usage\n\nRun the skill."
    text = render_skill_md(fm, body)
    fm2, body2 = parse_skill_md(text)
    text2 = render_skill_md(fm2, body2)
    # Second render should be identical to first
    assert text == text2
