"""Unit tests for Skill and SkillFile domain models."""
import pytest
from pydantic import ValidationError

from memorylayer_server.models.skill import (
    Skill,
    SkillCreateInput,
    SkillFile,
    SkillFileInput,
    SkillUpdateInput,
    validate_skill_name,
)


# --- validate_skill_name ---

@pytest.mark.parametrize("name", [
    "pdf-processing",
    "my-skill",
    "a",
    "abc123",
    "extract-tables-from-pdfs",
])
def test_valid_skill_names(name):
    assert validate_skill_name(name) == name


@pytest.mark.parametrize("name,exc_fragment", [
    ("PDF", "lowercase"),
    ("-pdf", "hyphen"),
    ("pdf-", "hyphen"),
    ("pdf--proc", "consecutive"),
    ("", "empty"),
    ("a" * 65, "64 chars"),
    ("has space", "lowercase"),
    ("has_underscore", "lowercase"),
])
def test_invalid_skill_names(name, exc_fragment):
    with pytest.raises(ValueError, match=exc_fragment):
        validate_skill_name(name)


# --- Skill model ---

def make_skill(**overrides):
    defaults = dict(
        id="skl_aabbccddeeff",
        workspace_id="ws_test",
        name="pdf-processing",
        description="Extract text and tables from PDF files",
    )
    defaults.update(overrides)
    return Skill(**defaults)


def test_skill_defaults():
    s = make_skill()
    assert s.version == "0.1.0"
    assert s.source_mode == "server"
    assert s.enabled is True
    assert s.body == ""
    assert s.manifest_hash == ""
    assert s.bundle_hash == ""
    assert s.tenant_id == ""
    assert s.user_id is None


def test_skill_name_validator_rejects_invalid():
    with pytest.raises(ValidationError, match="hyphen|lowercase|empty|64"):
        make_skill(name="PDF-Processing")


def test_skill_description_empty_rejected():
    with pytest.raises(ValidationError):
        make_skill(description="")


def test_skill_description_too_long():
    with pytest.raises(ValidationError):
        make_skill(description="x" * 1025)


def test_skill_compatibility_too_long():
    with pytest.raises(ValidationError):
        make_skill(compatibility="x" * 501)


def test_skill_source_mode_literal():
    s = make_skill(source_mode="mirrored")
    assert s.source_mode == "mirrored"
    with pytest.raises(ValidationError):
        make_skill(source_mode="invalid")


# --- SkillFile model ---

def test_skill_file_basic():
    sf = SkillFile(
        id="sklf_aabbccddeeff",
        skill_id="skl_aabbccddeeff",
        path="scripts/extract.py",
        kind="script",
        content=b"print('hello')",
        content_hash="abc123",
        size_bytes=14,
    )
    assert sf.kind == "script"
    assert sf.mime_type is None


def test_skill_file_kind_literal():
    with pytest.raises(ValidationError):
        SkillFile(
            id="sklf_aabbccddeeff",
            skill_id="skl_aabbccddeeff",
            path="foo.py",
            kind="unknown_kind",
            content=b"",
            content_hash="",
            size_bytes=0,
        )


# --- SkillCreateInput ---

def test_skill_create_input_valid():
    inp = SkillCreateInput(name="my-skill", description="A useful skill")
    assert inp.version == "0.1.0"
    assert inp.source_mode == "server"


def test_skill_create_input_invalid_name():
    with pytest.raises(ValidationError):
        SkillCreateInput(name="My Skill", description="desc")


# --- SkillUpdateInput ---

def test_skill_update_input_all_optional():
    inp = SkillUpdateInput()
    assert inp.description is None
    assert inp.enabled is None


def test_skill_update_input_validates_description():
    with pytest.raises(ValidationError):
        SkillUpdateInput(description="x" * 1025)


# --- SkillFileInput ---

def test_skill_file_input():
    inp = SkillFileInput(path="references/REFERENCE.md", content=b"# ref")
    assert inp.mime_type is None


# --- OSS_KNOWN_SUBTYPES includes skill subtypes ---

def test_oss_known_subtypes_has_skill():
    from memorylayer_server.models.memory import OSS_KNOWN_SUBTYPES
    assert "skill" in OSS_KNOWN_SUBTYPES["*"]
    assert "skill_reference" in OSS_KNOWN_SUBTYPES["*"]
