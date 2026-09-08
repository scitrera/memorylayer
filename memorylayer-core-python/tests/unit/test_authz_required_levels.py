"""Regression tests for resource/action -> required access level resolution.

Guards the fix for the bug where un-enumerated resources (e.g. skills) fell
through to the conservative MANAGE default, so an ordinary READ wrongly required
MANAGE(30). Reads now resolve to READ via the action-aware default; writes and
other non-read actions of un-enumerated resources stay at the conservative
MANAGE default (no security weakening), and explicit overrides are preserved.
"""

import pytest

from memorylayer_server.services.authorization.aether import (
    ACCESS_ADMIN,
    ACCESS_MANAGE,
    ACCESS_READ,
    ACCESS_READWRITE,
    get_required_access_level as req,
)


def test_skills_read_is_read_not_manage():
    # The reported bug: skills read previously required MANAGE(30).
    assert req("skills", "read") == ACCESS_READ


def test_skills_write_stays_conservative_and_mcp_is_read():
    # write of an un-enumerated resource stays MANAGE (policy decision, unchanged)
    assert req("skills", "write") == ACCESS_MANAGE
    assert req("skills", "mcp") == ACCESS_READ  # non-standard action, explicit READ


@pytest.mark.parametrize(
    "resource", ["applications", "data_providers", "collections", "mcp_servers", "knowledgebase", "graph", "users"]
)
def test_unmapped_resource_read_is_read(resource):
    assert req(resource, "read") == ACCESS_READ


@pytest.mark.parametrize("action,level", [
    ("read", ACCESS_READ),
    ("list", ACCESS_READ),
    # Non-read actions of un-enumerated resources stay MANAGE (conservative).
    ("write", ACCESS_MANAGE),
    ("create", ACCESS_MANAGE),
    ("execute", ACCESS_MANAGE),
    ("delete", ACCESS_MANAGE),
])
def test_action_defaults(action, level):
    assert req("some_unmapped_resource", action) == level


def test_unknown_action_stays_conservative():
    assert req("skills", "frobnicate") == ACCESS_MANAGE


def test_explicit_overrides_preserved():
    # Explicit map entries win over the action-aware default in BOTH directions:
    # documents write/delete are READWRITE even though the default for a non-read
    # action is the conservative MANAGE (see the authz map: a principal with write
    # access to a workspace manages that workspace's documents).
    assert req("documents", "write") == ACCESS_READWRITE
    assert req("documents", "delete") == ACCESS_READWRITE
    # Deliberately-strict explicit entries must not be relaxed.
    assert req("memories", "read") == ACCESS_READ
    assert req("memories", "write") == ACCESS_READWRITE
    assert req("admin", "read") == ACCESS_ADMIN
    assert req("admin", "write") == ACCESS_ADMIN


def test_wildcard_is_admin():
    assert req("anything", "*") == ACCESS_ADMIN
