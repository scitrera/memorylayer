"""Unit tests for services/skills/sync.py."""

from memorylayer_server.services.skills.sync import compute_sync_action

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def test_in_sync_when_all_hashes_match():
    result = compute_sync_action(_HASH_A, _HASH_B, _HASH_A, _HASH_B)
    assert result.action == "in_sync"


def test_pull_when_client_has_no_data():
    result = compute_sync_action(_HASH_A, _HASH_B, "", "")
    assert result.action == "pull"
    assert result.server_manifest_hash == _HASH_A
    assert result.server_bundle_hash == _HASH_B


def test_push_when_server_has_no_data():
    result = compute_sync_action("", "", _HASH_A, _HASH_B)
    assert result.action == "push"


def test_conflict_when_manifest_differs():
    result = compute_sync_action(_HASH_A, _HASH_B, _HASH_C, _HASH_B)
    assert result.action == "conflict"
    assert "manifest" in result.reason


def test_conflict_when_bundle_differs():
    result = compute_sync_action(_HASH_A, _HASH_B, _HASH_A, _HASH_D)
    assert result.action == "conflict"
    assert "bundle" in result.reason


def test_conflict_when_both_differ():
    result = compute_sync_action(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
    assert result.action == "conflict"
    assert "manifest" in result.reason
    assert "bundle" in result.reason


def test_in_sync_both_empty():
    result = compute_sync_action("", "", "", "")
    assert result.action == "in_sync"


def test_result_carries_server_hashes():
    result = compute_sync_action(_HASH_A, _HASH_B, _HASH_C, _HASH_D)
    assert result.server_manifest_hash == _HASH_A
    assert result.server_bundle_hash == _HASH_B
