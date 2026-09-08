"""A memory must never be OWNED by a machine (service/agent) connection principal.

_owner_user_id resolves the user_id stamped on a new memory: an explicit request
value wins, otherwise the context user — but a service (sv::) / agent (ag::)
principal is dropped to None so the memory is workspace-shared, not orphaned to a
principal no user recall will ever match.
"""

from memorylayer_server.api.v1.memories import _owner_user_id


def test_real_user_from_context_is_kept():
    assert _owner_user_id(None, "botwinick") == "botwinick"
    assert _owner_user_id(None, "dev") == "dev"


def test_explicit_request_user_wins():
    assert _owner_user_id("alice", "bob") == "alice"


def test_service_principal_from_context_is_dropped():
    assert _owner_user_id(None, "sv::platform-bridge::bridge-abc-botwinick") is None


def test_agent_principal_from_context_is_dropped():
    assert _owner_user_id(None, "ag::_sandbox::sahara") is None


def test_machine_principal_dropped_even_when_explicit():
    # A machine principal must not own a memory regardless of how it arrived.
    assert _owner_user_id("sv::platform-bridge::x", None) is None


def test_none_when_no_user():
    assert _owner_user_id(None, None) is None
