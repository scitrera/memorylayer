"""Integration tests for the representation read-surface endpoint (P4.3).

The endpoint (POST /v1/representation) is a THIN PASS-THROUGH to
``RepresentationService.get_representation``. These tests assert the
ENDPOINT-specific contract:

  * dark-gate: flag OFF (the default) -> 404 (surface disabled).
  * flag ON -> the endpoint returns the service's Representation for
    (observer, subject), serialized correctly (observations / profile /
    derived_beliefs / provenance present).
  * leakage-0 pass-through: the endpoint returns EXACTLY what the service
    assembled — no other-observer content is added by the endpoint (it adds no
    scoping logic of its own). We assert this with a stub service whose
    leakage-safe scope is known, so the test isolates the endpoint behavior from
    the registry/storage wiring (the service's own leakage-0 logic is covered by
    tests/unit/test_representation_service.py).
  * self path (observer == subject) serializes with is_self=True.

The service is stubbed via FastAPI ``dependency_overrides`` so the test is
deterministic and does not require the entity registry to be globally enabled.
The dark-gate flag is toggled on the app's shared Variables and restored after.
"""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from memorylayer_server.api.v1.representation import get_representation_svc
from memorylayer_server.config import MEMORYLAYER_REPRESENTATION_ENABLED
from memorylayer_server.models.entity_registry import Entity, EntityType
from memorylayer_server.models.representation import (
    Observation,
    Representation,
    RepresentationProfile,
    UserRepresentation,
)

WS = "test_workspace"


def _entity(name: str, ent_id: str) -> Entity:
    now = datetime.now(UTC)
    return Entity(
        id=ent_id,
        workspace_id=WS,
        entity_type=EntityType.PERSON,
        canonical_name=name,
        normalized_name=name.lower(),
        created_at=now,
        updated_at=now,
    )


class _StubRepresentationService:
    """Returns a fixed, leakage-safe Representation for (observer, subject).

    The stub's observations are ONLY the observer's own authored-about-subject
    statements (the intersection scope) — there is no other-observer content to
    leak, so if the endpoint returns exactly these, the pass-through preserved
    leakage-0 (added nothing)."""

    async def get_representation(
        self, workspace_id, observer, subject, *, observer_type=None, subject_type=None, limit=20, include_profile=True
    ) -> Representation:
        is_self = observer == subject
        obs = [
            Observation(memory_id="m1", content=f"{observer} on {subject}: positive", role="self"),
            Observation(memory_id="m2", content=f"{observer} on {subject}: reliable", role="self"),
        ]
        profile = (
            RepresentationProfile(summary="assembled profile", source_memory_ids=["m1", "m2"], derived=False)
            if include_profile
            else None
        )
        return Representation(
            observer=_entity(observer, "obs-1"),
            subject=_entity(subject, "obs-1" if is_self else "subj-1"),
            is_self=is_self,
            observations=obs,
            profile=profile,
            derived_beliefs=[],
            provenance={"scoping_mode": "self" if is_self else "intersection", "observation_count": len(obs)},
        )

    async def get_user_representation(
        self, user_id, *, limit=20, include_profile=True
    ) -> UserRepresentation:
        obs = [
            Observation(memory_id="u1", content=f"{user_id}: prefers dark mode", role="self"),
            Observation(memory_id="u2", content=f"{user_id}: likes concise answers", role="self"),
        ]
        profile = (
            RepresentationProfile(summary="user profile", source_memory_ids=["u1", "u2"], derived=False)
            if include_profile
            else None
        )
        return UserRepresentation(
            user_id=user_id,
            observations=obs,
            profile=profile,
            derived_beliefs=[],
            provenance={"scoping_mode": "user", "user_id": user_id, "origin_workspace_ids": ["ws-1", "ws-2"]},
        )


@pytest.fixture
def workspace_headers() -> dict[str, str]:
    return {"X-Workspace-ID": WS}


@pytest.fixture
def representation_enabled(test_client: TestClient):
    """Enable the dark-gated representation surface for the duration of a test."""
    v = test_client.app.state.v
    prev = v.get(MEMORYLAYER_REPRESENTATION_ENABLED, default=None)
    v.set(MEMORYLAYER_REPRESENTATION_ENABLED, "true")
    yield
    if prev is None:
        v.set(MEMORYLAYER_REPRESENTATION_ENABLED, "false")
    else:
        v.set(MEMORYLAYER_REPRESENTATION_ENABLED, prev)


@pytest.fixture
def stub_service(test_client: TestClient):
    """Override the representation service dependency with the deterministic stub."""
    app = test_client.app
    app.dependency_overrides[get_representation_svc] = lambda: _StubRepresentationService()
    yield
    app.dependency_overrides.pop(get_representation_svc, None)


class TestRepresentationDarkGate:
    def test_disabled_by_default_returns_404(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Flag OFF (default) -> the surface is dark (404), even with valid input."""
        # Defensive: ensure the flag is off for this assertion regardless of order.
        test_client.app.state.v.set(MEMORYLAYER_REPRESENTATION_ENABLED, "false")
        response = test_client.post(
            "/v1/representation",
            json={"observer": "Alice", "subject": "Bob"},
            headers=workspace_headers,
        )
        assert response.status_code == 404


class TestRepresentationEnabled:
    def test_returns_representation_for_observer_subject(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """Flag ON -> the endpoint returns the assembled Representation, serialized."""
        response = test_client.post(
            "/v1/representation",
            json={"observer": "Alice", "subject": "Bob"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        data = response.json()
        # DTO serializes correctly: observer/subject/observations/profile/derived/provenance.
        assert data["observer"]["canonical_name"] == "Alice"
        assert data["subject"]["canonical_name"] == "Bob"
        assert data["is_self"] is False
        assert len(data["observations"]) == 2
        assert data["profile"]["summary"] == "assembled profile"
        assert data["derived_beliefs"] == []
        assert data["provenance"]["scoping_mode"] == "intersection"

    def test_leakage_zero_passthrough(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """The endpoint returns EXACTLY the service's observations — it adds no
        content (so no other-observer leakage can be introduced by the endpoint)."""
        response = test_client.post(
            "/v1/representation",
            json={"observer": "Alice", "subject": "Bob"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        contents = {o["content"] for o in response.json()["observations"]}
        # Only Alice's own authored-about-Bob statements; nothing else injected.
        assert contents == {"Alice on Bob: positive", "Alice on Bob: reliable"}
        # Every returned observation is observer-authored ("self") — no foreign rows.
        assert all(o["role"] == "self" for o in response.json()["observations"])

    def test_self_path_is_self_true(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """observer == subject -> the self-report view (is_self True)."""
        response = test_client.post(
            "/v1/representation",
            json={"observer": "Alice", "subject": "Alice"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_self"] is True
        assert data["provenance"]["scoping_mode"] == "self"

    def test_include_profile_false_omits_profile(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """include_profile=False is threaded through to the service."""
        response = test_client.post(
            "/v1/representation",
            json={"observer": "Alice", "subject": "Bob", "include_profile": False},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        assert response.json()["profile"] is None


class TestUserRepresentationDarkGate:
    def test_disabled_by_default_returns_404(self, test_client: TestClient, workspace_headers: dict[str, str]) -> None:
        """Flag OFF (default) -> the user-scope surface is dark (404)."""
        test_client.app.state.v.set(MEMORYLAYER_REPRESENTATION_ENABLED, "false")
        response = test_client.post(
            "/v1/representation/user",
            json={"user_id": "user-A"},
            headers=workspace_headers,
        )
        assert response.status_code == 404


class TestUserRepresentationEnabled:
    def test_returns_user_representation(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """Flag ON -> the endpoint returns the assembled UserRepresentation."""
        response = test_client.post(
            "/v1/representation/user",
            json={"user_id": "user-A"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-A"
        assert len(data["observations"]) == 2
        assert data["profile"]["summary"] == "user profile"
        assert data["derived_beliefs"] == []
        assert data["provenance"]["scoping_mode"] == "user"
        assert data["provenance"]["origin_workspace_ids"] == ["ws-1", "ws-2"]

    def test_cross_user_isolation_passthrough(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        """The endpoint returns EXACTLY the requested user's observations — it
        adds no content, so no other-user data can be introduced by the endpoint
        (the forced user_id filter is the service's; this asserts pass-through)."""
        response = test_client.post(
            "/v1/representation/user",
            json={"user_id": "user-A"},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        contents = {o["content"] for o in response.json()["observations"]}
        assert contents == {"user-A: prefers dark mode", "user-A: likes concise answers"}
        # No other user's tokens anywhere.
        assert all("user-A" in o["content"] for o in response.json()["observations"])

    def test_include_profile_false_omits_profile(
        self, test_client: TestClient, workspace_headers: dict[str, str], representation_enabled, stub_service
    ) -> None:
        response = test_client.post(
            "/v1/representation/user",
            json={"user_id": "user-A", "include_profile": False},
            headers=workspace_headers,
        )
        assert response.status_code == 200
        assert response.json()["profile"] is None
