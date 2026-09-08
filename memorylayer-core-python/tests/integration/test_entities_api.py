"""Integration tests for the entity registry CRUD API (Phase 2.3b).

The registry ships DARK (``MEMORYLAYER_ENTITY_REGISTRY_ENABLED`` default OFF), so
the CRUD endpoints return 501 unless the flag is enabled. These tests toggle the
flag on the shared test ``Variables`` for the enabled-path cases and seed
entities directly through the registry service (accretion is off by default).
"""

import uuid
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from scitrera_app_framework import get_extension

from memorylayer_server.config import MEMORYLAYER_ENTITY_REGISTRY_ENABLED
from memorylayer_server.models.entity_registry import EntityType
from memorylayer_server.services.entity_registry import EXT_ENTITY_REGISTRY_SERVICE


@contextmanager
def _registry_enabled(v):
    """Temporarily flip the registry flag ON for the shared test Variables."""
    prior = v.get(MEMORYLAYER_ENTITY_REGISTRY_ENABLED, None)
    v.set(MEMORYLAYER_ENTITY_REGISTRY_ENABLED, "true")
    try:
        yield
    finally:
        if prior is None:
            v.set(MEMORYLAYER_ENTITY_REGISTRY_ENABLED, "false")
        else:
            v.set(MEMORYLAYER_ENTITY_REGISTRY_ENABLED, prior)


def test_entities_gated_when_disabled(test_client: TestClient, v) -> None:
    """With the flag OFF every CRUD endpoint returns 501."""
    # Ensure OFF (default), independent of test ordering.
    v.set(MEMORYLAYER_ENTITY_REGISTRY_ENABLED, "false")
    headers = {"X-Workspace-ID": f"ent_gate_{uuid.uuid4().hex[:8]}"}

    assert test_client.get("/v1/entities", headers=headers).status_code == 501
    assert test_client.get("/v1/entities/some-id", headers=headers).status_code == 501
    assert test_client.get("/v1/entities/resolve", params={"name": "Alice"}, headers=headers).status_code == 501
    assert (
        test_client.post(
            "/v1/entities/merge",
            json={"source_id": "a", "target_id": "b", "reason": "x"},
            headers=headers,
        ).status_code
        == 501
    )


@pytest.mark.asyncio
async def test_entities_crud_when_enabled(test_client: TestClient, v) -> None:
    """List / get / resolve / merge against a seeded registry with the flag ON."""
    workspace_id = f"ent_crud_{uuid.uuid4().hex[:8]}"
    headers = {"X-Workspace-ID": workspace_id}

    # Create a memory first so the workspace row exists (entity rows FK the
    # workspace; the memory-create endpoint auto-creates the workspace).
    seed = test_client.post("/v1/memories", json={"content": "seed memory"}, headers=headers)
    assert seed.status_code == 201

    registry = get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)

    # Seed two entities directly through the service (accretion is off).
    alice = await registry.upsert(workspace_id, "Alice", EntityType.PERSON, aliases=["Ally"])
    bob = await registry.upsert(workspace_id, "Bob", EntityType.PERSON)

    with _registry_enabled(v):
        # LIST
        listed = test_client.get("/v1/entities", params={"limit": 100}, headers=headers)
        assert listed.status_code == 200
        body = listed.json()
        names = {e["canonical_name"] for e in body["entities"]}
        assert {"Alice", "Bob"}.issubset(names)
        assert body["total_count"] == len(body["entities"])

        # GET one
        got = test_client.get(f"/v1/entities/{alice.id}", headers=headers)
        assert got.status_code == 200
        assert got.json()["entity"]["id"] == alice.id

        # GET unknown => 404
        assert test_client.get(f"/v1/entities/nope-{uuid.uuid4().hex}", headers=headers).status_code == 404

        # RESOLVE by canonical name
        res = test_client.get(
            "/v1/entities/resolve",
            params={"name": "Alice", "entity_type": "person"},
            headers=headers,
        )
        assert res.status_code == 200
        resolution = res.json()["resolution"]
        assert resolution["entity"]["id"] == alice.id
        assert resolution["matched_via"] == "exact"

        # RESOLVE by alias
        res_alias = test_client.get(
            "/v1/entities/resolve",
            params={"name": "Ally", "entity_type": "person"},
            headers=headers,
        )
        assert res_alias.status_code == 200
        assert res_alias.json()["resolution"]["entity"]["id"] == alice.id

        # RESOLVE miss => 404 (allow_create=False)
        res_miss = test_client.get(
            "/v1/entities/resolve",
            params={"name": "Nonexistent Person", "entity_type": "person"},
            headers=headers,
        )
        assert res_miss.status_code == 404

        # MERGE bob -> alice
        merged = test_client.post(
            "/v1/entities/merge",
            json={"source_id": bob.id, "target_id": alice.id, "reason": "duplicate"},
            headers=headers,
        )
        assert merged.status_code == 200
        assert merged.json()["entity"]["id"] == alice.id

        # After merge, default (active) list excludes the tombstoned source.
        relisted = test_client.get("/v1/entities", params={"limit": 100}, headers=headers)
        active_ids = {e["id"] for e in relisted.json()["entities"]}
        assert bob.id not in active_ids
        assert alice.id in active_ids
