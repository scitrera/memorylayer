"""Tests that updating a memory's content/aliases reconciles the full-text index.

With background tasks disabled (the test config), reconciliation runs inline via
the fallback path, so these assertions are deterministic. Queries go straight to
``storage.full_text_search`` to check the materialized FTS index directly.
"""

import uuid
from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Workspace


@pytest.fixture
async def iso_ws(storage_backend) -> str:
    ws = f"fts_reconcile_{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    await storage_backend.create_workspace(
        Workspace(id=ws, tenant_id="fts_tenant", name="FTS Reconcile Test", created_at=now, updated_at=now)
    )
    return ws


async def _fts_ids(storage_backend, ws: str, term: str) -> list[str]:
    return [m.id for m in await storage_backend.full_text_search(ws, term)]


@pytest.mark.asyncio
async def test_update_content_reconciles_fts(memory_service, storage_backend, iso_ws):
    m = await memory_service.remember(iso_ws, RememberInput(content="alpha zzqoldterm uniquemarker"))
    assert await _fts_ids(storage_backend, iso_ws, "zzqoldterm") == [m.id]

    await memory_service.update(iso_ws, m.id, content="beta zzqnewterm uniquemarker")

    # New content is searchable; the old term no longer matches (index reconciled).
    assert await _fts_ids(storage_backend, iso_ws, "zzqnewterm") == [m.id]
    assert await _fts_ids(storage_backend, iso_ws, "zzqoldterm") == []


@pytest.mark.asyncio
async def test_update_aliases_reconciles_fts(memory_service, storage_backend, iso_ws):
    m = await memory_service.remember(iso_ws, RememberInput(content="ceremonial structure", metadata={"aliases": ["zzqfoobar"]}))
    assert await _fts_ids(storage_backend, iso_ws, "zzqfoobar") == [m.id]

    await memory_service.update(iso_ws, m.id, metadata={"aliases": ["zzqmingtang"]})

    # New alias is searchable; the old alias is gone from the index.
    assert await _fts_ids(storage_backend, iso_ws, "zzqmingtang") == [m.id]
    assert await _fts_ids(storage_backend, iso_ws, "zzqfoobar") == []
