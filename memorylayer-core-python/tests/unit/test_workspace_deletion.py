"""Workspace deletion is truthful, complete, and backend-conformant."""

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import RememberInput
from memorylayer_server.models.workspace import Context, Workspace
from memorylayer_server.services.storage.in_memory import MemoryStorageBackend
from memorylayer_server.services.storage.sqlite import SQLiteStorageBackend


def _workspace(workspace_id: str) -> Workspace:
    now = datetime.now(UTC)
    return Workspace(
        id=workspace_id,
        tenant_id="test-tenant",
        name=workspace_id,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_in_memory_delete_workspace_removes_owned_resources():
    backend = MemoryStorageBackend()
    workspace_id = "delete-memory-workspace"
    await backend.create_workspace(_workspace(workspace_id))
    await backend.create_context(
        workspace_id,
        Context(id="delete-context", workspace_id=workspace_id, name="delete-context"),
    )
    memory = await backend.create_memory(workspace_id, RememberInput(content="remove me"))

    assert await backend.delete_workspace(workspace_id) is True
    assert await backend.get_workspace(workspace_id) is None
    assert await backend.get_context(workspace_id, "delete-context") is None
    assert await backend.get_memory(workspace_id, memory.id) is None
    assert await backend.delete_workspace(workspace_id) is False


@pytest.mark.asyncio
async def test_sqlite_delete_workspace_purges_every_workspace_scoped_table(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "workspace-delete.db"))
    await backend.connect()
    try:
        workspace_id = "delete-sqlite-workspace"
        await backend.create_workspace(_workspace(workspace_id))
        await backend.create_context(
            workspace_id,
            Context(id="delete-context", workspace_id=workspace_id, name="delete-context"),
        )
        await backend.create_memory(workspace_id, RememberInput(content="remove me"))

        assert await backend.delete_workspace(workspace_id) is True
        assert await backend.get_workspace(workspace_id) is None

        tables = await (
            await backend._connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ).fetchall()
        for row in tables:
            table_name = str(row["name"])
            quoted = '"' + table_name.replace('"', '""') + '"'
            columns = await (
                await backend._connection.execute(f"PRAGMA table_info({quoted})")
            ).fetchall()
            if not any(str(column["name"]) == "workspace_id" for column in columns):
                continue
            remaining = await (
                await backend._connection.execute(
                    f"SELECT COUNT(*) FROM {quoted} WHERE workspace_id = ?",
                    (workspace_id,),
                )
            ).fetchone()
            assert remaining[0] == 0, table_name

        assert await backend.delete_workspace(workspace_id) is False
    finally:
        await backend.disconnect()
