"""Atomic workspace deletion for SQLite-compatible storage backends."""

from typing import Any

_KNOWN_CHILDREN_BY_PARENT: dict[str, set[str]] = {
    "chat_threads": {"chat_messages"},
    "documents": {"document_pages"},
    "entities": {"entity_aliases", "entity_members"},
    "memories": {"contradictions", "entity_members", "memory_associations"},
}


def _quoted_identifier(identifier: str) -> str:
    """Quote an identifier discovered from the database schema."""
    return '"' + identifier.replace('"', '""') + '"'


async def purge_workspace(
    connection: Any,
    workspace_id: str,
    *,
    discover_foreign_keys: bool = True,
) -> bool:
    """Delete a workspace and every row carrying its workspace boundary.

    SQLite and libSQL schemas contain both foreign-keyed and intentionally
    denormalized workspace-scoped tables.  Discovering the latter from the
    schema keeps deletion complete as new workspace-owned resources are added.
    Foreign-key dependencies are ordered child-first; indirectly owned rows
    such as skill files and working memory are removed by their declared
    ``ON DELETE CASCADE`` constraints.

    The operation is one transaction.  A failure rolls back the entire purge,
    so callers never receive success for a partially deleted workspace.
    """
    await connection.execute("BEGIN")
    try:
        exists_cursor = await connection.execute(
            "SELECT 1 FROM workspaces WHERE id = ?",
            (workspace_id,),
        )
        if await exists_cursor.fetchone() is None:
            await connection.rollback()
            return False

        table_cursor = await connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        table_rows = await table_cursor.fetchall()
        table_names = [str(row["name"]) for row in table_rows]

        scoped_tables: set[str] = {"workspaces"}
        children_by_parent = {
            parent: set(children) for parent, children in _KNOWN_CHILDREN_BY_PARENT.items()
        }
        for table_name in table_names:
            quoted = _quoted_identifier(table_name)
            columns = await (await connection.execute(f"PRAGMA table_info({quoted})")).fetchall()
            if any(str(column["name"]) == "workspace_id" for column in columns):
                scoped_tables.add(table_name)

            if discover_foreign_keys:
                foreign_keys = await (
                    await connection.execute(f"PRAGMA foreign_key_list({quoted})")
                ).fetchall()
                for foreign_key in foreign_keys:
                    parent = str(foreign_key["table"])
                    children_by_parent.setdefault(parent, set()).add(table_name)

        ordered: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(table_name: str) -> None:
            if table_name in visited:
                return
            if table_name in visiting:
                # Cyclic constraints remain protected by the transaction and
                # will fail closed rather than producing partial deletion.
                return
            visiting.add(table_name)
            for child in sorted(children_by_parent.get(table_name, ())):
                if child in scoped_tables:
                    visit(child)
            visiting.remove(table_name)
            visited.add(table_name)
            ordered.append(table_name)

        for table_name in sorted(scoped_tables - {"workspaces"}):
            visit(table_name)

        for table_name in ordered:
            await connection.execute(
                f"DELETE FROM {_quoted_identifier(table_name)} WHERE workspace_id = ?",
                (workspace_id,),
            )
        await connection.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
        await connection.commit()
        return True
    except BaseException:
        await connection.rollback()
        raise
