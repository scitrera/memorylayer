"""
Workspace resolution on the default (OSS) authenticator.

Endpoints take ``workspace_id`` as a FastAPI ``Query`` parameter and then use it
in preference to the context's workspace. resolve_workspace auto-creates the
workspace it is given, so if it never sees the query parameter it creates a
DIFFERENT workspace than the handler writes to. Because
``chat_threads.workspace_id`` is a foreign key onto ``workspaces(id)``, the write
then fails with a bare IntegrityError surfaced as an opaque
500 "Failed to append messages".

MessagesAppendRequest has no ``workspace_id`` field at all, so every append to a
workspace other than "_default" hit this.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from memorylayer_server.services.authentication.default import (
    OpenAuthenticationService,
)


class _BodyWithWorkspace(BaseModel):
    workspace_id: str | None = None


class _BodyWithoutWorkspace(BaseModel):
    """Mirrors MessagesAppendRequest, which carries no workspace_id."""

    messages: list[str] = []


def _make_request(query: dict | None = None, headers: dict | None = None) -> MagicMock:
    req = MagicMock()
    req.query_params = query or {}
    req.headers = headers or {}
    # Explicitly a write: auto-creation is limited to unsafe methods, and these
    # tests are about WHICH workspace gets ensured, not whether it is.
    req.method = "POST"
    return req


def _make_service() -> tuple[OpenAuthenticationService, MagicMock]:
    session_service = MagicMock()
    session_service.get = AsyncMock(side_effect=Exception("not found"))
    workspace_service = MagicMock()
    workspace_service.ensure_workspace = AsyncMock()
    svc = OpenAuthenticationService(
        session_service=session_service,
        workspace_service=workspace_service,
    )
    return svc, workspace_service


@pytest.mark.asyncio
async def test_query_workspace_is_ensured_when_body_has_no_field():
    """The append case: workspace only ever appears in the query string."""
    svc, workspace_service = _make_service()
    request = _make_request(query={"workspace_id": "myspace"})

    ctx = await svc.build_context(request, _BodyWithoutWorkspace())

    assert ctx.workspace_id == "myspace"
    ensured = workspace_service.ensure_workspace.await_args.kwargs["workspace_id"]
    assert ensured == "myspace", (
        "the workspace the handler will write to must be the one auto-created; "
        f"ensured {ensured!r} instead"
    )


@pytest.mark.asyncio
async def test_body_workspace_still_wins_over_query():
    """Ordering matches the Aether authenticator: body, then query, then header."""
    svc, workspace_service = _make_service()
    request = _make_request(query={"workspace_id": "from-query"})

    ctx = await svc.build_context(request, _BodyWithWorkspace(workspace_id="from-body"))

    assert ctx.workspace_id == "from-body"
    assert workspace_service.ensure_workspace.await_args.kwargs["workspace_id"] == "from-body"


@pytest.mark.asyncio
async def test_header_workspace_used_when_body_and_query_absent():
    svc, workspace_service = _make_service()
    request = _make_request(headers={"X-Workspace-ID": "from-header"})

    ctx = await svc.build_context(request, _BodyWithoutWorkspace())

    assert ctx.workspace_id == "from-header"
    assert workspace_service.ensure_workspace.await_args.kwargs["workspace_id"] == "from-header"


@pytest.mark.asyncio
async def test_falls_back_to_default_workspace():
    svc, workspace_service = _make_service()

    ctx = await svc.build_context(_make_request(), _BodyWithoutWorkspace())

    assert ctx.workspace_id == "_default"
    assert workspace_service.ensure_workspace.await_args.kwargs["workspace_id"] == "_default"
