"""Authorization denials retain their HTTP status at every RPG route."""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from memorylayer_server_rpg.api.router import router


@pytest.mark.parametrize("route", router.routes, ids=lambda route: route.name)
async def test_rpg_route_preserves_authorization_denial(route):
    auth_service = SimpleNamespace(build_context=AsyncMock(return_value=SimpleNamespace(workspace_id="ws_denied")))
    authz_service = SimpleNamespace(require_authorization=AsyncMock(side_effect=HTTPException(403, "Access denied")))
    kwargs = {name: MagicMock() for name in inspect.signature(route.endpoint).parameters}
    kwargs.update(auth_service=auth_service, authz_service=authz_service)

    with pytest.raises(HTTPException) as error:
        await route.endpoint(**kwargs)

    assert error.value.status_code == 403
    assert error.value.detail == "Access denied"
    authz_service.require_authorization.assert_awaited_once()
    for name in ("rpg_service", "conflict_service"):
        if name in kwargs:
            assert not kwargs[name].mock_calls
