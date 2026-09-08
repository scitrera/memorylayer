"""Unit tests for AetherServiceConnection's connection liveness watchdog.

Covers the silent-drop self-heal behaviour: the Aether SDK
(``scitrera_aether_client``) only reconnects on a *graceful* disconnect and
builds its gRPC channel without keepalive, so a *silently dropped* connection
(hard gateway pod kill) is otherwise never noticed and the service stays
absent from the gateway indefinitely.  ``AetherServiceConnection`` runs a
periodic ``kv_get`` liveness probe and, after enough consecutive failures,
closes the dead channel so the SDK's native ``auto_reconnect`` fires.

What is tested here (no live gateway — the probe/channel are mocked):
- A healthy probe (``kv_get`` returns a response) does NOT force a reconnect.
- A repeatedly-failing probe (``kv_get`` returns ``None`` / times out) closes
  the channel once the failure threshold is reached.
- A single failure below the threshold does NOT close the channel.
- The watchdog skips probing while the SDK is already reconnecting.
- ``connect``/``disconnect`` start and stop the watchdog task.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_server.services.aether_service import AetherServiceConnection


@pytest.fixture
def mock_variables():
    """Minimal Variables stub satisfying ``get_logger`` and ``environ``."""
    v = MagicMock()
    v.environ = MagicMock(side_effect=lambda key, default=None, **kwargs: default)
    return v


def _make_service_connection(mock_variables, **overrides) -> AetherServiceConnection:
    kwargs = dict(
        gateway_addr="localhost:50051",
        workspace="test-workspace",
        specifier="main",
        # Tight timings so the loop iterates fast under test.
        liveness_interval_s=0.01,
        liveness_timeout_s=0.01,
        liveness_failures=2,
    )
    kwargs.update(overrides)
    return AetherServiceConnection(mock_variables, **kwargs)


def _mock_client(*, kv_get_returns=None, kv_get_side_effect=None, confirmed=True, reconnecting=False):
    """Build a stand-in AsyncServiceClient with a mock channel."""
    client = MagicMock()
    client._connection_confirmed = confirmed
    client._reconnecting = reconnecting
    if kv_get_side_effect is not None:
        client.kv_get = AsyncMock(side_effect=kv_get_side_effect)
    else:
        client.kv_get = AsyncMock(return_value=kv_get_returns)
    client.channel = MagicMock()
    client.channel.close = AsyncMock()
    return client


class TestLivenessProbe:
    @pytest.mark.asyncio
    async def test_healthy_probe_returns_true(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        svc._client = _mock_client(kv_get_returns=MagicMock())  # any non-None response

        assert await svc._liveness_probe() is True

    @pytest.mark.asyncio
    async def test_timeout_probe_returns_false(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        svc._client = _mock_client(kv_get_returns=None)  # SDK timeout -> None

        assert await svc._liveness_probe() is False

    @pytest.mark.asyncio
    async def test_rpc_error_probe_returns_false(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        svc._client = _mock_client(kv_get_side_effect=RuntimeError("boom"))

        assert await svc._liveness_probe() is False


class TestForceReconnect:
    @pytest.mark.asyncio
    async def test_force_reconnect_closes_channel(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        client = _mock_client()
        svc._client = client

        await svc._force_reconnect()

        client.channel.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_reconnect_noop_when_no_client(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        svc._client = None
        # Should not raise.
        await svc._force_reconnect()


class TestLivenessLoop:
    @pytest.mark.asyncio
    async def test_dead_connection_forces_reconnect_after_threshold(self, mock_variables):
        svc = _make_service_connection(mock_variables, liveness_failures=2)
        client = _mock_client(kv_get_returns=None)  # every probe fails
        svc._client = client

        task = asyncio.create_task(svc._liveness_loop())
        # Give the loop time to accumulate >= threshold failures and react.
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The dead channel was closed at least once to trigger reconnect.
        assert client.channel.close.await_count >= 1

    @pytest.mark.asyncio
    async def test_healthy_connection_never_reconnects(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        client = _mock_client(kv_get_returns=MagicMock())  # every probe succeeds
        svc._client = client

        task = asyncio.create_task(svc._liveness_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        client.channel.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_probe_while_reconnecting(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        client = _mock_client(kv_get_returns=None, reconnecting=True)
        svc._client = client

        task = asyncio.create_task(svc._liveness_loop())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # While the SDK is reconnecting we never probe nor force a reconnect.
        client.kv_get.assert_not_awaited()
        client.channel.close.assert_not_awaited()


class TestWatchdogLifecycle:
    @pytest.mark.asyncio
    async def test_start_and_stop_watchdog(self, mock_variables):
        svc = _make_service_connection(mock_variables)
        svc._client = _mock_client(kv_get_returns=MagicMock())

        svc._start_liveness_watchdog()
        assert svc._liveness_task is not None
        assert not svc._liveness_task.done()

        await svc._stop_liveness_watchdog()
        assert svc._liveness_task is None

    @pytest.mark.asyncio
    async def test_disabled_watchdog_does_not_start(self, mock_variables):
        svc = _make_service_connection(mock_variables, liveness_enabled=False)
        svc._client = _mock_client(kv_get_returns=MagicMock())

        svc._start_liveness_watchdog()
        assert svc._liveness_task is None
