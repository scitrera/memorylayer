"""Readiness and detached ownership under overlapping model startup."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from memorylayer_embed_server.services import _sparkrun_runner as runner_module
from memorylayer_embed_server.services import required_models as startup
from memorylayer_embed_server.services._sparkrun_runner import SparkrunVLLMRunner


def variables(providers, parallel=False):
    values = {"dual_embedding_service": SimpleNamespace(_single_vector=providers[0], _multi_vector=providers[1]),
              "cascade_transcriber": SimpleNamespace(providers=providers[2:])}
    return SimpleNamespace(
        environ=lambda key, **kw: parallel if key == "MEMORYLAYER_EMBED_PARALLEL_STARTUP" else True,
        get=lambda key, default=None: values.get(key, default), set=lambda key, value: values.__setitem__(key, value),
    )


async def test_parallel_preloads_overlap_and_readiness_waits_for_last(monkeypatch):
    entered = []
    release = [asyncio.Event() for _ in range(3)]
    all_entered = asyncio.Event()
    providers = []
    for index in range(3):
        async def preload(index=index):
            entered.append(index)
            if len(entered) == 3:
                all_entered.set()
            await release[index].wait()
        providers.append(SimpleNamespace(preload=preload, _runner=SimpleNamespace(model_name=str(index))))
    check = AsyncMock()
    monkeypatch.setattr(startup, "check_required_models", check)
    v = variables(providers, parallel=True)
    task = asyncio.create_task(startup.preload_required_models(v))
    try:
        await asyncio.wait_for(all_entered.wait(), 2)
        assert entered == [2, 0, 1]
        check.assert_not_awaited()
        release[0].set()
        release[2].set()
        await asyncio.sleep(0)
        assert not task.done()
        check.assert_not_awaited()
        release[1].set()
        assert await asyncio.wait_for(task, 2) == providers
        check.assert_awaited_once_with(v)
        assert v.get("required_model_startup")["mode"] == "parallel"
    finally:
        for event in release:
            event.set()
        await task


async def test_sequential_mode_preserves_ocr_first_order(monkeypatch):
    order = []
    providers = []
    for index in range(3):
        async def preload(index=index):
            order.append(index)
        providers.append(SimpleNamespace(preload=preload))
    check = AsyncMock()
    monkeypatch.setattr(startup, "check_required_models", check)
    v = variables(providers)
    await startup.preload_required_models(v)
    assert order == [2, 0, 1]
    assert v.get("required_model_startup")["mode"] == "sequential"
    check.assert_awaited_once()


async def test_failed_sibling_settles_other_preloads_before_returning(monkeypatch):
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    allow_cleanup = asyncio.Event()
    async def blocked():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await allow_cleanup.wait()
    async def fail():
        await entered.wait()
        raise RuntimeError("model failed")
    providers = [SimpleNamespace(preload=blocked), SimpleNamespace(preload=AsyncMock()), SimpleNamespace(preload=fail)]
    check = AsyncMock()
    monkeypatch.setattr(startup, "check_required_models", check)
    task = asyncio.create_task(startup.preload_required_models(variables(providers, parallel=True)))
    try:
        await asyncio.wait_for(cleaning.wait(), 2)
        assert not task.done()
        check.assert_not_awaited()
    finally:
        allow_cleanup.set()
    with pytest.raises(ExceptionGroup, match="TaskGroup") as exc:
        await task
    assert any(str(error) == "model failed" for error in exc.value.exceptions)


def make_runner(monkeypatch, api, name="model"):
    monkeypatch.setattr(runner_module, "_require_sparkrun", lambda: (api, None))
    monkeypatch.setattr(runner_module, "sparkrun_context", lambda: "shared-context")
    runner = SparkrunVLLMRunner(recipe_path="unused", role="embedding", model_name=name, host="127.0.0.1", port=18000)
    monkeypatch.setattr(runner, "_run_options", lambda api: name)
    monkeypatch.setattr(runner, "_pump_logs", AsyncMock())
    monkeypatch.setattr(runner, "wait_for_health", AsyncMock())
    return runner


def launch_result(name="model", reused=False):
    return SimpleNamespace(rc=0, dry_run=False, cluster_id=name, already_running=reused,
                           serve_command="vllm serve model", serve_port=18000, executor="local", runtime="vllm")


@pytest.mark.parametrize("reused", [False, True])
async def test_cancel_during_launch_waits_for_ownership_and_stops_only_owned_job(monkeypatch, reused):
    entered = threading.Event()
    release = threading.Event()
    def run(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release controller"
        return launch_result(reused=reused)
    api = SimpleNamespace(run=run, stop=MagicMock(return_value=SimpleNamespace(success=True)))
    runner = make_runner(monkeypatch, api)
    task = asyncio.create_task(runner.start())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()  # A second signal must not abandon launch cleanup.
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not runner.is_running
    runner.wait_for_health.assert_not_awaited()
    if reused:
        api.stop.assert_not_called()
    else:
        api.stop.assert_called_once_with(cluster_id="model", sctx="shared-context")


async def test_cancel_during_health_wait_stops_owned_engine(monkeypatch):
    entered = asyncio.Event()
    async def health():
        entered.set()
        await asyncio.Event().wait()
    api = SimpleNamespace(run=lambda *a, **k: launch_result(), stop=MagicMock(return_value=SimpleNamespace(success=True)))
    runner = make_runner(monkeypatch, api)
    monkeypatch.setattr(runner, "wait_for_health", health)
    task = asyncio.create_task(runner.start())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    api.stop.assert_called_once()
    assert not runner.is_running


async def test_controller_calls_serialize_while_health_waits_overlap(monkeypatch):
    first_entered = threading.Event()
    release_controller = threading.Event()
    health_entered = []
    release_health = asyncio.Event()
    active = 0
    peak = 0
    lock = threading.Lock()
    def run(name, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            first_entered.set()
            assert release_controller.wait(5)
            return launch_result(name)
        finally:
            with lock:
                active -= 1
    api = SimpleNamespace(run=run, stop=MagicMock(return_value=SimpleNamespace(success=True)))
    runners = [make_runner(monkeypatch, api, str(index)) for index in range(3)]
    for index, runner in enumerate(runners):
        async def health(index=index):
            health_entered.append(index)
            await release_health.wait()
        monkeypatch.setattr(runner, "wait_for_health", health)
    tasks = [asyncio.create_task(runner.start()) for runner in runners]
    try:
        assert await asyncio.to_thread(first_entered.wait, 2)
        assert active == peak == 1
        release_controller.set()
        async with asyncio.timeout(2):
            while len(health_entered) < 3:
                await asyncio.sleep(0.001)
        assert peak == 1
        assert all(not task.done() for task in tasks)
    finally:
        release_controller.set()
        release_health.set()
        await asyncio.gather(*tasks)
        for runner in runners:
            await runner.shutdown()
    assert api.stop.call_count == 3


async def test_failed_stop_remains_visible_and_can_be_retried(monkeypatch):
    api = SimpleNamespace(run=lambda *a, **k: launch_result(), stop=MagicMock(side_effect=[
        SimpleNamespace(success=False, errors=["temporary failure"]), SimpleNamespace(success=True)]))
    runner = make_runner(monkeypatch, api)
    await runner.start()
    with pytest.raises(RuntimeError, match="stop failed"):
        await runner.shutdown()
    assert runner._cluster_id == "model" and runner._owns_job
    await runner.shutdown()
    assert runner._cluster_id is None
