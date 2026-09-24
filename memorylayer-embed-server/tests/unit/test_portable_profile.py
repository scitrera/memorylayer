"""Contracts for shared-GPU deployment without loading model weights."""
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from memorylayer_embed_server.lifecycle.limits import RequestLimitsMiddleware
from memorylayer_embed_server.services._sparkrun_runner import SparkrunVLLMRunner
from memorylayer_embed_server.services.required_models import check_required_models
from memorylayer_embed_server.services.transcription.vllm_transcription import VLLMTranscriptionProvider


@pytest.mark.parametrize("data", [{"input": ["x"] * 9}, {"input": "x" * 6145}, {"max_tokens": 4097},
                                  {"dimensions": 2048}, {"images": ["a", "b"]}, {"images": ["invalid"]}])
def test_profile_rejects_unbounded_input(data):
    with pytest.raises((ValueError, OSError)):
        RequestLimitsMiddleware.validate(data)


def test_profile_accepts_page_and_rejects_large_decoded_image():
    for size, accepted in [((1275, 1650), True), ((2049, 10), False)]:
        buffer = io.BytesIO()
        Image.new("RGB", size).save(buffer, format="PNG")
        data = {"images": [base64.b64encode(buffer.getvalue()).decode()]}
        if accepted:
            RequestLimitsMiddleware.validate(data)
        else:
            with pytest.raises(ValueError):
                RequestLimitsMiddleware.validate(data)


async def test_ocr_retains_figure_only_raw_output_and_rejects_truncation():
    raw = "<|det|>image [10, 20, 300, 400]<|/det|>"
    provider = VLLMTranscriptionProvider(provider_name="unlimited-ocr", model_name="baidu/Unlimited-OCR", port=18012,
        max_tokens=4096, output_contract="unlimited_ocr", postprocess=lambda text: "", runner=MagicMock())
    provider._skip_subprocess = True
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason="stop")], usage=None)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response))))
    provider._ensure_started = AsyncMock(return_value=client)
    # Provider uses the injected runner's concurrency context.
    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def slot():
        yield
    provider._runner.concurrency_slot = slot
    result = await provider.transcribe_page(b"test", "ignored", max_tokens=4096)
    assert result.success and result.raw_content == raw and result.content == ""
    response.choices[0].finish_reason = "length"
    assert not (await provider.transcribe_page(b"test", "ignored", max_tokens=4096)).success


async def test_sparkrun_nonzero_result_fails_before_adopting_job(monkeypatch):
    import memorylayer_embed_server.services._sparkrun_runner as module
    api = SimpleNamespace(run=MagicMock(return_value=SimpleNamespace(rc=2, dry_run=False, cluster_id="bad")))
    monkeypatch.setattr(module, "_require_sparkrun", lambda: (api, None))
    monkeypatch.setattr(module, "sparkrun_context", lambda: "context")
    runner = SparkrunVLLMRunner(recipe_path="unused", role="embedding", model_name="m", host="127.0.0.1", port=1)
    monkeypatch.setattr(runner, "_run_options", lambda api: "options")
    with pytest.raises(RuntimeError, match="rc=2"):
        await runner.start()
    assert not runner.is_running


async def test_required_readiness_detects_dead_engine():
    v = MagicMock()
    v.environ.return_value = True
    dead = SimpleNamespace(_runner=SimpleNamespace(is_running=False))
    v.get.side_effect = lambda key, **kw: (SimpleNamespace(_single_vector=dead, _multi_vector=dead)
                                         if key == "dual_embedding_service" else SimpleNamespace(providers=[dead])
                                         if key == "cascade_transcriber" else kw.get("default"))
    with pytest.raises(RuntimeError, match="not running"):
        await check_required_models(v)


def test_long_base64_image_is_not_stat_ed_as_a_filename():
    from memorylayer_server.services.embedding.embed_server import _to_base64

    from memorylayer_embed_server.services.embedding.vllm_subprocess import VLLMSubprocessEmbeddingProvider
    encoded = "a" * 4096
    assert VLLMSubprocessEmbeddingProvider._image_to_data_url(encoded).endswith(encoded)
    assert _to_base64(encoded) == encoded


async def test_startup_failure_stops_only_a_new_owned_job(monkeypatch):
    import memorylayer_embed_server.services._sparkrun_runner as module

    for reused in (False, True):
        result = SimpleNamespace(rc=0, dry_run=False, cluster_id="test-job", already_running=reused,
                                 serve_command="vllm serve model", serve_port=18000, executor="local", runtime="vllm")
        api = SimpleNamespace(run=MagicMock(return_value=result), stop=MagicMock(return_value=SimpleNamespace(success=True)))
        monkeypatch.setattr(module, "_require_sparkrun", lambda: (api, None))
        monkeypatch.setattr(module, "sparkrun_context", lambda: "shared-context")
        runner = SparkrunVLLMRunner(recipe_path="unused", role="embedding", model_name="model", host="127.0.0.1", port=18000)
        monkeypatch.setattr(runner, "_run_options", lambda api: "options")
        monkeypatch.setattr(runner, "_pump_logs", AsyncMock())
        monkeypatch.setattr(runner, "wait_for_health", AsyncMock(side_effect=RuntimeError("engine failed")))
        with pytest.raises(RuntimeError, match="engine failed"):
            await runner.start()
        assert not runner.is_running
        if reused:
            api.stop.assert_not_called()
        else:
            api.stop.assert_called_once_with(cluster_id="test-job", sctx="shared-context")


async def test_dead_engine_is_detected_when_log_reader_fails(monkeypatch):
    import memorylayer_embed_server.services._sparkrun_runner as module

    observed = SimpleNamespace(for_host=lambda host: object(), observation_errors={}, running_cluster_ids=lambda: [])
    api = SimpleNamespace(logs=MagicMock(side_effect=RuntimeError("process is gone")), status=MagicMock(return_value=observed))
    monkeypatch.setattr(module, "_require_sparkrun", lambda: (api, None))
    runner = SparkrunVLLMRunner(recipe_path="unused", role="embedding", model_name="m", host="127.0.0.1", port=1)
    runner._cluster_id = "test-job"
    runner._alive = True
    await runner._pump_logs()
    assert not runner.is_running


def test_oversized_png_header_returns_validation_error_without_pixel_allocation():
    import struct
    import zlib

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    raw = bytearray(buffer.getvalue())
    raw[16:24] = struct.pack(">II", 100000, 100000)
    raw[29:33] = struct.pack(">I", zlib.crc32(raw[12:29]))
    with pytest.raises(ValueError, match="image budget"):
        RequestLimitsMiddleware.validate({"images": [base64.b64encode(raw).decode()]})


async def test_service_shutdown_stops_all_lanes_without_subprocess_handles(monkeypatch):
    import scitrera_app_framework.core.plugins as plugins

    import memorylayer_embed_server.dependencies as dependencies

    single = SimpleNamespace(shutdown=AsyncMock(side_effect=RuntimeError("stop failed")))
    multi = SimpleNamespace(shutdown=AsyncMock())
    ocr = SimpleNamespace(shutdown=AsyncMock())
    values = {"dual_embedding_service": SimpleNamespace(_single_vector=single, _multi_vector=multi),
              "cascade_transcriber": SimpleNamespace(providers=[ocr, ocr])}
    v = SimpleNamespace(get=lambda key, default=None: values.get(key, default))
    monkeypatch.setattr(dependencies, "get_variables", lambda value: value)
    monkeypatch.setattr(dependencies, "get_logger", lambda v: MagicMock())
    monkeypatch.setattr(dependencies, "async_plugins_stopping", AsyncMock())
    monkeypatch.setattr(plugins, "shutdown_all_plugins", MagicMock())
    await dependencies.shutdown_services(v)
    for provider in (single, multi, ocr):
        provider.shutdown.assert_awaited_once()
