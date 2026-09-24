"""Isolation and bounded concurrency contracts, without GPUs or model downloads."""
import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from memorylayer_embed_server.lifecycle.limits import RequestLimitsMiddleware
from memorylayer_embed_server.services.required_models import required_providers
from memorylayer_embed_server.services.transcription.cascade import CascadeTranscriber
from memorylayer_embed_server.services.transcription.base import TranscriptionAttempt


def variables(values):
    return SimpleNamespace(environ=lambda name, **kw: True, get=lambda key, default=None: values.get(key, default))


def test_profile_required_models_excludes_opposite_role(monkeypatch):
    single, multi = object(), object()
    ocr = [SimpleNamespace(PROVIDER_NAME=n) for n in ('unlimited-ocr', 'deepseek-ocr')]
    v = variables({'dual_embedding_service': SimpleNamespace(_single_vector=single, _multi_vector=multi),
                   'cascade_transcriber': SimpleNamespace(providers=ocr)})
    monkeypatch.setenv('MEMORYLAYER_EMBED_PROFILE', 'embedding')
    assert required_providers(v) == [single, multi]
    monkeypatch.setenv('MEMORYLAYER_EMBED_PROFILE', 'transcription')
    assert required_providers(v) == ocr
    ocr.pop()
    with pytest.raises(RuntimeError, match='requires'):
        required_providers(v)


async def test_profile_concurrency_queue_and_health_are_independent(monkeypatch):
    monkeypatch.setenv('MEMORYLAYER_EMBED_PROFILE', 'embedding')
    monkeypatch.setenv('MEMORYLAYER_EMBED_BOUNDED_PROFILE', '1')
    monkeypatch.setenv('MEMORYLAYER_EMBED_MAX_CONCURRENT', '2')
    monkeypatch.setenv('MEMORYLAYER_EMBED_MAX_QUEUED', '0')
    active = 0
    entered, release = asyncio.Event(), asyncio.Event()
    async def work(request):
        nonlocal active
        active += 1
        if active == 2:
            entered.set()
        await release.wait()
        active -= 1
        return JSONResponse({'ok': True})
    async def health(request):
        return JSONResponse({'ok': True})
    app = RequestLimitsMiddleware(Starlette(routes=[Route('/v1/embeddings', work, methods=['POST']), Route('/health', health)]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        tasks = [asyncio.create_task(client.post('/v1/embeddings', json={'input': 'x'})) for _ in range(2)]
        await asyncio.wait_for(entered.wait(), 2)
        assert (await client.get('/health')).status_code == 200
        assert (await client.post('/v1/transcribe', json={})).status_code == 404
        assert (await client.post('/v1/embeddings', json={'input': 'x'})).status_code == 503
        tasks[0].cancel()
        await asyncio.sleep(0)
        assert app.pending == 2  # cancellation doesn't release ongoing GPU work
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert results[1].status_code == 200 and app.pending == 0


async def test_forced_ocr_provider_never_falls_back():
    providers = [SimpleNamespace(PROVIDER_NAME=name, transcribe_page=AsyncMock(return_value=TranscriptionAttempt(
        model=name, provider=name, success=False, error='test failure'))) for name in ('unlimited-ocr', 'deepseek-ocr')]
    cascade = object.__new__(CascadeTranscriber)
    cascade.providers, cascade._v, cascade.logger = providers, None, logging.getLogger('test')
    result = await cascade.transcribe_page(b'test', provider_name='deepseek-ocr')
    assert not result.success and len(result.attempts) == 1
    providers[0].transcribe_page.assert_not_awaited()
    with pytest.raises(ValueError):
        await cascade.transcribe_page(b'test', provider_name='unknown')


def test_provider_specific_recipes_and_grounded_contract(monkeypatch):
    from scitrera_app_framework import Variables
    from memorylayer_embed_server.services.transcription.vllm_transcription import build_deepseek_ocr_vllm_provider
    monkeypatch.setenv('MEMORYLAYER_EMBED_PROFILE', 'transcription')
    monkeypatch.setenv('MEMORYLAYER_EMBED_DEEPSEEK_OCR_SPARKRUN_RECIPE', '/tmp/deepseek.yaml')
    monkeypatch.setenv('MEMORYLAYER_EMBED_DEEPSEEK_OCR_GROUNDED', 'true')
    p = build_deepseek_ocr_vllm_provider(v=Variables(), logger=logging.getLogger('test'), model_name='deepseek-ai/DeepSeek-OCR-2',
        max_tokens=4096, port=18011, gpu_memory_utilization=.4, startup_timeout_sec=600, cmd='vllm')
    assert str(p._runner.recipe_path) == '/tmp/deepseek.yaml'
    assert p.output_contract == 'deepseek_ocr' and p.text_first
    assert '--skip-mm-profiling' not in p._runner.extra_args and '--load-format' not in p._runner.extra_args
