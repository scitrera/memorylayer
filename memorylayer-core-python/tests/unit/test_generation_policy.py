"""Central generation policy, budget, streaming, and ledger invariants."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from memorylayer_server.models.generation import (
    EnrichmentPolicy,
    GenerationActivity,
    GenerationBudgetExceededError,
    GenerationNotAllowedError,
)
from memorylayer_server.models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from memorylayer_server.services.llm.registry import LLMProviderRegistry
from memorylayer_server.services.llm.service_default import LLMService


def _request(max_tokens: int = 32) -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content="classify this input")],
        max_tokens=max_tokens,
    )


def _provider() -> AsyncMock:
    provider = AsyncMock()
    provider.default_model = "test-model"
    provider.supports_streaming = True
    provider.complete.return_value = LLMResponse(
        content="done",
        model="test-model",
        prompt_tokens=4,
        completion_tokens=1,
        total_tokens=5,
        finish_reason="stop",
    )

    async def stream(_request):
        yield LLMStreamChunk(content="done", is_final=False)
        yield LLMStreamChunk(content="", is_final=True, finish_reason="stop")

    provider.complete_stream = stream
    return provider


@pytest.mark.asyncio
async def test_deterministic_policy_rejects_before_provider_selection():
    provider = _provider()
    service = LLMService(
        LLMProviderRegistry(providers={"default": provider}),
        policy=EnrichmentPolicy.DETERMINISTIC,
    )

    with pytest.raises(GenerationNotAllowedError) as raised:
        await service.complete(_request(), activity=GenerationActivity.MEMORY_CLASSIFICATION)

    assert raised.value.code == "generation_not_allowed"
    provider.complete.assert_not_called()


@pytest.mark.asyncio
async def test_adaptive_requires_an_explicit_allowed_activity():
    provider = _provider()
    service = LLMService(
        LLMProviderRegistry(providers={"default": provider}),
        policy=EnrichmentPolicy.ADAPTIVE,
        adaptive_activities={GenerationActivity.REFLECTION},
    )

    with pytest.raises(GenerationNotAllowedError):
        await service.complete(_request())
    with pytest.raises(GenerationNotAllowedError):
        await service.complete(_request(), activity=GenerationActivity.SYNTHESIS)

    await service.complete(_request(), activity=GenerationActivity.REFLECTION)
    provider.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_budget_is_atomic_under_concurrency():
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider:
        default_model = "blocking-model"
        supports_streaming = False

        async def complete(self, _request):
            entered.set()
            await release.wait()
            return LLMResponse(
                content="done",
                model="blocking-model",
                prompt_tokens=4,
                completion_tokens=1,
                total_tokens=5,
                finish_reason="stop",
            )

    service = LLMService(
        LLMProviderRegistry(providers={"default": BlockingProvider()}),
        policy=EnrichmentPolicy.GENERATIVE,
    )
    authorization = service.authorization(
        GenerationActivity.SYNTHESIS,
        workspace_id="ws_budget",
        operation_id="op_budget",
        max_calls=1,
    )
    first = asyncio.create_task(service.complete(_request(), authorization=authorization))
    await entered.wait()

    with pytest.raises(GenerationBudgetExceededError):
        await service.complete(_request(), authorization=authorization)
    release.set()
    await first

    ledger = await service.get_ledger("op_budget")
    assert ledger is not None
    assert (ledger.attempted, ledger.allowed, ledger.rejected, ledger.completed) == (2, 1, 1, 1)


@pytest.mark.asyncio
async def test_output_reservations_prevent_concurrent_budget_oversubscription():
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingProvider:
        default_model = "blocking-model"
        supports_streaming = False

        async def complete(self, _request):
            entered.set()
            await release.wait()
            return LLMResponse(
                content="small",
                model="blocking-model",
                prompt_tokens=4,
                completion_tokens=2,
                total_tokens=6,
                finish_reason="stop",
            )

    service = LLMService(
        LLMProviderRegistry(providers={"default": BlockingProvider()}),
        policy=EnrichmentPolicy.GENERATIVE,
    )
    authorization = service.authorization(
        GenerationActivity.SYNTHESIS,
        operation_id="op_output",
        max_calls=2,
        max_output_tokens=100,
    )
    first = asyncio.create_task(service.complete(_request(60), authorization=authorization))
    await entered.wait()
    with pytest.raises(GenerationBudgetExceededError):
        await service.complete(_request(60), authorization=authorization)
    release.set()
    await first


@pytest.mark.asyncio
async def test_streaming_uses_the_same_policy_boundary_and_ledger():
    provider = _provider()
    deterministic = LLMService(
        LLMProviderRegistry(providers={"default": provider}),
        policy=EnrichmentPolicy.DETERMINISTIC,
    )
    with pytest.raises(GenerationNotAllowedError):
        async for _chunk in deterministic.complete_stream(
            _request(),
            activity=GenerationActivity.SYNTHESIS,
        ):
            pass

    generative = LLMService(
        LLMProviderRegistry(providers={"default": provider}),
        policy=EnrichmentPolicy.GENERATIVE,
    )
    authorization = generative.authorization(
        GenerationActivity.SYNTHESIS,
        operation_id="op_stream",
        workspace_id="ws_stream",
    )
    chunks = [
        chunk
        async for chunk in generative.complete_stream(
            _request(),
            authorization=authorization,
        )
    ]
    ledger = await generative.get_ledger("op_stream")

    assert chunks
    assert ledger is not None
    assert ledger.completed == 1
    assert ledger.output_tokens > 0


@pytest.mark.asyncio
async def test_request_policy_cannot_elevate_server_policy():
    provider = _provider()
    service = LLMService(
        LLMProviderRegistry(providers={"default": provider}),
        policy=EnrichmentPolicy.DETERMINISTIC,
    )
    authorization = service.authorization(
        GenerationActivity.SYNTHESIS,
        policy=EnrichmentPolicy.GENERATIVE,
    )
    assert authorization.policy == EnrichmentPolicy.DETERMINISTIC

    with pytest.raises(GenerationNotAllowedError):
        await service.complete(_request(), authorization=authorization)
    provider.complete.assert_not_called()
