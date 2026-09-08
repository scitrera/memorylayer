"""Default LLM service with central generation authorization and accounting."""

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from logging import Logger
from uuid import uuid4

from scitrera_app_framework import Variables, get_extension, get_logger

from ...config import (
    DEFAULT_MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES,
    DEFAULT_MEMORYLAYER_ENRICHMENT_POLICY,
    DEFAULT_MEMORYLAYER_GENERATION_MAX_CALLS,
    DEFAULT_MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS,
    DEFAULT_MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS,
    MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES,
    MEMORYLAYER_ENRICHMENT_POLICY,
    MEMORYLAYER_GENERATION_MAX_CALLS,
    MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS,
    MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS,
)
from ...models.generation import (
    EnrichmentPolicy,
    GenerationActivity,
    GenerationAuthorization,
    GenerationBudgetExceededError,
    GenerationLedger,
    GenerationNotAllowedError,
    GenerationSummary,
)
from ...models.llm import LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk
from .._constants import EXT_METRICS_SERVICE
from .base import EXT_LLM_REGISTRY, LLMServicePluginBase
from .registry import LLMProviderRegistry


class LLMService:
    """High-level LLM service wrapping provider registry.

    Similar to EmbeddingService wrapping EmbeddingProvider.
    Adds convenience methods for common patterns like synthesis.
    Supports profile-based routing through the registry.
    """

    def __init__(
        self,
        registry: LLMProviderRegistry,
        v: Variables = None,
        *,
        policy: EnrichmentPolicy | str | None = None,
        adaptive_activities: set[GenerationActivity | str] | None = None,
    ):
        self.registry = registry
        self.logger = get_logger(v, name=self.__class__.__name__)
        configured_policy = policy or (
            v.environ(
                MEMORYLAYER_ENRICHMENT_POLICY,
                default=DEFAULT_MEMORYLAYER_ENRICHMENT_POLICY,
            )
            if v is not None
            else DEFAULT_MEMORYLAYER_ENRICHMENT_POLICY
        )
        self.policy = (
            configured_policy if isinstance(configured_policy, EnrichmentPolicy) else EnrichmentPolicy(str(configured_policy).lower())
        )
        configured_activities = (
            v.environ(
                MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES,
                default=DEFAULT_MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES,
            )
            if v is not None
            else DEFAULT_MEMORYLAYER_ADAPTIVE_GENERATION_ACTIVITIES
        )
        raw_activities = adaptive_activities
        if raw_activities is None:
            raw_activities = {value.strip() for value in str(configured_activities).split(",") if value.strip()}
        self.adaptive_activities = {GenerationActivity(value) for value in raw_activities}
        self.default_max_calls = self._config_int(v, MEMORYLAYER_GENERATION_MAX_CALLS, DEFAULT_MEMORYLAYER_GENERATION_MAX_CALLS)
        self.default_max_input_tokens = self._config_int(
            v,
            MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS,
            DEFAULT_MEMORYLAYER_GENERATION_MAX_INPUT_TOKENS,
        )
        self.default_max_output_tokens = self._config_int(
            v,
            MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS,
            DEFAULT_MEMORYLAYER_GENERATION_MAX_OUTPUT_TOKENS,
        )
        self._ledger_lock = asyncio.Lock()
        self._ledgers: dict[str, GenerationLedger] = {}
        self._reserved_output_tokens: dict[str, int] = {}
        self._unlabeled_calls = 0
        try:
            self.metrics = get_extension(EXT_METRICS_SERVICE, v) if v is not None else None
        except Exception:
            self.metrics = None

    def _record_metrics(
        self,
        auth: GenerationAuthorization,
        outcome: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: int | None = None,
    ) -> None:
        if self.metrics is None:
            return
        labels = {
            "activity": auth.activity.value,
            "policy": auth.policy.value,
            "outcome": outcome,
        }
        try:
            self.metrics.counter("memorylayer_generation_calls_total", labels=labels)
            if input_tokens:
                self.metrics.counter(
                    "memorylayer_generation_tokens_total",
                    input_tokens,
                    labels={**labels, "direction": "input"},
                )
            if output_tokens:
                self.metrics.counter(
                    "memorylayer_generation_tokens_total",
                    output_tokens,
                    labels={**labels, "direction": "output"},
                )
            if latency_ms is not None:
                self.metrics.histogram(
                    "memorylayer_generation_latency_seconds",
                    latency_ms / 1000,
                    labels=labels,
                )
        except Exception:
            self.logger.debug("Generation metric emission failed", exc_info=True)

    @staticmethod
    def _config_int(v: Variables | None, key: str, default: int) -> int:
        if v is None:
            return default
        return v.environ(key, default=default, type_fn=int)

    @staticmethod
    def _estimate_request_tokens(request: LLMRequest) -> int:
        characters = sum(len(message.content or "") for message in request.messages)
        return max(1, (characters + 3) // 4)

    @staticmethod
    def _lower_policy(a: EnrichmentPolicy, b: EnrichmentPolicy) -> EnrichmentPolicy:
        order = {
            EnrichmentPolicy.DETERMINISTIC: 0,
            EnrichmentPolicy.ADAPTIVE: 1,
            EnrichmentPolicy.GENERATIVE: 2,
        }
        return a if order[a] <= order[b] else b

    def is_generation_allowed(self, activity: GenerationActivity) -> bool:
        if self.policy == EnrichmentPolicy.DETERMINISTIC:
            return False
        if self.policy == EnrichmentPolicy.ADAPTIVE:
            return activity in self.adaptive_activities
        return True

    def authorization(
        self,
        activity: GenerationActivity,
        *,
        workspace_id: str = "_system",
        operation_id: str | None = None,
        policy: EnrichmentPolicy | None = None,
        max_calls: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> GenerationAuthorization:
        """Create a request authorization that cannot elevate server policy."""
        return GenerationAuthorization(
            policy=self._lower_policy(self.policy, policy or self.policy),
            operation_id=operation_id or f"gen_{uuid4().hex}",
            workspace_id=workspace_id,
            activity=activity,
            max_calls=self.default_max_calls if max_calls is None else max_calls,
            max_input_tokens=(
                self.default_max_input_tokens if max_input_tokens is None else min(max_input_tokens, self.default_max_input_tokens)
            ),
            max_output_tokens=(
                self.default_max_output_tokens if max_output_tokens is None else min(max_output_tokens, self.default_max_output_tokens)
            ),
        )

    def _effective_authorization(
        self,
        authorization: GenerationAuthorization | None,
        activity: GenerationActivity | None,
    ) -> GenerationAuthorization:
        if authorization is not None:
            effective_policy = self._lower_policy(self.policy, authorization.policy)
            return authorization.model_copy(update={"policy": effective_policy})
        if activity is None:
            self._unlabeled_calls += 1
            if self.policy != EnrichmentPolicy.GENERATIVE:
                raise GenerationNotAllowedError(
                    GenerationActivity.SYNTHESIS,
                    self.policy,
                    "an explicit generation activity is required",
                )
            activity = GenerationActivity.SYNTHESIS
        return self.authorization(activity)

    async def _reserve(
        self,
        request: LLMRequest,
        profile: str,
        authorization: GenerationAuthorization | None,
        activity: GenerationActivity | None,
    ) -> tuple[GenerationAuthorization, int, int]:
        auth = self._effective_authorization(authorization, activity)
        input_tokens = self._estimate_request_tokens(request)
        requested_output = request.max_completion_tokens or request.max_tokens or 0
        async with self._ledger_lock:
            ledger = self._ledgers.setdefault(
                auth.operation_id,
                GenerationLedger(
                    operation_id=auth.operation_id,
                    workspace_id=auth.workspace_id,
                    activity=auth.activity,
                    policy=auth.policy,
                ),
            )
            ledger.attempted += 1
            ledger.updated_at = datetime.now(UTC)
            allowed = auth.policy == EnrichmentPolicy.GENERATIVE or (
                auth.policy == EnrichmentPolicy.ADAPTIVE and auth.activity in self.adaptive_activities
            )
            if not allowed:
                ledger.rejected += 1
                self._record_metrics(auth, "rejected_policy", input_tokens=input_tokens)
                raise GenerationNotAllowedError(auth.activity, auth.policy, "activity is not authorized")
            if ledger.allowed >= auth.max_calls:
                ledger.rejected += 1
                self._record_metrics(auth, "rejected_budget", input_tokens=input_tokens)
                raise GenerationBudgetExceededError(auth.activity, "maximum call count reached")
            if auth.max_input_tokens is not None and ledger.input_tokens + input_tokens > auth.max_input_tokens:
                ledger.rejected += 1
                self._record_metrics(auth, "rejected_budget", input_tokens=input_tokens)
                raise GenerationBudgetExceededError(auth.activity, "input token budget exceeded")
            reserved_output = self._reserved_output_tokens.get(auth.operation_id, 0)
            if auth.max_output_tokens is not None and ledger.output_tokens + reserved_output + requested_output > auth.max_output_tokens:
                ledger.rejected += 1
                self._record_metrics(auth, "rejected_budget", input_tokens=input_tokens)
                raise GenerationBudgetExceededError(auth.activity, "output token budget exceeded")
            ledger.allowed += 1
            ledger.input_tokens += input_tokens
            ledger.profile = profile
            self._reserved_output_tokens[auth.operation_id] = reserved_output + requested_output
        return auth, requested_output, input_tokens

    async def _finish(
        self,
        auth: GenerationAuthorization,
        *,
        response: LLMResponse | None,
        elapsed_ms: int,
        failed: bool,
        profile: str,
        input_tokens: int = 0,
        output_tokens: int | None = None,
        reserved_output_tokens: int = 0,
    ) -> None:
        async with self._ledger_lock:
            ledger = self._ledgers[auth.operation_id]
            remaining_reservation = max(
                0,
                self._reserved_output_tokens.get(auth.operation_id, 0) - reserved_output_tokens,
            )
            if remaining_reservation:
                self._reserved_output_tokens[auth.operation_id] = remaining_reservation
            else:
                self._reserved_output_tokens.pop(auth.operation_id, None)
            ledger.latency_ms += elapsed_ms
            ledger.updated_at = datetime.now(UTC)
            ledger.profile = profile
            if failed:
                ledger.failed += 1
                recorded_output_tokens = output_tokens or 0
            else:
                ledger.completed += 1
                recorded_output_tokens = (
                    output_tokens if output_tokens is not None else (response.completion_tokens if response is not None else 0)
                )
                if response is not None:
                    ledger.provider = response.model
            ledger.output_tokens += recorded_output_tokens
            outcome = "failed" if failed else "completed"
            provider = ledger.provider
        self._record_metrics(
            auth,
            outcome,
            input_tokens=input_tokens,
            output_tokens=recorded_output_tokens,
            latency_ms=elapsed_ms,
        )
        self.logger.info(
            "Generation call %s activity=%s policy=%s operation=%s workspace=%s provider=%s profile=%s",
            outcome,
            auth.activity.value,
            auth.policy.value,
            auth.operation_id,
            auth.workspace_id,
            provider or "unknown",
            profile,
        )

    async def complete(
        self,
        request: LLMRequest,
        profile: str = "default",
        *,
        authorization: GenerationAuthorization | None = None,
        activity: GenerationActivity | None = None,
    ) -> LLMResponse:
        """Authorize, account for, and route one completion."""
        auth, reserved_output, input_tokens = await self._reserve(request, profile, authorization, activity)
        started = time.monotonic()
        try:
            response = await self.registry.complete(request, profile=profile)
        except Exception:
            await self._finish(
                auth,
                response=None,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                failed=True,
                profile=profile,
                input_tokens=input_tokens,
                reserved_output_tokens=reserved_output,
            )
            raise
        await self._finish(
            auth,
            response=response,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            failed=False,
            profile=profile,
            input_tokens=input_tokens,
            reserved_output_tokens=reserved_output,
        )
        return response

    async def complete_stream(
        self,
        request: LLMRequest,
        profile: str = "default",
        *,
        authorization: GenerationAuthorization | None = None,
        activity: GenerationActivity | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Authorize and account for a streaming completion."""
        auth, reserved_output, input_tokens = await self._reserve(request, profile, authorization, activity)
        started = time.monotonic()
        output_chars = 0
        failed = True
        try:
            async for chunk in self.registry.complete_stream(request, profile=profile):
                output_chars += len(chunk.content or "")
                yield chunk
            failed = False
        finally:
            await self._finish(
                auth,
                response=None,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                failed=failed,
                profile=profile,
                input_tokens=input_tokens,
                output_tokens=max(0, (output_chars + 3) // 4),
                reserved_output_tokens=reserved_output,
            )

    async def synthesize(
        self,
        prompt: str,
        context: str | None = None,
        max_tokens: int = None,
        temperature: float = None,
        temperature_factor: float = None,
        profile: str = "default",
        *,
        authorization: GenerationAuthorization | None = None,
        activity: GenerationActivity | None = None,
    ) -> str:
        """Simple synthesis - prompt with optional context.

        Args:
            prompt: User prompt/question
            context: Optional context to include
            max_tokens: Maximum response tokens (None = provider default)
            temperature: Explicit sampling temperature (overrides factor)
            temperature_factor: Multiplier against provider's default temperature
            profile: LLM provider profile to use

        Returns:
            Generated text
        """
        messages = []

        if context:
            messages.append(LLMMessage(role=LLMRole.SYSTEM, content=f"Use this context to inform your response:\n\n{context}"))

        messages.append(LLMMessage(role=LLMRole.USER, content=prompt))

        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            temperature_factor=temperature_factor,
        )

        response = await self.complete(
            request,
            profile=profile,
            authorization=authorization,
            activity=activity,
        )
        return response.content

    async def answer_question(
        self,
        question: str,
        memories: list[str],
        max_tokens: int = 500,
        profile: str = "default",
        *,
        authorization: GenerationAuthorization | None = None,
    ) -> str:
        """Answer question using memories as context.

        Args:
            question: User question
            memories: List of memory contents
            max_tokens: Maximum response tokens
            profile: LLM provider profile to use

        Returns:
            Generated answer
        """
        context = "\n\n".join([f"- {m}" for m in memories])

        system_prompt = f"""You are a helpful assistant with access to the user's memories.
Answer questions based on the provided memories. If the memories don't contain
relevant information, say so.

Memories:
{context}"""

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            LLMMessage(role=LLMRole.USER, content=question),
        ]

        request = LLMRequest(
            messages=messages,
            max_tokens=max_tokens,
            temperature_factor=0.7,  # Lower temp for factual answers
        )

        response = await self.complete(
            request,
            profile=profile,
            authorization=authorization,
            activity=GenerationActivity.SYNTHESIS,
        )
        return response.content

    async def get_ledger(self, operation_id: str) -> GenerationLedger | None:
        """Return an immutable snapshot of one operation ledger."""
        async with self._ledger_lock:
            ledger = self._ledgers.get(operation_id)
            return ledger.model_copy(deep=True) if ledger else None

    async def generation_summary(self, operation_id: str) -> GenerationSummary:
        ledger = await self.get_ledger(operation_id)
        if ledger is None:
            return GenerationSummary(policy=self.policy)
        return GenerationSummary(
            policy=ledger.policy,
            calls=ledger.allowed,
            input_tokens=ledger.input_tokens,
            output_tokens=ledger.output_tokens,
        )

    @property
    def unlabeled_calls(self) -> int:
        return self._unlabeled_calls

    @property
    def default_model(self) -> str:
        """Default model from the default profile provider."""
        return self.registry.get_provider("default").default_model

    @property
    def supports_streaming(self) -> bool:
        """Streaming support from the default profile provider."""
        return self.registry.get_provider("default").supports_streaming


class DefaultLLMServicePlugin(LLMServicePluginBase):
    """Plugin for default LLM service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> LLMService:
        registry: LLMProviderRegistry = self.get_extension(EXT_LLM_REGISTRY, v)
        return LLMService(registry=registry, v=v)
