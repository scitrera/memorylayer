"""LLM Provider Registry - profile-based provider routing."""

from collections.abc import AsyncIterator
from logging import Logger

from scitrera_app_framework import Variables

from ...config import (
    DEFAULT_MEMORYLAYER_TENANT_ID,
    MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS,
    MEMORYLAYER_TENANT_ID,
)
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk
from ..api_key_store import ApiKeyStore, get_api_key_store
from .attribution import (
    HEADER_SOURCE,
    HEADER_TENANT,
    SOURCE_MEMORYLAYER,
    host_allows_identity,
    parse_host_patterns,
)
from .base import LLMProvider, LLMProviderRegistryPluginBase
from .noop import NoOpLLMProvider

# Internal activities that route through a named profile, i.e. every literal passed as
# `profile=` by a service. Used only for startup diagnostics: to report which activities
# fall back to `default`, and to catch assignments naming an activity that does not exist
# (`MEMORYLAYER_LLM_ASSIGN_REFLECT` is a real example — the activity is `reflection`, so
# that spelling resolved to nothing and the assignment was silently discarded).
#
# Keep in sync with `profile=` call sites; drift only costs a diagnostic line, never
# routing behaviour.
INTERNAL_ACTIVITIES: tuple[str, ...] = (
    "agentic",
    "contradiction",
    "extraction",
    "inference",
    "ontology",
    "reflection",
    "reranker",
    "tier_generation",
    "titling",
)


class LLMProviderRegistry:
    """Registry of named LLM provider instances with profile-based routing."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        profile_map: dict[str, str] | None = None,
    ):
        self._providers = providers  # Must include "default"
        self._profile_map: dict[str, str] = profile_map or {}

    def get_provider(self, profile: str = "default") -> LLMProvider:
        """Get provider for a given profile. Falls back to default."""
        # Check assignment map first (e.g., "tier_generation" -> "cheap")
        provider_name = self._profile_map.get(profile, profile)
        # Try the resolved name, then fall back to default
        provider = self._providers.get(provider_name)
        if provider is None:
            provider = self._providers["default"]
        return provider

    async def complete(self, request: LLMRequest, profile: str = "default") -> LLMResponse:
        """Route completion request to the provider for the given profile."""
        provider = self.get_provider(profile)
        return await provider.complete(request)

    async def complete_stream(self, request: LLMRequest, profile: str = "default") -> AsyncIterator[LLMStreamChunk]:
        """Route streaming request to the provider for the given profile."""
        provider = self.get_provider(profile)
        async for chunk in provider.complete_stream(request):
            yield chunk

    @property
    def profile_names(self) -> list[str]:
        """Names of all registered provider profiles."""
        return list(self._providers.keys())

    @property
    def profile_map(self) -> dict[str, str]:
        """Current activity-to-profile assignment map."""
        return dict(self._profile_map)


def create_provider_from_config(
    name: str,
    provider_type: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_name: str | None = None,
    api_key_store: ApiKeyStore | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    embed_server_url: str | None = None,
    embed_server_transport: str | None = None,
    embed_server_aether_target: str | None = None,
    embed_server_timeout: float | None = None,
    extra_body: dict | None = None,
    reasoning_effort: str | None = None,
    default_headers: dict[str, str] | None = None,
    stamp_identity: bool = False,
    v: Variables = None,
) -> LLMProvider:
    """Create an LLM provider instance from configuration values.

    Args:
        name: Profile name (for logging).
        provider_type: One of 'openai', 'fireworks', 'anthropic', 'google', 'embed_server', 'noop'.
        model: Model identifier. If None, the provider's built-in default is used.
        base_url: Optional base URL (OpenAI-compatible only).
        api_key: Optional literal/static API key. When set it wins and the store
            is not consulted (static escape hatch).
        api_key_name: Optional override for the store key name to resolve when no
            literal ``api_key`` is given. Defaults to the provider's own key name.
        api_key_store: API key store used to resolve the key by name (dynamic).
        max_tokens: Optional default max tokens.
        temperature: Optional default temperature for this profile.
        embed_server_url: Per-profile embed-server URL (``embed_server`` type only).
        embed_server_transport: Per-profile ``'http'`` or ``'aether'`` (``embed_server`` only).
        embed_server_aether_target: Per-profile Aether target (``embed_server`` only).
        embed_server_timeout: Per-profile request timeout (``embed_server`` only).
        default_headers: Static HTTP headers sent on every request (OpenAI-compatible
            providers only) — e.g. caller-asserted attribution headers. Providers
            without a base_url (anthropic/google) ignore this. Dropped unless
            ``stamp_identity`` is set.
        stamp_identity: Whether this endpoint is allowed to receive caller-asserted
            identity headers, per the configured host allowlist. Defaults to False:
            these headers name our internal tenant, and every provider here speaks
            the same protocol as a public vendor, so permission must be explicit.
        v: Optional Variables instance for framework logging.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If provider_type is unknown.
    """
    # Build kwargs, only passing model if explicitly set so providers use their defaults
    model_kwarg = {"model": model} if model else {}

    if provider_type == "openai":
        from .openai import OpenAILLMProvider

        return OpenAILLMProvider(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            base_url=base_url,
            **model_kwarg,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            default_extra_body=extra_body,
            default_reasoning_effort=reasoning_effort,
            default_headers=default_headers,
            stamp_identity=stamp_identity,
            v=v,
        )
    elif provider_type == "anthropic":
        from .anthropic import AnthropicLLMProvider

        return AnthropicLLMProvider(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            **model_kwarg,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            v=v,
        )
    elif provider_type == "fireworks":
        from .fireworks import FireworksLLMProvider

        return FireworksLLMProvider(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            base_url=base_url,
            **model_kwarg,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            default_extra_body=extra_body,
            default_reasoning_effort=reasoning_effort,
            default_headers=default_headers,
            stamp_identity=stamp_identity,
            v=v,
        )
    elif provider_type == "google":
        from .google import GoogleLLMProvider

        return GoogleLLMProvider(
            api_key=api_key,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            **model_kwarg,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            v=v,
        )
    elif provider_type == "embed_server":
        from .embed_server import EmbedServerLLMProvider

        return EmbedServerLLMProvider(
            model=model,
            embed_server_url=embed_server_url,
            embed_server_transport=embed_server_transport,
            embed_server_aether_target=embed_server_aether_target,
            embed_server_timeout=embed_server_timeout,
            default_max_tokens=max_tokens,
            default_temperature=temperature,
            v=v,
        )
    elif provider_type == "noop":
        return NoOpLLMProvider(v=v)
    else:
        raise ValueError(f"Unknown provider type: {provider_type!r}")


class DefaultLLMProviderRegistryPlugin(LLMProviderRegistryPluginBase):
    """Default plugin that builds the LLM registry from environment variables.

    Profile configuration uses env vars of the form:
        MEMORYLAYER_LLM_PROFILE_<NAME>_PROVIDER=openai
        MEMORYLAYER_LLM_PROFILE_<NAME>_MODEL=gpt-4o-mini
        MEMORYLAYER_LLM_PROFILE_<NAME>_BASE_URL=...
        MEMORYLAYER_LLM_PROFILE_<NAME>_API_KEY=...
        MEMORYLAYER_LLM_PROFILE_<NAME>_MAX_TOKENS=4096
        MEMORYLAYER_LLM_PROFILE_<NAME>_TEMPERATURE=0.7

    Activity-to-profile assignment uses:
        MEMORYLAYER_LLM_ASSIGN_<ACTIVITY>=<profile_name>

    ACTIVITY must be one of INTERNAL_ACTIVITIES; an unrecognized name is ignored,
    so it is warned about at startup rather than failing silently.
    """

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> LLMProviderRegistry:
        """Build LLM provider registry from environment configuration.

        Uses ``v.import_from_env_by_prefix`` so that configuration can come
        from either real environment variables **or** values already loaded
        into the ``Variables`` instance (converged configuration).
        """
        # Multi-word fields must be checked before their shorter prefixes
        # (we sort by length descending below). For example,
        # ``embed_server_aether_target`` must win over ``base_url`` /
        # ``api_key`` even though we iterate a set in undefined order.
        known_fields = {
            "provider",
            "base_url",
            "api_key",
            "api_key_name",
            "model",
            "max_tokens",
            "temperature",
            "embed_server_url",
            "embed_server_transport",
            "embed_server_aether_target",
            "embed_server_timeout",
        }
        # Longest first ensures specific multi-word fields are stripped
        # rather than a shorter trailing word collapsing into the profile name.
        fields_sorted = sorted(known_fields, key=len, reverse=True)

        # Import MEMORYLAYER_LLM_PROFILE_* into Variables and get flattened dict.
        # Keys are lowercased with prefix stripped, e.g. "default_provider".
        profile_vars = v.import_from_env_by_prefix("MEMORYLAYER_LLM_PROFILE")

        # API key store: providers resolve their key lazily through this (env by
        # default; Aether-first in enterprise) so rotated keys are picked up
        # without a restart. A literal per-profile API_KEY still wins statically.
        api_key_store = get_api_key_store(v)

        # Static caller-asserted attribution headers stamped on every outgoing
        # OpenAI-compatible LLM call (so the MLflow AI Gateway can attribute the
        # traffic). MemoryLayer asserts only its OWN known identity: a static
        # source and its single-tenant slug. X-Scitrera-Tenant is omitted when
        # MEMORYLAYER_TENANT_ID is unset (empty). The per-task X-Scitrera-Task-Id
        # header is added at call time from the ambient task context.
        static_llm_headers: dict[str, str] = {HEADER_SOURCE: SOURCE_MEMORYLAYER}
        tenant_id = v.environ(MEMORYLAYER_TENANT_ID, DEFAULT_MEMORYLAYER_TENANT_ID)
        if tenant_id:
            static_llm_headers[HEADER_TENANT] = tenant_id

        # ...but ONLY to first-party endpoints. These headers name our internal
        # tenant, and every provider built here speaks the same OpenAI-compatible
        # protocol as a public vendor, so the provider type tells us nothing about
        # who is on the other end — only the base_url host does. An empty allowlist
        # stamps nothing at all; enterprise ships a default for its own gateway.
        #
        # Read with NO explicit default on purpose. ``v.environ(key, default)``
        # REGISTERS that default, overwriting any ``set_default_value`` an outer
        # layer installed earlier — so passing the OSS default here would silently
        # clobber the enterprise allowlist and stamp nothing, with no error to show
        # for it. Unset resolves to None, which parses to "no hosts", which IS the
        # OSS default (see DEFAULT_MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS).
        identity_hosts = parse_host_patterns(v.environ(MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS))
        if identity_hosts:
            logger.info(
                "LLM attribution headers %s stamped only for hosts matching %s",
                sorted(static_llm_headers),
                list(identity_hosts),
            )
        else:
            logger.info(
                "LLM attribution headers disabled: %s is empty, so no profile is "
                "stamped with caller-asserted identity",
                MEMORYLAYER_LLM_IDENTITY_HEADER_HOSTS,
            )

        # Discover profile names by stripping known field suffixes from keys
        profile_names: set[str] = set()
        for key in profile_vars:
            for fld in fields_sorted:
                if key.endswith(f"_{fld}"):
                    name = key[: -(len(fld) + 1)]
                    if name:
                        profile_names.add(name)
                    break

        # Build providers from discovered profiles
        providers: dict[str, LLMProvider] = {}
        for name in sorted(profile_names):
            provider_type = profile_vars.get(f"{name}_provider")
            if not provider_type:
                logger.warning("LLM profile '%s' missing PROVIDER, skipping", name)
                continue

            model = profile_vars.get(f"{name}_model")  # None = provider default

            max_tokens_raw = profile_vars.get(f"{name}_max_tokens")
            max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None

            temp_raw = profile_vars.get(f"{name}_temperature")
            temperature = float(temp_raw) if temp_raw is not None else None

            embed_server_timeout_raw = profile_vars.get(f"{name}_embed_server_timeout")
            embed_server_timeout = float(embed_server_timeout_raw) if embed_server_timeout_raw is not None else None

            # Optional per-profile extra_body (JSON object) merged into every
            # request — e.g. '{"enable_thinking": false}' to turn off a reasoning
            # model's thinking for this profile. Invalid JSON is logged + ignored.
            extra_body = None
            extra_body_raw = profile_vars.get(f"{name}_extra_body")
            if extra_body_raw:
                import json as _json

                try:
                    parsed = _json.loads(extra_body_raw)
                    extra_body = parsed if isinstance(parsed, dict) else None
                    if extra_body is None:
                        logger.warning("LLM profile '%s' extra_body is not a JSON object, ignoring", name)
                except (ValueError, TypeError):
                    logger.warning("LLM profile '%s' extra_body is not valid JSON, ignoring", name)

            # Optional per-profile reasoning_effort (e.g. "none" to disable a
            # reasoning model's thinking — the Fireworks-honored mechanism).
            reasoning_effort = profile_vars.get(f"{name}_reasoning_effort") or None

            # Identity stamping is decided per profile from its own base_url, so a
            # deployment can route some activities to the internal gateway and
            # others to a public vendor without the vendor seeing our tenant.
            profile_base_url = profile_vars.get(f"{name}_base_url")
            stamp_identity = host_allows_identity(profile_base_url, identity_hosts)
            if stamp_identity:
                # Logged per profile: this is a security-relevant decision, and an
                # operator should be able to confirm from the startup log exactly
                # which endpoints are told who we are.
                logger.info("LLM profile '%s' is first-party: stamping identity headers", name)

            provider = create_provider_from_config(
                name=name,
                provider_type=provider_type,
                model=model,
                base_url=profile_base_url,
                api_key=profile_vars.get(f"{name}_api_key"),
                api_key_name=profile_vars.get(f"{name}_api_key_name"),
                api_key_store=api_key_store,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                embed_server_url=profile_vars.get(f"{name}_embed_server_url"),
                embed_server_transport=profile_vars.get(f"{name}_embed_server_transport"),
                embed_server_aether_target=profile_vars.get(f"{name}_embed_server_aether_target"),
                embed_server_timeout=embed_server_timeout,
                extra_body=extra_body,
                default_headers=static_llm_headers,
                stamp_identity=stamp_identity,
                v=v,
            )

            # Post-init validation: provider must resolve to a non-None model
            if provider.default_model is None:
                logger.warning(
                    "LLM profile '%s' (%s) resolved to model=None, skipping",
                    name,
                    provider_type,
                )
                continue

            providers[name] = provider
            logger.info("LLM registry: profile '%s' (%s/%s)", name, provider_type, provider.default_model)

        if "default" not in providers:
            logger.info("No LLM profiles configured, using NoOp provider for 'default'")
            providers["default"] = NoOpLLMProvider(v=v)

        # Read activity-to-profile assignments via Variables
        assign_vars = v.import_from_env_by_prefix("MEMORYLAYER_LLM_ASSIGN")
        profile_map: dict[str, str] = {activity: str(profile_name).lower() for activity, profile_name in assign_vars.items()}
        for activity, profile_name in profile_map.items():
            logger.info("LLM registry: assign '%s' -> profile '%s'", activity, profile_name)

        # An assignment naming an activity nobody requests is a no-op that looks like a
        # working config. Say so instead of discarding it quietly.
        for activity in sorted(set(profile_map) - set(INTERNAL_ACTIVITIES)):
            logger.warning(
                "LLM registry: assignment for unknown activity '%s' has no effect (known activities: %s)",
                activity,
                ", ".join(INTERNAL_ACTIVITIES),
            )

        # Report which internal activities land on 'default'. Maintenance work
        # (summarization, relationship typing, extraction) runs on every stored memory, so
        # inheriting a large chat model here is a standing cost that is otherwise invisible.
        unrouted = [activity for activity in INTERNAL_ACTIVITIES if profile_map.get(activity, activity) not in providers]
        if unrouted:
            logger.info(
                "LLM registry: %s using profile 'default' (%s) — assign a cheaper profile with "
                "MEMORYLAYER_LLM_ASSIGN_<ACTIVITY> if that model is expensive",
                ", ".join(unrouted),
                providers["default"].default_model,
            )

        registry = LLMProviderRegistry(providers=providers, profile_map=profile_map)
        logger.info(
            "LLM registry initialized: %d profiles (%s), %d assignments",
            len(registry.profile_names),
            ", ".join(registry.profile_names),
            len(profile_map),
        )
        return registry
