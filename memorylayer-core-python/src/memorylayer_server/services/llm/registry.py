"""LLM Provider Registry - profile-based provider routing."""
from logging import Logger
from typing import AsyncIterator

from scitrera_app_framework import get_logger, Variables

from .base import LLMProvider, LLMProviderRegistryPluginBase, EXT_LLM_REGISTRY
from .noop import NoOpLLMProvider
from ...models.llm import LLMRequest, LLMResponse, LLMStreamChunk


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

    async def complete_stream(
        self, request: LLMRequest, profile: str = "default"
    ) -> AsyncIterator[LLMStreamChunk]:
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
    max_tokens: int | None = None,
    temperature: float | None = None,
    v: Variables = None,
) -> LLMProvider:
    """Create an LLM provider instance from configuration values.

    Args:
        name: Profile name (for logging).
        provider_type: One of 'openai', 'anthropic', 'google', 'noop'.
        model: Model identifier. If None, the provider's built-in default is used.
        base_url: Optional base URL (OpenAI-compatible only).
        api_key: Optional API key.
        max_tokens: Optional default max tokens.
        temperature: Optional default temperature for this profile.
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
            api_key=api_key, base_url=base_url, **model_kwarg,
            default_max_tokens=max_tokens, default_temperature=temperature, v=v,
        )
    elif provider_type == "anthropic":
        from .anthropic import AnthropicLLMProvider
        return AnthropicLLMProvider(
            api_key=api_key, **model_kwarg,
            default_max_tokens=max_tokens, default_temperature=temperature, v=v,
        )
    elif provider_type == "google":
        from .google import GoogleLLMProvider
        return GoogleLLMProvider(
            api_key=api_key, **model_kwarg,
            default_max_tokens=max_tokens, default_temperature=temperature, v=v,
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
    """
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger: Logger) -> LLMProviderRegistry:
        """Build LLM provider registry from environment configuration.

        Uses ``v.import_from_env_by_prefix`` so that configuration can come
        from either real environment variables **or** values already loaded
        into the ``Variables`` instance (converged configuration).
        """
        known_fields = {'provider', 'base_url', 'api_key', 'model', 'max_tokens', 'temperature'}

        # Import MEMORYLAYER_LLM_PROFILE_* into Variables and get flattened dict.
        # Keys are lowercased with prefix stripped, e.g. "default_provider".
        profile_vars = v.import_from_env_by_prefix('MEMORYLAYER_LLM_PROFILE')

        # Discover profile names by stripping known field suffixes from keys
        profile_names: set[str] = set()
        for key in profile_vars:
            for fld in known_fields:
                if key.endswith(f'_{fld}'):
                    name = key[:-(len(fld) + 1)]
                    if name:
                        profile_names.add(name)
                    break

        # Build providers from discovered profiles
        providers: dict[str, LLMProvider] = {}
        for name in sorted(profile_names):
            provider_type = profile_vars.get(f'{name}_provider')
            if not provider_type:
                logger.warning("LLM profile '%s' missing PROVIDER, skipping", name)
                continue

            model = profile_vars.get(f'{name}_model')  # None = provider default

            max_tokens_raw = profile_vars.get(f'{name}_max_tokens')
            max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None

            temp_raw = profile_vars.get(f'{name}_temperature')
            temperature = float(temp_raw) if temp_raw is not None else None

            provider = create_provider_from_config(
                name=name, provider_type=provider_type, model=model,
                base_url=profile_vars.get(f'{name}_base_url'),
                api_key=profile_vars.get(f'{name}_api_key'),
                max_tokens=max_tokens, temperature=temperature, v=v,
            )

            # Post-init validation: provider must resolve to a non-None model
            if provider.default_model is None:
                logger.warning(
                    "LLM profile '%s' (%s) resolved to model=None, skipping",
                    name, provider_type,
                )
                continue

            providers[name] = provider
            logger.info("LLM registry: profile '%s' (%s/%s)", name, provider_type, provider.default_model)

        if 'default' not in providers:
            logger.info("No LLM profiles configured, using NoOp provider for 'default'")
            providers['default'] = NoOpLLMProvider(v=v)

        # Read activity-to-profile assignments via Variables
        assign_vars = v.import_from_env_by_prefix('MEMORYLAYER_LLM_ASSIGN')
        profile_map: dict[str, str] = {
            activity: str(profile_name).lower()
            for activity, profile_name in assign_vars.items()
        }
        for activity, profile_name in profile_map.items():
            logger.info("LLM registry: assign '%s' -> profile '%s'", activity, profile_name)

        registry = LLMProviderRegistry(providers=providers, profile_map=profile_map)
        logger.info(
            "LLM registry initialized: %d profiles (%s), %d assignments",
            len(registry.profile_names), ', '.join(registry.profile_names), len(profile_map),
        )
        return registry
