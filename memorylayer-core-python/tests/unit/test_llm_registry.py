"""Unit tests for LLM Provider Registry."""
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from scitrera_app_framework.api import Variables

from memorylayer_server.services.llm.registry import (
    LLMProviderRegistry,
    DefaultLLMProviderRegistryPlugin,
    create_provider_from_config,
)
from memorylayer_server.services.llm.service_default import LLMService
from memorylayer_server.services.llm.noop import NoOpLLMProvider, LLMNotConfiguredError
from memorylayer_server.models.llm import (
    LLMMessage, LLMRequest, LLMResponse, LLMRole, LLMStreamChunk,
)


# ============================================
# Helper factories
# ============================================

def _mock_provider(name: str = "mock") -> AsyncMock:
    """Create a mock LLMProvider with standard defaults."""
    provider = AsyncMock()
    provider.default_model = f"{name}-model"
    provider.supports_streaming = True
    provider.complete.return_value = LLMResponse(
        content=f"response from {name}",
        model=f"{name}-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        finish_reason="stop",
    )

    async def _stream_gen(request):
        yield LLMStreamChunk(content="chunk1", is_final=False)
        yield LLMStreamChunk(content="", is_final=True, finish_reason="stop")

    provider.complete_stream = _stream_gen
    return provider


def _make_request() -> LLMRequest:
    return LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
        max_tokens=100,
    )


# ============================================
# LLMProviderRegistry Tests
# ============================================

class TestLLMProviderRegistry:
    """Tests for LLMProviderRegistry."""

    def test_get_provider_returns_default(self):
        """Registry with a default provider returns it when no profile specified."""
        default = _mock_provider("default")
        registry = LLMProviderRegistry(providers={"default": default})

        result = registry.get_provider()
        assert result is default

    def test_get_provider_returns_named(self):
        """Registry with multiple providers returns the right one by name."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )

        assert registry.get_provider("cheap") is cheap
        assert registry.get_provider("default") is default

    def test_get_provider_falls_back_to_default(self):
        """Unknown profile name falls back to the default provider."""
        default = _mock_provider("default")
        registry = LLMProviderRegistry(providers={"default": default})

        result = registry.get_provider("nonexistent")
        assert result is default

    def test_get_provider_uses_profile_map(self):
        """Profile map resolves activity names to provider names."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
            profile_map={"tier_generation": "cheap"},
        )

        result = registry.get_provider("tier_generation")
        assert result is cheap

    def test_get_provider_profile_map_then_fallback(self):
        """If mapped name is not in providers, falls back to default."""
        default = _mock_provider("default")
        registry = LLMProviderRegistry(
            providers={"default": default},
            profile_map={"tier_generation": "missing_profile"},
        )

        result = registry.get_provider("tier_generation")
        assert result is default

    @pytest.mark.asyncio
    async def test_complete_routes_to_profile(self):
        """complete() routes to the correct provider based on profile."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )

        request = _make_request()
        response = await registry.complete(request, profile="cheap")

        assert response.content == "response from cheap"
        cheap.complete.assert_called_once_with(request)
        default.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_stream_routes_to_profile(self):
        """complete_stream() routes to the correct provider based on profile."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )

        request = _make_request()
        chunks = []
        async for chunk in registry.complete_stream(request, profile="cheap"):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "chunk1"
        assert chunks[1].is_final is True

    def test_profile_names(self):
        """profile_names returns list of all provider names."""
        registry = LLMProviderRegistry(
            providers={
                "default": _mock_provider("default"),
                "cheap": _mock_provider("cheap"),
                "reasoning": _mock_provider("reasoning"),
            },
        )

        names = registry.profile_names
        assert sorted(names) == ["cheap", "default", "reasoning"]

    def test_profile_map_returns_copy(self):
        """profile_map returns a copy of the assignment map."""
        original_map = {"tier_generation": "cheap"}
        registry = LLMProviderRegistry(
            providers={"default": _mock_provider("default")},
            profile_map=original_map,
        )

        result = registry.profile_map
        assert result == {"tier_generation": "cheap"}
        # Mutating the returned map should not affect the registry
        result["new_key"] = "new_value"
        assert "new_key" not in registry.profile_map


# ============================================
# create_provider_from_config Tests
# ============================================

class TestCreateProviderFromConfig:
    """Tests for create_provider_from_config factory function."""

    def test_create_openai_provider(self):
        """Creates OpenAILLMProvider with correct params."""
        provider = create_provider_from_config(
            name="default",
            provider_type="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.example.com",
        )
        from memorylayer_server.services.llm.openai import OpenAILLMProvider
        assert isinstance(provider, OpenAILLMProvider)
        assert provider.model == "gpt-4o-mini"
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.example.com"

    def test_create_anthropic_provider(self):
        """Creates AnthropicLLMProvider with correct params."""
        provider = create_provider_from_config(
            name="default",
            provider_type="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
        )
        from memorylayer_server.services.llm.anthropic import AnthropicLLMProvider
        assert isinstance(provider, AnthropicLLMProvider)
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.api_key == "test-key"

    def test_create_google_provider(self):
        """Creates GoogleLLMProvider with correct params."""
        provider = create_provider_from_config(
            name="default",
            provider_type="google",
            model="gemini-3-flash-preview",
            api_key="test-key",
        )
        from memorylayer_server.services.llm.google import GoogleLLMProvider
        assert isinstance(provider, GoogleLLMProvider)
        assert provider.model == "gemini-3-flash-preview"
        assert provider.api_key == "test-key"

    def test_create_noop_provider(self):
        """Creates NoOpLLMProvider."""
        provider = create_provider_from_config(
            name="noop",
            provider_type="noop",
            model="unused",
        )
        assert isinstance(provider, NoOpLLMProvider)

    def test_create_unknown_raises(self):
        """ValueError for unknown provider type."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider_from_config(
                name="bad",
                provider_type="unknown_provider",
                model="model",
            )

    def test_create_with_max_tokens(self):
        """Passes default_max_tokens through to the provider."""
        provider = create_provider_from_config(
            name="default",
            provider_type="openai",
            model="gpt-4o-mini",
            api_key="test-key",
            max_tokens=2048,
        )
        assert provider.default_max_tokens == 2048


# ============================================
# DefaultLLMProviderRegistryPlugin Tests
# ============================================

class TestDefaultLLMProviderRegistryPlugin:
    """Tests for DefaultLLMProviderRegistryPlugin environment discovery."""

    @staticmethod
    def _make_v():
        """Create a real Variables instance that reads from os.environ."""
        return Variables()

    @patch.dict(os.environ, {
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER': 'openai',
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL': 'gpt-4o-mini',
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY': 'test-key',
    }, clear=False)
    def test_discover_single_profile(self):
        """One DEFAULT profile creates one provider."""
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        assert "default" in registry.profile_names
        assert len(registry.profile_names) == 1

    @patch.dict(os.environ, {
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER': 'openai',
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL': 'gpt-4o-mini',
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_API_KEY': 'test-key',
        'MEMORYLAYER_LLM_PROFILE_CHEAP_PROVIDER': 'openai',
        'MEMORYLAYER_LLM_PROFILE_CHEAP_MODEL': 'gpt-4o-mini',
        'MEMORYLAYER_LLM_PROFILE_CHEAP_API_KEY': 'test-key-2',
        'MEMORYLAYER_LLM_PROFILE_REASONING_PROVIDER': 'anthropic',
        'MEMORYLAYER_LLM_PROFILE_REASONING_MODEL': 'claude-sonnet-4-20250514',
        'MEMORYLAYER_LLM_PROFILE_REASONING_API_KEY': 'test-key-3',
    }, clear=False)
    def test_discover_multiple_profiles(self):
        """DEFAULT + CHEAP + REASONING profiles are all discovered."""
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        names = sorted(registry.profile_names)
        assert names == ["cheap", "default", "reasoning"]

    @patch.dict(os.environ, {}, clear=True)
    def test_no_profiles_creates_noop_default(self):
        """Empty env creates NoOp default provider."""
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        assert "default" in registry.profile_names
        provider = registry.get_provider("default")
        assert isinstance(provider, NoOpLLMProvider)

    @patch.dict(os.environ, {
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER': 'noop',
        'MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL': 'unused',
        'MEMORYLAYER_LLM_ASSIGN_TIER_GENERATION': 'cheap',
        'MEMORYLAYER_LLM_ASSIGN_EMBEDDING': 'fast',
    }, clear=False)
    def test_assignment_mapping(self):
        """MEMORYLAYER_LLM_ASSIGN_* creates profile_map entries."""
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        profile_map = registry.profile_map
        assert profile_map["tier_generation"] == "cheap"
        assert profile_map["embedding"] == "fast"

    @patch.dict(os.environ, {
        'MEMORYLAYER_LLM_PROFILE_BAD_MODEL': 'gpt-4o-mini',
        'MEMORYLAYER_LLM_PROFILE_BAD_API_KEY': 'test-key',
        # Missing PROVIDER
    }, clear=True)
    def test_missing_provider_type_skips(self):
        """Profile without PROVIDER is skipped, falls back to NoOp default."""
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        # "bad" profile should be skipped; only default (NoOp) remains
        provider = registry.get_provider("default")
        assert isinstance(provider, NoOpLLMProvider)
        mock_logger.warning.assert_any_call(
            "LLM profile '%s' missing PROVIDER, skipping", "bad"
        )

    @patch.dict(os.environ, {
        'MEMORYLAYER_LLM_PROFILE_BAD_PROVIDER': 'openai',
        'MEMORYLAYER_LLM_PROFILE_BAD_API_KEY': 'test-key',
        # Missing MODEL — provider uses its built-in default
    }, clear=True)
    def test_missing_model_uses_provider_default(self):
        """Profile without MODEL uses the provider's built-in default model."""
        from memorylayer_server.services.llm.openai import OpenAILLMProvider, DEFAULT_LLM_OPENAI_MODEL
        plugin = DefaultLLMProviderRegistryPlugin()
        mock_v = self._make_v()
        mock_logger = MagicMock()

        registry = plugin.initialize(mock_v, mock_logger)

        assert "bad" in registry.profile_names
        provider = registry.get_provider("bad")
        assert isinstance(provider, OpenAILLMProvider)
        assert provider.model == DEFAULT_LLM_OPENAI_MODEL

    @patch.dict(os.environ, {}, clear=True)
    def test_converged_config_via_variables(self):
        """Profiles set directly on Variables instance (no env vars) are discovered."""
        v = Variables()
        # Set profile config directly on Variables — simulates converged config
        v.set('MEMORYLAYER_LLM_PROFILE_DEFAULT_PROVIDER', 'noop')
        v.set('MEMORYLAYER_LLM_PROFILE_DEFAULT_MODEL', 'test-model')
        v.set('MEMORYLAYER_LLM_PROFILE_FAST_PROVIDER', 'noop')
        v.set('MEMORYLAYER_LLM_PROFILE_FAST_MODEL', 'fast-model')
        v.set('MEMORYLAYER_LLM_ASSIGN_EXTRACTION', 'fast')

        plugin = DefaultLLMProviderRegistryPlugin()
        mock_logger = MagicMock()

        registry = plugin.initialize(v, mock_logger)

        assert sorted(registry.profile_names) == ["default", "fast"]
        assert isinstance(registry.get_provider("default"), NoOpLLMProvider)
        assert isinstance(registry.get_provider("fast"), NoOpLLMProvider)
        assert registry.profile_map["extraction"] == "fast"


# ============================================
# LLMService with Registry Tests
# ============================================

class TestLLMServiceWithRegistry:
    """Tests for LLMService profile-based routing through registry."""

    @pytest.mark.asyncio
    async def test_complete_forwards_profile(self):
        """Verify profile param reaches registry from service.complete()."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )
        service = LLMService(registry=registry)

        request = _make_request()
        response = await service.complete(request, profile="cheap")

        assert response.content == "response from cheap"
        cheap.complete.assert_called_once_with(request)
        default.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesize_forwards_profile(self):
        """Verify synthesize passes profile through to registry."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )
        service = LLMService(registry=registry)

        result = await service.synthesize("Tell me something", profile="cheap")

        assert result == "response from cheap"
        cheap.complete.assert_called_once()
        default.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_profile_when_not_specified(self):
        """Calls without profile use 'default'."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )
        service = LLMService(registry=registry)

        request = _make_request()
        response = await service.complete(request)

        assert response.content == "response from default"
        default.complete.assert_called_once_with(request)
        cheap.complete.assert_not_called()

    def test_provider_property_returns_default(self):
        """service.provider returns default provider from registry."""
        default = _mock_provider("default")
        cheap = _mock_provider("cheap")
        registry = LLMProviderRegistry(
            providers={"default": default, "cheap": cheap},
        )
        service = LLMService(registry=registry)

        assert service.provider is default
