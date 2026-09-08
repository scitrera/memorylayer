"""Unit tests for the Fireworks (OpenAI-compatible) LLM provider."""

import pytest

from memorylayer_server.services.api_key_store import EnvApiKeyStore
from memorylayer_server.services.llm.fireworks import (
    DEFAULT_FIREWORKS_BASE_URL,
    FireworksLLMProvider,
)
from memorylayer_server.services.llm.openai import OpenAILLMProvider
from memorylayer_server.services.llm.registry import create_provider_from_config


class _FakeStore(EnvApiKeyStore):
    def __init__(self, mapping):
        super().__init__(None)
        self._mapping = mapping

    async def get_api_key(self, name, *, default=None):
        return self._mapping.get(name, default)


def test_is_openai_compatible_subclass():
    provider = FireworksLLMProvider(api_key="fw-key")
    assert isinstance(provider, OpenAILLMProvider)


def test_default_base_url_and_key_name():
    provider = FireworksLLMProvider(api_key="fw-key")
    assert provider.base_url == DEFAULT_FIREWORKS_BASE_URL
    assert FireworksLLMProvider.DEFAULT_API_KEY_NAME == "FIREWORKS_API_KEY"
    # base must be the inference root (SDK appends /chat/completions).
    assert provider.base_url.endswith("/inference/v1")


def test_base_url_override_respected():
    provider = FireworksLLMProvider(api_key="fw-key", base_url="https://proxy.local/v1")
    assert provider.base_url == "https://proxy.local/v1"


def test_resolves_fireworks_key_name_from_store():
    store = _FakeStore({"FIREWORKS_API_KEY": "fw-from-store"})
    provider = FireworksLLMProvider(api_key=None, api_key_store=store)
    assert provider.api_key is None  # no literal -> store path
    assert provider._key_ref.name == "FIREWORKS_API_KEY"


@pytest.mark.asyncio
async def test_ensure_client_uses_store_key():
    store = _FakeStore({"FIREWORKS_API_KEY": "fw-from-store"})
    provider = FireworksLLMProvider(api_key=None, api_key_store=store)
    built = []
    provider._build_client = lambda key: built.append(key) or f"client::{key}"
    client = await provider._ensure_client()
    assert client == "client::fw-from-store"
    assert built == ["fw-from-store"]


def test_create_provider_from_config_fireworks():
    provider = create_provider_from_config(
        name="fast",
        provider_type="fireworks",
        model="accounts/fireworks/models/qwen2p5-72b-instruct",
        api_key="fw-key",
    )
    assert isinstance(provider, FireworksLLMProvider)
    assert provider.model == "accounts/fireworks/models/qwen2p5-72b-instruct"
    assert provider.base_url == DEFAULT_FIREWORKS_BASE_URL


def test_create_provider_from_config_fireworks_base_url_override():
    provider = create_provider_from_config(
        name="fast",
        provider_type="fireworks",
        api_key="fw-key",
        base_url="https://proxy.local/v1",
    )
    assert provider.base_url == "https://proxy.local/v1"
