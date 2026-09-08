"""Unit tests for the API key store abstraction and provider wiring."""

import os
import sys
import types
from unittest.mock import patch

import pytest
from scitrera_app_framework.api import Variables


def _fake_openai_module():
    """A stand-in ``openai`` module whose AsyncOpenAI records its api_key."""
    mod = types.ModuleType("openai")

    class _FakeAsyncOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    mod.AsyncOpenAI = _FakeAsyncOpenAI
    return mod

from memorylayer_server.services.api_key_store import (
    ApiKeyRef,
    EnvApiKeyStore,
    resolve_api_key_ref,
)


# ============================================
# EnvApiKeyStore
# ============================================


class TestEnvApiKeyStore:
    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-env"}, clear=False)
    async def test_resolves_from_env(self):
        store = EnvApiKeyStore(Variables())
        assert await store.get_api_key("OPENAI_API_KEY") == "sk-from-env"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {}, clear=True)
    async def test_missing_returns_default(self):
        store = EnvApiKeyStore(Variables())
        assert await store.get_api_key("NOPE") is None
        assert await store.get_api_key("NOPE", default="fb") == "fb"

    @pytest.mark.asyncio
    async def test_no_variables_returns_default(self):
        store = EnvApiKeyStore(None)
        assert await store.get_api_key("ANY", default="fb") == "fb"


# ============================================
# resolve_api_key_ref / ApiKeyRef
# ============================================


class _FakeStore(EnvApiKeyStore):
    def __init__(self, mapping):
        super().__init__(None)
        self._mapping = mapping

    async def get_api_key(self, name, *, default=None):
        return self._mapping.get(name, default)


class TestResolveApiKeyRef:
    def test_literal_wins_and_skips_store(self):
        store = _FakeStore({"OPENAI_API_KEY": "sk-store"})
        static, ref = resolve_api_key_ref(
            api_key="sk-literal", api_key_name=None,
            api_key_store=store, default_name="OPENAI_API_KEY",
        )
        assert static == "sk-literal"
        assert ref is None

    def test_name_override(self):
        store = _FakeStore({})
        static, ref = resolve_api_key_ref(
            api_key=None, api_key_name="CUSTOM_KEY",
            api_key_store=store, default_name="OPENAI_API_KEY",
        )
        assert static is None
        assert isinstance(ref, ApiKeyRef)
        assert ref.name == "CUSTOM_KEY"

    def test_default_name_when_no_override(self):
        store = _FakeStore({})
        _static, ref = resolve_api_key_ref(
            api_key=None, api_key_name=None,
            api_key_store=store, default_name="ANTHROPIC_API_KEY",
        )
        assert ref.name == "ANTHROPIC_API_KEY"

    def test_no_store_no_key(self):
        static, ref = resolve_api_key_ref(
            api_key=None, api_key_name=None,
            api_key_store=None, default_name="OPENAI_API_KEY",
        )
        assert static is None and ref is None

    @pytest.mark.asyncio
    async def test_ref_resolves_through_store(self):
        store = _FakeStore({"OPENAI_API_KEY": "sk-store"})
        ref = ApiKeyRef(store, "OPENAI_API_KEY")
        assert await ref.resolve() == "sk-store"


# ============================================
# LLM provider dynamic resolution + client rebuild
# ============================================


class TestLLMProviderDynamicKey:
    @pytest.mark.asyncio
    async def test_static_key_skips_store(self):
        from memorylayer_server.services.llm.openai import OpenAILLMProvider

        store = _FakeStore({"OPENAI_API_KEY": "sk-store"})
        provider = OpenAILLMProvider(api_key="sk-literal", api_key_store=store)
        # literal wins -> no key ref, store never consulted
        assert provider.api_key == "sk-literal"
        assert provider._key_ref is None

    @pytest.mark.asyncio
    async def test_resolves_default_name_and_rebuilds_on_rotation(self):
        from memorylayer_server.services.llm.openai import OpenAILLMProvider

        mapping = {"OPENAI_API_KEY": "sk-v1"}
        store = _FakeStore(mapping)
        provider = OpenAILLMProvider(api_key=None, api_key_store=store)
        assert provider.api_key is None  # no literal
        assert provider._key_ref.name == "OPENAI_API_KEY"

        built = []
        provider._build_client = lambda key: built.append(key) or f"client::{key}"

        c1 = await provider._ensure_client()
        assert c1 == "client::sk-v1"
        # same key -> cached, no rebuild
        c2 = await provider._ensure_client()
        assert c2 is c1 or c2 == c1
        assert built == ["sk-v1"]

        # rotate the key -> client rebuilt
        mapping["OPENAI_API_KEY"] = "sk-v2"
        c3 = await provider._ensure_client()
        assert c3 == "client::sk-v2"
        assert built == ["sk-v1", "sk-v2"]


# ============================================
# Embedding provider keyless placeholder + store resolution
# ============================================


class TestEmbeddingProviderKey:
    @pytest.mark.asyncio
    async def test_keyless_uses_placeholder(self):
        from memorylayer_server.services.embedding.openai import OpenAIEmbeddingProvider

        store = _FakeStore({})  # nothing resolves
        provider = OpenAIEmbeddingProvider(api_key=None, api_key_store=store)
        with patch.dict(sys.modules, {"openai": _fake_openai_module()}):
            client = await provider._ensure_client()
        # OpenAI-compatible local servers ignore the key; SDK needs a value.
        assert client.api_key == "x"

    @pytest.mark.asyncio
    async def test_store_resolved_key_used(self):
        from memorylayer_server.services.embedding.openai import OpenAIEmbeddingProvider

        store = _FakeStore({"OPENAI_API_KEY": "sk-embed"})
        provider = OpenAIEmbeddingProvider(api_key=None, api_key_store=store)
        with patch.dict(sys.modules, {"openai": _fake_openai_module()}):
            client = await provider._ensure_client()
        assert client.api_key == "sk-embed"
