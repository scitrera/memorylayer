"""Unit tests for _init_single_vector_provider dispatch.

Covers ``EMBED_SERVER_SINGLE_VECTOR_PROVIDER``-driven selection between
vllm / openai / google / mock / colpali. Each branch is exercised by
substituting the underlying provider plugin / module so the test stays
free of torch, openai-client, and google-genai imports.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from memorylayer_embed_server.dependencies import _init_single_vector_provider
from memorylayer_embed_server.services.embedding.mock_providers import (
    MockSingleVectorProvider,
)


def _make_logger() -> MagicMock:
    return MagicMock(name="logger")


# ---------------------------------------------------------------------------
# Mock branch — purely numpy-backed, no imports to fake.
# ---------------------------------------------------------------------------


def test_single_provider_mock(monkeypatch):
    provider = _init_single_vector_provider(v=None, logger=_make_logger(), kind="mock")
    assert isinstance(provider, MockSingleVectorProvider)


# ---------------------------------------------------------------------------
# vLLM branch — install a fake module so the plugin import path resolves.
# ---------------------------------------------------------------------------


def _install_fake_vllm_plugin(monkeypatch, returns):
    fake_mod = types.ModuleType(
        "memorylayer_embed_server.services.embedding.vllm",
    )

    class _FakeVLLMPlugin:
        def initialize(self, v, logger):
            return returns

    fake_mod.VLLMEmbeddingProviderPlugin = _FakeVLLMPlugin
    monkeypatch.setitem(
        sys.modules,
        "memorylayer_embed_server.services.embedding.vllm",
        fake_mod,
    )


def test_single_provider_vllm(monkeypatch):
    sentinel = object()
    _install_fake_vllm_plugin(monkeypatch, sentinel)
    provider = _init_single_vector_provider(v=None, logger=_make_logger(), kind="vllm")
    assert provider is sentinel


def test_single_provider_default_kind_routes_to_vllm(monkeypatch):
    sentinel = object()
    _install_fake_vllm_plugin(monkeypatch, sentinel)
    provider = _init_single_vector_provider(v=None, logger=_make_logger(), kind="")
    assert provider is sentinel


# ---------------------------------------------------------------------------
# OpenAI branch — fake the upstream OSS plugin module.
# ---------------------------------------------------------------------------


def test_single_provider_openai(monkeypatch):
    sentinel = object()
    fake_mod = types.ModuleType("memorylayer_server.services.embedding.openai")

    class _FakeOpenAIPlugin:
        def initialize(self, v, logger):
            return sentinel

    fake_mod.OpenAIEmbeddingProviderPlugin = _FakeOpenAIPlugin
    monkeypatch.setitem(
        sys.modules, "memorylayer_server.services.embedding.openai", fake_mod
    )

    provider = _init_single_vector_provider(v=None, logger=_make_logger(), kind="openai")
    assert provider is sentinel


# ---------------------------------------------------------------------------
# Google branch — same shape as OpenAI.
# ---------------------------------------------------------------------------


def test_single_provider_google(monkeypatch):
    sentinel = object()
    fake_mod = types.ModuleType("memorylayer_server.services.embedding.google")

    class _FakeGooglePlugin:
        def initialize(self, v, logger):
            return sentinel

    fake_mod.GoogleEmbeddingProviderPlugin = _FakeGooglePlugin
    monkeypatch.setitem(
        sys.modules, "memorylayer_server.services.embedding.google", fake_mod
    )

    provider = _init_single_vector_provider(v=None, logger=_make_logger(), kind="google")
    assert provider is sentinel


# ---------------------------------------------------------------------------
# Unknown kind → None, with a warning.
# ---------------------------------------------------------------------------


def test_single_provider_unknown_returns_none_and_warns():
    logger = _make_logger()
    provider = _init_single_vector_provider(v=None, logger=logger, kind="not-a-provider")
    assert provider is None
    # The implementation logs at warning level with the offending kind.
    assert any(
        "EMBED_SERVER_SINGLE_VECTOR_PROVIDER" in str(call)
        and "not-a-provider" in str(call)
        for call in logger.warning.call_args_list
    )


# ---------------------------------------------------------------------------
# Import failure → None (does not raise out).
# ---------------------------------------------------------------------------


def test_single_provider_openai_import_failure_returns_none(monkeypatch):
    # Force the import to fail by purging any existing fake and replacing
    # with a module that raises on attribute access.
    monkeypatch.delitem(
        sys.modules,
        "memorylayer_server.services.embedding.openai",
        raising=False,
    )
    # Insert a sentinel module without the expected attribute so attribute
    # lookup fails (mimicking a broken/legacy install).
    fake_mod = types.ModuleType("memorylayer_server.services.embedding.openai")
    monkeypatch.setitem(
        sys.modules, "memorylayer_server.services.embedding.openai", fake_mod
    )
    logger = _make_logger()
    provider = _init_single_vector_provider(v=None, logger=logger, kind="openai")
    assert provider is None
