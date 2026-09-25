"""Startup warning for the built-in context-pack cursor secret."""

import logging

import pytest
from scitrera_app_framework import Variables

from memorylayer_server.api.v1.sessions import SessionsAPIPlugin, warn_if_default_context_cursor_secret
from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_CONTEXT_CURSOR_SECRET,
    MEMORYLAYER_CONTEXT_CURSOR_SECRET,
    MEMORYLAYER_CONTEXT_PACK_ENABLED,
)

LOGGER_NAME = "test.context_cursor_secret"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MEMORYLAYER_CONTEXT_CURSOR_SECRET, raising=False)
    monkeypatch.delenv(MEMORYLAYER_CONTEXT_PACK_ENABLED, raising=False)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING and MEMORYLAYER_CONTEXT_CURSOR_SECRET in r.getMessage()]


def test_warns_when_default_secret_in_use(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_if_default_context_cursor_secret(Variables(), logging.getLogger(LOGGER_NAME)) is True
    assert len(_warnings(caplog)) == 1


def test_warns_when_secret_explicitly_set_to_default(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(MEMORYLAYER_CONTEXT_CURSOR_SECRET, DEFAULT_MEMORYLAYER_CONTEXT_CURSOR_SECRET)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_if_default_context_cursor_secret(Variables(), logging.getLogger(LOGGER_NAME)) is True
    assert len(_warnings(caplog)) == 1


def test_no_warning_with_private_secret(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(MEMORYLAYER_CONTEXT_CURSOR_SECRET, "a-private-value")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_if_default_context_cursor_secret(Variables(), logging.getLogger(LOGGER_NAME)) is False
    assert _warnings(caplog) == []


def test_no_warning_when_context_packs_disabled(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv(MEMORYLAYER_CONTEXT_PACK_ENABLED, "false")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert warn_if_default_context_cursor_secret(Variables(), logging.getLogger(LOGGER_NAME)) is False
    assert _warnings(caplog) == []


def test_sessions_plugin_initialize_warns_and_returns_router(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        router = SessionsAPIPlugin().initialize(Variables(), logging.getLogger(LOGGER_NAME))
    assert router is not None
    assert len(_warnings(caplog)) == 1
