"""Regression tests for the multi-extension route fix (commit 85845eb).

Two seams are pinned:

1. **Plugin enablement gate** — ``LLMChatRoutePlugin.is_multi_extension`` must
   return ``False`` when ``MEMORYLAYER_EMBED_LLM_ENABLED`` is falsy and ``True``
   when it is truthy.  If the gate is removed (e.g. the method is changed to
   ``return True`` unconditionally, or the env-var check is dropped) the
   *disabled* assertion below will fail, re-exposing the 404 regression where
   every sibling router (embeddings, score, …) was shadowed.

2. **Consumer None-skip** — the router-registration loop inside
   ``FastApiPlugin.initialize`` must silently skip a ``None`` value produced by
   a disabled multi-extension plugin and still register the remaining real
   routers.  If the ``if router is None: continue`` guard is removed, the test
   catches the regression via one of two observable failure modes:
   - ``app.include_router(None)`` raises ``AttributeError`` / ``TypeError``,
     crashing the registration and leaving the real router un-registered
     (``GET /v1/embeddings`` would 404 or the app factory would raise).
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from scitrera_app_framework.api import Variables

from memorylayer_embed_server.api import EXT_MULTI_API_ROUTERS
from memorylayer_embed_server.api.v1.chat import LLMChatRoutePlugin
from memorylayer_embed_server.config import EMBED_SERVER_LLM_ENABLED
from memorylayer_embed_server.lifecycle.fastapi import FastApiPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_variables(**env_overrides: str) -> Variables:
    """Return a fresh ``Variables`` instance with only the given env overrides
    injected into local storage so real ``os.environ`` is not consulted for
    these keys.  All other keys fall through to the real environment as usual.
    """
    v = Variables()
    for key, value in env_overrides.items():
        v.set(key, value)
    return v


# ---------------------------------------------------------------------------
# Seam 1 — Plugin enablement gate
# ---------------------------------------------------------------------------


class TestLLMChatRoutePluginEnablementGate:
    """Pin LLMChatRoutePlugin.is_multi_extension against the env-flag gate.

    Regression: if ``is_multi_extension`` is hardcoded to ``True`` (or the
    env-var check is removed), the ``disabled`` assertion will start passing
    when it should fail, and the chat plugin will silently claim the
    ``EXT_MULTI_API_ROUTERS`` extension point, shadowing all sibling routers.
    """

    def test_returns_false_when_llm_enabled_flag_is_off(self):
        """``is_multi_extension`` must be False when the flag is 'false'."""
        plugin = LLMChatRoutePlugin()
        v = _make_variables(**{EMBED_SERVER_LLM_ENABLED: "false"})
        assert plugin.is_multi_extension(v) is False

    def test_returns_false_when_llm_enabled_flag_is_zero(self):
        """Common falsy env-var value '0' must also disable the plugin."""
        plugin = LLMChatRoutePlugin()
        v = _make_variables(**{EMBED_SERVER_LLM_ENABLED: "0"})
        assert plugin.is_multi_extension(v) is False

    def test_returns_false_with_default_when_env_var_absent(self):
        """Without any override the default (False) must disable the plugin."""
        plugin = LLMChatRoutePlugin()
        # Fresh Variables with no override; the default is registered via
        # DEFAULT_EMBED_SERVER_LLM_ENABLED=False inside is_multi_extension.
        # We must not have MEMORYLAYER_EMBED_LLM_ENABLED in the real env.
        v = Variables()
        # Guard: if the real env happens to set this to true, skip the test
        # rather than giving a misleading failure.
        raw = v[EMBED_SERVER_LLM_ENABLED]
        if raw not in (None, "", "false", "0", "no", "f", "n"):
            pytest.skip("MEMORYLAYER_EMBED_LLM_ENABLED is set truthy in the real env")
        result = plugin.is_multi_extension(v)
        assert result is False, (
            "Default should be False — DEFAULT_EMBED_SERVER_LLM_ENABLED is False"
        )

    def test_returns_true_when_llm_enabled_flag_is_on(self):
        """``is_multi_extension`` must be True when the flag is 'true'."""
        plugin = LLMChatRoutePlugin()
        v = _make_variables(**{EMBED_SERVER_LLM_ENABLED: "true"})
        assert plugin.is_multi_extension(v) is True

    def test_returns_true_when_llm_enabled_flag_is_one(self):
        """Common truthy env-var value '1' must also enable the plugin."""
        plugin = LLMChatRoutePlugin()
        v = _make_variables(**{EMBED_SERVER_LLM_ENABLED: "1"})
        assert plugin.is_multi_extension(v) is True

    def test_is_enabled_always_returns_false(self):
        """``is_enabled`` must always be False for this multi-extension plugin.

        Returning True from ``is_enabled`` would make this the *single*
        extension winner, shadowing all sibling routers — the exact 404
        regression described in the commit comment.
        """
        plugin = LLMChatRoutePlugin()
        v = _make_variables(**{EMBED_SERVER_LLM_ENABLED: "true"})
        assert plugin.is_enabled(v) is False

    def test_extension_point_name_is_multi_api_routers(self):
        """Plugin must target the correct extension point."""
        plugin = LLMChatRoutePlugin()
        v = Variables()
        assert plugin.extension_point_name(v) == EXT_MULTI_API_ROUTERS


# ---------------------------------------------------------------------------
# Seam 2 — Consumer None-skip in the registration loop
# ---------------------------------------------------------------------------


class TestConsumerNoneSkip:
    """Pin the ``if router is None: continue`` guard in FastApiPlugin.

    The production code path (lifecycle/fastapi.py ~L163-179) calls
    ``get_extensions(EXT_MULTI_API_ROUTERS, v)`` and iterates the returned
    dict, skipping any ``None`` value.  We patch ``get_extensions`` to return
    a dict that includes a disabled (None) slot alongside a real router, then
    assert:

    - The factory does not raise.
    - The real router's routes are reachable (not 404).
    - The disabled slot did not crash registration of the sibling.

    Regression: if the ``if router is None: continue`` guard is removed,
    ``app.include_router(None)`` raises, the except branch swallows it, and
    *no* routers are registered — ``GET /v1/embeddings`` returns 404.
    """

    def _make_minimal_v(self) -> Variables:
        """Minimal Variables that satisfies FastApiPlugin.initialize()."""
        v = Variables()
        # Disable model preloading so we don't need real services.
        v.set("MEMORYLAYER_EMBED_PRELOAD_MODELS", "false")
        v.set("MEMORYLAYER_EMBED_LLM_PRELOAD", "false")
        return v

    def _build_app_with_patched_extensions(
        self,
        extensions_dict: dict,
    ) -> FastAPI:
        """Invoke FastApiPlugin.initialize() with get_extensions mocked.

        The real initialize() builds a FastAPI app and registers routers via a
        deferred ``from scitrera_app_framework import get_extensions`` call.
        We patch the canonical location of that function so the local import
        inside initialize() picks up our stub.

        The lifespan context (service init/shutdown) is never invoked here
        because TestClient with ``raise_server_exceptions=False`` won't start
        the lifespan unless we use it as a context manager — and we don't need
        to for route-registration tests.
        """
        v = self._make_minimal_v()
        logger = logging.getLogger("test-none-skip")

        plugin = FastApiPlugin()

        # ``from scitrera_app_framework import get_extensions`` re-imports from
        # the package each time the enclosing function runs, so we must patch
        # the name at both the canonical source module AND the public package
        # namespace to guarantee the stub is seen regardless of import caching.
        with (
            patch(
                "scitrera_app_framework.core.plugins.get_extensions",
                return_value=extensions_dict,
            ),
            patch(
                "scitrera_app_framework.get_extensions",
                return_value=extensions_dict,
            ),
        ):
            app = plugin.initialize(v, logger)

        return app

    def test_none_router_is_skipped_and_real_router_is_registered(self):
        """A None slot must be silently skipped; the sibling router must be live."""
        # Build a real minimal router that we can probe.
        real_router = APIRouter()

        @real_router.get("/v1/embeddings")
        async def _stub():
            return {"object": "list", "data": []}

        extensions = {
            "disabled_llm": None,  # This is the None that must be skipped.
            "embeddings": real_router,
        }

        app = self._build_app_with_patched_extensions(extensions)
        assert app is not None, "FastApiPlugin.initialize() must return an app"

        client = TestClient(app)
        resp = client.get("/v1/embeddings")
        assert resp.status_code == 200, (
            "Real router must be registered even when a sibling slot is None. "
            f"Got {resp.status_code} — the None-skip guard is likely missing."
        )

    def test_all_none_routers_are_skipped_without_crash(self):
        """An all-None dict must not crash the factory."""
        extensions = {"disabled_a": None, "disabled_b": None}

        # Must not raise.
        app = self._build_app_with_patched_extensions(extensions)
        assert app is not None

    def test_multiple_real_routers_all_registered_alongside_none(self):
        """Two real routers + one None slot: all real routes must be reachable."""
        router_a = APIRouter()
        router_b = APIRouter()

        @router_a.get("/v1/embeddings")
        async def _embed():
            return {"object": "list", "data": []}

        @router_b.get("/v1/score")
        async def _score():
            return {"scores": []}

        extensions = {
            "disabled_llm": None,
            "embeddings": router_a,
            "score": router_b,
        }

        app = self._build_app_with_patched_extensions(extensions)
        client = TestClient(app)

        assert client.get("/v1/embeddings").status_code == 200, (
            "/v1/embeddings must be reachable"
        )
        assert client.get("/v1/score").status_code == 200, (
            "/v1/score must be reachable"
        )

    def test_empty_extensions_dict_does_not_crash(self):
        """An empty extensions dict is a valid (degenerate) case."""
        app = self._build_app_with_patched_extensions({})
        assert app is not None
