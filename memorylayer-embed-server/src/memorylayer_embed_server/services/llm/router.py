"""Multi-profile LLM router.

Holds N :class:`LLMProvider` instances (one per declared profile) and
resolves incoming ``model`` strings to the right provider. Routing
precedence:

  1. Exact (case-insensitive) match against any registered name —
     profile name, alias, or underlying model name.
  2. Fall back to ``default_profile`` if the lookup misses and one
     is configured.
  3. Raise :class:`UnknownModelError` (the FastAPI route turns this
     into a 404).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from .base import LLMProvider


class UnknownModelError(Exception):
    """Raised when ``LLMRoutingService.resolve`` can't find a target."""

    def __init__(self, requested_model: str | None, available: list[str]) -> None:
        self.requested_model = requested_model
        self.available = available
        super().__init__(f"Unknown model: {requested_model!r}. Available models: {available}")


class LLMRoutingService:
    """Hold profile providers + alias map; preload / shutdown all of them."""

    def __init__(
        self,
        profiles: Mapping[str, LLMProvider],
        *,
        default_profile: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._profiles: dict[str, LLMProvider] = dict(profiles)
        self._default_profile = default_profile if default_profile else None
        self.logger = logger or logging.getLogger(__name__)

        if self._default_profile is not None and self._default_profile not in self._profiles:
            raise ValueError(f"default_profile={self._default_profile!r} is not a declared profile (declared: {list(self._profiles)})")

        # Build the alias map (lowercased keys).
        self._alias_to_profile: dict[str, str] = {}
        for profile_name, provider in self._profiles.items():
            self._alias_to_profile[profile_name.lower()] = profile_name
            for served in provider.served_names:
                self._alias_to_profile.setdefault(served.lower(), profile_name)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def profiles(self) -> dict[str, LLMProvider]:
        return dict(self._profiles)

    @property
    def default_profile(self) -> str | None:
        return self._default_profile

    def has_profile(self, profile: str) -> bool:
        return profile in self._profiles

    def list_models(self) -> list[dict]:
        """OpenAI-compatible ``/v1/models`` payload."""
        out: list[dict] = []
        for provider in self._profiles.values():
            out.extend(provider.list_models())
        return out

    def all_routing_names(self) -> list[str]:
        """Every name that ``resolve()`` will accept (for error messages)."""
        return sorted(self._alias_to_profile)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, model: str | None) -> LLMProvider:
        if model:
            key = model.lower()
            profile_name = self._alias_to_profile.get(key)
            if profile_name is not None:
                return self._profiles[profile_name]
        if self._default_profile is not None:
            return self._profiles[self._default_profile]
        raise UnknownModelError(model, self.all_routing_names())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def preload(self) -> None:
        """Eagerly start every profile's subprocess. Failures don't abort siblings."""
        for name, provider in self._profiles.items():
            try:
                await provider.preload()
            except Exception as e:  # noqa: BLE001 - log + continue
                self.logger.warning(
                    "LLM profile %s preload failed (will retry lazily on first request): %s",
                    name,
                    e,
                )

    async def shutdown(self) -> None:
        """Shut down all profiles concurrently. Best-effort."""
        if not self._profiles:
            return
        results = await asyncio.gather(
            *(self._safe_shutdown(name, p) for name, p in self._profiles.items()),
            return_exceptions=True,
        )
        for name, result in zip(self._profiles, results):
            if isinstance(result, Exception):
                self.logger.warning("LLM profile %s shutdown raised: %s", name, result)

    async def _safe_shutdown(self, name: str, provider: LLMProvider) -> None:
        try:
            await provider.shutdown()
        except Exception as e:  # noqa: BLE001 - logged by caller
            raise RuntimeError(f"profile {name!r} shutdown failed") from e

    # ------------------------------------------------------------------
    # Load reporting (consumed by /health/load)
    # ------------------------------------------------------------------

    def get_load_snapshot(self) -> dict[str, dict]:
        """Per-profile snapshot keyed by ``llm_<profile>``."""
        out: dict[str, dict] = {}
        for name, provider in self._profiles.items():
            try:
                out[f"llm_{name}"] = provider.get_load_snapshot()
            except Exception as e:  # noqa: BLE001 - best-effort
                self.logger.debug("get_load_snapshot for %s failed: %s", name, e)
        return out


__all__ = ["LLMRoutingService", "UnknownModelError"]
