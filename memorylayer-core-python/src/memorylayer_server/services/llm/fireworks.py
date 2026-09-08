"""Fireworks AI LLM provider (OpenAI-compatible).

Fireworks serves an OpenAI-compatible chat-completions API, so this provider is
a thin subclass of :class:`OpenAILLMProvider` that only changes the defaults:

- ``base_url`` defaults to the Fireworks inference root. The OpenAI SDK appends
  ``/chat/completions``, so the base is ``.../inference/v1`` (NOT the full
  ``.../inference/v1/chat/completions`` endpoint).
- The API key resolves from the store under ``FIREWORKS_API_KEY`` by default.
"""

from scitrera_app_framework.api import Variables

from .openai import OpenAILLMProvider

# OpenAI-SDK base URL (it appends ``/chat/completions``); the chat-completions
# endpoint is therefore ``https://api.fireworks.ai/inference/v1/chat/completions``.
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Default model. Fireworks retires account-pathed model ids over time, so this
# needs periodic updating — configure ``MEMORYLAYER_LLM_PROFILE_<NAME>_MODEL``
# per profile to pin a specific model.
DEFAULT_LLM_FIREWORKS_MODEL = "accounts/fireworks/models/qwen3p7-plus"


class FireworksLLMProvider(OpenAILLMProvider):
    """Fireworks AI provider — OpenAI-compatible with Fireworks defaults."""

    DEFAULT_API_KEY_NAME = "FIREWORKS_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_LLM_FIREWORKS_MODEL,
        default_max_tokens: int | None = None,
        default_temperature: float | None = None,
        api_key_name: str | None = None,
        api_key_store=None,
        default_extra_body: dict | None = None,
        default_reasoning_effort: str | None = None,
        default_headers: dict[str, str] | None = None,
        stamp_identity: bool = False,
        v: Variables = None,
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url or DEFAULT_FIREWORKS_BASE_URL,
            model=model,
            default_max_tokens=default_max_tokens,
            default_temperature=default_temperature,
            api_key_name=api_key_name,
            api_key_store=api_key_store,
            default_extra_body=default_extra_body,
            default_reasoning_effort=default_reasoning_effort,
            default_headers=default_headers,
            stamp_identity=stamp_identity,
            v=v,
        )
