"""Caller-asserted identity headers must reach first-party endpoints only.

X-Scitrera-Source / X-Scitrera-Tenant / X-Scitrera-Task-Id name the operator's
internal tenant and its running task ids. Every provider built by the registry
speaks the same OpenAI-compatible protocol as a public vendor, so the provider
type says nothing about who is on the other end — only the base_url host does.

These tests pin the gate and, more importantly, that it fails CLOSED: no
allowlist stamps nothing, and a profile with no base_url (pointed at the vendor's
own endpoint by definition) is never stamped.
"""

import pytest

from memorylayer_server.models.llm import LLMMessage, LLMRequest, LLMRole
from memorylayer_server.services.llm.attribution import (
    HEADER_SOURCE,
    HEADER_TASK_ID,
    HEADER_TENANT,
    host_allows_identity,
    parse_host_patterns,
    task_attribution,
)
from memorylayer_server.services.llm.openai import OpenAILLMProvider
from memorylayer_server.services.llm.registry import create_provider_from_config

_HEADERS = {HEADER_SOURCE: "memorylayer", HEADER_TENANT: "acme"}


# --------------------------------------------------------------------------- #
# Pattern parsing
# --------------------------------------------------------------------------- #


def test_parse_splits_on_commas_and_whitespace_and_dedupes():
    assert parse_host_patterns("a.internal, b.internal  c.internal, a.internal") == (
        "a.internal",
        "b.internal",
        "c.internal",
    )


def test_parse_tolerates_a_pasted_base_url():
    """An operator should be able to paste a profile's base_url verbatim."""
    assert parse_host_patterns("https://gw.internal:8443/v1") == ("gw.internal",)


def test_parse_empty_is_empty():
    assert parse_host_patterns("") == ()
    assert parse_host_patterns(None) == ()


# --------------------------------------------------------------------------- #
# The gate — failing closed
# --------------------------------------------------------------------------- #


def test_no_allowlist_stamps_nothing():
    """The OSS default: an empty allowlist must not stamp even a plausible host."""
    assert host_allows_identity("https://gw.internal/v1", ()) is False


def test_no_base_url_is_never_stamped():
    """No base_url means the vendor's own endpoint (api.openai.com, Fireworks)."""
    patterns = parse_host_patterns("*")
    assert host_allows_identity(None, patterns) is False
    assert host_allows_identity("", patterns) is False
    assert host_allows_identity("   ", patterns) is False


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://api.fireworks.ai/inference/v1",
        "https://api.anthropic.com",
    ],
)
def test_public_providers_are_not_stamped_under_an_internal_allowlist(base_url):
    """The leak this gate exists to prevent."""
    patterns = parse_host_patterns("*.mt, gw.internal")
    assert host_allows_identity(base_url, patterns) is False


# --------------------------------------------------------------------------- #
# The gate — matching
# --------------------------------------------------------------------------- #


def test_exact_host_matches_and_ignores_scheme_port_and_path():
    patterns = parse_host_patterns("gw.internal")
    assert host_allows_identity("https://gw.internal:8443/v1/chat", patterns) is True


def test_wildcard_matches_any_depth_under_the_suffix():
    patterns = parse_host_patterns("*.mt")
    assert host_allows_identity("https://llm.mt/v1", patterns) is True
    assert host_allows_identity("https://gw.llm.mt/v1", patterns) is True


def test_wildcard_does_not_match_the_bare_suffix_or_a_lookalike():
    """`*.mt` must not match `mt` itself, nor a host merely ending in those letters."""
    patterns = parse_host_patterns("*.mt")
    assert host_allows_identity("https://mt/v1", patterns) is False
    assert host_allows_identity("https://evil-mt/v1", patterns) is False
    assert host_allows_identity("https://notmt/v1", patterns) is False


def test_matching_is_case_insensitive():
    patterns = parse_host_patterns("*.MT")
    assert host_allows_identity("https://GW.Mt/v1", patterns) is True


def test_bare_star_allows_everything():
    """Documented escape hatch — pinned so its blast radius stays visible."""
    assert host_allows_identity("https://api.openai.com/v1", parse_host_patterns("*")) is True


def test_malformed_url_is_not_stamped():
    assert host_allows_identity("http://[oops", parse_host_patterns("*")) is False


# --------------------------------------------------------------------------- #
# Provider — both header paths ride on the one decision
# --------------------------------------------------------------------------- #


def _kwargs(provider: OpenAILLMProvider, request: LLMRequest | None = None) -> dict:
    request = request or LLMRequest(messages=[LLMMessage(role=LLMRole.USER, content="hi")])
    return provider._build_kwargs(request, stream=False)


def test_provider_without_permission_drops_static_headers():
    provider = OpenAILLMProvider(
        api_key="k", base_url="https://api.openai.com/v1",
        default_headers=_HEADERS, stamp_identity=False,
    )
    assert provider.default_headers is None


def test_provider_without_permission_drops_the_task_header():
    """Gating only the static set would still stream task ids to a public vendor."""
    provider = OpenAILLMProvider(
        api_key="k", base_url="https://api.openai.com/v1",
        default_headers=_HEADERS, stamp_identity=False,
    )
    with task_attribution("task-123"):
        assert "extra_headers" not in _kwargs(provider)


def test_provider_with_permission_sends_both():
    provider = OpenAILLMProvider(
        api_key="k", base_url="https://gw.internal/v1",
        default_headers=_HEADERS, stamp_identity=True,
    )
    assert provider.default_headers == _HEADERS
    with task_attribution("task-123"):
        assert _kwargs(provider)["extra_headers"][HEADER_TASK_ID] == "task-123"


def test_stamping_defaults_to_off_when_constructed_directly():
    """A provider built outside the registry must not leak identity by omission."""
    provider = OpenAILLMProvider(api_key="k", default_headers=_HEADERS)
    assert provider.stamp_identity is False
    assert provider.default_headers is None


def test_explicit_per_call_headers_are_never_gated():
    """request.extra_headers is the caller's own choice, not ambient identity."""
    provider = OpenAILLMProvider(
        api_key="k", base_url="https://api.openai.com/v1", stamp_identity=False,
    )
    request = LLMRequest(
        messages=[LLMMessage(role=LLMRole.USER, content="hi")],
        extra_headers={"X-Caller-Set": "yes"},
    )
    assert _kwargs(provider, request)["extra_headers"] == {"X-Caller-Set": "yes"}


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("provider_type", ["openai", "fireworks"])
def test_registry_honors_the_flag_for_openai_compatible_providers(provider_type):
    denied = create_provider_from_config(
        name="p", provider_type=provider_type, base_url="https://api.openai.com/v1",
        api_key="k", default_headers=_HEADERS, stamp_identity=False,
    )
    allowed = create_provider_from_config(
        name="p", provider_type=provider_type, base_url="https://gw.internal/v1",
        api_key="k", default_headers=_HEADERS, stamp_identity=True,
    )
    assert denied.default_headers is None
    assert allowed.default_headers == _HEADERS


def test_registry_defaults_to_not_stamping():
    """create_provider_from_config callers must opt in explicitly."""
    provider = create_provider_from_config(
        name="p", provider_type="openai", base_url="https://gw.internal/v1",
        api_key="k", default_headers=_HEADERS,
    )
    assert provider.default_headers is None
