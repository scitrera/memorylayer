"""Unit tests for the shared ``VLLMSubprocessRunner`` helper.

These cover the parts of the runner that don't actually spawn a child
process: argv construction and free-port allocation. Subprocess
lifecycle (start / wait_for_health / shutdown) is exercised indirectly
through the embedding-side ``test_vllm_subprocess.py`` and the LLM-side
tests added in Phase C.
"""
from __future__ import annotations

import socket

import pytest

from memorylayer_embed_server.services._vllm_runner import (
    VLLMSubprocessRunner,
    find_free_port,
)


def _runner(role: str = "embedding", **overrides) -> VLLMSubprocessRunner:
    defaults = dict(
        role=role,
        model_name="test/mock",
        host="127.0.0.1",
        port=18099,
        dtype="auto",
        max_model_len=8192,
        gpu_memory_utilization=0.25,
        enforce_eager=False,
        startup_timeout_sec=5.0,
    )
    defaults.update(overrides)
    return VLLMSubprocessRunner(**defaults)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_role_rejected():
    with pytest.raises(ValueError):
        _runner(role="not-a-role")


def test_base_url_and_health_url_are_derived_from_host_port():
    r = _runner(host="10.0.0.5", port=23456)
    assert r.base_url == "http://10.0.0.5:23456/v1"
    assert r.health_url == "http://10.0.0.5:23456/health"


def test_initial_state_is_not_running():
    r = _runner()
    assert r.is_running is False
    assert r.pid is None


# ---------------------------------------------------------------------------
# Argv construction — embedding role
# ---------------------------------------------------------------------------


def test_embedding_argv_contains_pooling_and_embed_flags():
    r = _runner(role="embedding", model_name="Qwen/Qwen3-Embedding-0.6B")
    argv = r.build_argv()
    assert argv[0:3] == ["vllm", "serve", "Qwen/Qwen3-Embedding-0.6B"]
    assert "--runner" in argv and argv[argv.index("--runner") + 1] == "pooling"
    assert "--convert" in argv and argv[argv.index("--convert") + 1] == "embed"
    assert "--trust-remote-code" in argv
    assert "--max-model-len" in argv


def test_embedding_argv_includes_enforce_eager_when_set():
    r = _runner(role="embedding", enforce_eager=True)
    assert "--enforce-eager" in r.build_argv()


def test_embedding_argv_excludes_enforce_eager_by_default():
    r = _runner(role="embedding", enforce_eager=False)
    assert "--enforce-eager" not in r.build_argv()


# ---------------------------------------------------------------------------
# Argv construction — llm role
# ---------------------------------------------------------------------------


def test_llm_argv_omits_pooling_and_embed_flags():
    r = _runner(role="llm", model_name="Qwen/Qwen2.5-7B-Instruct")
    argv = r.build_argv()
    assert "--runner" not in argv
    assert "--convert" not in argv
    # vllm serve <model> still at the front.
    assert argv[0:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct"]


def test_llm_argv_includes_served_model_names():
    r = _runner(
        role="llm",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        served_model_names=["qwen", "qwen-7b", "qwen-2.5"],
    )
    argv = r.build_argv()
    assert "--served-model-name" in argv
    idx = argv.index("--served-model-name")
    # All three names follow the flag, in order.
    assert argv[idx + 1: idx + 4] == ["qwen", "qwen-7b", "qwen-2.5"]


def test_llm_argv_omits_served_model_name_flag_when_empty():
    r = _runner(role="llm", served_model_names=None)
    assert "--served-model-name" not in r.build_argv()


def test_argv_includes_tensor_parallel_when_greater_than_one():
    r = _runner(role="llm", tensor_parallel_size=2)
    argv = r.build_argv()
    assert "--tensor-parallel-size" in argv
    assert argv[argv.index("--tensor-parallel-size") + 1] == "2"


def test_argv_omits_tensor_parallel_when_one():
    r = _runner(role="llm", tensor_parallel_size=1)
    assert "--tensor-parallel-size" not in r.build_argv()


def test_argv_omits_max_model_len_when_none():
    r = _runner(role="llm", max_model_len=None)
    assert "--max-model-len" not in r.build_argv()


def test_argv_extra_args_appended_in_order():
    r = _runner(role="llm", extra_args=["--quantization", "fp8", "--seed", "42"])
    argv = r.build_argv()
    # The extras land contiguously somewhere in argv, in declared order.
    extras = ["--quantization", "fp8", "--seed", "42"]
    for i in range(len(argv) - len(extras) + 1):
        if argv[i: i + len(extras)] == extras:
            return
    pytest.fail(f"extra_args not appended contiguously: {argv}")


def test_cmd_override_replaces_vllm_executable():
    r = _runner(role="llm", cmd="/opt/vllm/bin/vllm")
    argv = r.build_argv()
    assert argv[0] == "/opt/vllm/bin/vllm"


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------


def test_find_free_port_returns_unbound_port_when_range_unspecified():
    port = find_free_port()
    assert 1024 < port < 65536
    # Sanity-check the port is actually bindable right now.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_find_free_port_within_range():
    # Pick a small range we expect to be empty in CI environments.
    port = find_free_port(low=29110, high=29130)
    assert 29110 <= port <= 29130


def test_find_free_port_skips_busy_ports():
    # Bind one port in a 3-port range and verify the helper picks a different one.
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied_port = occupied.getsockname()[1]
    try:
        chosen = find_free_port(low=occupied_port, high=occupied_port + 5)
        assert chosen != occupied_port
        assert occupied_port < chosen <= occupied_port + 5
    finally:
        occupied.close()


def test_find_free_port_raises_when_range_exhausted():
    # Occupy a 1-port range so the helper has nowhere to go.
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    occupied_port = occupied.getsockname()[1]
    try:
        with pytest.raises(RuntimeError):
            find_free_port(low=occupied_port, high=occupied_port)
    finally:
        occupied.close()


def test_find_free_port_invalid_range_rejected():
    with pytest.raises(ValueError):
        find_free_port(low=20000, high=19999)
