"""Encryption protocol for MCP server secrets (env/headers).

Defines a thin passthrough-by-default interface. Enterprise can plug in
the real encrypt_json/decrypt_json from memorylayer_saas.services.encryption
by calling register_encrypter() at startup.

This keeps the OSS package free of proprietary dependencies while enabling
at-rest encryption in Enterprise deployments.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_encrypt_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None
_decrypt_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None


def register_encrypter(
    encrypt_fn: Callable[[dict[str, Any]], dict[str, Any]],
    decrypt_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    """Register encrypt/decrypt functions. Called by Enterprise at startup."""
    global _encrypt_fn, _decrypt_fn
    _encrypt_fn = encrypt_fn
    _decrypt_fn = decrypt_fn


def encrypt_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Encrypt a dict of secrets. Passthrough when no encrypter is registered."""
    if _encrypt_fn is None or not data:
        return data
    return _encrypt_fn(data)


def decrypt_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Decrypt a dict of secrets. Passthrough when no encrypter is registered."""
    if _decrypt_fn is None or not data:
        return data
    return _decrypt_fn(data)
