"""Explicit service capabilities; the default preserves combined deployments."""
import os


def service_profile():
    profile = os.environ.get("MEMORYLAYER_EMBED_PROFILE", "combined")
    if profile not in {"combined", "embedding", "transcription"}:
        raise ValueError("MEMORYLAYER_EMBED_PROFILE must be combined, embedding or transcription")
    return profile


def embeddings_enabled():
    return service_profile() != "transcription"


def transcription_enabled():
    return service_profile() != "embedding"
