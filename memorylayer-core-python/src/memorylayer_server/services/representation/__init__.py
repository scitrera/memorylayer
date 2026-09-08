"""Representation service package (P3 perspective slice 1).

The perspective-assembly surface over the workspace memory set: it answers
"what observer O understands about subject S" by assembling a tight, scoped,
deterministic view of memories.

This is a SCOPED ASSEMBLY surface, NOT a recall channel. The default provider
never calls ``recall()`` and never injects members into recall. It resolves the
observer + subject via the entity registry, then scopes the memory set by an
INTERSECTION (the observer's self-authored memories ∩ the subject's mentions),
which is leakage-safe — it excludes turns about the subject authored by *other*
observers.

OSS ships the relational ``default`` provider (deterministic, no LLM); the
enterprise package adds LLM-derived beliefs + consolidation behind the same
plugin pattern. Selection via ``MEMORYLAYER_REPRESENTATION_PROVIDER`` (default
``"default"``).

Ships DARK: ``MEMORYLAYER_REPRESENTATION_ENABLED`` defaults to ``False`` so the
service can be registered without any consumer wiring until flipped. The service
method itself always works when called directly (the flag gates future wiring).
"""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_REPRESENTATION_SERVICE
from .base import RepresentationService, RepresentationServicePluginBase


def get_representation_service(v: Variables = None) -> RepresentationService:
    """Get the representation service instance."""
    return get_extension(EXT_REPRESENTATION_SERVICE, v)


__all__ = (
    "RepresentationService",
    "RepresentationServicePluginBase",
    "get_representation_service",
    "EXT_REPRESENTATION_SERVICE",
)
