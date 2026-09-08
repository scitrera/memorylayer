"""Typed structural entity relations."""

from .default import EntityRelationService, RelationRecallResult
from .metadata import KNOWLEDGE_WORK_METADATA_KEY, extract_metadata_relations

__all__ = (
    "EntityRelationService",
    "KNOWLEDGE_WORK_METADATA_KEY",
    "RelationRecallResult",
    "extract_metadata_relations",
)
