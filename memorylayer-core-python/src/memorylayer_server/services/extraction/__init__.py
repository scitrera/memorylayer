"""Extraction service package."""
from .base import (
    ExtractionService,
    ExtractionServicePluginBase,
    EXT_EXTRACTION_SERVICE,
    ExtractionCategory,
    ExtractionOptions,
    ExtractedMemory,
    ExtractionResult,
    CATEGORY_MAPPING,
)

from scitrera_app_framework import Variables, get_extension


def get_extraction_service(v: Variables = None) -> ExtractionService:
    """Get the extraction service instance."""
    return get_extension(EXT_EXTRACTION_SERVICE, v)


__all__ = (
    'ExtractionService',
    'ExtractionServicePluginBase',
    'get_extraction_service',
    'EXT_EXTRACTION_SERVICE',
    'ExtractionCategory',
    'ExtractionOptions',
    'ExtractedMemory',
    'ExtractionResult',
    'CATEGORY_MAPPING',
)
