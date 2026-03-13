"""Inference service package - entity insight derivation."""
from .base import (
    InferenceServicePluginBase,
    InferenceService,
    InferenceResult,
    EXT_INFERENCE_SERVICE,
)
from .default import DefaultInferenceService

from scitrera_app_framework import Variables, get_extension


def get_inference_service(v: Variables = None) -> DefaultInferenceService:
    """Get the inference service instance."""
    return get_extension(EXT_INFERENCE_SERVICE, v)


__all__ = (
    'InferenceService',
    'DefaultInferenceService',
    'InferenceServicePluginBase',
    'InferenceResult',
    'get_inference_service',
    'EXT_INFERENCE_SERVICE',
)
