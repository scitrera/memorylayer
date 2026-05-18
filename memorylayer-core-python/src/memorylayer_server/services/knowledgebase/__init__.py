"""Knowledgebase service package."""

from scitrera_app_framework import Variables, get_extension

from ...config import DEFAULT_MEMORYLAYER_KNOWLEDGEBASE_PROVIDER, MEMORYLAYER_KNOWLEDGEBASE_PROVIDER
from .._constants import (
    EXT_GRAPH_ANALYSIS_SERVICE,
    EXT_KNOWLEDGEBASE_SERVICE,
    EXT_STORAGE_BACKEND,
)
from .._plugin_factory import make_service_plugin_base

KnowledgebaseServicePluginBase = make_service_plugin_base(
    ext_name=EXT_KNOWLEDGEBASE_SERVICE,
    config_key=MEMORYLAYER_KNOWLEDGEBASE_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_KNOWLEDGEBASE_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND, EXT_GRAPH_ANALYSIS_SERVICE),
)


def get_knowledgebase_service(v: Variables = None):
    """Get the knowledgebase service instance."""
    return get_extension(EXT_KNOWLEDGEBASE_SERVICE, v)


__all__ = (
    "KnowledgebaseServicePluginBase",
    "get_knowledgebase_service",
    "EXT_KNOWLEDGEBASE_SERVICE",
)
