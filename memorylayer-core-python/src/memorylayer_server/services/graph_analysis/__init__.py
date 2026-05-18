"""Graph analysis service package."""

from scitrera_app_framework import Variables, get_extension

from ...config import DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER, MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER
from .._constants import EXT_GRAPH_ANALYSIS_SERVICE, EXT_STORAGE_BACKEND
from .._plugin_factory import make_service_plugin_base
from .base import GraphAnalysisService

GraphAnalysisServicePluginBase = make_service_plugin_base(
    ext_name=EXT_GRAPH_ANALYSIS_SERVICE,
    config_key=MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER,
    default_value=DEFAULT_MEMORYLAYER_GRAPH_ANALYSIS_PROVIDER,
    dependencies=(EXT_STORAGE_BACKEND,),
)


def get_graph_analysis_service(v: Variables = None) -> GraphAnalysisService:
    """Get the graph analysis service instance."""
    return get_extension(EXT_GRAPH_ANALYSIS_SERVICE, v)


__all__ = (
    "GraphAnalysisService",
    "GraphAnalysisServicePluginBase",
    "get_graph_analysis_service",
    "EXT_GRAPH_ANALYSIS_SERVICE",
)
