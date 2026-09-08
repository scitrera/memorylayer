"""Graph query service package (P2 Track A).

The recall/RAG-facing READ seam over the workspace association graph. Distinct
from ``graph_analysis`` (whole-graph analytics): this seam answers localized
queries — neighborhoods, k-hop subgraphs, shortest paths, typed-edge patterns,
relationship rollups — with backend-agnostic DTOs (``models.graph_query``).

OSS ships the relational ``default`` provider; the enterprise package adds an
Apache-AGE ``age`` provider behind the same plugin pattern. Selection via
``MEMORYLAYER_GRAPH_QUERY_PROVIDER`` (default ``"default"``).
"""

from scitrera_app_framework import Variables, get_extension

from .._constants import EXT_GRAPH_QUERY_SERVICE
from .base import GraphQueryService, GraphQueryServicePluginBase


def get_graph_query_service(v: Variables = None) -> GraphQueryService:
    """Get the graph query service instance."""
    return get_extension(EXT_GRAPH_QUERY_SERVICE, v)


__all__ = (
    "GraphQueryService",
    "GraphQueryServicePluginBase",
    "get_graph_query_service",
    "EXT_GRAPH_QUERY_SERVICE",
)
