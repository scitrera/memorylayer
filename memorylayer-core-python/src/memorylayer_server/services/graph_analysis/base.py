"""Graph Analysis Service — Base interface."""

from abc import ABC, abstractmethod

from ...models.graph_analysis import (
    Bridge,
    CentralNode,
    Community,
    GraphAnalysis,
    GraphSnapshot,
    GraphStats,
)


class GraphAnalysisService(ABC):
    """Interface for workspace association graph analysis."""

    @abstractmethod
    async def build_workspace_graph(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
    ) -> GraphSnapshot:
        """Build a graph snapshot for the workspace and return metadata."""
        pass

    @abstractmethod
    async def detect_communities(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> list[Community]:
        """Detect communities (clusters) within the workspace graph."""
        pass

    @abstractmethod
    async def compute_centrality(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> list[CentralNode]:
        """Compute degree and betweenness centrality for all nodes."""
        pass

    @abstractmethod
    async def get_bridges(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> list[Bridge]:
        """Find edges that bridge different communities."""
        pass

    @abstractmethod
    async def get_statistics(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> GraphStats:
        """Compute aggregate graph statistics for the workspace."""
        pass

    @abstractmethod
    async def analyze(
        self,
        workspace_id: str,
        context_id: str | None = None,
        include_rpg: bool = False,
    ) -> GraphAnalysis:
        """Run full graph analysis (snapshot + communities + centrality + bridges + stats)."""
        pass
