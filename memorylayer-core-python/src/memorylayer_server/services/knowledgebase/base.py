"""Knowledgebase service - abstract base interface."""

from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field

from ...models.graph_analysis import GraphStats


class KBGenerateOptions(BaseModel):
    """Options to control knowledgebase generation."""

    include_rpg: bool = Field(False, description="Include RPG nodes in graph analysis")
    max_communities: int = Field(50, description="Maximum communities to generate articles for")
    max_god_nodes: int = Field(20, description="Maximum god/central nodes to generate entity articles for")
    regenerate: bool = Field(False, description="Force regeneration even if KB exists")


class Article(BaseModel):
    """A single knowledgebase article."""

    id: str = Field(..., description="Unique article identifier")
    article_type: str = Field(..., description="Article type: 'index', 'community', or 'entity'")
    title: str = Field(..., description="Article title")
    content_md: str = Field(..., description="Markdown content (Obsidian-compatible)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    generated_at: datetime = Field(..., description="When this article was generated")


class Knowledgebase(BaseModel):
    """Metadata about a generated knowledgebase for a workspace."""

    workspace_id: str
    article_count: int = 0
    community_count: int = 0
    generated_at: datetime
    stats: GraphStats | None = None


class KnowledgebaseService(ABC):
    """Abstract interface for knowledgebase generation and retrieval."""

    @abstractmethod
    async def generate(
        self,
        workspace_id: str,
        context_id: str | None = None,
        options: KBGenerateOptions | None = None,
    ) -> Knowledgebase:
        """Generate (or regenerate) the knowledgebase for a workspace.

        Runs the full pipeline: graph analysis -> community labeling ->
        entity deep-dives -> index article -> storage.

        Args:
            workspace_id: Target workspace
            context_id: Optional context partition to restrict analysis
            options: Generation options (communities, god nodes, regenerate flag)

        Returns:
            Knowledgebase metadata for the generated KB
        """
        pass

    @abstractmethod
    async def get_knowledgebase(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> Knowledgebase | None:
        """Get knowledgebase metadata for a workspace without regenerating.

        Args:
            workspace_id: Target workspace
            context_id: Optional context filter

        Returns:
            Knowledgebase metadata or None if no KB has been generated
        """
        pass

    @abstractmethod
    async def get_article(
        self,
        workspace_id: str,
        article_id: str,
    ) -> Article | None:
        """Retrieve a single article by ID.

        Args:
            workspace_id: Workspace boundary
            article_id: Article identifier (e.g. 'index', 'community-0', 'entity-alice')

        Returns:
            Article or None if not found
        """
        pass

    @abstractmethod
    async def list_articles(
        self,
        workspace_id: str,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Article]:
        """List articles for a workspace, optionally filtered by type.

        Args:
            workspace_id: Workspace boundary
            article_type: Optional filter ('index', 'community', 'entity')
            limit: Maximum articles to return
            offset: Pagination offset

        Returns:
            List of articles
        """
        pass

    @abstractmethod
    async def export_vault(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> bytes:
        """Export the knowledgebase as an Obsidian-compatible zip vault.

        The zip contains:
          index.md
          communities/community-{id}.md
          entities/{slug}.md

        Args:
            workspace_id: Workspace boundary
            context_id: Optional context filter

        Returns:
            Zip file bytes
        """
        pass
