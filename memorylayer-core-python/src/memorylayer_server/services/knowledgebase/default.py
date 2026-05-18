"""Knowledgebase service - default implementation.

Generation pipeline:
1. Run GraphAnalysisService.analyze() to get communities, central nodes, bridges, stats
2. For each community (up to max_communities): use ReflectService to generate a topic
   label and summary from member memories
3. For each god node (up to max_god_nodes): use InferenceService (or ReflectService)
   to generate an entity deep-dive
4. Render all articles via ObsidianRenderer
5. Store articles via storage.store_kb_article()
6. Cache graph analysis via storage.store_graph_analysis()
"""

import io
import zipfile
from datetime import UTC, datetime
from logging import Logger

from scitrera_app_framework import get_extension, get_logger
from scitrera_app_framework.api import Variables

from ...models.graph_analysis import Community, GraphAnalysis
from ...models.memory import DetailLevel, ReflectInput
from .._constants import (
    EXT_GRAPH_ANALYSIS_SERVICE,
    EXT_INFERENCE_SERVICE,
    EXT_REFLECT_SERVICE,
    EXT_STORAGE_BACKEND,
)
from ..graph_analysis.base import GraphAnalysisService
from ..storage import StorageBackend
from . import KnowledgebaseServicePluginBase
from .base import Article, KBGenerateOptions, Knowledgebase
from .renderer import ObsidianRenderer


class DefaultKnowledgebaseService:
    """Default knowledgebase generation service."""

    def __init__(
        self,
        storage: StorageBackend,
        graph_service: GraphAnalysisService,
        reflect_service=None,
        inference_service=None,
        v: Variables = None,
    ):
        self.storage = storage
        self.graph_service = graph_service
        self.reflect_service = reflect_service
        self.inference_service = inference_service
        self.renderer = ObsidianRenderer()
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized DefaultKnowledgebaseService (reflect=%s, inference=%s)",
            reflect_service is not None,
            inference_service is not None,
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        workspace_id: str,
        context_id: str | None = None,
        options: KBGenerateOptions | None = None,
    ) -> Knowledgebase:
        """Run full KB generation pipeline for a workspace."""
        if options is None:
            options = KBGenerateOptions()

        self.logger.info(
            "Generating knowledgebase for workspace=%s (regenerate=%s)",
            workspace_id,
            options.regenerate,
        )

        # Optionally clear old articles
        if options.regenerate:
            try:
                deleted = await self.storage.delete_kb_articles(workspace_id)
                self.logger.info("Cleared %d existing KB articles for workspace=%s", deleted, workspace_id)
            except NotImplementedError:
                self.logger.debug("Storage backend does not support delete_kb_articles; skipping")

        # 1. Graph analysis
        analysis: GraphAnalysis = await self.graph_service.analyze(
            workspace_id=workspace_id,
            context_id=context_id,
            include_rpg=options.include_rpg,
        )
        self.logger.info(
            "Graph analysis complete: %d communities, %d central nodes, stats=%s",
            len(analysis.communities),
            len(analysis.central_nodes),
            analysis.stats,
        )

        # Cache graph analysis
        try:
            await self.storage.store_graph_analysis(workspace_id, analysis.model_dump(mode="json"))
        except NotImplementedError:
            self.logger.debug("Storage backend does not support store_graph_analysis; skipping cache")

        # Build a community lookup for bridge rendering
        community_by_id: dict[int, Community] = {c.id: c for c in analysis.communities}

        # 2. Generate community articles
        community_articles: list[Article] = []
        selected_communities = analysis.communities[: options.max_communities]
        for community in selected_communities:
            article = await self._generate_community_article(
                workspace_id=workspace_id,
                community=community,
                analysis=analysis,
            )
            community_articles.append(article)
            try:
                await self.storage.store_kb_article(
                    workspace_id=workspace_id,
                    article_id=article.id,
                    article_type=article.article_type,
                    title=article.title,
                    content_md=article.content_md,
                    metadata=article.metadata,
                )
            except NotImplementedError:
                self.logger.debug("Storage backend does not support store_kb_article")

        # 3. Generate entity/god-node articles
        entity_articles: list[Article] = []
        selected_god_nodes = analysis.central_nodes[: options.max_god_nodes]
        for node in selected_god_nodes:
            community = community_by_id.get(node.community_id)
            article = await self._generate_entity_article(
                workspace_id=workspace_id,
                memory_id=node.memory_id,
                community=community,
            )
            entity_articles.append(article)
            try:
                await self.storage.store_kb_article(
                    workspace_id=workspace_id,
                    article_id=article.id,
                    article_type=article.article_type,
                    title=article.title,
                    content_md=article.content_md,
                    metadata=article.metadata,
                )
            except NotImplementedError:
                self.logger.debug("Storage backend does not support store_kb_article")

        # 4. Generate index article
        workspace_obj = await self.storage.get_workspace(workspace_id)
        workspace_name = workspace_obj.name if workspace_obj else workspace_id

        # Build node-title lookup for god nodes using community labels
        node_titles: dict[str, str] = {}
        for node in selected_god_nodes:
            # Use memory ID prefix as fallback title
            node_titles[node.memory_id] = node.memory_id[:16]

        # Use community labels already set during community article generation
        for community in selected_communities:
            if community.label:
                for nid in community.central_node_ids:
                    if nid in node_titles:
                        # Keep memory id prefix; labels are for communities
                        pass

        index_md = self.renderer.render_index(
            workspace_name=workspace_name,
            stats=analysis.stats,
            communities=selected_communities,
            god_nodes=selected_god_nodes,
            node_titles=node_titles,
        )

        now = datetime.now(UTC)
        index_article = Article(
            id="index",
            article_type="index",
            title=workspace_name,
            content_md=index_md,
            metadata={"workspace_id": workspace_id},
            generated_at=now,
        )
        try:
            await self.storage.store_kb_article(
                workspace_id=workspace_id,
                article_id="index",
                article_type="index",
                title=workspace_name,
                content_md=index_md,
                metadata={"workspace_id": workspace_id},
            )
        except NotImplementedError:
            self.logger.debug("Storage backend does not support store_kb_article")

        total_articles = 1 + len(community_articles) + len(entity_articles)
        self.logger.info(
            "KB generation complete for workspace=%s: %d articles (%d communities, %d entities)",
            workspace_id,
            total_articles,
            len(community_articles),
            len(entity_articles),
        )

        return Knowledgebase(
            workspace_id=workspace_id,
            article_count=total_articles,
            community_count=len(community_articles),
            generated_at=now,
            stats=analysis.stats,
        )

    async def get_knowledgebase(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> Knowledgebase | None:
        """Return KB metadata without regenerating."""
        try:
            index_raw = await self.storage.get_kb_article(workspace_id, "index")
        except NotImplementedError:
            return None

        if not index_raw:
            return None

        try:
            articles_raw = await self.storage.list_kb_articles(workspace_id, limit=1000)
        except NotImplementedError:
            articles_raw = []

        community_count = sum(1 for a in articles_raw if a.get("article_type") == "community")
        generated_at_raw = index_raw.get("generated_at") or index_raw.get("created_at")
        generated_at = datetime.now(UTC)
        if generated_at_raw:
            try:
                if isinstance(generated_at_raw, str):
                    generated_at = datetime.fromisoformat(generated_at_raw)
                elif isinstance(generated_at_raw, datetime):
                    generated_at = generated_at_raw
            except (ValueError, TypeError):
                pass

        # Try to load cached graph stats
        stats = None
        try:
            analysis_raw = await self.storage.get_graph_analysis(workspace_id)
            if analysis_raw and "stats" in analysis_raw:
                from ...models.graph_analysis import GraphStats

                stats = GraphStats(**analysis_raw["stats"])
        except (NotImplementedError, Exception) as e:
            self.logger.debug("Could not load cached graph analysis: %s", e)

        return Knowledgebase(
            workspace_id=workspace_id,
            article_count=len(articles_raw),
            community_count=community_count,
            generated_at=generated_at,
            stats=stats,
        )

    async def get_article(
        self,
        workspace_id: str,
        article_id: str,
    ) -> Article | None:
        """Retrieve a single article by ID."""
        try:
            raw = await self.storage.get_kb_article(workspace_id, article_id)
        except NotImplementedError:
            return None

        if not raw:
            return None

        return self._raw_to_article(raw)

    async def list_articles(
        self,
        workspace_id: str,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Article]:
        """List articles, optionally filtered by type."""
        try:
            raws = await self.storage.list_kb_articles(
                workspace_id,
                article_type=article_type,
                limit=limit,
                offset=offset,
            )
        except NotImplementedError:
            return []

        return [self._raw_to_article(r) for r in raws]

    async def export_vault(
        self,
        workspace_id: str,
        context_id: str | None = None,
    ) -> bytes:
        """Export all articles as an Obsidian vault zip."""
        try:
            raws = await self.storage.list_kb_articles(workspace_id, limit=10000)
        except NotImplementedError:
            raws = []

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for raw in raws:
                article = self._raw_to_article(raw)
                path = self._article_zip_path(article)
                zf.writestr(path, article.content_md)

        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    async def _generate_community_article(
        self,
        workspace_id: str,
        community: Community,
        analysis: GraphAnalysis,
    ) -> Article:
        """Generate a markdown article for a single community."""
        member_ids = community.memory_ids[:20]
        members: list[dict] = []

        # Fetch member memory contents for context
        for mem_id in member_ids:
            try:
                mem = await self.storage.get_memory(workspace_id, mem_id, track_access=False)
                if mem:
                    members.append({"id": mem.id, "content": mem.content, "type": mem.type.value})
            except Exception as e:
                self.logger.debug("Could not fetch member memory %s: %s", mem_id, e)

        # Generate label + summary
        label, summary = await self._summarize_community(workspace_id, community, members)
        community.label = label

        # Bridges involving this community
        community_bridges = [b for b in analysis.bridges if b.source_community_id == community.id or b.target_community_id == community.id]

        content_md = self.renderer.render_community(
            community=community,
            summary=summary,
            members=members,
            bridges=community_bridges,
        )

        return Article(
            id=f"community-{community.id}",
            article_type="community",
            title=label,
            content_md=content_md,
            metadata={"community_id": community.id, "size": community.size},
            generated_at=datetime.now(UTC),
        )

    async def _generate_entity_article(
        self,
        workspace_id: str,
        memory_id: str,
        community: Community | None,
    ) -> Article:
        """Generate a markdown article for a god-node entity."""
        # Fetch the central memory
        mem = None
        try:
            mem = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
        except Exception as e:
            self.logger.debug("Could not fetch central memory %s: %s", memory_id, e)

        title = (mem.content[:60] if mem else memory_id[:16]).replace("\n", " ")
        slug = self.renderer.slugify(title)

        # Fetch associations as connections
        connections: list[dict] = []
        try:
            assocs = await self.storage.get_associations(workspace_id, memory_id, direction="both")
            for assoc in assocs[:20]:
                connections.append(
                    {
                        "target_id": assoc.target_id if assoc.source_id == memory_id else assoc.source_id,
                        "relationship": assoc.relationship,
                        "strength": assoc.strength,
                    }
                )
        except Exception as e:
            self.logger.debug("Could not fetch associations for %s: %s", memory_id, e)

        # Derive entity insights
        entity_card: dict | None = None
        if self.inference_service:
            try:
                result = await self.inference_service.derive_insights(
                    workspace_id=workspace_id,
                    subject_id=memory_id,
                )
                if result.insights:
                    entity_card = {"insights": [m.content for m in result.insights[:10]]}
            except Exception as e:
                self.logger.debug("Inference failed for entity %s: %s", memory_id, e)
        elif self.reflect_service:
            try:
                reflect_result = await self.reflect_service.reflect(
                    workspace_id=workspace_id,
                    input=ReflectInput(
                        query=f"What is notable about: {title}",
                        detail_level=DetailLevel.OVERVIEW,
                        include_sources=False,
                        depth=1,
                    ),
                )
                if reflect_result.reflection:
                    entity_card = {"insights": [reflect_result.reflection]}
            except Exception as e:
                self.logger.debug("Reflect fallback failed for entity %s: %s", memory_id, e)

        source_memories = []
        if mem:
            source_memories.append({"id": mem.id, "content": mem.content, "type": mem.type.value})

        content_md = self.renderer.render_entity(
            entity_id=memory_id,
            title=title,
            entity_card=entity_card,
            connections=connections,
            community=community,
            source_memories=source_memories,
        )

        return Article(
            id=f"entity-{slug}",
            article_type="entity",
            title=title,
            content_md=content_md,
            metadata={"memory_id": memory_id, "slug": slug},
            generated_at=datetime.now(UTC),
        )

    async def _summarize_community(
        self,
        workspace_id: str,
        community: Community,
        members: list[dict],
    ) -> tuple[str, str]:
        """Generate a topic label and summary for a community.

        Returns:
            (label, summary) tuple
        """
        if not members:
            label = f"Community {community.id}"
            summary = f"Empty community with {community.size} members."
            return label, summary

        # Build a short content preview for the prompt
        content_snippets = [m["content"][:150] for m in members[:10] if m.get("content")]
        combined = "\n".join(f"- {s}" for s in content_snippets)

        if self.reflect_service:
            try:
                result = await self.reflect_service.reflect(
                    workspace_id=workspace_id,
                    input=ReflectInput(
                        query=f"What is the common theme or topic of these memories?\n\n{combined}",
                        detail_level=DetailLevel.OVERVIEW,
                        include_sources=False,
                        depth=1,
                    ),
                )
                reflection = result.reflection or ""
                # Extract a short label from the first sentence
                first_sentence = reflection.split(".")[0].strip()
                label = first_sentence[:80] if first_sentence else f"Community {community.id}"
                summary = reflection
                return label, summary
            except Exception as e:
                self.logger.debug("Reflect failed for community %d: %s", community.id, e)

        # Fallback: simple summary from member content
        label = f"Community {community.id}"
        summary_parts = [f"A cluster of {community.size} memories (cohesion: {community.cohesion_score:.2f})."]
        summary_parts.append("\n**Sample memories:**")
        for snippet in content_snippets[:5]:
            summary_parts.append(f"- {snippet}")
        summary = "\n".join(summary_parts)
        return label, summary

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _raw_to_article(raw: dict) -> Article:
        """Convert a storage dict to an Article model."""
        generated_at = datetime.now(UTC)
        raw_ts = raw.get("generated_at") or raw.get("created_at")
        if raw_ts:
            try:
                if isinstance(raw_ts, str):
                    generated_at = datetime.fromisoformat(raw_ts)
                elif isinstance(raw_ts, datetime):
                    generated_at = raw_ts
            except (ValueError, TypeError):
                pass

        return Article(
            id=raw.get("article_id") or raw.get("id", ""),
            article_type=raw.get("article_type", ""),
            title=raw.get("title", ""),
            content_md=raw.get("content_md", ""),
            metadata=raw.get("metadata") or {},
            generated_at=generated_at,
        )

    @staticmethod
    def _article_zip_path(article: Article) -> str:
        """Map an article to its path inside the vault zip."""
        if article.article_type == "index":
            return "index.md"
        if article.article_type == "community":
            return f"communities/{article.id}.md"
        if article.article_type == "entity":
            return f"entities/{article.id.removeprefix('entity-')}.md"
        return f"{article.id}.md"


class DefaultKnowledgebaseServicePlugin(KnowledgebaseServicePluginBase):
    """Plugin registration for the default knowledgebase service."""

    PROVIDER_NAME = "default"

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_GRAPH_ANALYSIS_SERVICE)

    def initialize(self, v: Variables, logger: Logger) -> DefaultKnowledgebaseService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        graph_service: GraphAnalysisService = self.get_extension(EXT_GRAPH_ANALYSIS_SERVICE, v)

        # Optional services — may not be configured
        reflect_service = None
        try:
            reflect_service = get_extension(EXT_REFLECT_SERVICE, v)
        except Exception:
            logger.debug("ReflectService not available for KnowledgebaseService")

        inference_service = None
        try:
            inference_service = get_extension(EXT_INFERENCE_SERVICE, v)
        except Exception:
            logger.debug("InferenceService not available for KnowledgebaseService")

        return DefaultKnowledgebaseService(
            storage=storage,
            graph_service=graph_service,
            reflect_service=reflect_service,
            inference_service=inference_service,
            v=v,
        )
