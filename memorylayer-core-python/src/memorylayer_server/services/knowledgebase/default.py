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

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from logging import Logger

from scitrera_app_framework import get_extension, get_logger
from scitrera_app_framework.api import Variables

from ...config import (
    DEFAULT_MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD,
    DEFAULT_MEMORYLAYER_KB_CONTENT_VERSION_MODE,
    DEFAULT_MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS,
    DEFAULT_MEMORYLAYER_KB_MIN_COMMUNITY_SIZE,
    DEFAULT_MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS,
    DEFAULT_MEMORYLAYER_KB_SUMMARY_MAX_TOKENS,
    MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD,
    MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS,
    MEMORYLAYER_KB_CONTENT_VERSION_MODE,
    MEMORYLAYER_KB_MIN_COMMUNITY_SIZE,
    MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS,
    MEMORYLAYER_KB_SUMMARY_MAX_TOKENS,
)
from ...models.graph_analysis import Community, GraphAnalysis
from ...models.generation import GenerationActivity
from ...models.memory import DetailLevel, ReflectInput
from .._constants import (
    EXT_CONTRADICTION_SERVICE,
    EXT_GRAPH_ANALYSIS_SERVICE,
    EXT_INFERENCE_SERVICE,
    EXT_REFLECT_SERVICE,
    EXT_STORAGE_BACKEND,
)
from ..llm import EXT_LLM_SERVICE, LLMNotConfiguredError, LLMService
from ..graph_analysis.base import GraphAnalysisService
from ..storage import StorageBackend
from . import KnowledgebaseServicePluginBase
from . import differ as kb_differ
from .base import Article, KBGenerateOptions, Knowledgebase
from .citations import CitationReport, audit_citations, summary_snippet
from .coverage import compute_coverage
from .linkcheck import LinkIndex, article_vault_path, validate_articles
from .renderer import ObsidianRenderer


class DefaultKnowledgebaseService:
    """Default knowledgebase generation service."""

    def __init__(
        self,
        storage: StorageBackend,
        graph_service: GraphAnalysisService,
        reflect_service=None,
        inference_service=None,
        llm_service: LLMService | None = None,
        contradiction_service=None,
        v: Variables = None,
    ):
        self.storage = storage
        self.graph_service = graph_service
        self.reflect_service = reflect_service
        self.inference_service = inference_service
        # Soft: used only to flag which members are involved in an unresolved
        # contradiction (a per-run workspace fetch). Absent -> the frontmatter
        # simply omits the contradicted_by signal.
        self.contradiction_service = contradiction_service
        # Direct LLM handle for community label/summary generation (tight,
        # KB-specific prompts) — kept separate from reflect_service so the
        # user-facing reflect path is untouched.
        self.llm = llm_service
        self.renderer = ObsidianRenderer()
        # Communities smaller than this are skipped (Louvain singletons are just
        # individual memories — noise). See MEMORYLAYER_KB_MIN_COMMUNITY_SIZE.
        self.min_community_size = int(
            v.get(MEMORYLAYER_KB_MIN_COMMUNITY_SIZE, DEFAULT_MEMORYLAYER_KB_MIN_COMMUNITY_SIZE)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_MIN_COMMUNITY_SIZE
        )
        # Token budget for the community title+summary call — large enough for a
        # reasoning model to finish (see the config note).
        self.summary_max_tokens = int(
            v.get(MEMORYLAYER_KB_SUMMARY_MAX_TOKENS, DEFAULT_MEMORYLAYER_KB_SUMMARY_MAX_TOKENS)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_SUMMARY_MAX_TOKENS
        )
        self.community_max_members = int(
            v.get(MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS, DEFAULT_MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_COMMUNITY_MAX_MEMBERS
        )
        self.summary_budget_chars = int(
            v.get(MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS, DEFAULT_MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_SUMMARY_BUDGET_CHARS
        )
        # Incremental KB rendering (Lever 3) configuration.
        self.community_match_threshold = float(
            v.get(MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD, DEFAULT_MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_COMMUNITY_MATCH_THRESHOLD
        )
        self.content_version_mode = (
            v.get(MEMORYLAYER_KB_CONTENT_VERSION_MODE, DEFAULT_MEMORYLAYER_KB_CONTENT_VERSION_MODE)
            if v is not None
            else DEFAULT_MEMORYLAYER_KB_CONTENT_VERSION_MODE
        )
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

        # P2 strategy-C dirty-watermark skip-generate (Gate A). When NOT regenerating,
        # if a prior KB exists and the workspace's current change-watermark EXACTLY
        # matches the watermark recorded at the last successful generation, the entire
        # run (analyze + rendering + every LLM call) is redundant -> return the cached
        # KB via the existing get_knowledgebase read path. This stacks on top of P1.2's
        # per-article content-hash skip: the watermark skips the whole run when the
        # workspace is idle; the content-hash skips unchanged articles when a run
        # does proceed.
        #
        # FAIL-SAFE: skip ONLY when current_watermark is not None, a prior watermark
        # was recorded, they compare EQUAL, and a cached KB is actually returnable.
        # Any ambiguity (watermark unavailable, no prior watermark/KB, mismatch, or
        # any error) falls through to a normal full generation -- never skip and risk
        # serving stale data.
        current_watermark = await self._compute_change_watermark(workspace_id)
        if not options.regenerate and current_watermark is not None:
            try:
                prior_watermark = await self._get_stored_watermark(workspace_id)
                if prior_watermark is not None and prior_watermark == current_watermark:
                    cached = await self.get_knowledgebase(workspace_id, context_id=context_id)
                    if cached is not None:
                        self.logger.info(
                            "KB skip-generate for workspace=%s: change-watermark unchanged "
                            "(%s); returning cached KB (no analyze, no LLM)",
                            workspace_id,
                            current_watermark,
                        )
                        return cached
            except Exception as e:  # fail-safe: any doubt -> generate normally
                self.logger.debug("Watermark skip-generate check failed for %s: %s", workspace_id, e)

        # Incremental path (regenerate=False) uses the PRIOR cached analysis as the diff
        # baseline for community-stability matching. Load it BEFORE we overwrite the cache
        # with this run's analysis. On regenerate=True we force a full rebuild and ignore
        # any prior state, so there is no baseline to diff against.
        prior_communities: list[Community] = []
        if not options.regenerate:
            try:
                # get_graph_analysis returns a wrapper dict:
                # {"workspace_id": …, "analysis_json": {communities, stats, …}, "generated_at": …}
                # Unwrap "analysis_json" before parsing; calling parse on the wrapper would always
                # yield [] because the wrapper has no "communities" key (the blocker bug).
                prior_raw = await self.storage.get_graph_analysis(workspace_id)
                prior_analysis_json = (prior_raw or {}).get("analysis_json")
                prior_communities = kb_differ.parse_prior_communities(prior_analysis_json)
            except NotImplementedError:
                prior_communities = []
            except Exception as e:  # cache read is best-effort; fall back to full generation
                self.logger.debug("Could not load prior graph analysis for diff: %s", e)
                prior_communities = []

        # Full-rebuild escape hatch (regenerate=True) is now DEFERRED-DELETE, not eager
        # pre-delete. The old path eagerly called delete_kb_articles BEFORE analyze() +
        # LLM summarization, so any failure after the delete (analyze raising, an LLM
        # summary throwing) left the workspace with ZERO articles -- data loss. Instead we
        # reuse the incremental path's success-path-only GC discipline: every prior id is
        # an orphan candidate, and orphans are reclaimed by _gc_stale_articles ONLY when
        # gc_safe=True (all stores this run succeeded). A stale article is strictly safer
        # than a deleted live one. regenerate=True still FORCES fresh re-summarization
        # (force_regenerate bypasses the content-hash reuse below); only the DELETION
        # timing changes -- deferred to the all-articles-stored success path.
        force_regenerate = options.regenerate

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

        # Unresolved-contradiction member ids — fetched ONCE for the whole run and
        # threaded into article builders for the contradicted_by frontmatter signal
        # (soft; empty when no contradiction service).
        contradicted_ids = await self._fetch_contradicted_ids(workspace_id)

        # Match this run's communities to the prior run by member-set overlap so a matched
        # community inherits the PRIOR (stable) article id, decoupling the stored id from
        # the volatile Louvain idx. On regenerate=True prior_communities is empty -> all new.
        # Drop sub-threshold (typically singleton) communities before selection.
        # Louvain on a sparse association graph yields many size-1 clusters that
        # are just individual memories (noisy + heavily overlapping). Graph stats
        # keep the true Louvain partition; only article generation is filtered.
        eligible_communities = [
            c for c in analysis.communities if c.size >= self.min_community_size
        ]
        skipped = len(analysis.communities) - len(eligible_communities)
        if skipped:
            self.logger.info(
                "KB: %d/%d communities meet min_community_size=%d (skipped %d small/singleton)",
                len(eligible_communities), len(analysis.communities),
                self.min_community_size, skipped,
            )
        selected_communities = eligible_communities[: options.max_communities]
        capped = len(eligible_communities) - len(selected_communities)
        if capped:
            # Never drop silently: without this line a KB covering a third of the
            # workspace is indistinguishable from one covering all of it.
            self.logger.info(
                "KB: %d eligible communities exceed max_communities=%d and get no "
                "article (workspace=%s)",
                capped,
                options.max_communities,
                workspace_id,
            )
        match_result = kb_differ.match_communities(
            current=selected_communities,
            prior=prior_communities,
            threshold=self.community_match_threshold,
        )
        match_by_current_id: dict[int, kb_differ.CommunityMatch] = {
            m.community.id: m for m in match_result.matches
        }

        # Resolve the article id of EVERYTHING this run will produce before rendering
        # anything. A cross-reference (a bridge to a community, an association to a
        # memory) can then be emitted as a link only when the article it points at will
        # actually exist -- capped runs, the min-size filter and the god-node cap all
        # leave referenced things without a page. Community ids resolve with no I/O;
        # an entity id embeds a slug of the node's own content, so those memories are
        # prefetched here and handed to _generate_entity_article rather than refetched.
        selected_god_nodes = self._select_god_nodes(analysis, options)
        god_node_memories = await self._prefetch_memories(
            workspace_id, [node.memory_id for node in selected_god_nodes]
        )
        link_index = LinkIndex()
        for community in selected_communities:
            match = match_by_current_id.get(community.id)
            link_index.add_community(
                community.id,
                match.article_id if match else f"community-{community.id}",
            )
        for node in selected_god_nodes:
            link_index.add_entity(
                node.memory_id,
                self._entity_article_id(node.memory_id, god_node_memories.get(node.memory_id)),
            )
        # Additive producers register their own ids from inside _generate_extra_articles:
        # their articles link only to each other, and that runs before the index is
        # rendered, so nothing needs them resolvable earlier than that.

        # `current_article_ids` accumulates every id produced this run (community, entity,
        # index). GC (success path only) deletes prior ids not in this set. `gc_safe` flips
        # to False on ANY store failure so a partial run never deletes a live article.
        current_article_ids: set[str] = {"index"}
        gc_safe = True
        # Count REAL persistence failures (NotImplementedError backends are treated as
        # success by _store_article). Used to fail the run loudly rather than returning a
        # 200 with an in-memory article_count that was never actually persisted.
        store_failures = 0

        # 2. Generate community articles (content-hash skip -> reuse prior verbatim, no LLM)
        community_articles: list[Article] = []
        for community in selected_communities:
            match = match_by_current_id.get(community.id)
            article_id = match.article_id if match else f"community-{community.id}"
            article = await self._generate_community_article(
                workspace_id=workspace_id,
                community=community,
                analysis=analysis,
                article_id=article_id,
                prior_label=match.prior_label if match else None,
                force_regenerate=force_regenerate,
                contradicted_ids=contradicted_ids,
                link_index=link_index,
            )
            community_articles.append(article)
            current_article_ids.add(article.id)
            if not await self._store_article(workspace_id, article):
                gc_safe = False
                store_failures += 1

        # 3. Generate entity/god-node articles (key on stable node.memory_id; hash-skip)
        entity_articles: list[Article] = []
        for node in selected_god_nodes:
            community = community_by_id.get(node.community_id)
            article = await self._generate_entity_article(
                workspace_id=workspace_id,
                memory_id=node.memory_id,
                community=community,
                force_regenerate=force_regenerate,
                link_index=link_index,
                mem=god_node_memories.get(node.memory_id),
                contradicted_ids=contradicted_ids,
            )
            entity_articles.append(article)
            current_article_ids.add(article.id)
            if not await self._store_article(workspace_id, article):
                gc_safe = False
                store_failures += 1

        # 3.5 Additive article producers beyond communities/god-nodes. OSS yields
        # none (parity reference); an enterprise subclass overrides
        # _generate_extra_articles to add e.g. canonical-entity-registry articles.
        # Produced ids join current_article_ids BEFORE GC, so they survive
        # stale-article reclamation (which deletes any prior id not seen this run).
        extra_articles: list[Article] = []
        try:
            produced = await self._generate_extra_articles(
                workspace_id=workspace_id,
                analysis=analysis,
                community_by_id=community_by_id,
                force_regenerate=force_regenerate,
                contradicted_ids=contradicted_ids,
                link_index=link_index,
            )
        except Exception as e:  # fail-safe: extra producers never break core KB
            self.logger.error("Extra KB article producer failed for workspace=%s: %s", workspace_id, e)
            produced = []
        for article in produced:
            extra_articles.append(article)
            current_article_ids.add(article.id)
            if not await self._store_article(workspace_id, article):
                gc_safe = False
                store_failures += 1

        # 4. Generate index article (always current; never GC'd)
        workspace_obj = await self.storage.get_workspace(workspace_id)
        workspace_name = workspace_obj.name if workspace_obj else workspace_id

        # Titles come from the prefetched god-node memories -- the same string the entity
        # article is titled and slugged from -- so the index names the page the way the
        # page names itself instead of showing a truncated uuid.
        node_titles: dict[str, str] = {
            node.memory_id: self._entity_title(
                node.memory_id, god_node_memories.get(node.memory_id)
            )
            for node in selected_god_nodes
        }

        index_md = self.renderer.render_index(
            workspace_name=workspace_name,
            stats=analysis.stats,
            communities=selected_communities,
            god_nodes=selected_god_nodes,
            node_titles=node_titles,
            link_index=link_index,
        )

        # Link-integrity sweep over everything produced this run. Construction already
        # stops an unresolvable reference from becoming a link, so a break found here is
        # the case construction cannot see: an article REUSED verbatim from a prior run
        # (content-hash hit) that links to a target since reclaimed by GC. Report it
        # rather than fail -- a stale link is a quality signal, not a reason to withhold
        # an entire knowledgebase.
        produced_md: dict[str, str] = {
            article_vault_path(a.article_type, a.id): a.content_md
            for a in (*community_articles, *entity_articles, *extra_articles)
        }
        produced_md["index"] = index_md
        link_report = validate_articles(produced_md, set(produced_md))
        if link_report.ok:
            self.logger.info(
                "KB link check for workspace=%s: all %d wikilinks resolve",
                workspace_id,
                link_report.total,
            )
        else:
            self.logger.warning(
                "KB link check for workspace=%s: %d/%d wikilinks unresolved (e.g. %s)",
                workspace_id,
                link_report.broken_count,
                link_report.total,
                "; ".join(f"{b.source_path} -> {b.target}" for b in link_report.broken[:3]),
            )

        # Coverage: of the memories in the association graph, how many reached an
        # article. FULL community membership counts, not just the members rendered in
        # the body -- a memory has a home whether or not the summary quotes it.
        covered_memory_ids: set[str] = set()
        for community in selected_communities:
            covered_memory_ids.update(community.memory_ids)
        covered_memory_ids.update(node.memory_id for node in selected_god_nodes)
        for article in extra_articles:
            covered_memory_ids.update(article.metadata.get("member_ids") or ())
        coverage = compute_coverage(
            graph_nodes=analysis.stats.node_count if analysis.stats else 0,
            covered_memory_ids=covered_memory_ids,
            dropped_small_communities=skipped,
            dropped_capped_communities=capped,
        )
        self.logger.info(
            "KB coverage for workspace=%s: %.1f%% of graph memories have an article "
            "(%d/%d; %d uncovered, %d dropped below min size, %d dropped over cap)",
            workspace_id,
            coverage.ratio * 100,
            coverage.covered,
            coverage.graph_nodes,
            coverage.uncovered,
            coverage.dropped_small_communities,
            coverage.dropped_capped_communities,
        )

        now = datetime.now(UTC)
        # Record the change-watermark CAPTURED AT THE START of this run in the index
        # article metadata (the least-invasive, contract-stable per-workspace marker:
        # the index article is already written every run and read by get_knowledgebase).
        # Using the start-of-run snapshot is the fail-safe choice -- a write that lands
        # mid-run is NOT reflected in this KB, so leaving the stored watermark at the
        # start value means the NEXT run sees an advanced watermark and regenerates
        # (rather than skipping and serving a KB that missed that write).
        #
        # PARTIAL-FAILURE GUARD: only embed the watermark when ALL preceding article
        # stores succeeded (gc_safe=True). If any community/entity store failed, the KB
        # is incomplete; recording the watermark here would let the next idle run skip
        # via Gate A and permanently serve the incomplete KB (it would never self-heal
        # because the workspace is idle). Omitting the watermark on a partial failure
        # means _get_stored_watermark returns None next time -> fail-safe full regenerate.
        index_metadata: dict = {
            "workspace_id": workspace_id,
            **link_report.as_metadata(),
            **coverage.as_metadata(),
        }
        if current_watermark is not None and gc_safe:
            index_metadata["change_watermark"] = list(current_watermark)
        index_article = Article(
            id="index",
            article_type="index",
            title=workspace_name,
            content_md=index_md,
            metadata=index_metadata,
            generated_at=now,
        )
        if not await self._store_article(workspace_id, index_article):
            gc_safe = False
            store_failures += 1

        # 5. Stale-article GC — ONLY on the all-articles-stored success path. A stale article
        # is strictly safer than deleting a live one, so any store failure skips GC this fire.
        if gc_safe:
            await self._gc_stale_articles(workspace_id, current_article_ids)
        else:
            self.logger.warning(
                "Skipping stale-article GC for workspace=%s: a store_kb_article failed this run",
                workspace_id,
            )

        total_articles = 1 + len(community_articles) + len(entity_articles) + len(extra_articles)

        # Fail loudly if any article failed to persist. Returning a 200 with the in-memory
        # article_count while every store silently failed is what masked the metadata/meta
        # column bug: the API reported 50 articles while 0 were written and reads 404'd.
        if store_failures:
            raise RuntimeError(
                f"KB generation for workspace={workspace_id} failed to persist "
                f"{store_failures}/{total_articles} articles; see prior store errors"
            )

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
            coverage=coverage.as_metadata(),
        )

    async def _store_article(self, workspace_id: str, article: Article) -> bool:
        """Persist an article. Returns True on success (or unsupported backend), False on error.

        A NotImplementedError backend is treated as success (the article was "stored" as far
        as this run is concerned) so GC is not skipped on storage backends without KB support.
        A real store failure returns False so the caller skips GC this fire.
        """
        try:
            await self.storage.store_kb_article(
                workspace_id=workspace_id,
                article_id=article.id,
                article_type=article.article_type,
                title=article.title,
                content_md=article.content_md,
                metadata=article.metadata,
            )
            return True
        except NotImplementedError:
            self.logger.debug("Storage backend does not support store_kb_article")
            return True
        except Exception as e:
            self.logger.error("Failed to store KB article %s for workspace=%s: %s", article.id, workspace_id, e)
            return False

    async def _gc_stale_articles(self, workspace_id: str, current_article_ids: set[str]) -> None:
        """Delete prior articles not produced this run (orphans). Idempotent.

        Orphans are computed from the article-id set (community ids are volatile), so a
        dissolved/renumbered community's stale ``community-{id}`` article is reclaimed.
        """
        try:
            prior_raw = await self.storage.list_kb_articles(workspace_id, limit=10000)
        except NotImplementedError:
            return
        except Exception as e:
            self.logger.debug("Could not list KB articles for GC: %s", e)
            return

        prior_ids = {r.get("article_id") or r.get("id") for r in prior_raw}
        orphans = {aid for aid in prior_ids if aid and aid not in current_article_ids}
        for orphan_id in orphans:
            try:
                await self.storage.delete_kb_article(workspace_id, orphan_id)
                self.logger.info("GC deleted stale KB article %s for workspace=%s", orphan_id, workspace_id)
            except NotImplementedError:
                self.logger.debug("Storage backend does not support delete_kb_article; skipping GC")
                return
            except Exception as e:
                self.logger.error("Failed to GC stale KB article %s for workspace=%s: %s", orphan_id, workspace_id, e)

    async def _compute_change_watermark(self, workspace_id: str) -> tuple | None:
        """Current dirty-watermark for the workspace, or None (fail-safe).

        None means "cannot determine if unchanged" -> callers must do the work.
        A backend without watermark support (NotImplementedError) yields None.
        """
        try:
            wm = await self.storage.get_workspace_change_watermark(workspace_id)
        except NotImplementedError:
            return None
        except Exception as e:
            self.logger.debug("Change-watermark unavailable for %s: %s", workspace_id, e)
            return None
        # Normalize to a tuple so equality against the stored (JSON list) form is exact.
        return tuple(wm) if wm is not None else None

    async def _get_stored_watermark(self, workspace_id: str) -> tuple | None:
        """Watermark recorded in the prior index article's metadata, or None.

        None means no prior watermark was stored (e.g. first-ever generation or a
        KB generated before this feature) -> caller must not skip (fail-safe).
        """
        try:
            index_raw = await self.storage.get_kb_article(workspace_id, "index")
        except NotImplementedError:
            return None
        except Exception as e:
            self.logger.debug("Could not read prior index article for %s: %s", workspace_id, e)
            return None
        if not index_raw:
            return None
        stored = (index_raw.get("metadata") or {}).get("change_watermark")
        if not isinstance(stored, (list, tuple)):
            return None
        return tuple(stored)

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

        # Try to load cached graph stats.
        # get_graph_analysis returns {"workspace_id": …, "analysis_json": {…}, "generated_at": …};
        # unwrap "analysis_json" before reading "stats" (the wrapper dict never has "stats" itself).
        stats = None
        try:
            analysis_wrapper = await self.storage.get_graph_analysis(workspace_id)
            analysis_inner = (analysis_wrapper or {}).get("analysis_json") or {}
            if analysis_inner and "stats" in analysis_inner:
                from ...models.graph_analysis import GraphStats

                stats = GraphStats(**analysis_inner["stats"])
        except (NotImplementedError, Exception) as e:
            self.logger.debug("Could not load cached graph analysis: %s", e)

        # Coverage is recorded on the index article at generation time. Surfacing it here
        # matters because the watermark skip-generate path returns THIS object: without
        # it an idle workspace would report no coverage at all rather than the coverage
        # of the KB it is actually serving.
        index_meta = index_raw.get("metadata") or {}
        coverage = {k: v for k, v in index_meta.items() if k.startswith("coverage_")} or None

        return Knowledgebase(
            workspace_id=workspace_id,
            article_count=len(articles_raw),
            community_count=community_count,
            generated_at=generated_at,
            stats=stats,
            coverage=coverage,
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
        article_id: str | None = None,
        prior_label: str | None = None,
        force_regenerate: bool = False,
        contradicted_ids: set[str] | None = None,
        link_index: LinkIndex | None = None,
    ) -> Article:
        """Generate a markdown article for a single community.

        ``article_id`` is the stable id this community should be stored under (the prior
        article id for a matched community); defaults to ``community-{id}`` for new ones.
        If the recomputed content hash matches the prior article's stored ``content_key``,
        the prior article is reused verbatim and NO LLM call is made.

        ``force_regenerate`` (regenerate=True) bypasses the content-hash reuse so a fresh
        LLM summary is always produced; the article still overwrites by ``article_id``.
        """
        article_id = article_id or f"community-{community.id}"
        member_ids = community.memory_ids[: self.community_max_members]
        members: list[dict] = []
        content_versions: dict[str, str] = {}

        # Fetch member memory contents for context (and their content versions for the hash)
        for mem_id in member_ids:
            try:
                mem = await self.storage.get_memory(workspace_id, mem_id, track_access=False)
                if mem:
                    members.append(self._member_dict(mem))
                    content_versions[mem_id] = self._content_version(mem)
            except Exception as e:
                self.logger.debug("Could not fetch member memory %s: %s", mem_id, e)

        # Content-hash skip: reuse the prior article verbatim if nothing it renders changed.
        # force_regenerate (regenerate=True) bypasses reuse so every article is freshly
        # re-summarized; only the deferred-delete timing changed, not force semantics.
        content_key = kb_differ.community_content_key(community, content_versions)
        if not force_regenerate:
            reuse = await self._maybe_reuse_article(workspace_id, article_id, content_key)
            if reuse is not None:
                community.label = prior_label or community.label
                return reuse

        # Generate label + summary (the LLM call). The report is audited against the
        # numbered member block the model was shown (measurement only — never rewrites
        # the summary); it flags any out-of-range (invented) reference number.
        label, summary, report = await self._summarize_community(workspace_id, community, members)
        community.label = label
        if report.invalid:
            self.logger.warning(
                "KB community %s summary cited %d out-of-range reference(s): %s",
                community.id, report.invalid, report.invalid_ids[:5],
            )

        # Bridges involving this community
        community_bridges = [b for b in analysis.bridges if b.source_community_id == community.id or b.target_community_id == community.id]

        # OKF v0.1 + trust/provenance frontmatter (Obsidian properties). Leads with
        # the OKF-recommended fields (type/title/description/tags/timestamp), then
        # the provenance aggregate (confidence/tags/sources) and grounding coverage.
        now = datetime.now(UTC)
        fm_fields = {
            "type": "community",
            "title": label,
            "description": summary_snippet(summary),
            "timestamp": now.isoformat(),
            "member_count": len(members),
            "cohesion": round(community.cohesion_score, 3),
            **self._provenance_fields(members, contradicted_ids),
            "citation_coverage": report.as_metadata()["citation_coverage"],
        }
        content_md = self.renderer.render_frontmatter(fm_fields) + self.renderer.render_community(
            community=community,
            summary=summary,
            members=members,
            bridges=community_bridges,
            link_index=link_index,
        )

        return Article(
            id=article_id,
            article_type="community",
            title=label,
            content_md=content_md,
            metadata={
                "community_id": community.id,
                "size": community.size,
                "member_ids": self.member_citation_ids(members),
                **report.as_metadata(),
                kb_differ.CONTENT_KEY_FIELD: content_key,
            },
            generated_at=now,
        )

    async def _generate_entity_article(
        self,
        workspace_id: str,
        memory_id: str,
        community: Community | None,
        force_regenerate: bool = False,
        link_index: LinkIndex | None = None,
        mem=None,
        contradicted_ids: set[str] | None = None,
    ) -> Article:
        """Generate a markdown article for a god-node entity.

        ``force_regenerate`` (regenerate=True) bypasses the content-hash reuse so a fresh
        article is always produced; it still overwrites by ``article_id``.
        """
        # generate() prefetches this to build the link index; fetch only when this method
        # is driven directly (tests, or a future single-article refresh path).
        if mem is None:
            try:
                mem = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
            except Exception as e:
                self.logger.debug("Could not fetch central memory %s: %s", memory_id, e)

        title = self._entity_title(memory_id, mem)
        # The pure slug is kept in metadata["slug"] for display; the article id carries a
        # memory_id suffix (see _entity_article_id for why).
        slug = self.renderer.slugify(title)
        article_id = self._entity_article_id(memory_id, mem)

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

        # Content-hash skip: reuse the prior article verbatim if node + connections unchanged.
        # force_regenerate (regenerate=True) bypasses reuse so the article is freshly built.
        content_version = self._content_version(mem) if mem else ""
        content_key = kb_differ.entity_content_key(memory_id, content_version, connections)
        if not force_regenerate:
            reuse = await self._maybe_reuse_article(workspace_id, article_id, content_key)
            if reuse is not None:
                return reuse

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

        # _member_dict (rather than a bare id/content/type) so the trust signals reach
        # the provenance aggregation below, exactly as they do for a community article.
        source_memories = [self._member_dict(mem)] if mem else []

        # OKF v0.1 core fields + trust/provenance, matching what community and
        # registry-entity articles already carry. Without this a god-node article was
        # the only page in the vault with NO front matter, leaving the exported bundle
        # non-conformant (`type` is OKF's one required field).
        now = datetime.now(UTC)
        if entity_card and entity_card.get("insights"):
            description_source = entity_card["insights"][0]
        else:
            description_source = mem.content if mem else ""
        fm_fields = {
            "type": "entity",
            "title": title,
            "description": summary_snippet(description_source),
            "timestamp": now.isoformat(),
            "connection_count": len(connections),
            **self._provenance_fields(source_memories, contradicted_ids),
        }
        content_md = self.renderer.render_frontmatter(fm_fields) + self.renderer.render_entity(
            entity_id=memory_id,
            title=title,
            entity_card=entity_card,
            connections=connections,
            community=community,
            source_memories=source_memories,
            link_index=link_index,
        )

        return Article(
            id=article_id,
            article_type="entity",
            title=title,
            content_md=content_md,
            metadata={
                "memory_id": memory_id,
                "slug": slug,
                kb_differ.CONTENT_KEY_FIELD: content_key,
            },
            generated_at=now,
        )

    def _select_god_nodes(self, analysis: GraphAnalysis, options: KBGenerateOptions) -> list:
        """The central ("god") nodes to render as entity articles.

        Base = the top ``max_god_nodes`` central nodes — memory *proxies* for the
        most-connected memories. An enterprise subclass with a real entity registry
        overrides this to suppress the proxies (return ``[]``) so the KB's entity
        surface is the canonical typed entity graph, not central-memory stand-ins.
        """
        return analysis.central_nodes[: options.max_god_nodes]

    async def _prefetch_memories(self, workspace_id: str, memory_ids: list[str]) -> dict:
        """Fetch memories once, for reuse by link-index construction AND rendering.

        A memory that cannot be fetched is simply absent: callers fall back to an
        id-derived title, exactly as they did when each fetched individually.
        """
        out: dict = {}
        for memory_id in memory_ids:
            try:
                mem = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
            except Exception as e:  # noqa: BLE001 - a missing node only loses its title
                self.logger.debug("Could not prefetch memory %s: %s", memory_id, e)
                continue
            if mem:
                out[memory_id] = mem
        return out

    def _entity_title(self, memory_id: str, mem) -> str:
        """Display title for a god-node entity article."""
        return (mem.content[:60] if mem else memory_id[:16]).replace("\n", " ")

    def _entity_article_id(self, memory_id: str, mem) -> str:
        """Article id for a god-node entity article.

        The ONE place this id is derived, used both to build the link index and to
        generate the article, so a link and the file it points at cannot be derived
        differently -- which is precisely what broke before: links were built from a
        bare slug while every filename carried a disambiguating id suffix.

        That suffix is required: distinct entities can slugify identically
        ("Foo Bar"/"Foo-Bar" -> "foo-bar", "C++"/"C" -> "c", non-ASCII "café"/"北京"
        -> "caf"/"unknown"). Without it the second god-node's article would overwrite
        the first under the same article_id and the first memory_id would become
        undiscoverable (GC + content-hash reuse are keyed on article_id too).
        """
        slug = self.renderer.slugify(self._entity_title(memory_id, mem))
        return f"entity-{slug}-{memory_id[:8]}"

    async def _generate_extra_articles(
        self,
        workspace_id: str,
        analysis: GraphAnalysis,
        community_by_id: dict[int, Community],
        force_regenerate: bool = False,
        contradicted_ids: set[str] | None = None,
        link_index: LinkIndex | None = None,
    ) -> list[Article]:
        """Additive article producers beyond communities / god-nodes / index.

        OSS returns none — this is the parity reference: the default KB is exactly
        communities + memory god-nodes + index. An enterprise subclass overrides
        this to add canonical-entity-registry ("proper entity graph") articles.
        Anything returned is stored and its id joins the run's
        ``current_article_ids`` before GC, so it is not reclaimed as stale.
        Overrides MUST be fail-safe: an exception here is caught in ``generate``
        and treated as "no extra articles" (OSS behavior preserved).
        """
        return []

    @staticmethod
    def _member_dict(mem) -> dict:
        """Shape a member Memory into the dict used for rendering + provenance.

        Carries content/type for the summary + renderer and the trust signals
        (importance, tags, source document) aggregated into article frontmatter.
        """
        return {
            "id": mem.id,
            "content": mem.content,
            "type": mem.type.value,
            "importance": getattr(mem, "importance", None),
            "tags": list(getattr(mem, "tags", None) or []),
            "source_document_id": getattr(mem, "source_document_id", None),
        }

    @staticmethod
    def _provenance_fields(members: list[dict], contradicted_ids: set[str] | None = None) -> dict:
        """Aggregate member trust/provenance signals for article YAML frontmatter.

        confidence = mean member importance; tags = union (capped); sources =
        distinct source-document ids (count + sample); contradicted_by = how many
        members are involved in an unresolved contradiction (when a contradiction
        set is supplied). Fields with no data are omitted so the frontmatter only
        carries what's actually known.
        """
        fields: dict = {}
        importances = [
            m["importance"] for m in members if isinstance(m.get("importance"), (int, float))
        ]
        if importances:
            fields["confidence"] = round(sum(importances) / len(importances), 3)
        tags = sorted({t for m in members for t in (m.get("tags") or []) if t})
        if tags:
            fields["tags"] = tags[:12]
        sources = sorted({m["source_document_id"] for m in members if m.get("source_document_id")})
        if sources:
            fields["source_count"] = len(sources)
            fields["sources"] = sources[:8]
        if contradicted_ids:
            n = sum(1 for m in members if m.get("id") in contradicted_ids)
            if n:
                fields["contradicted_by"] = n
        return fields

    async def _fetch_contradicted_ids(self, workspace_id: str) -> set[str]:
        """Memory ids involved in an UNRESOLVED contradiction (once per generate()).

        Soft + fail-safe: no contradiction service, an unsupported backend, or any
        error yields an empty set, so the ``contradicted_by`` frontmatter signal is
        simply absent (never blocks generation).
        """
        if self.contradiction_service is None:
            return set()
        try:
            records = await self.contradiction_service.get_unresolved(workspace_id, limit=1000)
        except NotImplementedError:
            return set()
        except Exception as e:  # noqa: BLE001 - provenance signal is best-effort
            self.logger.debug("Contradiction fetch failed for %s: %s", workspace_id, e)
            return set()
        ids: set[str] = set()
        for r in records or []:
            a = getattr(r, "memory_a_id", None)
            b = getattr(r, "memory_b_id", None)
            if a:
                ids.add(a)
            if b:
                ids.add(b)
        return ids

    def _content_version(self, mem) -> str:
        """Per-member content version for the article content-hash.

        Default mode ``updated_at`` uses the member's last-update timestamp (cheap, already
        loaded). Mode ``content`` hashes the member content (robust to clock skew / no-op
        touches). Any other value falls back to ``updated_at``.
        """
        if self.content_version_mode == "content":
            content = getattr(mem, "content", "") or ""
            return hashlib.sha256(content.encode("utf-8")).hexdigest()
        updated_at = getattr(mem, "updated_at", None)
        return updated_at.isoformat() if updated_at is not None else ""

    async def _maybe_reuse_article(
        self,
        workspace_id: str,
        article_id: str,
        content_key: str,
    ) -> Article | None:
        """Return the prior article verbatim if its stored content_key matches.

        A hit means the article's rendered inputs are unchanged, so the prior (already
        LLM-generated) article is reused with NO LLM call. Returns None on any miss/error.
        """
        try:
            prior_raw = await self.storage.get_kb_article(workspace_id, article_id)
        except NotImplementedError:
            return None
        except Exception as e:
            self.logger.debug("Could not load prior article %s for reuse: %s", article_id, e)
            return None

        if not prior_raw:
            return None
        prior_meta = prior_raw.get("metadata") or {}
        if prior_meta.get(kb_differ.CONTENT_KEY_FIELD) != content_key:
            return None
        self.logger.debug("Reusing unchanged KB article %s (content_key hit)", article_id)
        return self._raw_to_article(prior_raw)

    # Prompt for the community SUMMARY. Strict about output hygiene: some models
    # otherwise emit preamble / chain-of-thought ("The user wants...", "Let's
    # analyze...") that used to be stored verbatim as the article body.
    # One call returns "title\n\nsummary". The LLM gateway strips the model's
    # reasoning and returns clean content, so no preamble cleanup is needed — the
    # only requirement is a token budget large enough for a reasoning model to
    # finish before it's truncated (see MEMORYLAYER_KB_SUMMARY_MAX_TOKENS).
    _COMMUNITY_PROMPT = (
        "You are writing one section of a knowledge base from a cluster of related "
        "memories. Each memory below is numbered, e.g. [m1], [m2].\n"
        "Respond in EXACTLY this structure and nothing else:\n"
        "- First line: a short topic title, 3-6 words (no quotes, no trailing punctuation).\n"
        "- Then a blank line.\n"
        "- Then a concise 2-5 sentence summary of the common topic and key points, in "
        "neutral, factual, third person. Ground each claim by citing the reference "
        "number(s) it draws from in square brackets, e.g. [m1] or [m2][m5]. Only cite "
        "numbers shown in the list below; never invent a number.\n\n"
        "Memories:\n{memories}"
    )

    # Per-member content cap inside the budgeted member block.
    _MEMBER_SNIPPET_CHARS = 400

    def _included_members(self, members: list[dict]) -> list[dict]:
        """The members actually shown to the model, in citation order.

        Non-empty content only, prefix-truncated at ``self.summary_budget_chars``
        so a large cluster can't blow the context window. The i-th (1-based) entry
        is exactly what ``[m{i}]`` refers to — the single source of truth for both
        the numbered block and the ``member_ids`` citation map in article metadata.
        """
        included: list[dict] = []
        used = 0
        for m in members:
            content = (m.get("content") or "").strip()
            if not content:
                continue
            line_len = len(f"[m{len(included) + 1}] {content[: self._MEMBER_SNIPPET_CHARS]}")
            if included and used + line_len > self.summary_budget_chars:
                break
            included.append(m)
            used += line_len
        return included

    def member_citation_ids(self, members: list[dict]) -> list[str]:
        """Ordered member memory ids aligned to the ``[m1..mN]`` citation numbers.

        ``member_citation_ids(members)[n - 1]`` is the memory id cited by ``[m{n}]``
        (empty string when that member has no id, to preserve positional alignment),
        letting the UI turn a citation into a link to that memory.
        """
        return [str(m.get("id") or "") for m in self._included_members(members)]

    def _build_member_block(self, members: list[dict]) -> tuple[str, list[str], int]:
        """Build the numbered, budget-bounded member block for a summary prompt.

        Each member is rendered as ``[m<n>] <content up to _MEMBER_SNIPPET_CHARS>``
        with a CONSECUTIVE 1-based number (in ``_included_members`` order). The
        numbers let the model ground its claims reliably (small integers, unlike
        long opaque ids). Also returns short plain snippets for the deterministic
        fallback. Shared by community and (enterprise) entity summarization.

        Returns:
            (combined_block, snippets, member_count) — ``member_count`` is the
            number of members actually included (= the max valid reference ``[m{n}]``).
        """
        lines: list[str] = []
        snippets: list[str] = []
        for n, m in enumerate(self._included_members(members), start=1):
            content = (m.get("content") or "").strip()
            lines.append(f"[m{n}] {content[: self._MEMBER_SNIPPET_CHARS]}")
            snippets.append(content[:150])
        return "\n".join(lines), snippets, len(lines)

    async def _summarize_community(
        self,
        workspace_id: str,
        community: Community,
        members: list[dict],
    ) -> tuple[str, str, CitationReport]:
        """Generate a topic label + grounded summary for a community in ONE LLM call.

        Members are fed numbered (``[m1]``, ``[m2]`` …) within a char budget
        (deterministic prefix truncation, so a large community can't blow the
        context window), and the prompt asks the model to cite the reference numbers
        it uses. The gateway separates/strips reasoning so the content is clean as
        long as the token budget lets it finish. Falls back to a deterministic
        summary when no LLM is configured or the call fails.

        Returns:
            (label, summary, citation_report) — the report is audited against the
            number of members actually shown to the model.
        """
        if not members:
            return (
                f"Community {community.id}",
                f"Empty community with {community.size} members.",
                CitationReport(),
            )

        combined, snippets, member_count = self._build_member_block(members)

        def _fallback() -> tuple[str, str, CitationReport]:
            parts = [
                f"A cluster of {community.size} memories "
                f"(cohesion: {community.cohesion_score:.2f}).",
                "\n**Sample memories:**",
            ]
            parts.extend(f"- {s}" for s in snippets[:5])
            return f"Community {community.id}", "\n".join(parts), CitationReport()

        if not self.llm or not combined:
            return _fallback()

        try:
            raw = await self.llm.synthesize(
                prompt=self._COMMUNITY_PROMPT.format(memories=combined),
                max_tokens=self.summary_max_tokens,
                profile="reflection",
                activity=GenerationActivity.SYNTHESIS,
            )
        except LLMNotConfiguredError:
            return _fallback()
        except Exception as e:
            self.logger.debug("LLM community summarize failed for %d: %s", community.id, e)
            return _fallback()

        parsed = self._split_title_summary((raw or "").strip(), community.id)
        if parsed is None:
            return _fallback()
        label, summary = parsed
        return label, summary, audit_citations(summary, member_count)

    @staticmethod
    def _split_title_summary(text: str, community_id: int) -> tuple[str, str] | None:
        """Parse the model's ``title\\n\\nsummary`` response.

        Light structural parsing only — NOT reasoning cleanup (the gateway
        already returns clean content). Returns ``None`` for empty input so the
        caller falls back. If the model ignored the format (first line is a full
        sentence, not a short title), the whole text becomes the summary and a
        generic title is used.
        """
        if not text:
            return None
        first, _, rest = text.partition("\n")
        title = first.strip().strip("'\"").rstrip(".").strip("'\"").strip()
        summary = rest.strip() or text
        if len(title) > 80 or len(title.split()) > 12:
            title, summary = f"Community {community_id}", text
        return (title or f"Community {community_id}"), summary

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
        """Map an article to its path inside the vault zip.

        Delegates to the same placement rule the link index uses, so an exported
        filename and the wikilinks pointing at it cannot drift apart.
        """
        return article_vault_path(article.article_type, article.id) + ".md"


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

        # Optional: direct LLM handle for KB-specific community label/summary
        # generation (tight prompts, no reflect/recall). Best-effort — falls back
        # to a deterministic summary when absent.
        llm_service = None
        try:
            llm_service = get_extension(EXT_LLM_SERVICE, v)
        except Exception:
            logger.debug("LLMService not available for KnowledgebaseService")

        # Optional: contradiction service backs the contradicted_by frontmatter
        # signal. Absent -> the signal is simply omitted.
        contradiction_service = None
        try:
            contradiction_service = get_extension(EXT_CONTRADICTION_SERVICE, v)
        except Exception:
            logger.debug("ContradictionService not available for KnowledgebaseService")

        return DefaultKnowledgebaseService(
            storage=storage,
            graph_service=graph_service,
            reflect_service=reflect_service,
            inference_service=inference_service,
            llm_service=llm_service,
            contradiction_service=contradiction_service,
            v=v,
        )
