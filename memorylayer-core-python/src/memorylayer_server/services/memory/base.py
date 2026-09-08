from ...config import (
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
    DEFAULT_MEMORYLAYER_MEMORY_SERVICE,
    MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
    MEMORYLAYER_MEMORY_SERVICE,
)
from .._constants import (
    EXT_CACHE_SERVICE,
    EXT_CONTRADICTION_SERVICE,
    EXT_DEDUPLICATION_SERVICE,
    EXT_EMBEDDING_PROVIDER,
    EXT_EXTRACTION_SERVICE,
    EXT_MEMORY_SERVICE,
    EXT_RERANKER_SERVICE,
    EXT_SEMANTIC_TIERING_SERVICE,
    EXT_STORAGE_BACKEND,
)
from .._plugin_factory import make_service_plugin_base

# Recall overfetch multiplier for reranker candidate pool
MEMORYLAYER_MEMORY_RECALL_OVERFETCH = "MEMORYLAYER_MEMORY_RECALL_OVERFETCH"
DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH = 3

# Maximum memories discovered via association graph expansion
MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION = "MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION"
DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION = 50

# Default include_associations for recall (graph expansion enabled by default)
MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS = "MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS"
DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS = True

# Default traverse_depth for recall (multi-hop graph traversal)
MEMORYLAYER_MEMORY_TRAVERSE_DEPTH = "MEMORYLAYER_MEMORY_TRAVERSE_DEPTH"
DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH = 2

# Hybrid retrieval: fuse keyword (full-text/BM25) and vector results via
# Reciprocal Rank Fusion in the core recall path. On by default — lexical
# matching recovers exact/rare terms that pure vector similarity misses.
MEMORYLAYER_HYBRID_SEARCH_ENABLED = "MEMORYLAYER_HYBRID_SEARCH_ENABLED"
DEFAULT_MEMORYLAYER_HYBRID_SEARCH_ENABLED = True

# Reciprocal Rank Fusion constant. Larger values flatten the contribution of
# deep ranks; 60 is the canonical default from the RRF literature.
MEMORYLAYER_HYBRID_RRF_K = "MEMORYLAYER_HYBRID_RRF_K"
DEFAULT_MEMORYLAYER_HYBRID_RRF_K = 60

# Backlink salience: log-scale boost applied to recall scores based on a
# memory's incoming-association count (graph in-degree). Well-connected "hub"
# memories rank higher. 0.0 disables the boost.
# DEFAULT is 0.0: on a dense similar_to (embedding-KNN) graph, in-degree
# measures genericness, which is anti-correlated with relevance (LoCoMo
# 0.463->0.235 with boost on). Opt in by setting the env var to e.g. 0.2.
MEMORYLAYER_BACKLINK_BOOST_WEIGHT = "MEMORYLAYER_BACKLINK_BOOST_WEIGHT"
DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT = 0.0

# Alias hop: small multiplicative boost when a query term matches one of a
# memory's metadata aliases (metadata["aliases"]). Aliases are also folded into
# the full-text index so alias-only queries retrieve the canonical memory.
MEMORYLAYER_ALIAS_BOOST_WEIGHT = "MEMORYLAYER_ALIAS_BOOST_WEIGHT"
DEFAULT_MEMORYLAYER_ALIAS_BOOST_WEIGHT = 0.1

# MMR (Maximal Marginal Relevance) result diversification. When enabled, the
# final recall selection greedily trades a little relevance for diversity so the
# top-k is not spent on near-duplicate memories (see services/reranker/mmr.py).
# Ships DARK (default OFF) — mutually exclusive with the cross-encoder reranker
# in the current cut; validate on LoCoMo before enabling by default.
MEMORYLAYER_RERANK_MMR_ENABLED = "MEMORYLAYER_RERANK_MMR_ENABLED"
DEFAULT_MEMORYLAYER_RERANK_MMR_ENABLED = False
# Relevance/diversity balance in [0, 1]. 1.0 = pure relevance (no
# diversification); lower values diversify more aggressively.
MEMORYLAYER_RERANK_MMR_LAMBDA = "MEMORYLAYER_RERANK_MMR_LAMBDA"
DEFAULT_MEMORYLAYER_RERANK_MMR_LAMBDA = 0.5

# Query intent routing: rule-based (no LLM) classification of the query into
# entity/temporal/event/general, used to softly tune retrieval (recency,
# alias/backlink boosts, graph expansion) and apply an explicit-date window when
# one is confidently parsed. Distinct from LLM query rewrite (default off).
MEMORYLAYER_QUERY_INTENT_ENABLED = "MEMORYLAYER_QUERY_INTENT_ENABLED"
DEFAULT_MEMORYLAYER_QUERY_INTENT_ENABLED = True

# Query-aware association expansion. When enabled, graph-discovered (associated)
# memories are re-scored by their similarity to the *query* instead of
# parent_score * edge_strength * hop_decay. This lets expansion surface a
# relevant memory the vector arm missed while preventing topically-similar but
# off-query neighbors from displacing direct hits in the top-k (the dominant
# failure mode of naive similarity-edge expansion). Neighbors whose query
# similarity falls below MEMORYLAYER_ASSOC_QUERY_FLOOR are dropped entirely.
# Disabled by default to preserve legacy expansion behavior.
MEMORYLAYER_ASSOC_QUERY_AWARE = "MEMORYLAYER_ASSOC_QUERY_AWARE"
DEFAULT_MEMORYLAYER_ASSOC_QUERY_AWARE = False

MEMORYLAYER_ASSOC_QUERY_FLOOR = "MEMORYLAYER_ASSOC_QUERY_FLOOR"
DEFAULT_MEMORYLAYER_ASSOC_QUERY_FLOOR = 0.0

# A2: association consensus boost. When > 0, the association graph is used as a
# RANKING signal only — already-retrieved memories that are graph-connected to
# OTHER retrieved memories get a score boost, and NO new candidates are injected
# into the result pool. This tests whether the graph is useful as a consensus /
# mutual-support signal without the dilution that candidate injection causes.
# Mutually exclusive with expansion: when this weight is > 0 it replaces the
# graph-expansion step. 0.0 disables (legacy expansion path runs instead).
MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT = "MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT"
DEFAULT_MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT = 0.0

# Entity-anchored retrieval channel. When enabled, recall extracts entities from
# the query and fetches a candidate set restricted to memories whose speaker or
# entity set intersects those query entities, ranked by vector similarity, then
# RRF-fuses that ranked list with the global vector ranks (rank discipline: the
# channel can promote a gold memory the vector arm missed but cannot crowd out
# direct hits). Requires entity metadata populated at ingest. Default ON:
# validated on LoCoMo over the enterprise PG backend (real embeddings, dim 1960)
# with backlink boost at 0.0 — recall@5 hybrid 0.461 -> 0.536 (+16%), every
# question type improved (multi_hop +73%, open_domain +41%), zero regressions.
# Set to "false" to restore the pre-channel pure-hybrid behaviour.
MEMORYLAYER_ENTITY_ANCHOR_ENABLED = "MEMORYLAYER_ENTITY_ANCHOR_ENABLED"
DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_ENABLED = True

# Reciprocal Rank Fusion constant for the entity-anchored channel.
MEMORYLAYER_ENTITY_ANCHOR_RRF_K = "MEMORYLAYER_ENTITY_ANCHOR_RRF_K"
DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_RRF_K = 60

# Fact retrieval channel. When enabled, recall fetches a candidate set restricted
# to fact memories (subtype="fact") ranked by vector similarity and RRF-fuses that
# ranked list with the global vector ranks — modeled exactly on the entity-anchored
# channel (rank discipline: the channel can promote a gold fact the vector arm
# missed but cannot crowd out direct hits). The primary vector arm is kept PURE of
# fact memories so the date-grounded raw turns retain their temporal phrasing; only
# the fact arm sees facts. Default off to preserve production recall behaviour.
# This is the in-core form of a two-store RRF arrangement (separate turn and fact
# stores, fused at query time) that measurably improved multi-hop retrieval offline.
MEMORYLAYER_FACT_CHANNEL_ENABLED = "MEMORYLAYER_FACT_CHANNEL_ENABLED"
DEFAULT_MEMORYLAYER_FACT_CHANNEL_ENABLED = False
# NOTE — production ingest caveat: fact extraction+storage currently runs ONLY
# on the inline post-store path (_inline_auto_enrich).  The background
# auto_enrich task does NOT yet extract or store facts, so when a task service
# is configured, facts are populated only via inline ingest or an explicit
# ingest_fact call.  Enabling this channel without inline ingest active will
# yield an empty fact candidate pool.

# Reciprocal Rank Fusion constant for the fact channel.
MEMORYLAYER_FACT_CHANNEL_RRF_K = "MEMORYLAYER_FACT_CHANNEL_RRF_K"
DEFAULT_MEMORYLAYER_FACT_CHANNEL_RRF_K = 60

# Depth of the fact candidate pool (per query). Mirrors the entity-anchor pool
# size: the fact-restricted set is small, so a generous pool is cheap and lets the
# channel promote a gold fact that sits deep in the vector ranking.
MEMORYLAYER_FACT_CHANNEL_POOL = "MEMORYLAYER_FACT_CHANNEL_POOL"
DEFAULT_MEMORYLAYER_FACT_CHANNEL_POOL = 200

# Cue-anchor retrieval channel (Memora-inspired abstraction+cue indexing). A cue
# anchor is a short "[entity] + [aspect]" semantic key generated per memory and
# embedded (see ExtractionService.generate_cue_anchors). When enabled, recall
# searches the cue anchors by vector similarity, dereferences each hit back to
# its primary memory, and RRF-fuses that ranked list with the vector ranks —
# modeled exactly on the fact channel (rank discipline: the channel can promote a
# gold memory whose facet the query matches but that scores low on raw content
# similarity, without crowding out direct hits). This is a BOLT-ON arm: the
# primary content-embedding arm is untouched. Cue generation + storage is
# enterprise-backed (OSS storage/extraction base methods are no-ops), gated by
# this same flag so there is zero ingest cost when off. Ships DARK (default OFF)
# — byte-identical to today until a LoCoMo regression gate validates it.
MEMORYLAYER_CUE_CHANNEL_ENABLED = "MEMORYLAYER_CUE_CHANNEL_ENABLED"
DEFAULT_MEMORYLAYER_CUE_CHANNEL_ENABLED = False

# Reciprocal Rank Fusion constant for the cue channel.
MEMORYLAYER_CUE_CHANNEL_RRF_K = "MEMORYLAYER_CUE_CHANNEL_RRF_K"
DEFAULT_MEMORYLAYER_CUE_CHANNEL_RRF_K = 60

# Depth of the cue candidate pool (per query). Mirrors the fact/entity-anchor
# pool sizing: the cue-anchor set is small, so a generous pool is cheap and lets
# the channel promote a gold memory whose cue sits deep in the ranking.
MEMORYLAYER_CUE_CHANNEL_POOL = "MEMORYLAYER_CUE_CHANNEL_POOL"
DEFAULT_MEMORYLAYER_CUE_CHANNEL_POOL = 50

# Graph-traversal recall channel (P4 channel G). When enabled AND a graph query
# service is wired, recall resolves the query's entities to canonical registry
# entities (allow_create=False — a query string must NEVER create an entity) and,
# for each resolved entity, pulls a SMALL, BOUNDED neighborhood of MENTIONED
# memories via ``GraphQueryService.entity_neighborhood(..., memory_limit=SMALL)``,
# ranks them by vector similarity, and RRF-fuses that ranked list with the global
# vector ranks (rank discipline: it can promote a graph-adjacent gold memory the
# vector arm missed but cannot crowd out direct hits).
#
# This is the cross-source / relationship-traversal channel over the canonical
# entity layer — NOT a chat-recall expansion. The killed registry-recall flood
# (LoCoMo 0.54 -> 0.13) came from dumping ~all turns mentioning an entity into the
# pool; this channel MUST NOT repeat that. The neighborhood ``memory_limit`` is
# therefore the recall overfetch budget (``limit * recall_overfetch``), capped at
# MEMORYLAYER_GRAPH_RECALL_POOL — a SMALL bound, never 100/unbounded. See
# ``_fuse_graph_results``: the bound is the anti-flood guardrail.
#
# Tier note: the OSS relational ``entity_neighborhood`` (over the registry tables)
# proves the channel; the enterprise AGE backend answers it richly over the C1
# Entity/MENTIONS graph. Ships DARK (default OFF): byte-identical to today until a
# LoCoMo regression gate validates the bounded pool.
MEMORYLAYER_GRAPH_RECALL_ENABLED = "MEMORYLAYER_GRAPH_RECALL_ENABLED"
DEFAULT_MEMORYLAYER_GRAPH_RECALL_ENABLED = False

# Reciprocal Rank Fusion constant for the graph-traversal channel.
MEMORYLAYER_GRAPH_RECALL_RRF_K = "MEMORYLAYER_GRAPH_RECALL_RRF_K"
DEFAULT_MEMORYLAYER_GRAPH_RECALL_RRF_K = 60

# Hard upper bound on the graph neighborhood candidate pool (per query). The
# anti-flood ceiling: the per-entity ``memory_limit`` passed to
# ``entity_neighborhood`` is ``min(limit * recall_overfetch, this)``, so even a
# huge requested limit cannot turn the graph arm into the killed member-dump.
# Mirrors the fact/entity-anchor pool sizing — small and cheap.
MEMORYLAYER_GRAPH_RECALL_POOL = "MEMORYLAYER_GRAPH_RECALL_POOL"
DEFAULT_MEMORYLAYER_GRAPH_RECALL_POOL = 50

# Query-intent-driven recall channel selection (P4.1). When enabled, the
# classified query intent selects WHICH of the already-enabled fuse arms fire on
# a given query (a per-intent channel matrix), instead of every enabled arm
# firing unconditionally. Channel selection only SUBTRACTS arms — it never adds
# candidates, never changes how arms fuse, and never touches pool purity. The
# existing per-channel flags remain a GLOBAL KILL-SWITCH ABOVE intent: a
# flag-off channel never fires regardless of what the matrix selects.
#
# Default OFF: when off, behavior is byte-identical to today (all enabled arms
# fire), so this is a strict no-op until validated. Cardinal safety rule — any
# unmatched intent, missing intent, or low-confidence classification degrades to
# the FULL default arm set (today's behavior), never to fewer arms; the worst
# case is "same as today", never a regression. The LoCoMo regression gate
# (recall@5 >= 0.536 with no per-category regression) must hold before flipping
# the default on.
MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED = "MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED"
DEFAULT_MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED = False


# noinspection PyAbstractClass
MemoryServicePluginBase = make_service_plugin_base(
    ext_name=EXT_MEMORY_SERVICE,
    config_key=MEMORYLAYER_MEMORY_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_MEMORY_SERVICE,
    dependencies=(
        EXT_STORAGE_BACKEND,
        EXT_EMBEDDING_PROVIDER,
        EXT_CACHE_SERVICE,
        EXT_SEMANTIC_TIERING_SERVICE,
        EXT_DEDUPLICATION_SERVICE,
        EXT_RERANKER_SERVICE,
        EXT_CONTRADICTION_SERVICE,
        EXT_EXTRACTION_SERVICE,
    ),
    extra_defaults={
        MEMORYLAYER_MEMORY_RECALL_OVERFETCH: DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
        MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION: DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
        MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS: DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
        MEMORYLAYER_MEMORY_TRAVERSE_DEPTH: DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
        MEMORYLAYER_HYBRID_SEARCH_ENABLED: DEFAULT_MEMORYLAYER_HYBRID_SEARCH_ENABLED,
        MEMORYLAYER_HYBRID_RRF_K: DEFAULT_MEMORYLAYER_HYBRID_RRF_K,
        MEMORYLAYER_BACKLINK_BOOST_WEIGHT: DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
        MEMORYLAYER_ALIAS_BOOST_WEIGHT: DEFAULT_MEMORYLAYER_ALIAS_BOOST_WEIGHT,
        MEMORYLAYER_RERANK_MMR_ENABLED: DEFAULT_MEMORYLAYER_RERANK_MMR_ENABLED,
        MEMORYLAYER_RERANK_MMR_LAMBDA: DEFAULT_MEMORYLAYER_RERANK_MMR_LAMBDA,
        MEMORYLAYER_QUERY_INTENT_ENABLED: DEFAULT_MEMORYLAYER_QUERY_INTENT_ENABLED,
        MEMORYLAYER_ASSOC_QUERY_AWARE: DEFAULT_MEMORYLAYER_ASSOC_QUERY_AWARE,
        MEMORYLAYER_ASSOC_QUERY_FLOOR: DEFAULT_MEMORYLAYER_ASSOC_QUERY_FLOOR,
        MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT: DEFAULT_MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
        MEMORYLAYER_ENTITY_ANCHOR_ENABLED: DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
        MEMORYLAYER_ENTITY_ANCHOR_RRF_K: DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_RRF_K,
        MEMORYLAYER_FACT_CHANNEL_ENABLED: DEFAULT_MEMORYLAYER_FACT_CHANNEL_ENABLED,
        MEMORYLAYER_FACT_CHANNEL_RRF_K: DEFAULT_MEMORYLAYER_FACT_CHANNEL_RRF_K,
        MEMORYLAYER_FACT_CHANNEL_POOL: DEFAULT_MEMORYLAYER_FACT_CHANNEL_POOL,
        # Cue-anchor retrieval channel (Memora-inspired) ships DARK (default OFF).
        MEMORYLAYER_CUE_CHANNEL_ENABLED: DEFAULT_MEMORYLAYER_CUE_CHANNEL_ENABLED,
        MEMORYLAYER_CUE_CHANNEL_RRF_K: DEFAULT_MEMORYLAYER_CUE_CHANNEL_RRF_K,
        MEMORYLAYER_CUE_CHANNEL_POOL: DEFAULT_MEMORYLAYER_CUE_CHANNEL_POOL,
        MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED: DEFAULT_MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
        # Graph-traversal recall channel (P4 channel G) ships DARK (default OFF).
        MEMORYLAYER_GRAPH_RECALL_ENABLED: DEFAULT_MEMORYLAYER_GRAPH_RECALL_ENABLED,
        MEMORYLAYER_GRAPH_RECALL_RRF_K: DEFAULT_MEMORYLAYER_GRAPH_RECALL_RRF_K,
        MEMORYLAYER_GRAPH_RECALL_POOL: DEFAULT_MEMORYLAYER_GRAPH_RECALL_POOL,
        # Entity registry accretion ships DARK (default OFF).
        MEMORYLAYER_ENTITY_REGISTRY_ENABLED: DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
        # Registry-backed entity-expansion recall channel ships DARK (default OFF).
        MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED: DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
        MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K: DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
        MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL: DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
        MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE: DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
    },
)
