"""
Memory Service - Core business logic for memory operations.

Operations:
- remember: Store new memory with automatic embedding and classification
- recall: Query memories with vector search and optional LLM enhancement
- forget: Soft or hard delete memories
- decay: Reduce memory importance over time
- get: Retrieve single memory by ID
"""

import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import DEBUG, Logger
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from scitrera_app_framework import Variables, ext_parse_bool, get_extension, get_logger

from ...models import (
    DetailLevel,
    Memory,
    MemoryMutation,
    MemoryMutationResult,
    MemoryReplaceInput,
    MemoryRevision,
    MemoryScope,
    MemoryStatus,
    MemorySubtype,
    MemoryType,
    RecallInput,
    RecallMode,
    RecallResult,
    RememberInput,
    SearchTolerance,
)
from ...models.generation import (
    EnrichmentPolicy,
    GenerationActivity,
    GenerationAuthorization,
    GenerationBudgetExceededError,
    GenerationNotAllowedError,
    GenerationSummary,
)
from ...models.versioned_resource import (
    VersionedResourceConflictError,
    VersionedResourceNotFoundError,
)
from ...utils import compute_content_hash, cosine_similarity, generate_id, to_utc_iso
from .._constants import EXT_ENTITY_REGISTRY_SERVICE, EXT_GRAPH_QUERY_SERVICE, EXT_METRICS_SERVICE, EXT_TASK_SERVICE
from ..cache import EXT_CACHE_SERVICE
from ..contradiction import EXT_CONTRADICTION_SERVICE, ContradictionService
from ..decay import EXT_DECAY_SERVICE, DecayService
from ..deduplication import EXT_DEDUPLICATION_SERVICE, DeduplicationAction, DeduplicationService
from ..embedding import EXT_EMBEDDING_SERVICE, EmbeddingService
from ..entity_relation import KNOWLEDGE_WORK_METADATA_KEY, EntityRelationService
from ..extraction import (
    EXT_EXTRACTION_SERVICE,
    ExtractionService,
    extract_marker_candidates,
    segment_content,
)
from ..extraction import (
    classify_content as deterministic_classify_content,
)
from ..llm import EXT_LLM_SERVICE, LLMService
from ..reranker import EXT_RERANKER_SERVICE, RerankerService
from ..reranker.mmr import mmr_select
from ..semantic_tiering import EXT_SEMANTIC_TIERING_SERVICE, SemanticTieringService
from ..storage import EXT_STORAGE_BACKEND, StorageBackend, StorageCapabilityError
from ..versioned_resources import VersionedResourceService
from .budget import pack_recall_memories
from .confidence import retrieval_confidence
from .scope_classifier import HeuristicScopeClassifier, ScopeClassification
from .versioning import canonical_hash

if TYPE_CHECKING:
    from ..tasks import TaskService

from ...config import (
    DEFAULT_CONTEXT_ID,
    DEFAULT_MEMORYLAYER_ENRICH_ON_MERGE,
    DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    DEFAULT_MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
    DEFAULT_MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
    DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_MODE,
    DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_PENALTY,
    DEFAULT_MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT,
    DEFAULT_MEMORYLAYER_RELATIONAL_RECALL_ENABLED,
    DEFAULT_MEMORYLAYER_RERANK_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED,
    DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
    DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
    DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_LOOSE,
    DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_MODERATE,
    DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_STRICT,
    DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED,
    DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD,
    DEFAULT_MEMORYLAYER_USER_SCOPE_SUBTYPES,
    DEFAULT_RECENCY_HALF_LIFE_HOURS,
    DEFAULT_RECENCY_WEIGHT,
    GLOBAL_USER_WORKSPACE_ID,
    GLOBAL_WORKSPACE_ID,
    MEMORYLAYER_ENRICH_ON_MERGE,
    MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
    MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
    MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
    MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
    MEMORYLAYER_RECALL_SUPERSESSION_MODE,
    MEMORYLAYER_RECALL_SUPERSESSION_PENALTY,
    MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT,
    MEMORYLAYER_RELATIONAL_RECALL_ENABLED,
    MEMORYLAYER_RERANK_MAX_TOKENS,
    MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED,
    MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
    MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
    MEMORYLAYER_TOLERANCE_FLOOR_LOOSE,
    MEMORYLAYER_TOLERANCE_FLOOR_MODERATE,
    MEMORYLAYER_TOLERANCE_FLOOR_STRICT,
    MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED,
    MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD,
    MEMORYLAYER_USER_SCOPE_SUBTYPES,
    SUPERSESSION_MODE_DEMOTE,
    SUPERSESSION_MODE_EXCLUDE,
    SUPERSESSION_MODE_OFF,
)
from ..association import (
    DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    EXT_ASSOCIATION_SERVICE,
    MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD,
    AssociationService,
)
from .base import (
    DEFAULT_MEMORYLAYER_ALIAS_BOOST_WEIGHT,
    DEFAULT_MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
    DEFAULT_MEMORYLAYER_ASSOC_QUERY_AWARE,
    DEFAULT_MEMORYLAYER_ASSOC_QUERY_FLOOR,
    DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
    DEFAULT_MEMORYLAYER_CUE_CHANNEL_ENABLED,
    DEFAULT_MEMORYLAYER_CUE_CHANNEL_POOL,
    DEFAULT_MEMORYLAYER_CUE_CHANNEL_RRF_K,
    DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_RRF_K,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
    DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
    DEFAULT_MEMORYLAYER_FACT_CHANNEL_ENABLED,
    DEFAULT_MEMORYLAYER_FACT_CHANNEL_POOL,
    DEFAULT_MEMORYLAYER_FACT_CHANNEL_RRF_K,
    DEFAULT_MEMORYLAYER_GRAPH_RECALL_ENABLED,
    DEFAULT_MEMORYLAYER_GRAPH_RECALL_POOL,
    DEFAULT_MEMORYLAYER_GRAPH_RECALL_RRF_K,
    DEFAULT_MEMORYLAYER_HYBRID_RRF_K,
    DEFAULT_MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    DEFAULT_MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
    DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
    DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
    DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
    DEFAULT_MEMORYLAYER_QUERY_INTENT_ENABLED,
    DEFAULT_MEMORYLAYER_RERANK_MMR_ENABLED,
    DEFAULT_MEMORYLAYER_RERANK_MMR_LAMBDA,
    MEMORYLAYER_ALIAS_BOOST_WEIGHT,
    MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
    MEMORYLAYER_ASSOC_QUERY_AWARE,
    MEMORYLAYER_ASSOC_QUERY_FLOOR,
    MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
    MEMORYLAYER_CUE_CHANNEL_ENABLED,
    MEMORYLAYER_CUE_CHANNEL_POOL,
    MEMORYLAYER_CUE_CHANNEL_RRF_K,
    MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
    MEMORYLAYER_ENTITY_ANCHOR_RRF_K,
    MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
    MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
    MEMORYLAYER_FACT_CHANNEL_ENABLED,
    MEMORYLAYER_FACT_CHANNEL_POOL,
    MEMORYLAYER_FACT_CHANNEL_RRF_K,
    MEMORYLAYER_GRAPH_RECALL_ENABLED,
    MEMORYLAYER_GRAPH_RECALL_POOL,
    MEMORYLAYER_GRAPH_RECALL_RRF_K,
    MEMORYLAYER_HYBRID_RRF_K,
    MEMORYLAYER_HYBRID_SEARCH_ENABLED,
    MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
    MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
    MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
    MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
    MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
    MEMORYLAYER_QUERY_INTENT_ENABLED,
    MEMORYLAYER_RERANK_MMR_ENABLED,
    MEMORYLAYER_RERANK_MMR_LAMBDA,
    MemoryServicePluginBase,
)
from .entities import extract_entities, extract_query_entities
from .query_intent import ENTITY, EVENT, GENERAL, TEMPORAL, QueryIntent, classify_query_intent

# Intent-routing nudge magnitudes (soft routing; see _route_by_intent)
_INTENT_TEMPORAL_RECENCY_WEIGHT = 0.5  # recency weight applied to temporal/event queries when caller left it unset
_INTENT_ENTITY_BOOST_MULTIPLIER = 1.5  # amplifies alias/backlink boosts for entity queries

# Recall fuse-arm channel labels (P4.1 intent channel selection). These name the
# fuse arms that the per-intent matrix can gate. A and B are the proven backbone
# (dense vector + keyword/BM25) and appear in every selected set; D and E are the
# entity-anchor and fact arms. Channels G/C/F are intentionally NOT wired here
# (later slices). The labels match the architecture doc's arm IDs.
_CHANNEL_DENSE = "A"  # primary dense vector arm (always fires)
_CHANNEL_KEYWORD = "B"  # keyword / BM25 arm (always fires)
_CHANNEL_ENTITY = "D"  # entity-anchored arm (_fuse_entity_results)
_CHANNEL_FACT = "E"  # fact arm (_fuse_fact_results)
_CHANNEL_GRAPH = "G"  # graph-traversal arm (_fuse_graph_results); cross-source/relationship-traversal

# The full default arm set = today's behavior (every wired arm fires). This is the
# safe-degrade target: any uncertain/unknown/missing intent falls back to this so
# the worst case is "same as today", never fewer arms than today.
#
# G (graph-traversal) is intentionally NOT a member of this set, nor of the
# per-intent matrix below. The proven backbone arms (A/B/D/E) are governed by the
# channel selector; the dark graph arm is governed instead by its OWN flag plus a
# SEPARATE permitted-intent set (``_GRAPH_CHANNEL_INTENTS``) consulted only when
# intent-channel-select is ON. Keeping G out of these sets preserves the exact
# P4.1 selection contract (A/B/D/E) byte-for-byte.
_CHANNEL_FULL_DEFAULT = frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT})

# Intents for which the graph-traversal arm (G) is on-task when intent-channel-
# select is ON: the cross-source / relationship-traversal cases (entity/event).
# Consulted ONLY by the graph arm's call-site gate (see _recall_rag); G's own flag
# (default OFF) remains the master gate above this.
_GRAPH_CHANNEL_INTENTS = frozenset({ENTITY, EVENT})

# Per-intent channel matrix (P4.1). Maps a recognized query-intent label to the
# set of fuse arms permitted to fire for that intent. Dense (A) + keyword (B) are
# the proven backbone and are present in every set. Entity/event queries also get
# the entity-anchor (D) and fact (E) arms. GENERAL (the rule-based classifier's
# catch-all) KEEPS the entity-anchor (D): the P4.1 LoCoMo flip-gate (2026-06-04)
# showed the classifier dumps ~41/50 multi_hop queries into `general`, and routing
# them to A+B only dropped D — the very arm delivering the validated +16% multi_hop
# lift — craterng multi_hop −0.127. D is cheap and broadly helpful, so it stays in
# the catch-all; only the fact arm (E) is subtracted from temporal/general.
# Readable + easy to tune; only SUBTRACTS arms from today's full default set.
#
# NOTE: this is consulted only when the intent-channel-select flag is ON (default
# OFF — the gate found channel-selection is at best recall-neutral on LoCoMo, so it
# stays dark; this matrix only ensures it is non-harmful if enabled). A query may
# carry MULTIPLE intent labels — the selected set is their UNION. Any label absent
# from this matrix degrades the whole query to _CHANNEL_FULL_DEFAULT (safe-degrade).
_INTENT_CHANNEL_MATRIX: dict[str, frozenset[str]] = {
    ENTITY: frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT}),
    EVENT: frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY, _CHANNEL_FACT}),
    TEMPORAL: frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD}),
    GENERAL: frozenset({_CHANNEL_DENSE, _CHANNEL_KEYWORD, _CHANNEL_ENTITY}),
}

# Internal constant for LLM recall token budget
_INTERNAL_LLM_RECALL_TOKEN_BUDGET = 2048

# Depth of the entity-anchored candidate pool (per query). The entity-restricted
# set is small, so a generous pool is cheap and lets the channel promote a gold
# memory that sits deep in the vector ranking (the offline lab fuses against the
# full per-entity ranking; this mirrors that depth instead of the reranker overfetch).
_ENTITY_ANCHOR_POOL = 200

# A relation path requires both a recognized structural intent and an exact
# canonical entity seed.  Give that inspectable signal enough weight to beat a
# vector-only rank-1 result; equal-weight RRF can merely tie when the evidence
# memory was outside the vector pool, leaving a UUID tie-break to decide rank 1.
_RELATION_RECALL_RRF_WEIGHT = 2.0


@dataclass
class ScopeBoosts:
    """Configuration for locality-based score boosting."""

    same_context: float = 1.5  # 50% boost for same context
    same_workspace: float = 1.2  # 20% boost for same workspace
    global_workspace: float = 1.0  # No boost for global


class MemoryService:
    """
    Core memory service implementing remember/recall/forget operations.

    This service coordinates between:
    - Storage backend (PostgreSQL or SQLite)
    - Embedding service (for vector generation)
    - LLM service (for query rewriting and re-ranking)
    - Cache (for recent memories)
    """

    @staticmethod
    def _exponential_freshness(age: float, half_life: float) -> float:
        """Compute exponential decay factor for freshness/recency scoring.

        Args:
            age: Age in the same time unit as half_life (hours, days, etc.)
            half_life: Time until the factor reaches 0.5

        Returns:
            Decay factor in (0.0, 1.0]
        """
        return math.exp(-math.log(2) * age / half_life)

    def _generation_allowed(self, activity: GenerationActivity) -> bool:
        """Preserve compatibility with lightweight test and plugin LLM facades."""
        if self.llm_service is None:
            return True
        checker = getattr(self.llm_service, "is_generation_allowed", None)
        return checker(activity) if checker is not None else True

    def __init__(
        self,
        storage: StorageBackend,
        embedding_service: EmbeddingService,
        deduplication_service: DeduplicationService,
        association_service: AssociationService | None = None,
        cache: Any | None = None,
        v: Variables = None,
        tier_generation_service: SemanticTieringService | None = None,
        llm_service: LLMService | None = None,
        reranker_service: RerankerService | None = None,
        decay_service: DecayService | None = None,
        contradiction_service: ContradictionService | None = None,
        task_service: Optional["TaskService"] = None,
        extraction_service: ExtractionService | None = None,
        entity_registry_service: Any | None = None,
        graph_query_service: Any | None = None,
    ):
        self.storage = storage
        self.embedding = embedding_service
        self.deduplication = deduplication_service
        self.association_service = association_service
        self.cache = cache
        self.tier_generation_service = tier_generation_service
        self.llm_service = llm_service
        self.reranker_service = reranker_service
        self.decay_service = decay_service
        self.contradiction_service = contradiction_service
        self.task_service = task_service
        self.extraction_service = extraction_service
        self.entity_registry_service = entity_registry_service
        self.graph_query_service = graph_query_service
        self.entity_relation_service = EntityRelationService(
            storage,
            entity_registry=entity_registry_service,
        )
        self.v = v
        self.logger = get_logger(v, name=self.__class__.__name__)

        # Get auto-association threshold from config
        self.auto_association_threshold = v.get(
            MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD, DEFAULT_MEMORYLAYER_ASSOCIATION_SIMILARITY_THRESHOLD
        )

        # Fact decomposition config
        self.fact_decomposition_enabled = v.get(
            MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
            DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_ENABLED,
        )
        # Whether a merge re-runs enrichment on the surviving memory.
        self.enrich_on_merge = v.get(
            MEMORYLAYER_ENRICH_ON_MERGE,
            DEFAULT_MEMORYLAYER_ENRICH_ON_MERGE,
        )
        self.fact_decomposition_min_length = v.get(
            MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
            DEFAULT_MEMORYLAYER_FACT_DECOMPOSITION_MIN_LENGTH,
        )
        # Completion cap for the LLM rerank (comma-separated indices); env-tunable.
        self.rerank_max_tokens = v.get(
            MEMORYLAYER_RERANK_MAX_TOKENS,
            DEFAULT_MEMORYLAYER_RERANK_MAX_TOKENS,
        )

        # User-scope subtype mapping: when on, PREFERENCE/DIRECTIVE memories are
        # auto-routed to user scope (the _global_user workspace) on remember()
        # even without an explicit scope. An explicit scope=USER always wins;
        # this knob only governs the implicit subtype-based mapping. Read via
        # v.environ so env overrides are type-coerced correctly.
        self.user_scope_subtypes_enabled = v.environ(
            MEMORYLAYER_USER_SCOPE_SUBTYPES,
            default=DEFAULT_MEMORYLAYER_USER_SCOPE_SUBTYPES,
            type_fn=ext_parse_bool,
        )

        # Slice 2: automatic preference-vs-episodic classification. KNOB 1 is the
        # master enable (default OFF — opt-in; a misclassified episodic memory
        # wrongly follows the user everywhere). When OFF, the classifier never
        # runs (zero added latency, byte-identical to Slice 1). KNOB 2 is the
        # confidence floor: only promote to USER scope when the classifier's
        # confidence meets it. See config.py for full knob docs.
        self.user_scope_autoclassify_enabled = v.environ(
            MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED,
            default=DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.user_scope_autoclassify_threshold = v.environ(
            MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD,
            default=DEFAULT_MEMORYLAYER_USER_SCOPE_AUTOCLASSIFY_THRESHOLD,
            type_fn=float,
        )
        # OSS tier: conservative deterministic heuristic. The enterprise memory
        # service overrides ``_classify_user_scope`` with an LLM-backed,
        # prompt-injection-hardened classifier; this attribute is the OSS
        # default that hook uses. Instantiated unconditionally (cheap, no
        # network); it only runs when the master knob above is on.
        self.scope_classifier = HeuristicScopeClassifier()

        # Recall overfetch multiplier for reranker candidate pool
        self.recall_overfetch = v.get(
            MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
            DEFAULT_MEMORYLAYER_MEMORY_RECALL_OVERFETCH,
        )

        # What recall does with memories a later memory has superseded.
        self._supersession_mode = str(v.get(MEMORYLAYER_RECALL_SUPERSESSION_MODE, DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_MODE)).lower()
        if self._supersession_mode not in (SUPERSESSION_MODE_OFF, SUPERSESSION_MODE_DEMOTE, SUPERSESSION_MODE_EXCLUDE):
            self.logger.warning(
                "Unknown %s value %r; supersession disabled. Valid: %s, %s, %s",
                MEMORYLAYER_RECALL_SUPERSESSION_MODE,
                self._supersession_mode,
                SUPERSESSION_MODE_OFF,
                SUPERSESSION_MODE_DEMOTE,
                SUPERSESSION_MODE_EXCLUDE,
            )
            self._supersession_mode = SUPERSESSION_MODE_OFF
        self._supersession_penalty = float(v.get(MEMORYLAYER_RECALL_SUPERSESSION_PENALTY, DEFAULT_MEMORYLAYER_RECALL_SUPERSESSION_PENALTY))

        # Maximum memories discovered via association graph expansion
        self.max_graph_expansion = v.get(
            MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
            DEFAULT_MEMORYLAYER_MEMORY_MAX_GRAPH_EXPANSION,
        )

        # Default graph traversal settings for recall
        self.default_include_associations = v.get(
            MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
            DEFAULT_MEMORYLAYER_MEMORY_INCLUDE_ASSOCIATIONS,
        )
        self.default_traverse_depth = v.get(
            MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
            DEFAULT_MEMORYLAYER_MEMORY_TRAVERSE_DEPTH,
        )

        # Query-aware association expansion: re-score graph-discovered neighbors
        # by their similarity to the query (see _expand_with_associations).
        # Read via v.environ so env overrides are type-coerced correctly.
        self.assoc_query_aware = v.environ(
            MEMORYLAYER_ASSOC_QUERY_AWARE,
            default=DEFAULT_MEMORYLAYER_ASSOC_QUERY_AWARE,
            type_fn=ext_parse_bool,
        )
        self.assoc_query_floor = v.environ(
            MEMORYLAYER_ASSOC_QUERY_FLOOR,
            default=DEFAULT_MEMORYLAYER_ASSOC_QUERY_FLOOR,
            type_fn=float,
        )
        # A2: consensus boost weight — when > 0, use the graph as a ranking-only
        # signal (boost connected retrieved memories, inject nothing).
        self.assoc_consensus_boost_weight = v.environ(
            MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
            default=DEFAULT_MEMORYLAYER_ASSOC_CONSENSUS_BOOST_WEIGHT,
            type_fn=float,
        )

        # Hybrid retrieval: fuse keyword (full-text/BM25) + vector via RRF
        self.hybrid_search_enabled = v.get(
            MEMORYLAYER_HYBRID_SEARCH_ENABLED,
            DEFAULT_MEMORYLAYER_HYBRID_SEARCH_ENABLED,
        )
        self.hybrid_rrf_k = v.get(
            MEMORYLAYER_HYBRID_RRF_K,
            DEFAULT_MEMORYLAYER_HYBRID_RRF_K,
        )

        # Backlink salience: log-scale boost by incoming-association count.
        # Read via v.environ(type_fn=float) so an env-string override is coerced
        # (a bare v.get returned the raw string -> crashed the recall-time float
        # comparison).
        self.backlink_boost_weight = v.environ(
            MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
            default=DEFAULT_MEMORYLAYER_BACKLINK_BOOST_WEIGHT,
            type_fn=float,
        )

        # Alias hop: boost when a query term matches a memory's metadata aliases.
        # Same env-string coercion as backlink above.
        self.alias_boost_weight = v.environ(
            MEMORYLAYER_ALIAS_BOOST_WEIGHT,
            default=DEFAULT_MEMORYLAYER_ALIAS_BOOST_WEIGHT,
            type_fn=float,
        )

        # MMR result diversification (default OFF). When enabled, the final recall
        # selection greedily trades relevance for diversity to avoid near-duplicate
        # top-k results. Read via v.environ so env overrides are type-coerced.
        self.rerank_mmr_enabled = v.environ(
            MEMORYLAYER_RERANK_MMR_ENABLED,
            default=DEFAULT_MEMORYLAYER_RERANK_MMR_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.rerank_mmr_lambda = v.environ(
            MEMORYLAYER_RERANK_MMR_LAMBDA,
            default=DEFAULT_MEMORYLAYER_RERANK_MMR_LAMBDA,
            type_fn=float,
        )

        # Query intent routing (rule-based, no LLM)
        self.query_intent_enabled = v.get(
            MEMORYLAYER_QUERY_INTENT_ENABLED,
            DEFAULT_MEMORYLAYER_QUERY_INTENT_ENABLED,
        )

        # Query-intent-driven recall channel selection (P4.1). When enabled, the
        # classified intent selects WHICH already-enabled fuse arms fire per query
        # (see _select_channels / _INTENT_CHANNEL_MATRIX). Default OFF -> every
        # enabled arm fires (byte-identical to today). Read via v.environ so env
        # overrides are type-coerced.
        self.intent_channel_select_enabled = v.environ(
            MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
            default=DEFAULT_MEMORYLAYER_INTENT_CHANNEL_SELECT_ENABLED,
            type_fn=ext_parse_bool,
        )

        # Entity-anchored retrieval channel: RRF-fuse entity-restricted candidates
        # (memories whose speaker/entities intersect the query entities) with the
        # vector arm. Read via v.environ so env overrides are type-coerced.
        self.entity_anchor_enabled = v.environ(
            MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
            default=DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.entity_anchor_rrf_k = v.environ(
            MEMORYLAYER_ENTITY_ANCHOR_RRF_K,
            default=DEFAULT_MEMORYLAYER_ENTITY_ANCHOR_RRF_K,
            type_fn=int,
        )

        # Entity registry accretion (entity registry slice 1). Ships DARK
        # (default OFF). When enabled and a registry service is wired, ingest-time
        # enrichment resolves extracted entity strings to canonical entities and
        # accretes member rows. Read via v.environ so env overrides coerce.
        self.entity_registry_enabled = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
            default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_ENABLED,
            type_fn=ext_parse_bool,
        )

        # Registry-backed entity-expansion recall channel (the registry's
        # RETRIEVAL consumer; distinct from the accretion flag above). Ships DARK
        # (default OFF). When enabled and a registry service is wired, recall
        # resolves the query's entities to canonical entities (allow_create=False)
        # and RRF-fuses each entity's member memories as an additive arm —
        # surfacing alias/merge-linked members the metadata entity-anchor misses.
        self.entity_registry_recall_enabled = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
            default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.entity_registry_recall_rrf_k = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
            default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_RRF_K,
            type_fn=int,
        )
        self.entity_registry_recall_pool = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
            default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_POOL,
            type_fn=int,
        )
        # Query-relevance floor for registry members (anti-flood). Members below
        # this cosine similarity to the query are dropped before RRF fusion.
        self.entity_registry_recall_min_relevance = v.environ(
            MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
            default=DEFAULT_MEMORYLAYER_ENTITY_REGISTRY_RECALL_MIN_RELEVANCE,
            type_fn=float,
        )

        # Graph-traversal recall channel (P4 channel G). Ships DARK (default OFF).
        # When enabled AND a graph_query_service is wired, recall resolves the
        # query's entities to canonical entities and RRF-fuses a SMALL, BOUNDED
        # neighborhood of MENTIONED memories (via entity_neighborhood) as an
        # additive arm. The pool bound (graph_recall_pool) is the anti-flood
        # guardrail — see _fuse_graph_results. Read via v.environ so env overrides
        # coerce.
        self.graph_recall_enabled = v.environ(
            MEMORYLAYER_GRAPH_RECALL_ENABLED,
            default=DEFAULT_MEMORYLAYER_GRAPH_RECALL_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.graph_recall_rrf_k = v.environ(
            MEMORYLAYER_GRAPH_RECALL_RRF_K,
            default=DEFAULT_MEMORYLAYER_GRAPH_RECALL_RRF_K,
            type_fn=int,
        )
        self.graph_recall_pool = v.environ(
            MEMORYLAYER_GRAPH_RECALL_POOL,
            default=DEFAULT_MEMORYLAYER_GRAPH_RECALL_POOL,
            type_fn=int,
        )

        # Fact retrieval channel: RRF-fuse a fact-restricted candidate set
        # (subtype="fact" memories ranked by vector similarity) with the vector
        # arm, while keeping the primary vector arm pure of facts. Modeled exactly
        # on the entity-anchored channel. Read via v.environ so env overrides are
        # type-coerced.
        self.fact_channel_enabled = v.environ(
            MEMORYLAYER_FACT_CHANNEL_ENABLED,
            default=DEFAULT_MEMORYLAYER_FACT_CHANNEL_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.fact_channel_rrf_k = v.environ(
            MEMORYLAYER_FACT_CHANNEL_RRF_K,
            default=DEFAULT_MEMORYLAYER_FACT_CHANNEL_RRF_K,
            type_fn=int,
        )
        self.fact_channel_pool = v.environ(
            MEMORYLAYER_FACT_CHANNEL_POOL,
            default=DEFAULT_MEMORYLAYER_FACT_CHANNEL_POOL,
            type_fn=int,
        )

        # Cue-anchor retrieval channel (Memora-inspired abstraction+cue indexing):
        # RRF-fuse a candidate set reached via cue-anchor vector similarity
        # (dereferenced back to the primary memories) with the vector arm. Bolt-on
        # arm — the primary content-embedding arm is untouched. Cue generation +
        # storage is enterprise-backed (OSS base methods are no-ops), gated by the
        # same flag so there is zero cost when off. Read via v.environ so env
        # overrides are type-coerced. Ships DARK (default OFF).
        self.cue_channel_enabled = v.environ(
            MEMORYLAYER_CUE_CHANNEL_ENABLED,
            default=DEFAULT_MEMORYLAYER_CUE_CHANNEL_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.cue_channel_rrf_k = v.environ(
            MEMORYLAYER_CUE_CHANNEL_RRF_K,
            default=DEFAULT_MEMORYLAYER_CUE_CHANNEL_RRF_K,
            type_fn=int,
        )
        self.cue_channel_pool = v.environ(
            MEMORYLAYER_CUE_CHANNEL_POOL,
            default=DEFAULT_MEMORYLAYER_CUE_CHANNEL_POOL,
            type_fn=int,
        )

        # LLM query rewriting config
        self.llm_query_rewrite_enabled = v.get(
            MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
            DEFAULT_MEMORYLAYER_LLM_QUERY_REWRITE_ENABLED,
        )
        self.default_recall_token_budget = v.environ(
            MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT,
            default=DEFAULT_MEMORYLAYER_RECALL_TOKEN_BUDGET_DEFAULT,
            type_fn=int,
        )
        self.retrieval_confidence_enabled = v.environ(
            MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED,
            default=DEFAULT_MEMORYLAYER_RETRIEVAL_CONFIDENCE_ENABLED,
            type_fn=ext_parse_bool,
        )
        self.relational_recall_enabled = v.environ(
            MEMORYLAYER_RELATIONAL_RECALL_ENABLED,
            default=DEFAULT_MEMORYLAYER_RELATIONAL_RECALL_ENABLED,
            type_fn=ext_parse_bool,
        )

        # Freshness annotation config
        self.freshness_half_life_days = v.get(
            MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
            DEFAULT_MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS,
        )

        # Configurable scope boosts
        self.default_scope_boosts = ScopeBoosts(
            same_context=v.get(
                MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
                DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_CONTEXT,
            ),
            same_workspace=v.get(
                MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
                DEFAULT_MEMORYLAYER_SCOPE_BOOST_SAME_WORKSPACE,
            ),
        )

        # Best-effort metrics handle so swallowed post-store sub-step failures
        # become observable (silent degradation -> a real counter). Resolved
        # lazily and tolerant of a missing/unwired metrics extension; None means
        # we fall back to a stable WARNING log event only.
        try:
            self.metrics = get_extension(EXT_METRICS_SERVICE, v) if v is not None else None
        except Exception:
            self.metrics = None
        self.entity_relation_service.metrics = self.metrics

        # Server-configurable per-tolerance relevance floors (see
        # _get_relevance_threshold). Defaults match the historical hardcoded
        # values, so behavior is unchanged unless an operator overrides them.
        self.tolerance_floors = {
            SearchTolerance.STRICT: v.get(
                MEMORYLAYER_TOLERANCE_FLOOR_STRICT,
                DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_STRICT,
            ),
            SearchTolerance.MODERATE: v.get(
                MEMORYLAYER_TOLERANCE_FLOOR_MODERATE,
                DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_MODERATE,
            ),
            SearchTolerance.LOOSE: v.get(
                MEMORYLAYER_TOLERANCE_FLOOR_LOOSE,
                DEFAULT_MEMORYLAYER_TOLERANCE_FLOOR_LOOSE,
            ),
        }

        self.logger.info(
            "Initialized MemoryService (auto_association_threshold=%.2f, fact_decomposition=%s, recall_overfetch=%s, max_graph_expansion=%s, include_associations=%s, traverse_depth=%s)",
            self.auto_association_threshold,
            self.fact_decomposition_enabled,
            self.recall_overfetch,
            self.max_graph_expansion,
            self.default_include_associations,
            self.default_traverse_depth,
        )

    def _recall_cache_key(self, workspace_id: str, query: str, input: RecallInput) -> str:
        """Generate a deterministic cache key for recall results.

        Every result-affecting field MUST be part of the key. Pagination
        (``offset``), the temporal window/order (``event_after``/``event_before``/
        ``time_order``), the creation-time window, the global-scope toggles,
        ``min_relevance``, and the user/entity scoping all change the returned
        set; omitting any of them returns a stale cached result for a different
        request (e.g. paging or a time window silently ignored on a cache hit).
        """

        def _iso(dt) -> str | None:
            return dt.isoformat() if dt is not None and hasattr(dt, "isoformat") else (None if dt is None else str(dt))

        filter_data = json.dumps(
            {
                "types": [t.value if hasattr(t, "value") else str(t) for t in (input.types or [])],
                "subtypes": [s.value if hasattr(s, "value") else str(s) for s in (input.subtypes or [])],
                "tags": sorted(input.tags or []),
                "mode": input.mode.value if input.mode and hasattr(input.mode, "value") else str(input.mode),
                "tolerance": input.tolerance.value if input.tolerance and hasattr(input.tolerance, "value") else str(input.tolerance),
                "limit": input.limit,
                "offset": input.offset,
                "context_id": input.context_id,
                "user_id": input.user_id,
                "observer_id": input.observer_id,
                "subject_id": input.subject_id,
                "include_global": input.include_global,
                "include_global_user": input.include_global_user,
                "min_relevance": input.min_relevance,
                "recency_weight": input.recency_weight,
                "include_associations": input.include_associations,
                "traverse_depth": input.traverse_depth,
                "max_expansion": input.max_expansion,
                "rag_threshold": input.rag_threshold,
                "created_after": _iso(input.created_after),
                "created_before": _iso(input.created_before),
                "event_after": _iso(input.event_after),
                "event_before": _iso(input.event_before),
                "time_order": input.time_order,
                "include_archived": input.include_archived,
                "exclude_ids": sorted(input.exclude_ids or []),
                "detail_level": input.detail_level.value
                if input.detail_level and hasattr(input.detail_level, "value")
                else str(input.detail_level),
                "budget_tokens": input.budget_tokens,
                "include_confidence": input.include_confidence,
                "include_relations": input.include_relations,
            },
            sort_keys=True,
        )
        hash_input = f"{query}|{filter_data}"
        key_hash = compute_content_hash(hash_input)[:16]
        return f"recall:{workspace_id}:{key_hash}"

    async def _classify_user_scope(self, content: str) -> ScopeClassification:
        """Classify whether ``content`` is a durable user preference/trait.

        OSS tier: delegate to the conservative deterministic heuristic. The
        enterprise memory service OVERRIDES this method with a prompt-injection-
        hardened, LLM-backed classifier. This is the single seam enterprise has
        to override to "make it good"; the routing logic in
        ``_route_user_scope`` stays shared.

        CARDINAL: this method MUST NOT raise. Any failure must degrade to a
        non-preference verdict (the heuristic already self-guards). The caller
        treats a non-preference / sub-threshold verdict as "keep workspace".
        """
        try:
            return await self.scope_classifier.classify(content)
        except Exception:  # noqa: BLE001 - classification must never break the write
            self.logger.warning(
                "Scope classifier raised unexpectedly; defaulting to workspace scope (fail-safe)",
                exc_info=True,
            )
            return ScopeClassification(False, 0.0, "classifier exception (fail-safe)")

    # noinspection PyShadowingBuiltins
    async def _route_user_scope(
        self,
        workspace_id: str,
        input: RememberInput,
        user_id: str | None,
    ) -> tuple[str, RememberInput]:
        """Resolve the effective storage scope for a remember() call.

        Cross-workspace USER-scope memory. Decides whether a memory should land
        in its origin workspace (WORKSPACE scope, today's behavior) or in the
        user-scoped global workspace (USER scope, partitioned by user_id and
        recalled via RecallInput.include_global_user).

        Precedence (scope is TRI-STATE; see RememberInput.scope):
          1. Explicit ``input.scope == USER`` always wins (-> USER).
          2. Explicit ``input.scope == WORKSPACE`` always wins (-> WORKSPACE);
             bypasses both the subtype map and the classifier.
          3. Scope UNSET (None): if the subtype-mapping knob is on, the
             user-global subtypes (PREFERENCE, DIRECTIVE) map to USER scope.
          4. Scope UNSET and subtype did not decide: if the autoclassify knob
             (Slice 2) is on, run the preference-vs-episodic classifier; route
             to USER only when it returns is_user_preference=True with
             confidence >= the threshold knob AND a user_id is available.
          5. Otherwise WORKSPACE scope (unchanged).

        When USER scope resolves, a user_id is REQUIRED: the cross-user read
        boundary is the forced user_id filter on the recall fan-out, so a
        user-scope row with no user_id would be unfilterable (visible to every
        user). An explicit ``scope=USER`` with no user_id raises; an IMPLICIT
        promotion (subtype map OR classifier) with no user_id is scoped DOWN to
        the origin workspace (no silent unfilterable global write).

        FAIL-SAFE (Slice 2 cardinal): the classifier never raises and never
        blocks the write; on any uncertainty/error the memory stays
        workspace-scoped. The classifier is gated behind the master knob, so
        when the knob is off there is ZERO added latency.

        Returns the (possibly rewritten) workspace_id and a RememberInput with
        the effective user_id stamped and ``metadata['origin_workspace_id']``
        recording provenance. WORKSPACE scope returns the inputs unchanged.
        """
        # Already in the user bucket (e.g. a caller passing workspace_id
        # explicitly): nothing to route, leave as-is.
        if workspace_id == GLOBAL_USER_WORKSPACE_ID:
            return workspace_id, input

        # Explicit WORKSPACE: caller opted out of any auto-routing. Bypass the
        # subtype map and the classifier entirely (explicit intent wins).
        if input.scope == MemoryScope.WORKSPACE:
            return workspace_id, input

        explicit_user = input.scope == MemoryScope.USER
        scope_unset = input.scope is None
        subtype_user = (
            scope_unset
            and self.user_scope_subtypes_enabled
            and input.subtype in (MemorySubtype.PREFERENCE.value, MemorySubtype.DIRECTIVE.value)
        )

        # Slice 2: automatic classification. ONLY when scope is unset, the
        # subtype map did not already decide, and the master knob is on. Gated
        # so a disabled classifier adds zero latency. The classifier decision is
        # purely advisory toward USER scope; the user_id guard below still
        # applies (an unfilterable global row is never written).
        classified_user = False
        if scope_unset and not subtype_user and self.user_scope_autoclassify_enabled:
            classification = await self._classify_user_scope(input.content)
            if classification.is_user_preference and classification.confidence >= self.user_scope_autoclassify_threshold:
                classified_user = True
                self.logger.debug(
                    "Autoclassify -> user preference (confidence=%.3f >= %.3f): %s",
                    classification.confidence,
                    self.user_scope_autoclassify_threshold,
                    classification.reason,
                )
            else:
                self.logger.debug(
                    "Autoclassify -> keep workspace (is_pref=%s confidence=%.3f, threshold=%.3f): %s",
                    classification.is_user_preference,
                    classification.confidence,
                    self.user_scope_autoclassify_threshold,
                    classification.reason,
                )

        if not (explicit_user or subtype_user or classified_user):
            return workspace_id, input

        effective_user_id = user_id or input.user_id
        if not effective_user_id:
            if explicit_user:
                # Hard reject: caller explicitly asked for user scope but gave
                # us no user_id to partition/filter on.
                raise ValueError("scope=USER requires a user_id; refusing to write an unfilterable user-global memory")
            # Implicit promotion (subtype map OR classifier) with no user_id:
            # scope down to the origin workspace rather than writing an
            # unfilterable global row. A false-negative here is cheap.
            self.logger.debug(
                "User-scope promotion skipped (subtype=%s, classified=%s) — no user_id; storing in origin workspace %s",
                input.subtype,
                classified_user,
                workspace_id,
            )
            return workspace_id, input

        # Route to the user bucket. Preserve provenance and stamp the effective
        # user_id so the forced read filter keeps this memory user-private.
        routed_metadata = dict(input.metadata or {})
        routed_metadata.setdefault("origin_workspace_id", workspace_id)
        routed_input = input.model_copy(update={"user_id": effective_user_id, "metadata": routed_metadata})
        self.logger.info(
            "Routing memory to user scope: origin_workspace=%s -> %s, user_id=%s, subtype=%s, classified=%s",
            workspace_id,
            GLOBAL_USER_WORKSPACE_ID,
            effective_user_id,
            input.subtype,
            classified_user,
        )
        return GLOBAL_USER_WORKSPACE_ID, routed_input

    async def remember(
        self,
        workspace_id: str,
        input: RememberInput,
        user_id: str | None = None,
        inline: bool = False,
    ) -> Memory:
        """
        Store a new memory.

        Hot path (always synchronous):
        1. Generate content hash and embedding
        2. Check for duplicates (SKIP/UPDATE/MERGE/CREATE)
        3. Classify memory type if not provided
        4. Store in backend

        Post-store (conditional on decomposition):
        - If decomposable: schedule fact decomposition (background by default).
          The decomposition handler owns the per-fact pipeline for each
          extracted fact (dedup, store, associate, contradict, tier-gen).
        - If not decomposable: run post-store pipeline directly
          (cache invalidation, tier gen, contradiction check, auto-association).

        Args:
            workspace_id: Target workspace
            input: Memory content and metadata
            user_id: Optional user ID override
            inline: If True, run all post-store work synchronously including
                    fact decomposition. Default False (background/eventual).
        """
        # User-scope routing (Slice 1). Resolve BEFORE the content hash / dedup /
        # storage calls below, since those are all workspace-scoped: routing a
        # memory to _global_user means dedup must run against that workspace, not
        # the origin. May rewrite workspace_id, stamp the effective user_id onto
        # the input, and record origin_workspace_id in metadata for provenance.
        workspace_id, input = await self._route_user_scope(workspace_id, input, user_id)

        # Explicit relation input is a transactional capability requirement.
        # Reject it before embedding, deduplication, or durable memory creation
        # instead of storing a memory and only then discovering that the backend
        # cannot persist its relation evidence.
        metadata_relations_requested = KNOWLEDGE_WORK_METADATA_KEY in (input.metadata or {})
        if (input.relations or metadata_relations_requested) and not self.storage.supports_capability("entity_relations"):
            raise StorageCapabilityError("entity_relations")

        self.logger.info(
            "Storing memory in workspace: %s, type: %s, content length: %s",
            workspace_id,
            input.type,
            len(input.content),
        )

        # 1. Generate content hash
        content_hash = compute_content_hash(input.content)

        # 2. Generate embedding (needed for deduplication)
        start_time = datetime.now(UTC)
        embedding = await self.embedding.embed(input.content)
        if self.logger.isEnabledFor(DEBUG):
            self.logger.debug("Generated embedding in %s ms", (datetime.now(UTC) - start_time).total_seconds() * 1000)

        # 3. Check for duplicates using DeduplicationService
        dedup_result = await self.deduplication.check_duplicate(
            content=input.content, content_hash=content_hash, embedding=embedding, workspace_id=workspace_id
        )

        if dedup_result.action == DeduplicationAction.SKIP:
            # Exact duplicate found, return existing memory
            self.logger.info("Found duplicate memory: %s (%s)", dedup_result.existing_memory_id, dedup_result.reason)
            existing = await self.storage.get_memory(workspace_id, dedup_result.existing_memory_id, track_access=False)
            return await self._write_entity_relations(existing, input.relations)

        elif dedup_result.action == DeduplicationAction.UPDATE:
            # Semantic duplicate found, update existing memory
            self.logger.info("Updating existing memory: %s (%s)", dedup_result.existing_memory_id, dedup_result.reason)
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=input.content,
                embedding=embedding,
                importance=max(input.importance, 0.5),  # Boost importance on update
            )
            return await self._write_entity_relations(updated, input.relations)

        elif dedup_result.action == DeduplicationAction.MERGE:
            self.logger.info("Merging with existing memory: %s (%s)", dedup_result.existing_memory_id, dedup_result.reason)
            existing = await self.storage.get_memory(workspace_id, dedup_result.existing_memory_id, track_access=False)
            updated = await self._merge_memories(workspace_id, existing, input.content, input.tags, input.metadata, input.importance)
            return await self._write_entity_relations(updated, input.relations)

        # 4. No duplicate - proceed with creating new memory
        self.logger.debug("Creating new memory (%s)", dedup_result.reason)

        # 5. Track whether type was auto-classified (for LLM reclassification later)
        type_was_auto = input.type is None

        # 6. Classify memory type if not provided
        classification = deterministic_classify_content(
            input.content,
            explicit_type=input.type,
            explicit_subtype=input.subtype,
        )
        memory_type = classification.memory_type
        self.logger.debug(
            "Deterministically classified memory type: %s (%s)",
            memory_type,
            classification.confidence,
        )

        # 7. Create memory object with generated fields.
        # Perspective seed: derive observer_id from speaker when not explicitly
        # set. Done inline here so the value is persisted in the single
        # storage.create_memory() call below (no second update needed).
        seeded_observer_id, seeded_subject_id = self._seed_perspective_ids(
            content=input.content,
            observer_id=input.observer_id,
            subject_id=input.subject_id,
        )
        memory_data = RememberInput(
            content=input.content,
            tenant_id=input.tenant_id,
            logical_key=input.logical_key,
            type=memory_type,
            subtype=input.subtype,
            importance=input.importance,
            tags=input.tags,
            metadata={
                **self._maybe_add_entity_metadata(input.content, input.metadata),
                "deterministic_classification": {
                    "confidence": classification.confidence,
                    "matched_rules": list(classification.matched_rules),
                    "explicit": classification.explicitly_classified,
                },
            },
            refinement_metadata=input.refinement_metadata,
            associations=input.associations,
            context_id=input.context_id,
            user_id=user_id or input.user_id,
            pinned=input.pinned,
            observer_id=seeded_observer_id,
            subject_id=seeded_subject_id,
            event_time=input.event_time,
            # Carry source provenance forward so out-of-band producers (document
            # ingest, the email adapter) keep their document/page/dataset/thread
            # linkage through the shared remember() path instead of losing it on
            # the RememberInput reconstruction above.
            source_document_id=input.source_document_id,
            source_page_id=input.source_page_id,
            source_dataset_id=input.source_dataset_id,
            source_thread_id=input.source_thread_id,
            session_id=input.session_id,
            source_memory_id=input.source_memory_id,
            relations=input.relations,
        )

        # Store in backend (backend will create Memory object)
        memory = await self.storage.create_memory(workspace_id, memory_data)

        # Update with embedding (if backend doesn't handle it)
        if memory.embedding is None:
            memory = await self.storage.update_memory(
                workspace_id,
                memory.id,
                embedding=embedding,
            )

        self.logger.info("Stored memory: %s", memory.id)

        # Post-store: conditional on decomposition. Delegated to the shared
        # enqueue_post_store hook so every producer of a stored memory (remember()
        # here, and out-of-band producers such as document ingestion) runs the
        # identical decompose-or-enrich lifecycle.
        await self.enqueue_post_store(
            workspace_id,
            memory,
            embedding,
            inline=inline,
            classify_type=type_was_auto,
        )
        return await self._write_entity_relations(memory, input.relations)

    async def _write_entity_relations(self, memory: Memory, relations) -> Memory:
        report = await self.entity_relation_service.write_for_memory(
            memory,
            list(relations or []),
            include_patterns=True,
        )
        if any((report.resolved, report.unresolved, report.rejected, report.duplicate)):
            return memory.model_copy(update={"relation_write_result": report})
        return memory

    async def remember_versioned(
        self,
        workspace_id: str,
        input: RememberInput,
        *,
        tenant_id: str,
        user_id: str | None,
        operation_id: str,
        expected_etag: str,
    ) -> MemoryMutationResult:
        """Create a keyed semantic memory with exact retry and CAS semantics."""

        if not input.logical_key:
            raise ValueError("conditional memory create requires logical_key")
        if input.type is None:
            raise ValueError("conditional memory create requires an explicit type")
        if input.associations:
            raise ValueError(
                "conditional memory create does not accept associations; attach child relationships after the memory is committed"
            )

        workspace_id, routed = await self._route_user_scope(
            workspace_id,
            input.model_copy(update={"tenant_id": tenant_id}),
            user_id,
        )
        metadata_relations_requested = KNOWLEDGE_WORK_METADATA_KEY in (routed.metadata or {})
        if (routed.relations or metadata_relations_requested) and not self.storage.supports_capability("entity_relations"):
            raise StorageCapabilityError("entity_relations")
        request_hash = canonical_hash(
            {
                "action": "create",
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "input": routed.model_dump(mode="json"),
                "expected_etag": expected_etag,
            }
        )
        if replay := await self.storage.get_memory_operation(tenant_id, workspace_id, operation_id, request_hash):
            return replay

        embedding = await self.embedding.embed(routed.content)
        observer_id, subject_id = self._seed_perspective_ids(
            content=routed.content,
            observer_id=routed.observer_id,
            subject_id=routed.subject_id,
        )
        now = datetime.now(UTC)
        memory = Memory(
            id=generate_id("mem"),
            logical_key=routed.logical_key,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            context_id=routed.context_id or DEFAULT_CONTEXT_ID,
            user_id=user_id or routed.user_id,
            observer_id=observer_id,
            subject_id=subject_id,
            content=routed.content,
            content_hash=compute_content_hash(routed.content),
            type=routed.type,
            subtype=routed.subtype,
            importance=routed.importance,
            tags=routed.tags,
            metadata=self._maybe_add_entity_metadata(routed.content, routed.metadata),
            refinement_metadata=routed.refinement_metadata,
            embedding=embedding,
            pinned=routed.pinned,
            session_id=routed.session_id,
            source_memory_id=routed.source_memory_id,
            source_document_id=routed.source_document_id,
            source_page_id=routed.source_page_id,
            source_dataset_id=routed.source_dataset_id,
            source_thread_id=routed.source_thread_id,
            event_time=routed.event_time,
            created_at=now,
            updated_at=now,
        )
        result = await self.storage.mutate_memory(
            MemoryMutation(
                action="create",
                memory=memory,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )
        if not result.replayed:
            await self.enqueue_post_store(
                workspace_id,
                result.memory,
                embedding,
                inline=False,
                classify_type=False,
            )
        related = await self._write_entity_relations(result.memory, routed.relations)
        return result.model_copy(update={"memory": related})

    async def replace_versioned(
        self,
        workspace_id: str,
        memory_id: str,
        input: MemoryReplaceInput,
        *,
        tenant_id: str,
        operation_id: str,
        expected_etag: str,
    ) -> MemoryMutationResult:
        """Replace the complete refinement-owned semantic memory document."""

        request_hash = canonical_hash(
            {
                "action": "replace",
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "memory_id": memory_id,
                "input": input.model_dump(mode="json"),
                "expected_etag": expected_etag,
            }
        )
        if replay := await self.storage.get_memory_operation(tenant_id, workspace_id, operation_id, request_hash):
            return replay
        current = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
        if current is None or current.tenant_id != tenant_id or not current.logical_key:
            raise VersionedResourceNotFoundError("keyed memory not found")
        embedding = await self.embedding.embed(input.content)
        desired = current.model_copy(
            update={
                **input.model_dump(),
                "content_hash": compute_content_hash(input.content),
                "embedding": embedding,
                "updated_at": datetime.now(UTC),
            }
        )
        result = await self.storage.mutate_memory(
            MemoryMutation(
                action="replace",
                memory=desired,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )
        if not result.replayed:
            await self._reconcile_fts_index(workspace_id, memory_id)
        return result

    async def delete_versioned(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        tenant_id: str,
        operation_id: str,
        expected_etag: str,
    ) -> MemoryMutationResult:
        return await self._change_versioned_deleted_state(
            action="delete",
            workspace_id=workspace_id,
            memory_id=memory_id,
            tenant_id=tenant_id,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )

    async def restore_versioned(
        self,
        workspace_id: str,
        memory_id: str,
        *,
        tenant_id: str,
        operation_id: str,
        expected_etag: str,
    ) -> MemoryMutationResult:
        return await self._change_versioned_deleted_state(
            action="restore",
            workspace_id=workspace_id,
            memory_id=memory_id,
            tenant_id=tenant_id,
            operation_id=operation_id,
            expected_etag=expected_etag,
        )

    async def _change_versioned_deleted_state(
        self,
        *,
        action: str,
        workspace_id: str,
        memory_id: str,
        tenant_id: str,
        operation_id: str,
        expected_etag: str,
    ) -> MemoryMutationResult:
        request_hash = canonical_hash(
            {
                "action": action,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "memory_id": memory_id,
                "expected_etag": expected_etag,
            }
        )
        if replay := await self.storage.get_memory_operation(tenant_id, workspace_id, operation_id, request_hash):
            return replay
        current = await self.storage.get_memory(
            workspace_id,
            memory_id,
            track_access=False,
            include_deleted=True,
        )
        if current is None or current.tenant_id != tenant_id or not current.logical_key:
            raise VersionedResourceNotFoundError("keyed memory not found")
        if action == "restore" and current.deleted_at is None:
            raise VersionedResourceConflictError("memory is not deleted")
        now = datetime.now(UTC)
        desired = current.model_copy(
            update={
                "updated_at": now,
                "deleted_at": None if action == "restore" else now,
            }
        )
        result = await self.storage.mutate_memory(
            MemoryMutation(
                action=action,
                memory=desired,
                operation_id=operation_id,
                request_hash=request_hash,
                expected_etag=expected_etag,
            )
        )
        if action == "restore" and not result.replayed:
            # Delete/replace deactivates evidence in the storage transaction.
            # A restored metadata-first memory still carries its authoritative
            # source profile, so deterministically replay it to reactivate the
            # same idempotent evidence rows (SQLite and PostgreSQL both support
            # inactive-evidence reactivation in upsert_entity_relation).
            restored = await self._write_entity_relations(result.memory, [])
            result = result.model_copy(update={"memory": restored})
        return result

    async def list_revision_page(
        self,
        tenant_id: str,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int,
        page_token: str | None = None,
    ) -> tuple[list[MemoryRevision], str | None]:
        cursor_scope = VersionedResourceService.cursor_scope("memory-revisions", tenant_id, workspace_id, memory_id)
        before = VersionedResourceService.decode_cursor(page_token, cursor_scope) if page_token else None
        revisions = await self.storage.list_memory_revisions(
            tenant_id,
            workspace_id,
            memory_id,
            limit=limit + 1,
            before_sequence=before,
        )
        has_more = len(revisions) > limit
        revisions = revisions[:limit]
        next_token = VersionedResourceService.encode_cursor(revisions[-1].sequence, cursor_scope) if has_more else None
        return revisions, next_token

    async def enqueue_post_store(
        self,
        workspace_id: str,
        memory: Memory,
        embedding: list[float],
        *,
        job_id: str | None = None,
        inline: bool = False,
        classify_type: bool = False,
        force_decompose: bool = False,
        decompose: bool | None = None,
    ) -> bool:
        """Run the post-store lifecycle for ANY producer of a stored memory.

        This is the shared hook extracted from remember()'s post-store branch.
        If the memory will be decomposed, only decomposition is dispatched -- the
        decomposition handler owns the per-fact pipeline. Otherwise the post-store
        pipeline runs directly on this memory.

        Decomposition dispatch is a plain ``schedule_task("decompose_facts", ...)``
        so it works with every task provider (in-memory / asyncio / Aether) and
        never depends on a workflow-engine rule (no OSS regression).

        Args:
            workspace_id: Target workspace.
            memory: The already-stored memory.
            embedding: Pre-computed embedding vector for ``memory``.
            job_id: Optional ingestion/job correlation id, threaded into the
                ``decompose_facts`` payload for downstream completion events.
                ``None`` for the standalone remember() path.
            inline: If True, run all post-store work synchronously including
                fact decomposition.
            classify_type: If True, request LLM-based type reclassification in
                the auto-enrich task (only honored on the non-decompose branch).
            decompose: Tri-state opt-out. ``None`` (the default) leaves the
                decision to the usual heuristic; ``False`` suppresses
                decomposition for this memory. Deliberately NOT a bool: callers
                pass a per-document flag that is frequently unset, and an unset
                flag must mean "no opinion" rather than "off". ``True`` is
                accepted but does not bypass the content-shape checks — use
                ``force_decompose`` for that.

        Returns:
            True when background fact decomposition was SCHEDULED and is still
            outstanding. This is the authoritative answer to "will this memory
            be decomposed?" — the decision depends on the enabled flag, content
            shape and memory type, so callers that need it (document ingestion,
            to track its knowledge phase) must take it from here rather than
            re-deriving the heuristic and drifting from it. False when nothing
            was scheduled, and also for the inline path, which has already
            finished decomposing by the time it returns.
        """
        # ``decompose`` is a TRI-STATE, not a boolean: None means "no opinion --
        # apply the usual heuristic", which is what every existing caller passes
        # implicitly. Only an explicit False suppresses, so an unset per-document
        # flag can never be mistaken for a decision to turn decomposition off.
        # ``force_decompose`` remains the one-way override in the other
        # direction (visual-only pages, whose real content is the page image and
        # which therefore fail the text heuristic).
        generation_allowed = self._generation_allowed(GenerationActivity.FACT_DECOMPOSITION)
        if decompose is False or not generation_allowed:
            should_decompose = False
        else:
            should_decompose = self._should_decompose(
                memory.content,
                memory.type,
                force=force_decompose,
            )

        if should_decompose:
            if inline:
                await self._decompose_and_process_inline(workspace_id, memory, embedding)
                # Already done, so nothing is outstanding for a caller to track.
                return False
            elif self.task_service:
                try:
                    await self.task_service.schedule_task(
                        "decompose_facts",
                        {"memory_id": memory.id, "workspace_id": workspace_id, "job_id": job_id},
                    )
                    self.logger.debug("Scheduled fact decomposition for memory %s", memory.id)
                    return True
                except Exception as e:
                    self.logger.warning(
                        "Failed to schedule decomposition for %s, running post-store on composite: %s",
                        memory.id,
                        e,
                    )
                    # Fallback: run post-store pipeline on composite
                    await self._post_store_pipeline(workspace_id, memory, embedding, inline=False)
            else:
                # No task service and not inline — run post-store on composite as fallback
                await self._post_store_pipeline(workspace_id, memory, embedding, inline=False)
        else:
            await self._post_store_pipeline(workspace_id, memory, embedding, inline=inline, classify_type=classify_type)
            if not generation_allowed:
                await self._store_deterministic_segments(workspace_id, memory)
        # Every remaining path ran the composite pipeline instead of scheduling
        # decomposition, so there is no outstanding decomposition to wait on.
        return False

    async def _store_deterministic_segments(self, workspace_id: str, memory: Memory) -> list[Memory]:
        """Persist source-linked episodic retrieval fragments without inventing facts."""
        segments = segment_content(
            memory.content,
            max_chars=max(256, self.fact_decomposition_min_length),
        )
        markers = extract_marker_candidates(memory.content)
        existing = await self.storage.list_source_memories(workspace_id, memory.id)
        existing_by_key = {
            (
                child.metadata.get("deterministic_kind", "segment"),
                child.metadata.get("source_span_start"),
                child.metadata.get("source_span_end"),
                " ".join(child.content.casefold().split()),
            ): child
            for child in existing
        }
        created: list[Memory] = []
        # A single segment would duplicate the raw source without adding a useful
        # retrieval boundary. Marker candidates remain useful even for short raw
        # records and are handled below.
        for segment in segments if len(segments) > 1 else []:
            key = (
                "segment",
                segment.source_span_start,
                segment.source_span_end,
                " ".join(segment.content.casefold().split()),
            )
            if key in existing_by_key:
                child = existing_by_key[key]
                if child.embedding is None:
                    embedding = await self.embedding.embed(child.content)
                    await self.storage.update_memory(workspace_id, child.id, embedding=embedding)
                continue
            metadata = {
                **(memory.metadata or {}),
                "deterministic_kind": "segment",
                "segmentation_method": segment.method,
                "source_span_start": segment.source_span_start,
                "source_span_end": segment.source_span_end,
            }
            child = await self.storage.create_memory(
                workspace_id,
                RememberInput(
                    content=segment.content,
                    type=MemoryType.EPISODIC,
                    subtype=memory.subtype,
                    importance=memory.importance,
                    tags=memory.tags,
                    metadata=metadata,
                    context_id=memory.context_id,
                    tenant_id=memory.tenant_id,
                    user_id=memory.user_id,
                    observer_id=memory.observer_id,
                    subject_id=memory.subject_id,
                    session_id=memory.session_id,
                    source_memory_id=memory.id,
                    event_time=memory.event_time,
                ),
            )
            embedding = await self.embedding.embed(segment.content)
            updated = await self.storage.update_memory(
                workspace_id,
                child.id,
                embedding=embedding,
            )
            created.append(updated or child)
            existing_by_key[key] = updated or child
        marker_subtypes = {
            "decision": MemorySubtype.DECISION.value,
            "directive": MemorySubtype.DIRECTIVE.value,
            "todo": MemorySubtype.WORKFLOW.value,
            "next": MemorySubtype.WORKFLOW.value,
            "blocked": MemorySubtype.PROBLEM.value,
            "resolved": MemorySubtype.FIX.value,
        }
        for marker in markers:
            key = (
                f"candidate:{marker.kind}",
                marker.source_span_start,
                marker.source_span_end,
                " ".join(marker.content.casefold().split()),
            )
            if key in existing_by_key:
                child = existing_by_key[key]
                if child.embedding is None:
                    embedding = await self.embedding.embed(child.content)
                    await self.storage.update_memory(workspace_id, child.id, embedding=embedding)
                continue
            child = await self.storage.create_memory(
                workspace_id,
                RememberInput(
                    content=marker.content,
                    type=MemoryType.EPISODIC,
                    subtype=marker_subtypes.get(marker.kind, memory.subtype),
                    importance=memory.importance,
                    tags=memory.tags,
                    metadata={
                        **(memory.metadata or {}),
                        "deterministic_kind": f"candidate:{marker.kind}",
                        "extraction_rule": marker.rule_id,
                        "source_span_start": marker.source_span_start,
                        "source_span_end": marker.source_span_end,
                    },
                    context_id=memory.context_id,
                    tenant_id=memory.tenant_id,
                    user_id=memory.user_id,
                    observer_id=memory.observer_id,
                    subject_id=memory.subject_id,
                    session_id=memory.session_id,
                    source_memory_id=memory.id,
                    event_time=memory.event_time,
                ),
            )
            embedding = await self.embedding.embed(marker.content)
            updated = await self.storage.update_memory(
                workspace_id,
                child.id,
                embedding=embedding,
            )
            created.append(updated or child)
            existing_by_key[key] = updated or child
        return created

    def _record_post_store_failure(self, step: str, workspace_id: str) -> None:
        """Record an observable signal for a swallowed post-store sub-step failure.

        Increments a metrics counter (labeled by sub-step) when a metrics
        service is wired so silent degradation in the post-store pipeline becomes
        measurable. Counter failures are themselves swallowed — observability
        must never break the store path. The caller still logs the underlying
        exception; this only adds the stable, aggregatable signal.

        Note: workspace_id is retained in the signature so callers remain
        unchanged; it is intentionally NOT included as a metric label to avoid
        unbounded cardinality in multi-tenant deployments.
        """
        if self.metrics is not None:
            try:
                self.metrics.counter(
                    "memorylayer_post_store_step_failures_total",
                    labels={"step": step},
                )
            except Exception:
                self.logger.debug("Metrics counter failed for post-store step: %s", step)

    async def _post_store_pipeline(
        self,
        workspace_id: str,
        memory: Memory,
        embedding: list[float],
        inline: bool = False,
        classify_type: bool = False,
    ) -> None:
        """Run post-store normalization: cache invalidation, association, contradiction, tier gen.

        This is the reusable pipeline that runs after a memory is stored.
        Used by both remember() (for non-decomposable memories) and ingest_fact()
        (for each decomposed fact).

        Args:
            workspace_id: Target workspace
            memory: The stored memory
            embedding: Pre-computed embedding vector
            inline: If True, run all steps synchronously.
                    If False (default), schedule as background tasks where possible.
            classify_type: If True, request LLM-based type reclassification
                    in the auto-enrich task.
        """
        # Cache invalidation (always inline, trivial cost)
        if self.cache:
            try:
                await self.cache.clear_prefix(f"recall:{workspace_id}:")
                await self.cache.clear_prefix(f"assoc:{workspace_id}:")
            except Exception as e:
                self.logger.debug("Cache invalidation failed: %s", e)
                self._record_post_store_failure("cache_invalidation", workspace_id)

        # Tier generation
        if self.tier_generation_service:
            try:
                if inline:
                    await self.tier_generation_service.generate_tiers(memory.id, workspace_id)
                else:
                    await self.tier_generation_service.request_tier_generation(memory.id, workspace_id)
            except Exception as e:
                self.logger.warning("Failed tier generation for %s: %s", memory.id, e)
                self._record_post_store_failure("tier_generation", workspace_id)

        generation_allowed = self._generation_allowed(GenerationActivity.CONTRADICTION_DETECTION)

        # Contradiction detection is optional enrichment and is not queued when
        # policy rejects the generative activity.
        if self.contradiction_service and generation_allowed:
            try:
                await self.contradiction_service.check_new_memory(workspace_id, memory.id)
            except Exception as e:
                self.logger.warning("Failed contradiction check for %s: %s", memory.id, e)
                self._record_post_store_failure("contradiction_check", workspace_id)

        # Auto-enrich (association + optional type classification)
        if inline or not self.task_service:
            await self._inline_auto_enrich(
                workspace_id,
                memory,
                embedding,
                classify_type=classify_type and generation_allowed,
                allow_generation=generation_allowed,
            )
        else:
            try:
                await self.task_service.schedule_task(
                    "auto_enrich",
                    {
                        "memory_id": memory.id,
                        "workspace_id": workspace_id,
                        "content": memory.content,
                        "classify_type": classify_type,
                        "allow_generation": generation_allowed,
                    },
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to schedule auto-enrich for %s, falling back to inline: %s",
                    memory.id,
                    e,
                )
                self._record_post_store_failure("auto_enrich_schedule", workspace_id)
                await self._inline_auto_enrich(workspace_id, memory, embedding, classify_type=classify_type)

    async def ingest_fact(
        self,
        workspace_id: str,
        input: RememberInput,
        embedding: list[float] | None = None,
        source_memory_id: str | None = None,
        inline: bool = False,
    ) -> Memory | None:
        """Process a single memory through the full pipeline: dedup, store, post-store.

        This is the isolated per-fact pipeline. Called by:
        - FactDecompositionTaskHandler for each decomposed fact (background)
        - _decompose_and_process_inline() for inline decomposition
        - Any future scenario needing atomic memory ingestion with full treatment

        Args:
            workspace_id: Target workspace
            input: Memory content and metadata
            embedding: Pre-computed embedding (generated if None)
            source_memory_id: Parent memory ID (for decomposed facts)
            inline: If True, run post-store pipeline synchronously

        Returns:
            The stored Memory, or None if deduplicated away (SKIP)
        """
        # Generate embedding if not provided
        if embedding is None:
            embedding = await self.embedding.embed(input.content)

        # Dedup check
        content_hash = compute_content_hash(input.content)
        dedup_result = await self.deduplication.check_duplicate(
            content=input.content,
            content_hash=content_hash,
            embedding=embedding,
            workspace_id=workspace_id,
        )

        if dedup_result.action == DeduplicationAction.SKIP:
            self.logger.info(
                "Fact is duplicate (SKIP): %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason,
            )
            return None

        if dedup_result.action == DeduplicationAction.UPDATE:
            self.logger.info(
                "Fact updates existing memory: %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason,
            )
            updated = await self.storage.update_memory(
                workspace_id=workspace_id,
                memory_id=dedup_result.existing_memory_id,
                content=input.content,
                embedding=embedding,
                importance=max(input.importance, 0.5),
            )
            await self._post_store_pipeline(workspace_id, updated, embedding, inline=inline)
            return updated

        if dedup_result.action == DeduplicationAction.MERGE:
            self.logger.info(
                "Fact merges with existing memory: %s (%s)",
                dedup_result.existing_memory_id,
                dedup_result.reason,
            )
            existing = await self.storage.get_memory(
                workspace_id,
                dedup_result.existing_memory_id,
                track_access=False,
            )
            updated = await self._merge_memories(workspace_id, existing, input.content, input.tags, input.metadata, input.importance)
            # The survivor was enriched when it was first stored, so re-running
            # the pipeline here is mostly re-derivation: a batched relationship
            # classification whose edges then collide with uq_association. Off
            # by default; see MEMORYLAYER_ENRICH_ON_MERGE for the trade-off.
            if self.enrich_on_merge:
                await self._post_store_pipeline(
                    workspace_id,
                    updated,
                    updated.embedding,
                    inline=inline,
                )
            else:
                self.logger.debug(
                    "Skipping post-store enrichment for merged memory %s (MEMORYLAYER_ENRICH_ON_MERGE disabled)",
                    updated.id,
                )
            return updated

        # CREATE: store new memory
        memory_type = input.type
        if memory_type is None:
            memory_type = await self._classify_memory_type(input.content)

        memory_data = RememberInput(
            content=input.content,
            type=memory_type,
            subtype=input.subtype,
            importance=input.importance,
            tags=input.tags,
            metadata=self._maybe_add_entity_metadata(input.content, input.metadata),
            associations=input.associations,
            context_id=input.context_id,
            user_id=input.user_id,
        )

        memory = await self.storage.create_memory(workspace_id, memory_data)

        if memory.embedding is None:
            memory = await self.storage.update_memory(
                workspace_id,
                memory.id,
                embedding=embedding,
            )

        # Set source_memory_id if this fact came from decomposition
        if source_memory_id:
            memory = await self.storage.update_memory(
                workspace_id,
                memory.id,
                source_memory_id=source_memory_id,
            )

        self.logger.info("Stored fact memory: %s", memory.id)

        # Run post-store pipeline (no decomposition — facts are atomic)
        await self._post_store_pipeline(workspace_id, memory, embedding, inline=inline)

        return memory

    async def _decompose_and_process_inline(
        self,
        workspace_id: str,
        memory: Memory,
        embedding: list[float],
    ) -> list[Memory]:
        """Decompose a composite memory and process each fact inline.

        Used when remember(inline=True) and the memory qualifies for decomposition.
        Runs the full per-fact pipeline (dedup, store, associate, etc.) synchronously
        for each decomposed fact, then archives the parent.

        Args:
            workspace_id: Target workspace
            memory: The composite memory to decompose
            embedding: Pre-computed embedding of the composite

        Returns:
            List of created fact memories
        """
        # Write-time temporal normalization: anchor relative dates in the source
        # content to the memory's effective time (event_time if known, else when
        # it was recorded), so decomposed facts resolve "last Tuesday" etc. to
        # absolute dates and can surface their own event_time.
        reference_time = memory.event_time or memory.created_at or datetime.now(UTC)
        facts = await self.extraction_service.decompose_to_facts(memory.content, reference_time=reference_time)

        if len(facts) <= 1:
            # Already atomic — just run post-store pipeline on the original
            await self._post_store_pipeline(workspace_id, memory, embedding, inline=True)
            return [memory]

        from ...models.association import AssociateInput

        created = []
        for fact in facts:
            # Determine type/subtype: prefer fact-level overrides, fall back to parent
            fact_type = memory.type
            fact_subtype = memory.subtype
            try:
                if fact.get("type"):
                    fact_type = MemoryType(fact["type"])
            except ValueError:
                pass
            if fact.get("subtype"):
                fact_subtype = fact["subtype"]

            fact_input = RememberInput(
                content=fact["content"],
                type=fact_type,
                subtype=fact_subtype,
                importance=memory.importance,
                tags=memory.tags,
                metadata={**(memory.metadata or {}), "decomposed_from": memory.id},
                context_id=memory.context_id,
                user_id=memory.user_id,
                # Carry the fact's resolved absolute date (write-time temporal
                # normalization). Absent/None leaves event_time unset (do not
                # overwrite with created_at).
                event_time=fact.get("event_time"),
            )
            result = await self.ingest_fact(
                workspace_id,
                fact_input,
                source_memory_id=memory.id,
                inline=True,
            )
            if result:
                created.append(result)

        # Create PART_OF associations from each fact to the parent
        for fact_mem in created:
            try:
                assoc_input = AssociateInput(
                    source_id=fact_mem.id,
                    target_id=memory.id,
                    relationship="part_of",
                    strength=1.0,
                    metadata={"auto_generated": True, "source": "fact_decomposition"},
                )
                await self.storage.create_association(workspace_id, assoc_input)
            except Exception as e:
                self.logger.warning(
                    "Failed PART_OF association %s->%s: %s",
                    fact_mem.id,
                    memory.id,
                    e,
                )

        # Archive the parent memory
        try:
            await self.storage.update_memory(
                workspace_id,
                memory.id,
                status=MemoryStatus.ARCHIVED.value,
            )
            self.logger.info(
                "Decomposed memory %s into %d facts inline, archived parent",
                memory.id,
                len(created),
            )
        except Exception as e:
            self.logger.warning("Failed to archive parent memory %s: %s", memory.id, e)

        return created

    async def _inline_auto_enrich(
        self,
        workspace_id: str,
        memory: Memory,
        embedding: list[float],
        classify_type: bool = False,
        allow_generation: bool = True,
    ) -> None:
        """
        Fallback inline auto-enrich: association + optional type classification.

        Used when the task service is not available to schedule background
        auto-enrich with LLM classification.
        """
        if self.association_service:
            try:
                similar_memories = await self.storage.search_memories(
                    workspace_id=workspace_id,
                    query_embedding=embedding,
                    limit=5,
                    min_relevance=self.auto_association_threshold,
                )
            except Exception as e:
                self.logger.warning("Failed to search similar memories for %s: %s", memory.id, e)
                self._record_post_store_failure("auto_enrich", workspace_id)
                similar_memories = []

            for similar_memory, score in similar_memories or []:
                if similar_memory.id != memory.id:  # Don't self-associate
                    try:
                        await self.association_service.auto_associate(
                            workspace_id=workspace_id,
                            new_memory_id=memory.id,
                            similar_memories=[(similar_memory.id, score)],
                            threshold=self.auto_association_threshold,
                            allow_generation=allow_generation,
                        )
                    except Exception as e:
                        self.logger.warning(
                            "Failed to auto-associate %s with %s: %s",
                            memory.id,
                            similar_memory.id,
                            e,
                        )

        # Type classification
        if classify_type and self.extraction_service:
            try:
                classified_type, classified_subtype = await self.extraction_service.classify_content(
                    memory.content,
                )
                if classified_type != memory.type:
                    update_kwargs: dict = {"type": classified_type.value}
                    if classified_subtype is not None:
                        update_kwargs["subtype"] = classified_subtype.value
                    await self.storage.update_memory(
                        workspace_id=workspace_id,
                        memory_id=memory.id,
                        **update_kwargs,
                    )
                    self.logger.info(
                        "Reclassified memory %s from %s to %s",
                        memory.id,
                        memory.type,
                        classified_type,
                    )
            except Exception as e:
                self.logger.debug("Inline type classification skipped for %s: %s", memory.id, e)

        # Knowledge-graph fact extraction (multi-hop). Runs here, in the enrichment
        # task, because LLM fact extraction is far too slow for inline ingest. The
        # base ExtractionService.extract_facts returns [] (no-op) until an LLM-backed
        # provider overrides it; extracted (subject, relation, object) triples are
        # persisted on metadata['facts'] for the entity/fact retrieval channel. The
        # next step (mapping each relation onto an ontology type and persisting a
        # typed association edge for graph traversal) layers on top via the existing
        # association_service — see .slop/INCREMENTAL_KG_DESIGN.md.
        # Skip for fact memories themselves: they are atomic, already-extracted
        # triples, so re-extracting would recurse (fact -> facts -> ...).
        if (
            allow_generation
            and (self.entity_anchor_enabled or self.fact_channel_enabled)
            and self.extraction_service
            and memory.subtype != MemorySubtype.FACT.value
        ):
            try:
                facts = await self.extraction_service.extract_facts(memory.content)
                if facts:
                    # Entity-anchor channel: persist triples on metadata['facts'].
                    if self.entity_anchor_enabled:
                        merged = {**(memory.metadata or {}), "facts": facts}
                        await self.storage.update_memory(
                            workspace_id=workspace_id,
                            memory_id=memory.id,
                            metadata=merged,
                        )
                    # Fact channel: store each triple as its own subtype="fact"
                    # memory so it becomes a retrievable candidate for the fact
                    # arm. Reuses the SAME `facts` result (no second extract_facts
                    # call). NOTE: OSS extract_facts returns [] (no LLM), so this
                    # is a structural no-op in OSS until an LLM provider overrides
                    # it — OSS proves the shape; enterprise makes it real.
                    if self.fact_channel_enabled:
                        await self._store_fact_memories(workspace_id, memory, facts)
            except Exception as e:
                self.logger.debug("Fact extraction skipped for %s: %s", memory.id, e)

        # Cue-anchor generation (Memora-inspired abstraction+cue indexing). Runs
        # here, in the enrichment task, because cue generation is LLM-backed (too
        # slow for inline ingest). Flag-gated DARK (default OFF) via the SAME
        # cue-channel flag, so there is zero cost when the channel is off. The base
        # ExtractionService.generate_cue_anchors returns [] (no-op) and the base
        # StorageBackend.store_cue_anchors is a no-op, so this is a structural
        # no-op in OSS until an enterprise provider/backend makes it real. Skip for
        # fact memories: they are atomic, already-normalised triples.
        #
        # NOTE — production ingest caveat (mirrors the fact channel): this runs on
        # the INLINE post-store path only. When a task service is configured the
        # background auto_enrich task runs instead and does NOT yet generate cue
        # anchors, so cues are populated only via inline ingest.
        if allow_generation and self.cue_channel_enabled and self.extraction_service and memory.subtype != MemorySubtype.FACT.value:
            try:
                cues = await self.extraction_service.generate_cue_anchors(memory.content)
                if cues:
                    cue_texts = [c["cue"] for c in cues]
                    embeddings = await self.embedding.embed_batch(cue_texts)
                    rows: list[dict] = []
                    for cue, embedding in zip(cues, embeddings):
                        entity_id = await self._resolve_cue_entity_id(workspace_id, cue.get("entity"))
                        rows.append(
                            {
                                "cue": cue["cue"],
                                "embedding": embedding,
                                "entity_id": entity_id,
                            }
                        )
                    await self.storage.store_cue_anchors(
                        workspace_id=workspace_id,
                        memory_id=memory.id,
                        cues=rows,
                    )
            except Exception as e:
                self.logger.debug("Cue-anchor generation skipped for %s: %s", memory.id, e)

        # Entity registry accretion (entity registry slice 1). Flag-gated DARK
        # (default OFF). For each extracted entity string, resolve it to a
        # canonical entity (exact+alias+create on OSS) and attach this memory as
        # a member. Non-destructive: failures are logged, never block ingest.
        if self.entity_registry_enabled and self.entity_registry_service is not None:
            await self._accrete_entities(workspace_id, memory)

    async def _resolve_cue_entity_id(self, workspace_id: str, entity_name: str | None) -> str | None:
        """Resolve a cue's ``entity`` string to a canonical Entity id, or ``None``.

        Links a cue anchor to the entity it names so the cue arm and the
        entity-registry / entity-anchored arm share one entity vocabulary. Only
        attempts resolution when the entity registry is available AND enabled
        (mirrors ``_accrete_entities`` gating); otherwise returns ``None`` so the
        cue is stored without an entity link.

        Uses ``allow_create=False`` — cues never CREATE entities (they are a
        read-only handle onto the vocabulary accretion already produces). A
        ``LookupError`` (no existing entity) or any registry failure degrades to
        ``None``. ``promote=True`` makes resolution NAME-FIRST so a cue naming a
        person resolves onto the same PERSON node regardless of the cue's
        (unknown) type.
        """
        if not entity_name or not entity_name.strip():
            return None
        if not (self.entity_registry_enabled and self.entity_registry_service is not None):
            return None

        from ...models.entity_registry import EntityType

        try:
            resolution = await self.entity_registry_service.resolve(
                workspace_id,
                entity_name.strip(),
                entity_type=EntityType.CONCEPT,
                allow_create=False,
                promote=True,
            )
            return resolution.entity.id
        except LookupError:
            return None
        except Exception as e:
            self.logger.debug("Cue entity resolution skipped for %r: %s", entity_name, e)
            return None

    async def _accrete_entities(self, workspace_id: str, memory: Memory, *, force_extract: bool = False) -> None:
        """Resolve extracted entity strings to canonical entities + add members.

        Reads the speaker/entities/entity_types already folded onto ``metadata``
        by ``_maybe_add_entity_metadata`` (falling back to the shared regex util /
        extraction service). Each string is resolved (exact+alias+create) and
        the memory is attached as a member whose role distinguishes self-authorship
        from third-party mention:

        * ``role="self"`` — the entity IS the speaker/observer of this memory,
          i.e. ``name == speaker`` (the extracted speaker string) OR
          ``name == memory.observer_id`` (the perspective anchor). Self members
          represent turns the entity authored; they are excluded from the
          registry-recall expansion channel (which requests ``role="mention"``
          only) so a speaker's own history does not crowd out third-party
          observations about them.
        * ``role="mention"`` — all other entities; the entity is talked-about in
          this turn and is a legitimate recall expansion candidate.

        Entity typing: when the provider supplied ``entity_types`` (typed NER,
        e.g. GLiNER2), each name takes its mapped ``EntityType``; otherwise the
        heuristic applies — the speaker is a PERSON and all other spans default
        to CONCEPT.

        ``force_extract`` re-runs the CURRENT extraction service over the content,
        ignoring any pre-folded ``metadata.entities``. The backfill uses this so a
        workspace whose memories were ingested under an OLDER extractor (e.g. the
        regex extractor, which folded untyped spans + no ``entity_types``) is
        re-accreted with the current extractor (e.g. GLiNER2's typed NER) instead of
        silently reusing the stale untyped fold.

        Beyond the content regex, the memory FIELDS are consulted as the
        AUTHORITATIVE perspective anchors: ``observer_id`` (when set) always
        accretes as a PERSON ``role="self"`` member and ``subject_id`` (when set)
        as a PERSON ``role="mention"`` member — so perspective works on
        prefix-less sources (email/doc) where ``observer_id`` is set from the
        sender but there is no ``[ts] Name:`` dialogue prefix in the content.
        Never raises.
        """
        from ...models.entity_registry import EntityType

        try:
            md = memory.metadata or {}
            speaker = md.get("speaker")
            entities = md.get("entities")
            entity_types = md.get("entity_types")
            if force_extract or (speaker is None and entities is None):
                # Extract on demand: either nothing was pre-folded, or the caller
                # forces a fresh extraction with the CURRENT extractor (backfill).
                if self.extraction_service is not None:
                    extracted = self.extraction_service.extract_entities(memory.content or "")
                else:
                    extracted = extract_entities(memory.content or "")
                speaker = extracted.get("speaker")
                entities = extracted.get("entities")
                entity_types = extracted.get("entity_types")

            speaker = speaker or None
            names = list(entities or [])
            type_map = entity_types or {}

            # Track (entity_id, role) pairs already added so the authoritative
            # observer_id/subject_id anchors below do not re-resolve / re-add a
            # member that the content-regex loop already produced (storage is
            # idempotent on (entity, memory, role), but skipping avoids a
            # redundant resolve()/add_member() round-trip).
            added: set[tuple[str, str]] = set()

            for name in names:
                if not name:
                    continue
                if name in type_map:
                    # Typed provider supplied an EntityType value for this name.
                    try:
                        etype = EntityType(type_map[name])
                    except (ValueError, KeyError):
                        etype = await self._intended_entity_type(workspace_id, name, speaker)
                else:
                    etype = await self._intended_entity_type(workspace_id, name, speaker)
                try:
                    # promote=True enables NAME-FIRST resolution: a normalized name
                    # maps to ONE canonical entity in the workspace regardless of
                    # type, with CONCEPT promoted/merged to PERSON when intended.
                    # This unifies a person who both SPEAKS (PERSON/self) and is
                    # MENTIONED (CONCEPT/mention) onto a single PERSON node so
                    # get_representation(observer, subject) (default subject_type=
                    # PERSON) finds the mention members instead of an empty node.
                    resolution = await self.entity_registry_service.resolve(
                        workspace_id,
                        name,
                        entity_type=etype,
                        source_memory_id=memory.id,
                        observer_id=memory.observer_id,
                        promote=True,
                    )
                    # Distinguish self-authorship from third-party mention.
                    # An entity is "self" when it IS the speaker of this turn
                    # or matches the perspective anchor (observer_id). All
                    # other entities are "mention" — they are talked-about and
                    # are the legitimate recall expansion candidates.
                    member_role = "self" if (name == speaker or (memory.observer_id and name == memory.observer_id)) else "mention"
                    await self.entity_registry_service.add_member(
                        workspace_id,
                        resolution.entity.id,
                        memory.id,
                        role=member_role,
                    )
                    added.add((resolution.entity.id, member_role))
                except Exception as e:
                    self.logger.debug("Entity accretion skipped for %r on %s: %s", name, memory.id, e)

            # Authoritative perspective anchors from the memory FIELDS (not the
            # content regex). These make perspective work on prefix-less sources
            # (email/doc) where ``observer_id`` is set from the sender but the
            # content carries no ``[ts] Name:`` dialogue prefix, so the observer
            # name never appears in the regex-extracted ``entities`` list.
            #
            #   * observer_id -> PERSON, role="self": the observer is ALWAYS the
            #     self-anchor of their own memory, with or without a content
            #     prefix. This is the authoritative perspective key.
            #   * subject_id  -> PERSON, role="mention": the memory is *about*
            #     this subject; the subject is observed, not the author, so it
            #     accretes as a mention (a legitimate recall/expansion candidate,
            #     and the third-party-view scope in get_representation).
            #
            # observer_id/subject_id are free-text NAMES today (seeded from the
            # speaker / set by the caller); resolving them by name is correct for
            # the current model (the deeper FK-to-entity-id migration is out of
            # scope). promote=True keeps these on the same PERSON node as any
            # regex-extracted self/mention members for the same name.
            for anchor_name, anchor_role in (
                (memory.observer_id, "self"),
                (memory.subject_id, "mention"),
            ):
                if not anchor_name:
                    continue
                try:
                    anchor_res = await self.entity_registry_service.resolve(
                        workspace_id,
                        anchor_name,
                        entity_type=EntityType.PERSON,
                        source_memory_id=memory.id,
                        observer_id=memory.observer_id,
                        promote=True,
                    )
                    key = (anchor_res.entity.id, anchor_role)
                    if key in added:
                        continue
                    await self.entity_registry_service.add_member(
                        workspace_id,
                        anchor_res.entity.id,
                        memory.id,
                        role=anchor_role,
                    )
                    added.add(key)
                except Exception as e:
                    self.logger.debug(
                        "Entity accretion (anchor %s=%r) skipped on %s: %s",
                        anchor_role,
                        anchor_name,
                        memory.id,
                        e,
                    )
        except Exception as e:
            self.logger.debug("Entity accretion skipped for %s: %s", memory.id, e)

    async def backfill_entity_registry(
        self,
        workspace_id: str,
        *,
        batch_size: int = 200,
        created_after: datetime | None = None,
        max_memories: int | None = None,
        reset: bool = False,
    ) -> dict:
        """Replay ingest-time entity accretion over the workspace's EXISTING memories.

        For workspaces whose memories were ingested BEFORE the entity registry was
        enabled, no entity rows were ever created. This walks the memory set and
        calls the exact ingest-time path (``_accrete_entities``) on each memory, so
        the registry is populated from history — the same extraction (GLiNER2 NER /
        regex), the same resolve(promote=True) + add_member(role) semantics.

        Idempotent and safe to re-run: ``resolve`` is exact-match-first (an already
        canonicalized name re-resolves to the same entity, no duplicate) and
        ``add_member`` is idempotent on ``(entity, memory, role)``. Re-extraction is
        the only repeated cost. ``_accrete_entities`` never raises, so one bad memory
        never aborts the sweep.

        No-op (returns zeroed counters) when the registry is disabled/unavailable —
        mirrors the ingest gate so this is safe to call unconditionally.

        Args:
            workspace_id: the workspace to backfill (the hard isolation boundary).
            batch_size: memories fetched per page (offset pagination over the stable
                memory set; accretion never inserts/deletes memories).
            created_after: only backfill memories created at/after this instant
                (default: epoch — the whole workspace).
            max_memories: optional cap on memories processed this call (for a bounded
                first pass); ``None`` processes them all.
            reset: when True, WIPE the workspace's existing entity registry (entities
                + aliases + members) BEFORE re-accreting — for a pristine rebuild
                after changing the extractor. Best-effort: a backend that can't clear
                (NotImplementedError) proceeds with an additive backfill instead.

        Returns:
            ``{"workspace_id", "cleared", "scanned", "accreted"}`` counts.
        """
        result = {"workspace_id": workspace_id, "cleared": 0, "scanned": 0, "accreted": 0}
        if not (self.entity_registry_enabled and self.entity_registry_service is not None):
            self.logger.info(
                "Entity-registry backfill skipped for %s: registry disabled/unavailable",
                workspace_id,
            )
            return result

        if reset:
            try:
                result["cleared"] = await self.entity_registry_service.clear_workspace(workspace_id)
                self.logger.info(
                    "Entity-registry backfill reset for %s: cleared %d existing entities",
                    workspace_id,
                    result["cleared"],
                )
            except NotImplementedError:
                self.logger.warning(
                    "Entity-registry backfill reset requested for %s but the backend cannot "
                    "clear entities; proceeding additively (no wipe)",
                    workspace_id,
                )
            except Exception as e:  # noqa: BLE001 - wipe is best-effort; never abort the backfill
                self.logger.error(
                    "Entity-registry backfill reset failed for %s: %s; proceeding additively",
                    workspace_id,
                    e,
                )

        after = created_after or datetime(1970, 1, 1, tzinfo=UTC)
        offset = 0
        while True:
            try:
                # Enumerate ids cheaply (abstract detail); the full Memory (content,
                # metadata, observer_id, subject_id) is loaded per id below.
                rows = await self.storage.get_recent_memories(
                    workspace_id,
                    created_after=after,
                    limit=batch_size,
                    offset=offset,
                    detail_level="abstract",
                )
            except NotImplementedError:
                self.logger.warning(
                    "Entity-registry backfill: storage backend lacks get_recent_memories; aborting for %s",
                    workspace_id,
                )
                break
            except Exception as e:
                self.logger.error(
                    "Entity-registry backfill: memory enumeration failed for %s at offset %d: %s",
                    workspace_id,
                    offset,
                    e,
                )
                break
            if not rows:
                break

            for row in rows:
                mem_id = row.get("id") if isinstance(row, dict) else getattr(row, "id", None)
                if not mem_id:
                    continue
                result["scanned"] += 1
                try:
                    memory = await self.storage.get_memory(workspace_id, mem_id, track_access=False)
                except Exception as e:
                    self.logger.debug("Entity-registry backfill: get_memory %s failed: %s", mem_id, e)
                    memory = None
                if memory is None:
                    continue
                # Ingest-time accretion path (never raises, idempotent), but FORCE a
                # fresh extraction with the current extractor so a workspace ingested
                # under an older extractor is re-typed (e.g. regex spans -> GLiNER2 NER)
                # instead of reusing the stale pre-folded metadata.entities.
                await self._accrete_entities(workspace_id, memory, force_extract=True)
                result["accreted"] += 1
                if max_memories is not None and result["scanned"] >= max_memories:
                    self.logger.info(
                        "Entity-registry backfill for %s hit max_memories=%d (scanned=%d, accreted=%d)",
                        workspace_id,
                        max_memories,
                        result["scanned"],
                        result["accreted"],
                    )
                    return result

            offset += len(rows)

        self.logger.info(
            "Entity-registry backfill complete for %s: scanned=%d accreted=%d",
            workspace_id,
            result["scanned"],
            result["accreted"],
        )
        return result

    async def _intended_entity_type(self, workspace_id: str, name: str, speaker: str | None):
        """Decide the intended ``EntityType`` for an accreted name (no LLM).

        PERSON when the name IS this turn's speaker, OR when a PERSON entity with
        that normalized name already exists in the workspace (a later mention of a
        known speaker is that person); otherwise CONCEPT. Combined with name-first
        ``resolve(promote=True)``, this keeps self + mention members for the same
        person on a single PERSON node. Best-effort: a registry/storage lookup
        failure falls back to the original speaker-only heuristic.
        """
        from ...models.entity_registry import EntityType

        if name == speaker:
            return EntityType.PERSON
        try:
            from ..entity_registry._normalize import normalize_entity_name

            normalized = normalize_entity_name(name)
            existing = await self.storage.find_entity_by_normalized_name(workspace_id, EntityType.PERSON.value, normalized)
            if existing is not None:
                return EntityType.PERSON
        except Exception as e:
            self.logger.debug("Intended-type PERSON probe failed for %r: %s", name, e)
        return EntityType.CONCEPT

    async def _store_fact_memories(self, workspace_id: str, memory: Memory, facts: list[dict]) -> None:
        """Store each extracted (subject, relation, object) triple as a fact memory.

        Each triple is joined to a string and prepended with the source memory's
        date (``event_time`` or ``created_at``) for temporal grounding, then stored
        with ``subtype="fact"`` and ``metadata={"kind":"fact","source_id":<parent>}``
        so the fact retrieval channel (``_fuse_fact_results``) can surface it. Never
        raises — per-fact failures are logged and skipped.
        """
        source_dt = memory.event_time or memory.created_at
        date_prefix = source_dt.strftime("[%d %B, %Y] ") if source_dt else ""
        for fact in facts:
            triple = (fact.get("subject", ""), fact.get("relation", ""), fact.get("object", ""))
            fact_text = " ".join(part for part in triple if part).strip()
            if not fact_text:
                continue
            fact_content = f"{date_prefix}{fact_text}" if date_prefix else fact_text
            try:
                # Fact channel: enterprise extract_facts fills subject from the
                # LLM-extracted triple; OSS returns [] so this is structural
                # plumbing only. observer_id propagates from the source turn so
                # fact memories carry whose perspective produced them.
                await self.ingest_fact(
                    workspace_id,
                    RememberInput(
                        content=fact_content,
                        type=MemoryType.SEMANTIC,
                        subtype=MemorySubtype.FACT.value,
                        metadata={"kind": "fact", "source_id": memory.id},
                        context_id=memory.context_id,
                        user_id=memory.user_id,
                        observer_id=memory.observer_id,  # inherit source turn's perspective
                        # subject_id: set by enterprise extract_facts (LLM triple extraction);
                        # None in OSS (no-op until an LLM-backed provider overrides extract_facts).
                    ),
                    source_memory_id=memory.id,
                    inline=True,
                )
            except Exception as e:
                self.logger.debug("Fact memory store skipped for %s (%r): %s", memory.id, fact_text[:80], e)

    async def recall(
        self,
        workspace_id: str,
        input: RecallInput,
        user_id: str | None = None,
    ) -> RecallResult:
        """
        Query memories using vector similarity and optional filters.

        Modes:
        - RAG: Pure vector similarity (fast, ~30ms)
        - LLM: Query rewriting + tiered search (accurate, ~500ms)
        - HYBRID: RAG first, LLM if insufficient (balanced)
        """
        self.logger.info("Recalling memories in workspace: %s, mode: %s, query: %s", workspace_id, input.mode, input.query[:50])

        start_time = datetime.now(UTC)

        # Resolve None → server defaults for mode, tolerance, and detail_level
        effective_mode = input.mode if input.mode is not None else RecallMode.RAG
        effective_tolerance = input.tolerance if input.tolerance is not None else SearchTolerance.MODERATE
        effective_detail_level = input.detail_level if input.detail_level is not None else DetailLevel.FULL
        if (
            effective_mode in {RecallMode.LLM, RecallMode.AGENTIC}
            and self.llm_service
            and not self._generation_allowed(GenerationActivity.QUERY_REWRITING)
        ):
            raise GenerationNotAllowedError(
                GenerationActivity.QUERY_REWRITING,
                self.llm_service.policy,
                f"{effective_mode.value} recall explicitly requires generation",
            )
        generation_operation_ids: list[str] = []
        generation_operation_prefix = generate_id("recall_gen")

        # Determine effective tolerance threshold
        relevance_threshold = self._get_relevance_threshold(effective_tolerance, input.min_relevance)

        # Phase 4: Check recall cache
        cache_key = None
        if self.cache:
            cache_key = self._recall_cache_key(workspace_id, input.query, input)
            try:
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    self.logger.debug("Recall cache hit for key: %s", cache_key)
                    return RecallResult(**cached)
            except Exception as e:
                self.logger.debug("Cache get failed: %s", e)

        # Query intent routing (rule-based, no LLM): classify the query and softly
        # tune retrieval, filling only params the caller left unset. Entity intent
        # amplifies alias/backlink boosts via per-recall weight overrides.
        intent: QueryIntent | None = None
        intent_labels: list[str] | None = None
        alias_weight: float | None = None
        backlink_weight: float | None = None
        if self.query_intent_enabled and input.query.strip() not in ("*", "**", ""):
            intent = classify_query_intent(input.query)
            intent_labels = sorted(intent.labels)
            input, alias_weight, backlink_weight = self._route_by_intent(input, intent)

        # RAG mode: Pass 1 - Pure vector similarity
        if effective_mode == RecallMode.RAG:
            result = await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
                alias_weight=alias_weight,
                backlink_weight=backlink_weight,
                intent=intent,
            )
            result.mode_used = RecallMode.RAG

        # LLM mode: Query rewriting + enhanced search
        elif effective_mode == RecallMode.LLM:
            result = await self._recall_llm(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
                alias_weight=alias_weight,
                backlink_weight=backlink_weight,
                intent=intent,
                generation_operation_prefix=generation_operation_prefix,
                generation_operation_ids=generation_operation_ids,
            )
            result.mode_used = RecallMode.LLM

        # HYBRID mode: Try RAG first, fall back to LLM if insufficient
        else:
            result = await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
                alias_weight=alias_weight,
                backlink_weight=backlink_weight,
                intent=intent,
            )

            # Check if RAG results are sufficient.
            # Compare the top result's RETRIEVAL confidence against rag_threshold,
            # NOT importance. importance is a stored authoring weight (default
            # 0.5); rag_threshold is a RAG confidence gate (default 0.8), so the
            # old `importance < rag_threshold` check almost always fell through to
            # the expensive LLM path. boosted_score is the post-boost/rerank
            # relevance score attached by _recall_rag (apply_scope_boosts always
            # sets it); fall back to relevance_score, then importance, if unset.
            top = result.memories[0] if result.memories else None
            top_score = None
            if top is not None:
                top_score = top.boosted_score if top.boosted_score is not None else top.relevance_score
                if top_score is None:
                    top_score = top.importance
            can_rewrite = bool(self.llm_service and self._generation_allowed(GenerationActivity.QUERY_REWRITING))
            if (top is None or top_score < input.rag_threshold) and can_rewrite:
                self.logger.debug("RAG insufficient, trying LLM mode")

                result = await self._recall_llm(
                    workspace_id=workspace_id,
                    input=input,
                    relevance_threshold=relevance_threshold,
                    alias_weight=alias_weight,
                    backlink_weight=backlink_weight,
                    intent=intent,
                    generation_operation_prefix=generation_operation_prefix,
                    generation_operation_ids=generation_operation_ids,
                )
                result.mode_used = RecallMode.LLM
            else:
                result.mode_used = RecallMode.RAG

        # Record the classified intent on the result for observability.
        if intent_labels is not None:
            result.query_intent = intent_labels

        relation_unresolved = False
        if self.relational_recall_enabled and input.include_relations:
            relation_result = await self.entity_relation_service.recall(
                workspace_id,
                input.query,
                max_edges=min(80, max(10, input.limit * 4)),
                max_memories=min(40, max(10, input.limit * 2)),
            )
            relation_unresolved = relation_result.unresolved_seed
            if relation_result.memories:
                memory_by_id = {memory.id: memory for memory in result.memories}
                scores: dict[str, float] = {}
                for rank, memory in enumerate(result.memories):
                    scores[memory.id] = scores.get(memory.id, 0.0) + 1.0 / (61 + rank)
                for rank, memory in enumerate(relation_result.memories):
                    memory_by_id.setdefault(memory.id, memory)
                    scores[memory.id] = scores.get(memory.id, 0.0) + _RELATION_RECALL_RRF_WEIGHT / (61 + rank)
                result.memories = [
                    memory_by_id[memory_id] for memory_id in sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))
                ][: input.limit]
                result.total_count = max(result.total_count, len(memory_by_id))
                result.relation_paths = relation_result.paths

        # Calculate search-only latency (vector/LLM search phase)
        search_latency_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
        result.search_latency_ms = search_latency_ms

        # Resolve None → server defaults for graph traversal
        effective_include_associations = (
            input.include_associations if input.include_associations is not None else self.default_include_associations
        )
        effective_traverse_depth = input.traverse_depth if input.traverse_depth is not None else self.default_traverse_depth
        effective_max_expansion = input.max_expansion if input.max_expansion is not None else self.max_graph_expansion

        # Association expansion (Phase 3A)
        assoc_ms = 0
        if effective_include_associations or effective_traverse_depth > 0:
            t0 = datetime.now(UTC)
            if self.assoc_consensus_boost_weight > 0.0:
                # A2: ranking-only — boost retrieved memories that are connected
                # to other retrieved memories; inject no new candidates. Note this
                # inherits the same association-enablement gate as expansion (it
                # only runs when include_associations/traverse_depth requested it),
                # so it is a no-op on plain vector recalls by design.
                result.memories = await self._apply_consensus_boost(
                    workspace_id=workspace_id,
                    memories=result.memories,
                    weight=self.assoc_consensus_boost_weight,
                )
            else:
                # Query-aware expansion needs the query vector so discovered
                # neighbors can be scored against the query. Skip the embed for
                # trivial/wildcard queries (no meaningful similarity to score).
                expansion_query_embedding = None
                if self.assoc_query_aware and input.query.strip() not in ("*", "**", ""):
                    expansion_query_embedding = await self.embedding.embed(input.query)
                result.memories = await self._expand_with_associations(
                    workspace_id=workspace_id,
                    memories=result.memories,
                    traverse_depth=effective_traverse_depth,
                    include_associations=effective_include_associations,
                    max_expansion=effective_max_expansion,
                    query_embedding=expansion_query_embedding,
                )
            assoc_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)

        # Reranking across all modes (Phase 3B)
        # Skip reranking for wildcard/trivial queries where ranking is meaningless
        rerank_ms = 0
        trivial_query = input.query.strip() in ("*", "", "**")
        if (self.reranker_service or self.rerank_mmr_enabled) and len(result.memories) > input.limit and not trivial_query:
            t0 = datetime.now(UTC)
            result.memories = await self._apply_reranking(
                query=input.query,
                memories=result.memories,
                limit=input.limit,
            )
            rerank_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)
        elif len(result.memories) > input.limit:
            # Truncate without reranking (trivial query or no reranker)
            result.memories = result.memories[: input.limit]

        # Apply detail_level filtering if requested
        detail_ms = 0
        if effective_detail_level != DetailLevel.FULL:
            t0 = datetime.now(UTC)
            filtered_memories = self._apply_detail_level(result.memories, effective_detail_level)
            result.memories = filtered_memories
            detail_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)

        # Increment access counts (and boost importance) in parallel
        access_ms = 0
        if result.memories:
            t0 = datetime.now(UTC)
            access_tasks = [self.increment_access(workspace_id, m.id) for m in result.memories]
            await asyncio.gather(*access_tasks, return_exceptions=True)
            access_ms = int((datetime.now(UTC) - t0).total_seconds() * 1000)

        # Annotate memories with trust scores and set drift_caveat if any are low-trust
        if result.memories:
            self._annotate_trust(result.memories)
            low_trust = [m for m in result.memories if m.trust_score is not None and m.trust_score < 0.5]
            if low_trust:
                result.drift_caveat = (
                    f"{len(low_trust)} of {len(result.memories)} recalled memories have low trust scores and may be stale or unreliable."
                )

        # Annotate memories with freshness scores and staleness warnings
        if result.memories:
            self._annotate_freshness(result.memories)
            freshness_scores = [m.freshness_score for m in result.memories if m.freshness_score is not None]
            if freshness_scores:
                result.freshness_metadata = {
                    "avg_freshness": round(sum(freshness_scores) / len(freshness_scores), 4),
                    "min_freshness": round(min(freshness_scores), 4),
                    "max_freshness": round(max(freshness_scores), 4),
                    "severe_count": sum(1 for m in result.memories if m.staleness_warning == "severe"),
                    "moderate_count": sum(1 for m in result.memories if m.staleness_warning == "moderate"),
                }

        # Collapse raw-parent / deterministic-segment siblings before enforcing
        # a response budget so one source cannot consume multiple result slots.
        result.memories = self._collapse_source_siblings(result.memories)

        effective_budget = input.budget_tokens
        if effective_budget is None and self.default_recall_token_budget > 0:
            effective_budget = self.default_recall_token_budget
        result.memories, result.budget_summary = pack_recall_memories(
            result.memories,
            effective_budget,
            effective_detail_level,
        )
        if input.include_confidence and self.retrieval_confidence_enabled:
            confidence, reasons = retrieval_confidence(
                input.query,
                result.memories,
                unresolved_entity=relation_unresolved,
            )
            result.retrieval_confidence = confidence
            result.confidence_reasons = reasons
        if generation_operation_ids and self.llm_service:
            summaries = [await self.llm_service.generation_summary(operation_id) for operation_id in generation_operation_ids]
            result.generation_summary = GenerationSummary(
                policy=self.llm_service.policy,
                calls=sum(summary.calls for summary in summaries),
                input_tokens=sum(summary.input_tokens for summary in summaries),
                output_tokens=sum(summary.output_tokens for summary in summaries),
            )
        elif input.trace:
            policy = self.llm_service.policy if self.llm_service else EnrichmentPolicy.DETERMINISTIC
            result.generation_summary = GenerationSummary(policy=policy)

        # Calculate total latency
        total_latency_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

        self.logger.info(
            "Recalled %s memories in %s ms "
            "(search: %s ms, associations: %s ms, rerank: %s ms, detail_filter: %s ms, access_tracking: %s ms) "
            "using %s mode (detail_level: %s)",
            len(result.memories),
            total_latency_ms,
            search_latency_ms,
            assoc_ms,
            rerank_ms,
            detail_ms,
            access_ms,
            result.mode_used,
            effective_detail_level.value,
        )

        # Phase 4: Cache recall result
        if self.cache and cache_key:
            try:
                await self.cache.set(cache_key, result.model_dump(), ttl_seconds=300)
            except Exception as e:
                self.logger.debug("Cache set failed: %s", e)

        return result

    @staticmethod
    def _collapse_source_siblings(memories: list[Memory]) -> list[Memory]:
        """Keep the first ranked item from each raw-source family."""
        child_ids_by_parent = {memory.source_memory_id for memory in memories if memory.source_memory_id}
        seen: set[str] = set()
        collapsed: list[Memory] = []
        for memory in memories:
            family = memory.source_memory_id or memory.id
            if memory.id in child_ids_by_parent:
                family = memory.id
            if family in seen:
                continue
            seen.add(family)
            collapsed.append(memory)
        return collapsed

    async def _merge_memories(
        self,
        workspace_id: str,
        existing: Memory,
        new_content: str,
        new_tags: list,
        new_metadata: dict,
        new_importance: float,
    ) -> Memory:
        """
        Merge new memory content into an existing memory.

        Strategy:
        - Keep newer content as primary (new_content replaces existing)
        - Tag union from both memories
        - Deep-merge metadata dicts (new overrides old)
        - Importance: max(existing, new) * 1.1 capped at 1.0
        - Store old content_hash in metadata.merged_from for provenance
        - Re-embed merged content
        - Re-generate semantic tiers
        """
        # Tag union
        merged_tags = sorted(set(existing.tags) | set(new_tags))

        # Deep-merge metadata: old base, new overrides, plus provenance
        merged_metadata = {**existing.metadata, **new_metadata}
        merged_metadata["merged_from"] = existing.content_hash

        # Importance boost capped at 1.0
        merged_importance = min(max(existing.importance, new_importance) * 1.1, 1.0)

        # Re-embed the new content
        new_embedding = await self.embedding.embed(new_content)

        updated = await self.storage.update_memory(
            workspace_id=workspace_id,
            memory_id=existing.id,
            content=new_content,
            embedding=new_embedding,
            importance=merged_importance,
            tags=merged_tags,
            metadata=merged_metadata,
        )

        # Merge changed content + metadata; reconcile the full-text index.
        await self._reconcile_fts_index(workspace_id, existing.id)

        # Re-generate semantic tiers for merged memory
        if self.tier_generation_service:
            try:
                await self.tier_generation_service.generate_tiers(existing.id, workspace_id)
            except Exception as e:
                self.logger.warning("Tier generation failed after merge for %s: %s", existing.id, e)

        return updated

    def _compute_trust_score(self, memory: Memory) -> tuple[float, dict]:
        """
        Compute a composite trust score for a memory.

        Components:
        - Freshness (0.3 weight): exponential decay same as recency scoring
        - Access frequency (0.2): min(access_count / 10, 1.0)
        - Decay factor (0.2): current memory.decay_factor value
        - Verification (0.15): 1.0 if pinned or metadata.verified, else 0.5
        - Source reliability (0.15): 1.0 session commit, 0.8 manual, 0.6 extraction

        Returns:
            Tuple of (trust_score, trust_signals dict)
        """
        now = datetime.now(UTC)

        # Freshness component: exponential decay (aligned with configurable freshness_half_life_days)
        age_hours = (now - memory.created_at).total_seconds() / 3600.0
        half_life = self.freshness_half_life_days * 24.0
        freshness = self._exponential_freshness(age_hours, half_life)

        # Access frequency
        access_freq = min(memory.access_count / 10.0, 1.0)

        # Decay factor (already 0.0-1.0)
        decay = memory.decay_factor

        # Verification
        if memory.pinned or memory.metadata.get("verified"):
            verification = 1.0
        else:
            verification = 0.5

        # Source reliability
        if memory.source_memory_id:
            # Came from session commit decomposition
            source_reliability = 1.0
        elif memory.metadata.get("source") == "manual" or (
            not memory.source_memory_id and not memory.source_document_id and not memory.source_thread_id
        ):
            source_reliability = 0.8
        else:
            # Extracted from document/thread
            source_reliability = 0.6

        trust_score = 0.3 * freshness + 0.2 * access_freq + 0.2 * decay + 0.15 * verification + 0.15 * source_reliability
        # Clamp to [0.0, 1.0]
        trust_score = max(0.0, min(1.0, trust_score))

        trust_signals = {
            "freshness": round(freshness, 4),
            "access_frequency": round(access_freq, 4),
            "decay_factor": round(decay, 4),
            "verification": round(verification, 4),
            "source_reliability": round(source_reliability, 4),
        }

        return trust_score, trust_signals

    def _annotate_trust(self, memories: list[Memory]) -> list[Memory]:
        """
        Annotate memories with trust scores in place.

        Sets trust_score and trust_signals on each memory.
        """
        for memory in memories:
            trust_score, trust_signals = self._compute_trust_score(memory)
            memory.trust_score = trust_score
            memory.trust_signals = trust_signals
        return memories

    def _route_by_intent(self, input: RecallInput, intent: QueryIntent) -> tuple[RecallInput, float | None, float | None]:
        """Translate query intent into soft retrieval nudges (caller-set values win).

        Returns the (possibly adjusted) RecallInput plus optional per-recall alias
        and backlink boost-weight overrides for entity queries. The only
        result-dropping nudge — a temporal window — is applied solely from an
        explicitly parsed date range; fuzzy temporal cues just raise recency.
        """
        updates: dict[str, object] = {}

        # Temporal / event: emphasize recency when the caller didn't set it.
        if (intent.has(TEMPORAL) or intent.has(EVENT)) and input.recency_weight is None:
            updates["recency_weight"] = _INTENT_TEMPORAL_RECENCY_WEIGHT

        # Explicitly parsed date range -> temporal window (the one hard filter).
        if intent.event_after is not None and input.event_after is None:
            updates["event_after"] = intent.event_after
        if intent.event_before is not None and input.event_before is None:
            updates["event_before"] = intent.event_before

        # Entity: amplify alias/backlink boosts and ensure graph expansion is on.
        alias_weight: float | None = None
        backlink_weight: float | None = None
        if intent.has(ENTITY):
            alias_weight = self.alias_boost_weight * _INTENT_ENTITY_BOOST_MULTIPLIER
            backlink_weight = self.backlink_boost_weight * _INTENT_ENTITY_BOOST_MULTIPLIER
            if input.include_associations is None:
                updates["include_associations"] = True

        adjusted = input.model_copy(update=updates) if updates else input
        return adjusted, alias_weight, backlink_weight

    def _select_channels(self, intent: QueryIntent | None) -> frozenset[str]:
        """Return the fuse arms permitted to fire for this query's intent (P4.1).

        This is the channel-selection seam. It is a PURE SUBTRACTION layer that
        sits BELOW the per-channel kill-switch flags: the caller still gates each
        arm on its own enablement flag (``... and channel in selected``), so a
        flag-off channel never fires regardless of what this returns.

        No-op guarantee: when ``intent_channel_select_enabled`` is OFF this always
        returns the FULL default set, so every enabled arm fires exactly as today
        (byte-identical behavior). The flag is the only switch that turns
        per-intent gating on.

        Cardinal safety (degrade to today's behavior on uncertainty): a MISSING
        intent (``None`` — e.g. query-intent routing disabled, or a wildcard query
        that skips classification), an EMPTY label set, or ANY label not present
        in ``_INTENT_CHANNEL_MATRIX`` (an unknown / future / low-confidence label)
        all degrade to ``_CHANNEL_FULL_DEFAULT``. A misclassification therefore
        degrades to the proven default (A+B+D+E), never to fewer arms than today.

        When every label is recognized, the selected set is the UNION of each
        label's matrix entry (a multi-label query gets the widest applicable set).
        """
        if not self.intent_channel_select_enabled:
            return _CHANNEL_FULL_DEFAULT

        # Missing intent or empty labels -> safe-degrade to today's behavior.
        if intent is None or not intent.labels:
            return _CHANNEL_FULL_DEFAULT

        selected: set[str] = set()
        for label in intent.labels:
            matrix_entry = _INTENT_CHANNEL_MATRIX.get(label)
            if matrix_entry is None:
                # Unknown / unmatched / low-confidence label: degrade to full set.
                return _CHANNEL_FULL_DEFAULT
            selected |= matrix_entry

        # Defensive: an empty union (should not happen given non-empty matrix
        # entries) also degrades rather than starving the query.
        return frozenset(selected) if selected else _CHANNEL_FULL_DEFAULT

    def _graph_channel_selected(self, intent: QueryIntent | None) -> bool:
        """Whether the dark graph-traversal arm (G) is permitted for this query (P4 channel G).

        SEPARATE from ``_select_channels`` (which governs the proven A/B/D/E
        backbone): G is dark and intent-narrowed, so it has its own gate.

        When intent-channel-select is OFF this returns True unconditionally — the
        graph arm's OWN flag (``graph_recall_enabled``, default OFF) is then the sole
        gate, so behavior is byte-identical to today until that flag is flipped.

        When intent-channel-select is ON, the arm is restricted to its on-task
        intents (``_GRAPH_CHANNEL_INTENTS`` = entity/event — the cross-source /
        multi-hop-ish cases). A missing intent (None / empty labels) does NOT enable
        the arm here: unlike the safe-degrade-to-full rule for the proven backbone,
        the dark graph arm stays narrow (no benefit to firing it on an unclassified
        query), which is the conservative choice for an unvalidated channel.
        """
        if not self.intent_channel_select_enabled:
            return True
        if intent is None or not intent.labels:
            return False
        return bool(intent.labels & _GRAPH_CHANNEL_INTENTS)

    async def _recall_browse(
        self,
        workspace_id: str,
        input: RecallInput,
    ) -> RecallResult:
        """Browse memories without embedding — used for wildcard queries like '*'.

        Returns recent memories ordered by creation time, skipping the
        embedding + vector search path entirely.
        """
        self.logger.debug("Wildcard query detected, using browse mode for workspace %s", workspace_id)
        far_past = datetime(2000, 1, 1, tzinfo=UTC)
        recent = await self.storage.get_recent_memories(
            workspace_id=workspace_id,
            created_after=far_past,
            limit=input.limit,
            detail_level="full",
            offset=input.offset,
        )
        # get_recent_memories returns dicts; fetch full Memory objects by ID
        memories = []
        for entry in recent:
            mem = await self.storage.get_memory(workspace_id, entry["id"], track_access=False)
            if mem is not None:
                memories.append(mem)

        return RecallResult(
            memories=memories,
            total_count=len(memories),
            query_tokens=0,
            search_latency_ms=0,
            mode_used=RecallMode.RAG,
        )

    async def _recall_rag(
        self,
        workspace_id: str,
        input: RecallInput,
        relevance_threshold: float,
        alias_weight: float | None = None,
        backlink_weight: float | None = None,
        intent: QueryIntent | None = None,
    ) -> RecallResult:
        """Pure vector similarity search.

        ``alias_weight``/``backlink_weight`` optionally override the service-level
        boost weights for this recall (used by entity-intent routing); when None
        the configured defaults apply.

        ``intent`` is the already-classified query intent (from the recall entry
        point). When intent-channel selection is enabled it gates WHICH enabled
        fuse arms fire for this query (see ``_select_channels``); when disabled or
        None it has no effect (the channel selector returns the full default set,
        so every enabled arm fires exactly as today).
        """
        # Channel selection (P4.1): which fuse arms may fire for this query's
        # intent. This is a PURE SUBTRACTION below the per-channel kill-switch
        # flags — each arm below is still gated on its own enablement flag AND
        # its channel being selected. When the feature flag is OFF this is the
        # full default set, so the AND short-circuits to today's behavior.
        selected_channels = self._select_channels(intent)
        # Wildcard/browse: skip embedding for trivial queries like "*"
        if input.query.strip() in ("*", "**", ""):
            return await self._recall_browse(workspace_id, input)

        # Generate query embedding
        query_embedding = await self.embedding.embed(input.query)

        # Overfetch to give the reranker a larger candidate pool
        overfetch_limit = input.limit * self.recall_overfetch

        # Search memories in current workspace
        include_archived = getattr(input, "include_archived", False)
        entity_filters = {}
        if getattr(input, "observer_id", None) is not None:
            entity_filters["observer_id"] = input.observer_id
        if getattr(input, "subject_id", None) is not None:
            entity_filters["subject_id"] = input.subject_id
        if getattr(input, "user_id", None) is not None:
            entity_filters["user_id"] = input.user_id

        date_filters = {}
        if getattr(input, "created_after", None) is not None:
            date_filters["created_after"] = (
                input.created_after.isoformat() if hasattr(input.created_after, "isoformat") else str(input.created_after)
            )
        if getattr(input, "created_before", None) is not None:
            date_filters["created_before"] = (
                input.created_before.isoformat() if hasattr(input.created_before, "isoformat") else str(input.created_before)
            )

        results = await self.storage.search_memories(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=overfetch_limit,
            offset=input.offset,
            min_relevance=relevance_threshold,
            types=[t.value for t in input.types] if input.types else None,
            subtypes=list(input.subtypes) if input.subtypes else None,
            tags=input.tags if input.tags else None,
            include_archived=include_archived,
            **entity_filters,
            **date_filters,
        )

        # Search _global workspace if enabled and not already searching it
        global_results = []
        if input.include_global and workspace_id != GLOBAL_WORKSPACE_ID:
            global_results = await self.storage.search_memories(
                workspace_id=GLOBAL_WORKSPACE_ID,
                query_embedding=query_embedding,
                limit=overfetch_limit,
                offset=input.offset,
                min_relevance=relevance_threshold,
                types=[t.value for t in input.types] if input.types else None,
                subtypes=list(input.subtypes) if input.subtypes else None,
                tags=input.tags if input.tags else None,
                include_archived=include_archived,
                **entity_filters,
                **date_filters,
            )

        # Search _global_user workspace filtered by user_id. This lets per-user
        # preferences (stored at workspace=_global_user, user_id=<uid>) surface
        # in any workspace the same user is acting in, without leaking across
        # users. Requires user_id to be set — otherwise the filter would return
        # everything in _global_user, which is not intended.
        global_user_results = []
        if input.include_global_user and input.user_id and workspace_id != GLOBAL_USER_WORKSPACE_ID:
            # The user_id filter flows through entity_filters only when the
            # caller set it on the RecallInput; we need to force it here
            # regardless so cross-user leakage cannot happen.
            global_user_filters = dict(entity_filters)
            global_user_filters["user_id"] = input.user_id
            global_user_results = await self.storage.search_memories(
                workspace_id=GLOBAL_USER_WORKSPACE_ID,
                query_embedding=query_embedding,
                limit=overfetch_limit,
                offset=input.offset,
                min_relevance=relevance_threshold,
                types=[t.value for t in input.types] if input.types else None,
                subtypes=list(input.subtypes) if input.subtypes else None,
                tags=input.tags if input.tags else None,
                include_archived=include_archived,
                **global_user_filters,
                **date_filters,
            )

        # Combine results
        all_results = results + global_results + global_user_results

        # Evidence contract: track which retrieval signals surfaced each memory
        # so the returned results can explain *why* they matched.
        signals: dict[str, set[str]] = {}
        for memory, _score in all_results:
            signals.setdefault(memory.id, set()).add("vector")

        # Hybrid retrieval: fuse in keyword (full-text/BM25) candidates via RRF
        # so exact/rare terms the vector arm missed can still surface and rank.
        # Channel B: in the default matrix B fires for every intent (it is part of
        # the proven backbone), so this AND only ever subtracts when a future
        # matrix entry drops it; the flag-off path keeps it byte-identical.
        if self.hybrid_search_enabled and _CHANNEL_KEYWORD in selected_channels:
            all_results = await self._fuse_keyword_results(
                workspace_id=workspace_id,
                input=input,
                vector_results=all_results,
                overfetch_limit=overfetch_limit,
                include_archived=include_archived,
                signals=signals,
            )

        # Entity-anchored fusion: RRF-fuse an entity-restricted candidate set
        # (memories whose speaker/entities intersect the query entities, ranked by
        # vector similarity) with the vector ranks. Done BEFORE boosts/truncation
        # so it affects the top-k. Rank discipline (RRF) lets the channel promote a
        # gold memory the vector arm missed without crowding out direct hits.
        # Channel D (entity-anchor): the per-channel flag is the GLOBAL kill-switch
        # (flag-off -> never fires regardless of intent); the channel-selection
        # term only further restricts WHICH intents fire it when the feature is on.
        if self.entity_anchor_enabled and _CHANNEL_ENTITY in selected_channels and input.query.strip() not in ("*", "**", ""):
            all_results = await self._fuse_entity_results(
                workspace_id=workspace_id,
                input=input,
                query_embedding=query_embedding,
                vector_results=all_results,
                overfetch_limit=overfetch_limit,
                signals=signals,
            )

        # Fact channel fusion: RRF-fuse a fact-restricted candidate set
        # (subtype="fact" memories ranked by vector similarity) with the vector
        # ranks, modeled exactly on the entity-anchored arm. Done AFTER the entity
        # arm and BEFORE exclude-id/truncation so it affects the top-k.
        #
        # Pool purity (load-bearing): the PRIMARY vector arm must NOT contain fact
        # memories — mixing date-stripped facts into the turn pool reintroduces the
        # temporal regression. So we strip subtype=="fact" out of the running
        # results BEFORE fusing the fact arm in; only the fact arm sees facts. The
        # entity/keyword arms fetch their own candidates independently, so this
        # filter does not starve them. No-op for backends/builds without facts.
        # Channel E (fact): same kill-switch precedence as channel D. The pool
        # purity strip (removing subtype=="fact" from the primary arm) is part of
        # the fact arm and stays coupled to it — when the fact arm does not fire
        # (flag off OR intent does not select it) the strip is skipped too, exactly
        # as today, so no behavior changes for unselected queries.
        if self.fact_channel_enabled and _CHANNEL_FACT in selected_channels and input.query.strip() not in ("*", "**", ""):
            all_results = [(m, s) for m, s in all_results if m.subtype != MemorySubtype.FACT.value]
            all_results = await self._fuse_fact_results(
                workspace_id=workspace_id,
                input=input,
                query_embedding=query_embedding,
                vector_results=all_results,
                signals=signals,
            )

        # Cue-anchor fusion (Memora-inspired): RRF-fuse a candidate set reached via
        # cue-anchor vector similarity (short "[entity] + [aspect]" keys generated
        # per memory, dereferenced back to their primary memories) with the vector
        # ranks, modeled exactly on the fact arm. Done AFTER the fact arm and BEFORE
        # exclude-id/truncation so it affects the top-k. Bolt-on arm — the primary
        # content-embedding arm is untouched (no pool-purity strip needed). The
        # per-channel flag is the GLOBAL kill-switch (flag-off -> never fires, no
        # storage call). Ships DARK (default OFF) -> recall byte-identical. No-op for
        # wildcard queries and for workspaces without cue anchors (OSS base no-op).
        if self.cue_channel_enabled and input.query.strip() not in ("*", "**", ""):
            all_results = await self._fuse_cue_results(
                workspace_id=workspace_id,
                input=input,
                query_embedding=query_embedding,
                vector_results=all_results,
                signals=signals,
            )

        # Registry-backed entity expansion: resolve the query's entities to
        # canonical registry entities (allow_create=False) and RRF-fuse each
        # entity's MEMBER memories — which include memories linked via aliases and
        # fuzzy/LLM-merged surface forms — with the vector ranks. Done AFTER the
        # entity-anchor/fact arms and BEFORE exclude-id/truncation so it affects
        # the top-k. Independent of (additive to) the metadata entity-anchor arm;
        # both can be on. No-op for wildcard queries.
        if self.entity_registry_recall_enabled and input.query.strip() not in ("*", "**", ""):
            all_results = await self._fuse_registry_results(
                workspace_id=workspace_id,
                input=input,
                query_embedding=query_embedding,
                vector_results=all_results,
                signals=signals,
            )

        # Graph-traversal fusion (P4 channel G): resolve the query's entities to
        # canonical entities and RRF-fuse a SMALL, BOUNDED neighborhood of the
        # memories each entity MENTIONS (via GraphQueryService.entity_neighborhood)
        # with the vector ranks. Done AFTER the registry arm and BEFORE exclude-id/
        # truncation so it affects the top-k. This is the cross-source /
        # relationship-traversal channel over the canonical-entity layer — NOT a
        # chat-recall expansion; the neighborhood pool is hard-bounded
        # (graph_recall_pool) so it can never become the killed member-dump flood.
        # Channel G (graph): the per-channel flag is the GLOBAL kill-switch (flag-off
        # -> never fires regardless of intent, and no graph_query_service call is
        # made). G is NOT part of the A/B/D/E channel-selection matrix (so that
        # contract stays byte-identical); instead, WHEN intent-channel-select is ON,
        # the graph arm is additionally restricted to its on-task intents
        # (_GRAPH_CHANNEL_INTENTS = entity/event — the cross-source/multi-hop-ish
        # cases). When intent-channel-select is OFF, the own-flag is the sole gate.
        # Default OFF -> recall byte-identical.
        if self.graph_recall_enabled and self._graph_channel_selected(intent) and input.query.strip() not in ("*", "**", ""):
            all_results = await self._fuse_graph_results(
                workspace_id=workspace_id,
                input=input,
                query_embedding=query_embedding,
                vector_results=all_results,
                signals=signals,
            )

        # Filter out already-surfaced memory IDs
        exclude_ids = getattr(input, "exclude_ids", None)
        if exclude_ids:
            exclude_set = set(exclude_ids)
            all_results = [(m, s) for m, s in all_results if m.id not in exclude_set]

        # Apply scope boosts
        context_id = input.context_id if input.context_id else DEFAULT_CONTEXT_ID
        boosted_memories = self.apply_scope_boosts(
            all_results,
            query_context_id=context_id,
            query_workspace_id=workspace_id,
            boosts=None,  # Use default boosts
        )

        # Apply recency boost
        effective_recency_weight = input.recency_weight if input.recency_weight is not None else DEFAULT_RECENCY_WEIGHT
        boosted_memories = self.apply_recency_boost(
            boosted_memories,
            recency_weight=effective_recency_weight,
        )

        # Effective boost weights (entity-intent routing may override per-recall).
        effective_alias_weight = alias_weight if alias_weight is not None else self.alias_boost_weight
        effective_backlink_weight = backlink_weight if backlink_weight is not None else self.backlink_boost_weight

        # Apply alias hop: boost memories whose metadata aliases match query terms.
        if effective_alias_weight > 0.0:
            boosted_memories = self.apply_alias_boost(
                input.query,
                boosted_memories,
                weight=effective_alias_weight,
                signals=signals,
            )

        # Apply backlink salience: boost well-connected (high in-degree) memories
        # over the full candidate pool, before truncating to the requested limit.
        if effective_backlink_weight > 0.0:
            boosted_memories = await self.apply_backlink_boost(
                workspace_id,
                boosted_memories,
                weight=effective_backlink_weight,
                signals=signals,
            )

        # Temporal: filter by effective event-time window and/or order by time,
        # over the full candidate pool before truncating to the requested limit.
        if input.event_after is not None or input.event_before is not None or input.time_order:
            boosted_memories = self._apply_temporal_filter_order(boosted_memories, input)

        # Supersession: demote or drop memories a later memory has superseded. Applied
        # over the full pool, before truncation, so a demoted memory can lose its slot.
        boosted_memories = await self.apply_supersession(workspace_id, boosted_memories, signals)

        # Rerank the over-fetched candidate pool, then take the top limit. The
        # overfetch (recall_overfetch x limit) exists precisely to give the
        # reranker a larger pool; truncating to limit *before* reranking (the
        # prior behaviour) discarded that pool and starved the reranker. When no
        # reranker is configured this is a plain top-limit slice. Returning <=
        # limit preserves the contract enterprise cold-tier fallback relies on.
        memories = await self._apply_reranking(input.query, boosted_memories, input.limit)

        # Annotate the evidence contract on the returned memories.
        for memory in memories:
            matched = signals.get(memory.id)
            if matched:
                memory.match_signals = sorted(matched)

        return RecallResult(
            memories=memories,
            total_count=len(all_results),
            query_tokens=0,
            search_latency_ms=0,  # Will be set by caller
            mode_used=RecallMode.RAG,
        )

    async def _fuse_keyword_results(
        self,
        workspace_id: str,
        input: RecallInput,
        vector_results: list[tuple[Memory, float]],
        overfetch_limit: int,
        include_archived: bool,
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Fetch keyword (full-text/BM25) candidates and RRF-fuse with vector results.

        Keyword candidates are pulled from the same workspace scopes as the
        vector arm (current + _global + per-user _global_user) and post-filtered
        in Python to honour the same RecallInput predicates, since the storage
        full_text_search contract does not take structured filters. When a
        ``signals`` map is given, keyword-matched ids are recorded for the recall
        evidence contract. On any keyword-side failure this degrades gracefully
        to the vector results.
        """
        try:
            keyword_memories = await self._keyword_candidates(
                workspace_id=workspace_id,
                input=input,
                overfetch_limit=overfetch_limit,
                include_archived=include_archived,
            )
        except Exception as e:
            # Degrade gracefully, but warn: a persistent keyword-arm failure
            # silently turns hybrid into vector-only on every query, which is a
            # real (often invisible) quality regression worth surfacing.
            self.logger.warning("Hybrid keyword arm failed; degrading to vector-only results: %s", e)
            return vector_results

        if not keyword_memories:
            return vector_results

        if signals is not None:
            for memory in keyword_memories:
                signals.setdefault(memory.id, set()).add("keyword")

        return self._rrf_fuse(vector_results, keyword_memories, k=self.hybrid_rrf_k)

    def _seed_perspective_ids(
        self,
        content: str,
        observer_id: str | None,
        subject_id: str | None,
    ) -> tuple[str | None, str | None]:
        """Derive observer_id / subject_id from the memory content when not explicitly set.

        This is the OSS "perspective seed" — a cheap, regex-based heuristic that
        populates the perspective fields for free during the hot ingest path so
        the model starts accreting data without requiring callers to supply them.

        Rules (all non-destructive: explicit caller values always win):
        - ``observer_id``: if None and the content has a ``[timestamp] Speaker:``
          prefix, set observer_id = speaker name. This is a high-confidence
          attribution (structural, not inferred).
        - ``subject_id``: left as-is. The "self-report" default
          (subject_id == observer_id) is semantically correct for single-speaker
          turns, but silently assigning it would conflate "speaker observed
          something about themselves" with "no subject known" — callers who need
          the self-report case should set it explicitly. The enterprise
          extract_facts path populates subject_id from LLM-extracted triples.

        Returns the (observer_id, subject_id) pair to store, never raising.
        """
        if observer_id is not None:
            # Caller explicitly set observer_id — never overwrite.
            return observer_id, subject_id

        if not content:
            return observer_id, subject_id

        try:
            if self.extraction_service is not None:
                extracted = self.extraction_service.extract_entities(content)
            else:
                extracted = extract_entities(content)
            speaker = extracted.get("speaker")
            if speaker:
                self.logger.debug("Perspective seed: observer_id derived from speaker %r", speaker)
                return speaker, subject_id
        except Exception as e:
            self.logger.debug("Perspective seed skipped for content (len=%d): %s", len(content), e)

        return observer_id, subject_id

    def _maybe_add_entity_metadata(self, content: str, metadata: dict | None) -> dict | None:
        """Fold extracted entities into a memory's metadata at ingest time.

        Populates ``metadata['speaker']``, ``metadata['entities']`` and (when the
        provider types them) ``metadata['entity_types']`` so the entity-anchored
        recall channel (``_fuse_entity_results``) has something to match on in
        production AND registry accretion (``_accrete_entities``) has typed
        entities — without this, entity metadata only existed in the eval harness
        and the channel was a no-op outside benchmarks.

        The extraction is sourced from the pluggable ``ExtractionService``
        (canonical home; providers may override with typed NER/LLM extraction),
        falling back to the shared regex util only when no service is wired. This
        is THE hook that lets the selected provider (e.g. GLiNER2) drive both the
        anchor channel and registry accretion. The richer LLM fact graph runs
        separately in the async ``auto_enrich`` task (see ``_inline_auto_enrich``).

        Runs whenever the entity-anchor channel OR the entity registry is enabled:
        accretion needs the typed metadata even when only the registry flag is on.
        No-op (returns ``metadata`` unchanged) when neither flag is set, the
        content is empty, or nothing was extracted. Never clobbers caller-provided
        ``speaker``/``entities``/``entity_types`` keys.
        """
        if (not self.entity_anchor_enabled and not self.entity_registry_enabled) or not content:
            return metadata
        if self.extraction_service is not None:
            extracted = self.extraction_service.extract_entities(content)
        else:
            extracted = extract_entities(content)
        if not extracted.get("speaker") and not extracted.get("entities"):
            return metadata
        md = dict(metadata or {})
        if extracted["speaker"] and "speaker" not in md:
            md["speaker"] = extracted["speaker"]
        if extracted["entities"] and "entities" not in md:
            md["entities"] = extracted["entities"]
        # entity_types carries the per-entity EntityType values from typed
        # providers (regex/default returns {}). Stored so accretion can assign
        # proper types; back-compat consumers ignore the extra key.
        entity_types = extracted.get("entity_types")
        if entity_types and "entity_types" not in md:
            md["entity_types"] = entity_types
        return md

    async def _fuse_entity_results(
        self,
        workspace_id: str,
        input: RecallInput,
        query_embedding: list[float],
        vector_results: list[tuple[Memory, float]],
        overfetch_limit: int,
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Fetch entity-anchored candidates and RRF-fuse with the vector results.

        The query entities (capitalized proper-noun spans minus question words)
        restrict the candidate set to memories whose ``speaker`` or ``entities``
        metadata intersects them; the storage backend ranks that set by vector
        similarity. Those ranks are RRF-fused (``entity_anchor_rrf_k``) with the
        vector ranks, mirroring the hybrid keyword arm. When the query has no
        entities, or the backend has no entity metadata, this is a no-op. On any
        entity-side failure it degrades gracefully to the vector results.

        The entity arm is fetched at a deeper pool than the vector overfetch
        (``_ENTITY_ANCHOR_POOL``): the speaker-restricted set is small (only
        memories spoken by the query entities), and the channel's value is
        promoting a gold memory that sits *deep* in the vector ranking. The
        offline lab (``.slop/probe_entity_fusion.py``) confirms the
        SPEAKER-restricted candidate set is the discriminating signal: ORing in
        the broader mention set (``metadata['entities']``) dilutes the ranking and
        *lowers* recall (0.527 -> 0.501), since within a conversation the entity
        is mentioned by most turns. So the channel anchors on speaker only.

        The entity arm intentionally does NOT apply a relevance floor: its whole
        value is promoting a gold memory that scores *low* on raw vector similarity
        (the multi-hop case). RRF rank discipline keeps that from crowding out
        direct hits, so a floor would only suppress the signal we want.
        """
        entities = extract_query_entities(input.query)
        if not entities:
            return vector_results

        entity_pool = max(overfetch_limit, _ENTITY_ANCHOR_POOL)
        try:
            entity_results = await self.storage.search_memories_by_entities(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                entities=entities,
                limit=entity_pool,
            )
        except NotImplementedError:
            # Backend does not support entity-anchored search; degrade silently.
            return vector_results
        except Exception as e:
            self.logger.warning("Entity-anchored arm failed; degrading to vector-only results: %s", e)
            return vector_results

        if not entity_results:
            return vector_results

        # entity_results is already ranked best-first by vector similarity within
        # the entity-restricted set; pass the memories as the second RRF arm.
        entity_memories = [memory for memory, _score in entity_results]

        if signals is not None:
            for memory in entity_memories:
                signals.setdefault(memory.id, set()).add("entity")

        return self._rrf_fuse(vector_results, entity_memories, k=self.entity_anchor_rrf_k)

    async def _fuse_fact_results(
        self,
        workspace_id: str,
        input: RecallInput,
        query_embedding: list[float],
        vector_results: list[tuple[Memory, float]],
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Fetch fact-memory candidates and RRF-fuse with the vector results.

        Clone of ``_fuse_entity_results`` for the fact channel: instead of an
        entity-restricted set, the candidate set is restricted to fact memories
        (``subtype="fact"``) ranked by vector similarity, then RRF-fused
        (``fact_channel_rrf_k``) with the vector ranks. The caller has already
        stripped facts out of ``vector_results`` (pool purity), so the two arms
        are disjoint: the vector arm carries date-grounded raw turns, the fact arm
        carries clean normalised triples. This is the in-core form of a two-store
        RRF arrangement (separate turn and fact stores, fused at query time).

        The fact arm intentionally does NOT apply a relevance floor (same rationale
        as the entity arm): its whole value is promoting a gold fact that scores
        *low* on raw vector similarity (the multi-hop case). RRF rank discipline
        keeps that from crowding out direct hits, so a floor would only suppress
        the signal we want.

        When there are no fact memories, this is a no-op. On any fact-side failure
        it degrades gracefully to the vector results.
        """
        fact_pool = max(input.limit * self.recall_overfetch, self.fact_channel_pool)
        try:
            fact_results = await self.storage.search_memories(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                limit=fact_pool,
                subtypes=[MemorySubtype.FACT.value],
                min_relevance=0.0,
                include_archived=getattr(input, "include_archived", False),
            )
        except Exception as e:
            self.logger.warning("Fact channel arm failed; degrading to vector-only results: %s", e)
            return vector_results

        if not fact_results:
            return vector_results

        # fact_results is already ranked best-first by vector similarity within the
        # fact-restricted set; pass the memories as the second RRF arm.
        fact_memories = [memory for memory, _score in fact_results]

        if signals is not None:
            for memory in fact_memories:
                signals.setdefault(memory.id, set()).add("fact")

        return self._rrf_fuse(vector_results, fact_memories, k=self.fact_channel_rrf_k)

    async def _fuse_cue_results(
        self,
        workspace_id: str,
        input: RecallInput,
        query_embedding: list[float],
        vector_results: list[tuple[Memory, float]],
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Fetch cue-anchor candidates and RRF-fuse with the vector results.

        Clone of ``_fuse_fact_results`` for the cue channel (Memora-inspired
        abstraction+cue indexing): instead of a subtype-restricted vector search,
        the candidate set is the primary memories reached via cue-anchor vector
        similarity — short "[entity] + [aspect]" keys generated per memory, embedded
        and stored separately, then dereferenced back to their parent memory (see
        StorageBackend.search_cue_anchors). Those ranks are RRF-fused
        (``cue_channel_rrf_k``) with the vector ranks. This is a BOLT-ON arm: the
        primary content-embedding arm is untouched.

        The cue arm intentionally does NOT apply a relevance floor (same rationale
        as the fact/entity arms): its whole value is promoting a memory whose
        FACET the query matches even though the memory scores *low* on raw content
        similarity. RRF rank discipline keeps that from crowding out direct hits,
        so a floor would only suppress the signal we want.

        Cue anchors reached via similarity may point at the same memory more than
        once, so the dereferenced list is deduped by memory id (keeping best-first
        order). When the workspace has no cue anchors (OSS base no-op, or an
        enterprise workspace ingested with the channel off), this is a no-op. On
        any cue-side failure it degrades gracefully to the vector results.
        """
        cue_pool = max(input.limit * self.recall_overfetch, self.cue_channel_pool)
        try:
            cue_results = await self.storage.search_cue_anchors(
                workspace_id=workspace_id,
                query_embedding=query_embedding,
                limit=cue_pool,
            )
        except Exception as e:
            self.logger.warning("Cue channel arm failed; degrading to vector-only results: %s", e)
            return vector_results

        if not cue_results:
            return vector_results

        # cue_results is already ranked best-first by cue-anchor similarity, with
        # each anchor dereferenced to its primary memory. Dedup by memory id
        # (multiple cues can point at one memory), preserving best-first order.
        cue_memories: list[Memory] = []
        seen: set[str] = set()
        for memory, _score in cue_results:
            if memory.id in seen:
                continue
            seen.add(memory.id)
            cue_memories.append(memory)

        if signals is not None:
            for memory in cue_memories:
                signals.setdefault(memory.id, set()).add("cue")

        return self._rrf_fuse(vector_results, cue_memories, k=self.cue_channel_rrf_k)

    async def _fuse_registry_results(
        self,
        workspace_id: str,
        input: RecallInput,
        query_embedding: list[float],
        vector_results: list[tuple[Memory, float]],
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Resolve query entities to canonical registry entities and RRF-fuse their members.

        Clone of ``_fuse_entity_results`` for the registry channel: the query
        entities (capitalized proper-noun spans minus question words — the SAME
        extraction the entity-anchor arm uses) are resolved against the registry
        with ``allow_create=False`` (a query string must NEVER create an entity);
        each resolved canonical entity's MEMBER memories are pulled, ranked by
        vector similarity to the query, and RRF-fused
        (``entity_registry_recall_rrf_k``) with the vector ranks.

        The value over the metadata-string entity-anchor arm: members include
        memories linked via aliases and fuzzy/LLM-merged surface forms, so this
        promotes a gold member whose text never contained the query string
        verbatim. OSS-first (exact+alias registry); enterprise inherits richer
        canonical entities via the same path.

        Entity-type inference at query time: accretion stores the speaker as
        ``PERSON`` and all other spans as ``CONCEPT`` (see ``_accrete_entities``).
        A query has no speaker, so each query entity is resolved against BOTH types
        and the resolved entity ids are deduped — this matches members regardless
        of how the surface form was typed at ingest.

        Like the entity/fact arms, this applies NO relevance floor (its value is
        promoting a member that scores low on raw vector similarity; RRF rank
        discipline prevents crowding). Bounded by ``entity_registry_recall_pool``.
        A per-entity sub-cap (pool / n_entities, min 1) prevents a single
        high-member CONCEPT entity from monopolising the pool and starving more
        selective PERSON entities. Fail-safe: any error degrades to the vector
        results (never breaks recall). When the query has no entities, none
        resolve, or no members are found, this is a no-op.
        """
        if self.entity_registry_service is None:
            return vector_results

        import math

        from ...models.entity_registry import EntityType

        try:
            names = extract_query_entities(input.query)
            if not names:
                return vector_results

            pool = self.entity_registry_recall_pool

            # Resolve each query entity to its canonical entity id. A query string
            # must never create an entity -> allow_create=False (mandatory). The
            # miss path raises LookupError; treat it (and any per-entity error) as
            # "no match" and continue. Resolve against both accretion-used types
            # (PERSON for speakers, CONCEPT for other spans) and dedup ids.
            entity_ids: list[str] = []
            seen_ids: set[str] = set()
            for name in names:
                for etype in (EntityType.PERSON, EntityType.CONCEPT):
                    try:
                        resolution = await self.entity_registry_service.resolve(
                            workspace_id,
                            name,
                            etype,
                            allow_create=False,
                        )
                    except LookupError:
                        continue
                    except Exception as e:
                        self.logger.debug("Registry recall: resolve(%r, %s) failed: %s", name, etype, e)
                        continue
                    eid = resolution.entity.id
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        entity_ids.append(eid)

            if not entity_ids:
                return vector_results

            # Per-entity sub-cap: divide the pool evenly across resolved entities
            # (min 1) so a high-member CONCEPT entity cannot monopolise the pool
            # and starve a more selective PERSON entity. Overall pool bound is
            # still enforced after round-robin collection.
            per_entity = max(1, math.ceil(pool / len(entity_ids)))

            # Collect member memory ids across the resolved entities, bounded by
            # per_entity per canonical entity and the overall pool. Only "mention"
            # role members are fetched: "self" members (speaker-authored turns)
            # are intentionally excluded so a speaker's own history does not crowd
            # out third-party observations about them in the expansion result.
            member_ids: list[str] = []
            seen_members: set[str] = set()
            for eid in entity_ids:
                if len(member_ids) >= pool:
                    break
                members = await self.entity_registry_service.list_members(workspace_id, eid, role="mention", limit=per_entity)
                for member in members:
                    mid = member.memory_id
                    if mid not in seen_members:
                        seen_members.add(mid)
                        member_ids.append(mid)
                        if len(member_ids) >= pool:
                            break

            if not member_ids:
                return vector_results

            # Fetch member memories and rank by vector similarity to the query
            # (same as the entity arm — no relevance floor). track_access=False:
            # this is an internal read that must not perturb decay tracking.
            # TECH_DEBT: this is N+1 over the (bounded) pool; replace with a
            # batched get_memories call before production enablement.
            ranked: list[tuple[Memory, float]] = []
            for mid in member_ids:
                memory = await self.storage.get_memory(workspace_id, mid, track_access=False)
                if memory is None or not memory.embedding:
                    continue
                relevance = cosine_similarity(query_embedding, memory.embedding)
                ranked.append((memory, relevance))

            if not ranked:
                return vector_results

            # Anti-flood relevance floor: a frequent canonical entity owns hundreds
            # of members; injecting the low-relevance tail floods the RRF pool and
            # destroys recall (measured LoCoMo 0.517 -> 0.083 unfiltered). Drop
            # members below the query-relevance floor. Default floor 0.0 keeps the
            # legacy behavior; a positive floor makes the channel additive.
            floor = self.entity_registry_recall_min_relevance
            if floor > 0.0:
                ranked = [pair for pair in ranked if pair[1] >= floor]
                if not ranked:
                    return vector_results

            ranked.sort(key=lambda pair: pair[1], reverse=True)
            member_memories = [memory for memory, _score in ranked]

            if signals is not None:
                for memory in member_memories:
                    signals.setdefault(memory.id, set()).add("registry")

            return self._rrf_fuse(vector_results, member_memories, k=self.entity_registry_recall_rrf_k)
        except Exception as e:
            self.logger.debug("Registry-expansion arm failed; degrading to vector-only results: %s", e)
            return vector_results

    async def _fuse_graph_results(
        self,
        workspace_id: str,
        input: RecallInput,
        query_embedding: list[float],
        vector_results: list[tuple[Memory, float]],
        signals: dict[str, set[str]] | None = None,
    ) -> list[tuple[Memory, float]]:
        """Resolve query entities and RRF-fuse a BOUNDED graph neighborhood (P4 channel G).

        The cross-source / relationship-traversal arm over the canonical-entity
        layer. The query entities (capitalized proper-noun spans minus question
        words — the SAME extraction the entity-anchor/registry arms use) are
        resolved to canonical entities with ``allow_create=False`` (a query string
        must NEVER create an entity); for each resolved entity, a SMALL, BOUNDED
        neighborhood of the memories it MENTIONS is pulled via
        ``GraphQueryService.entity_neighborhood(..., memory_limit=SMALL)``, ranked by
        vector similarity, and RRF-fused (``graph_recall_rrf_k``) with the vector
        ranks.

        ANTI-FLOOD GUARDRAIL (load-bearing — the cardinal lesson of the killed
        registry-recall flood, LoCoMo 0.54 -> 0.13): the registry-recall flood
        dumped ~all turns mentioning an entity (a ~1000-turn weakly-relevant set)
        into the pool. RRF caps a channel's marginal DISPLACEMENT but NOT a channel
        whose entire pool is noise — so the graph arm's candidate pool MUST be
        SMALL. The neighborhood ``memory_limit`` passed to ``entity_neighborhood``
        is ``min(input.limit * recall_overfetch, graph_recall_pool)`` — the recall
        overfetch budget, hard-capped at ``graph_recall_pool`` (default 50), NEVER
        100 and NEVER unbounded. The total fused candidate count is additionally
        clipped to that same bound across all resolved entities. This is a SEPARATE
        RRF pool (mirrors ``_fuse_entity_results``); it is never merged into the
        dense/vector pool.

        Fusion decision (chat-only LoCoMo measurement): this fuses
        ``memories_for_entity`` (the entity's bounded MENTIONED memories) — the
        only graph product that yields recall candidates for a chat-only corpus.
        ``entities_co_mentioned`` (the cross-source bridge) carries entity ids, not
        memory candidates, so it is the VALUE for cross-source retrieval but has
        nothing to fuse in a single-source chat measurement; it is intentionally
        NOT expanded here (expanding co-mentioned entities into their member
        memories would reintroduce the unbounded-flood risk). Like the entity/fact/
        registry arms, NO relevance floor is applied (RRF rank discipline prevents
        crowding). Fail-safe: a missing graph_query_service, no resolvable entities,
        an empty neighborhood, or ANY error degrades to the vector results (never
        breaks recall) and — when the service is missing or no entity resolves — is
        a clean no-op with zero graph calls.

        Tier note: the OSS relational ``entity_neighborhood`` (over the registry
        tables) proves the channel; the enterprise AGE backend answers it richly
        over the C1 Entity/MENTIONS graph.
        """
        if self.graph_query_service is None:
            return vector_results
        # Resolving a query entity NAME -> canonical entity id requires the
        # registry resolver (the graph service keys on entity_id, not name). With
        # no registry wired we cannot anchor the traversal -> clean no-op.
        if self.entity_registry_service is None:
            return vector_results

        from ...models.entity_registry import EntityType

        try:
            names = extract_query_entities(input.query)
            if not names:
                return vector_results

            # ANTI-FLOOD bound: the overfetch budget, hard-capped at graph_recall_pool.
            # This is the per-entity entity_neighborhood memory_limit AND the overall
            # graph-candidate ceiling. Bounded, never 100/unbounded.
            pool = min(input.limit * self.recall_overfetch, self.graph_recall_pool)
            if pool < 1:
                pool = 1

            # Resolve each query entity to its canonical entity id (allow_create=False
            # — a query string must never create an entity). Resolve against both
            # accretion-used types (PERSON for speakers, CONCEPT for other spans) and
            # dedup ids — mirrors the registry arm.
            entity_ids: list[str] = []
            seen_ids: set[str] = set()
            for name in names:
                for etype in (EntityType.PERSON, EntityType.CONCEPT):
                    try:
                        resolution = await self.entity_registry_service.resolve(
                            workspace_id,
                            name,
                            etype,
                            allow_create=False,
                        )
                    except LookupError:
                        continue
                    except Exception as e:
                        self.logger.debug("Graph recall: resolve(%r, %s) failed: %s", name, etype, e)
                        continue
                    eid = resolution.entity.id
                    if eid not in seen_ids:
                        seen_ids.add(eid)
                        entity_ids.append(eid)

            if not entity_ids:
                return vector_results

            # Collect bounded neighborhood memory ids across resolved entities. Each
            # entity_neighborhood call is itself bounded by memory_limit=pool; the
            # running collection is additionally clipped to pool overall so multiple
            # resolved entities cannot stack into an unbounded set.
            member_ids: list[str] = []
            seen_members: set[str] = set()
            for eid in entity_ids:
                if len(member_ids) >= pool:
                    break
                neighborhood = await self.graph_query_service.entity_neighborhood(
                    workspace_id,
                    eid,
                    hops=1,
                    memory_limit=pool,
                )
                for mention in neighborhood.memories_for_entity:
                    mid = mention.memory_id
                    if mid not in seen_members:
                        seen_members.add(mid)
                        member_ids.append(mid)
                        if len(member_ids) >= pool:
                            break

            if not member_ids:
                return vector_results

            # Fetch neighborhood memories and rank by vector similarity to the query
            # (no relevance floor — same as the entity/registry arms). track_access=
            # False: an internal read must not perturb decay tracking.
            ranked: list[tuple[Memory, float]] = []
            for mid in member_ids:
                memory = await self.storage.get_memory(workspace_id, mid, track_access=False)
                if memory is None or not memory.embedding:
                    continue
                relevance = cosine_similarity(query_embedding, memory.embedding)
                ranked.append((memory, relevance))

            if not ranked:
                return vector_results

            ranked.sort(key=lambda pair: pair[1], reverse=True)
            graph_memories = [memory for memory, _score in ranked]

            if signals is not None:
                for memory in graph_memories:
                    signals.setdefault(memory.id, set()).add("graph")

            return self._rrf_fuse(vector_results, graph_memories, k=self.graph_recall_rrf_k)
        except Exception as e:
            self.logger.debug("Graph-traversal arm failed; degrading to vector-only results: %s", e)
            return vector_results

    async def _keyword_candidates(
        self,
        workspace_id: str,
        input: RecallInput,
        overfetch_limit: int,
        include_archived: bool,
    ) -> list[Memory]:
        """Gather BM25-ranked keyword candidates across the recall workspace scopes."""
        scopes = [workspace_id]
        if input.include_global and workspace_id != GLOBAL_WORKSPACE_ID:
            scopes.append(GLOBAL_WORKSPACE_ID)
        if input.include_global_user and input.user_id and workspace_id != GLOBAL_USER_WORKSPACE_ID:
            scopes.append(GLOBAL_USER_WORKSPACE_ID)

        ranked: list[Memory] = []
        seen: set[str] = set()
        for scope in scopes:
            hits = await self.storage.full_text_search(
                workspace_id=scope,
                query=input.query,
                limit=overfetch_limit,
                offset=input.offset,
            )
            for memory in hits:
                if memory.id in seen:
                    continue
                if not self._keyword_hit_passes_filters(memory, input, include_archived):
                    continue
                seen.add(memory.id)
                ranked.append(memory)
        return ranked

    @staticmethod
    def _keyword_hit_passes_filters(memory: Memory, input: RecallInput, include_archived: bool) -> bool:
        """Replicate vector-search predicates for a keyword candidate.

        full_text_search returns full Memory rows without honouring structured
        filters, so we apply them here to keep the keyword arm consistent with
        the vector arm (no leaking archived or filtered-out memories).
        """
        if not include_archived and memory.status != MemoryStatus.ACTIVE:
            return False
        if input.types and memory.type not in input.types:
            return False
        if input.subtypes and memory.subtype not in input.subtypes:
            return False
        if input.tags and not all(tag in memory.tags for tag in input.tags):
            return False
        if input.observer_id is not None and memory.observer_id != input.observer_id:
            return False
        if input.subject_id is not None and memory.subject_id != input.subject_id:
            return False
        if input.user_id is not None and memory.user_id != input.user_id:
            return False
        if input.created_after is not None and memory.created_at < input.created_after:
            return False
        if input.created_before is not None and memory.created_at > input.created_before:
            return False
        return True

    def _rrf_fuse(
        self,
        vector_results: list[tuple[Memory, float]],
        keyword_memories: list[Memory],
        k: int,
    ) -> list[tuple[Memory, float]]:
        """Reciprocal Rank Fusion of the vector and keyword rankings.

        Each list contributes 1 / (k + rank) per item; contributions are summed
        across lists and normalized to [0, 1] (an item ranked first in both arms
        scores 1.0). The fused score replaces the per-item base relevance that
        feeds the downstream scope/recency boosts and reranker.
        """
        memory_by_id: dict[str, Memory] = {}
        rrf_scores: dict[str, float] = {}

        # Vector arm ranked by descending relevance score.
        vec_sorted = sorted(vector_results, key=lambda pair: pair[1], reverse=True)
        for rank, (memory, _score) in enumerate(vec_sorted):
            memory_by_id.setdefault(memory.id, memory)
            rrf_scores[memory.id] = rrf_scores.get(memory.id, 0.0) + 1.0 / (k + rank + 1)

        # Keyword arm already in BM25 rank order.
        for rank, memory in enumerate(keyword_memories):
            memory_by_id.setdefault(memory.id, memory)
            rrf_scores[memory.id] = rrf_scores.get(memory.id, 0.0) + 1.0 / (k + rank + 1)

        # Normalize by the maximum achievable score (rank 0 in both arms).
        max_score = 2.0 / (k + 1)
        fused = [(memory_by_id[mid], score / max_score) for mid, score in rrf_scores.items()]
        fused.sort(key=lambda pair: pair[1], reverse=True)
        return fused

    async def _recall_llm(
        self,
        workspace_id: str,
        input: RecallInput,
        relevance_threshold: float,
        alias_weight: float | None = None,
        backlink_weight: float | None = None,
        intent: QueryIntent | None = None,
        generation_operation_prefix: str | None = None,
        generation_operation_ids: list[str] | None = None,
    ) -> RecallResult:
        """
        LLM-enhanced retrieval with query rewriting.

        Steps:
        1. Use LLM to rewrite query for better semantic match
        2. Perform RAG search with rewritten query

        Note: Re-ranking is handled at the top level in recall() for all modes.
        """
        import time

        start_time = time.time()

        # Check if LLM service is available
        if not self.llm_service:
            self.logger.warning("LLM service not available, falling back to RAG")
            return await self._recall_rag(
                workspace_id=workspace_id,
                input=input,
                relevance_threshold=relevance_threshold,
                alias_weight=alias_weight,
                backlink_weight=backlink_weight,
                intent=intent,
            )

        # Step 1: LLM Query Rewriting
        rewritten_query = input.query
        if self.llm_query_rewrite_enabled:
            # Serialize context list to a string for the rewriter
            context_str = None
            if input.context:
                context_str = "\n".join(f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in input.context)
            operation_id = f"{generation_operation_prefix or generate_id('recall_gen')}:rewrite"
            authorization = self.llm_service.authorization(
                GenerationActivity.QUERY_REWRITING,
                workspace_id=workspace_id,
                operation_id=operation_id,
            )
            if generation_operation_ids is not None:
                generation_operation_ids.append(operation_id)
            rewritten_query = await self._rewrite_query_with_llm(
                input.query,
                context_str,
                authorization=authorization,
            )
            self.logger.info(
                "LLM query rewrite: '%s' -> '%s'",
                input.query[:50],
                rewritten_query[:50] if rewritten_query != input.query else "(unchanged)",
            )

        # Step 2: Search with rewritten query (fetch more candidates for re-ranking).
        # Copy ALL filter/scope/intent/weight fields from the caller's input so
        # LLM mode honours the same predicates as RAG/HYBRID. Only `limit` and
        # `min_relevance` are intentionally overridden here for overfetch.
        search_input = RecallInput(
            query=rewritten_query,
            types=input.types,
            subtypes=input.subtypes,
            tags=input.tags,
            context_id=input.context_id,
            user_id=input.user_id,
            observer_id=input.observer_id,
            subject_id=input.subject_id,
            include_global=input.include_global,
            include_global_user=input.include_global_user,
            mode=RecallMode.RAG,  # Use RAG for initial retrieval
            tolerance=input.tolerance,
            limit=min(input.limit * self.recall_overfetch, 50),  # Overfetch for reranker
            offset=input.offset,
            min_relevance=max(0.2, relevance_threshold - 0.3),  # Lower threshold for candidates
            recency_weight=input.recency_weight,
            include_associations=input.include_associations,
            traverse_depth=input.traverse_depth,
            max_expansion=input.max_expansion,
            created_after=input.created_after,
            created_before=input.created_before,
            event_after=input.event_after,
            event_before=input.event_before,
            time_order=input.time_order,
            context=input.context,
            rag_threshold=input.rag_threshold,
            include_archived=input.include_archived,
            detail_level=input.detail_level,
            exclude_ids=input.exclude_ids,
        )

        rag_result = await self._recall_rag(
            workspace_id=workspace_id,
            input=search_input,
            relevance_threshold=max(0.2, relevance_threshold - 0.3),
            alias_weight=alias_weight,
            backlink_weight=backlink_weight,
            intent=intent,
        )

        if not rag_result.memories:
            # No candidates found
            return RecallResult(
                memories=[],
                total_count=0,
                search_latency_ms=int((time.time() - start_time) * 1000),
                mode_used=RecallMode.LLM,
                query_rewritten=rewritten_query,
                sufficiency_reached=False,
            )

        # Reranking is now handled at the top level in recall() for all modes
        search_latency_ms = int((time.time() - start_time) * 1000)

        self.logger.info("LLM recall complete: %d candidates in %d ms", len(rag_result.memories), search_latency_ms)

        return RecallResult(
            memories=rag_result.memories,
            total_count=len(rag_result.memories),
            search_latency_ms=search_latency_ms,
            mode_used=RecallMode.LLM,
            query_rewritten=rewritten_query,
            sufficiency_reached=len(rag_result.memories) >= input.limit,
        )

    async def _rewrite_query_with_llm(
        self,
        query: str,
        context: str | None = None,
        *,
        authorization: GenerationAuthorization | None = None,
    ) -> str:
        """
        Use LLM to rewrite query for better semantic search.

        Expands abbreviations, adds synonyms, clarifies intent.
        """
        prompt = f"""Rewrite the following search query to improve semantic search results.
Expand abbreviations, add relevant synonyms, and clarify the intent.
Keep it concise (under 100 words). Return ONLY the rewritten query, no explanation.

Original query: {query}"""

        if context:
            prompt += f"\n\nContext: {context}"

        try:
            rewritten = await self.llm_service.synthesize(
                prompt=prompt,
                max_tokens=_INTERNAL_LLM_RECALL_TOKEN_BUDGET,
                temperature_factor=0.4,  # Low temperature for consistency
                profile="default",
                authorization=authorization,
                activity=GenerationActivity.QUERY_REWRITING,
            )
            # Clean up response
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten if rewritten else query
        except (GenerationNotAllowedError, GenerationBudgetExceededError):
            raise
        except Exception as e:
            self.logger.warning("Query rewriting failed: %s, using original", e)
            return query

    async def _rerank_with_llm(self, query: str, memories: list, limit: int) -> list:
        """
        Re-rank memories by relevance to query.

        Uses dedicated RerankerService if available, falls back to LLM-based reranking.
        Returns top-k most relevant memories.
        """
        if len(memories) <= limit:
            return memories

        # Try dedicated reranker service first (faster, more accurate)
        if self.reranker_service:
            try:
                # Get initial scores for adaptive sizing
                initial_scores = [getattr(mem, "relevance", 0.5) for mem in memories]

                # Use adaptive reranking
                results = await self.reranker_service.rerank_objects_adaptive(
                    query=query,
                    objects=memories,
                    content_fn=lambda m: m.content,
                    score_fn=lambda m: getattr(m, "relevance", 0.5),
                    requested_k=limit,
                )

                if results:
                    self.logger.debug("Reranker service: %d candidates -> %d results", len(memories), len(results))
                    return [r.document for r in results]

            except Exception as e:
                self.logger.warning("Reranker service failed: %s, falling back to LLM", e)

        # Fall back to LLM-based reranking
        if not self.llm_service:
            self.logger.warning("No reranker or LLM service available, using original order")
            return memories[:limit]

        # Build context with memory summaries
        memory_summaries = []
        for i, mem in enumerate(memories[:20]):  # Limit to top 20 for LLM context
            content_preview = mem.content[:200] if len(mem.content) > 200 else mem.content
            memory_summaries.append(f"[{i}] {content_preview}")

        summaries_text = "\n".join(memory_summaries)

        prompt = f"""Given the search query and candidate memories, rank them by relevance.
Return ONLY a comma-separated list of indices (e.g., "3,0,5,2") for the {limit} most relevant memories.
Most relevant first. No explanation.

Query: {query}

Candidate memories:
{summaries_text}

Top {limit} indices (comma-separated):"""

        try:
            response = await self.llm_service.synthesize(
                prompt=prompt,
                max_tokens=self.rerank_max_tokens,
                temperature_factor=0.15,  # Very low for deterministic ranking
                profile="default",
                activity=GenerationActivity.RERANKING,
            )

            # Parse indices from response
            indices = []
            for part in response.strip().split(","):
                try:
                    idx = int(part.strip().strip("[]"))
                    if 0 <= idx < len(memories):
                        indices.append(idx)
                except ValueError:
                    continue

            if indices:
                # Return memories in ranked order
                ranked = [memories[i] for i in indices[:limit]]
                # Fill remaining slots if needed
                if len(ranked) < limit:
                    remaining = [m for i, m in enumerate(memories) if i not in indices]
                    ranked.extend(remaining[: limit - len(ranked)])
                return ranked

        except Exception as e:
            self.logger.warning("LLM re-ranking failed: %s, using original order", e)

        # Fallback to original order
        return memories[:limit]

    async def _apply_reranking(
        self,
        query: str,
        memories: list[Memory],
        limit: int,
    ) -> list[Memory]:
        """Apply reranking to memories using the reranker service.

        Uses reranker_service if available, falls back to truncation.

        Args:
            query: The original search query
            memories: Memories to rerank
            limit: Maximum number to return

        Returns:
            Reranked and truncated list of memories
        """
        # MMR result diversification (default OFF). When enabled it reorders the
        # candidate pool to trade a little relevance for diversity, then takes the
        # top-limit — avoiding a top-k spent on near-duplicates. Mutually exclusive
        # with the cross-encoder reranker in this cut, so it runs first and short-
        # circuits. Any failure falls through to the reranker/truncation path.
        if self.rerank_mmr_enabled and len(memories) >= 2:
            try:
                return await self._apply_mmr(memories, limit)
            except Exception as e:
                self.logger.warning("MMR diversification failed, falling back: %s", e)

        if not self.reranker_service or len(memories) <= limit:
            return memories[:limit]

        try:
            reranked = await self.reranker_service.rerank_objects_adaptive(
                query=query,
                objects=memories,
                content_fn=lambda m: m.content,
                score_fn=lambda m: getattr(m, "boosted_score", None) or getattr(m, "relevance_score", 0.5),
                requested_k=limit,
            )
            if reranked:
                return [r.document for r in reranked]
            return memories[:limit]
        except Exception as e:
            self.logger.warning("Reranking failed, falling back to truncation: %s", e)
            return memories[:limit]

    async def _apply_mmr(self, memories: list[Memory], limit: int) -> list[Memory]:
        """Diversify the candidate pool with Maximal Marginal Relevance.

        Relevance per memory is the post-boost recall score (``boosted_score``,
        falling back to ``relevance_score``). Diversity uses each memory's
        embedding: memories already carrying an ``embedding`` are used as-is, and
        any missing ones are best-effort batch-embedded from their content. If
        embeddings cannot be obtained they are passed as ``None``, which
        ``mmr_select`` treats as maximally novel (degrading to a relevance order).

        Returns the top-``limit`` memories in MMR-selected order.
        """
        relevance = [getattr(m, "boosted_score", None) or getattr(m, "relevance_score", None) or 0.5 for m in memories]
        embeddings: list[list[float] | None] = [getattr(m, "embedding", None) for m in memories]

        missing = [i for i, e in enumerate(embeddings) if not e]
        if missing:
            try:
                fetched = await self.embedding.embed_batch([memories[i].content for i in missing])
                for i, emb in zip(missing, fetched):
                    embeddings[i] = emb
            except Exception as e:
                # Best-effort: leave the missing ones as None; mmr_select copes.
                self.logger.debug("MMR embedding backfill failed, using relevance-only for %d items: %s", len(missing), e)

        order = mmr_select(relevance, embeddings, lambda_param=self.rerank_mmr_lambda, k=limit)
        return [memories[i] for i in order]

    async def forget(
        self,
        workspace_id: str,
        memory_id: str,
        hard: bool = False,
        reason: str | None = None,
    ) -> bool:
        """
        Delete or soft-delete a memory.

        Soft delete: Sets deleted_at timestamp
        Hard delete: Removes from database entirely

        Returns True if the memory existed and was deleted, False if no such
        memory exists in the workspace (genuine not-found). Actual storage
        failures propagate as exceptions so the API can distinguish a 404
        (not-found) from a 500 (storage error); the prior bool-only contract
        conflated the two.
        """
        self.logger.info("Forgetting memory: %s in workspace: %s, hard: %s", memory_id, workspace_id, hard)

        # Existence check first so a falsy delete result means "not found",
        # not "storage error" (storage errors raise and propagate).
        existing = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
        if existing is None:
            self.logger.info("Memory not found, nothing to forget: %s", memory_id)
            return False

        success = await self.storage.delete_memory(workspace_id=workspace_id, memory_id=memory_id, hard=hard)

        if success:
            self.logger.info("Memory forgotten: %s", memory_id)
        else:
            self.logger.warning("Failed to forget memory: %s", memory_id)

        return success

    async def decay(
        self,
        workspace_id: str,
        memory_id: str,
        decay_rate: float = 0.1,
    ) -> Memory | None:
        """
        Reduce memory importance by decay_rate.

        Used for implementing memory decay over time.
        """
        self.logger.debug("Decaying memory: %s by rate: %s", memory_id, decay_rate)

        # Get current memory
        memory = await self.storage.get_memory(workspace_id, memory_id)
        if not memory:
            self.logger.warning("Memory not found for decay: %s", memory_id)
            return None

        # Calculate new importance (apply decay directly)
        new_importance = max(0.0, memory.importance - decay_rate)

        # Update memory
        updated = await self.storage.update_memory(workspace_id=workspace_id, memory_id=memory_id, importance=new_importance)

        self.logger.debug("Decayed memory: %s, new importance: %s", memory_id, new_importance)

        return updated

    async def _reconcile_fts_index(self, workspace_id: str, memory_id: str) -> None:
        """Reconcile a memory's full-text index after a content/alias change.

        Prefers write-behind via the task mechanism (task type "reindex_memory");
        falls back to an inline reindex when no task could be scheduled (e.g. the
        task service is absent or disabled), so the index never silently goes
        stale. Reindex itself is a no-op on backends with no materialized FTS.
        """
        task_id = None
        if self.task_service is not None:
            try:
                task_id = await self.task_service.schedule_task(
                    "reindex_memory",
                    {"workspace_id": workspace_id, "memory_id": memory_id},
                )
            except Exception as e:
                self.logger.debug("Failed to schedule reindex_memory task: %s", e)

        if not task_id:
            try:
                await self.storage.reindex_memory(workspace_id, memory_id)
            except Exception as e:
                self.logger.warning("Inline FTS reindex failed for %s: %s", memory_id, e)

    async def update(
        self,
        workspace_id: str,
        memory_id: str,
        **updates,
    ) -> Memory | None:
        """
        Update a memory, recomputing content_hash and embedding when content changes.

        Args:
            workspace_id: Workspace the memory belongs to
            memory_id: Memory identifier
            **updates: Fields to update (content, type, importance, tags, etc.)

        Returns:
            Updated Memory object, or None if not found
        """
        self.logger.info("Updating memory: %s in workspace: %s", memory_id, workspace_id)

        # If content changed, recompute content_hash and regenerate embedding
        if "content" in updates:
            updates["content_hash"] = compute_content_hash(updates["content"])
            updates["embedding"] = await self.embedding.embed(updates["content"])

        # Normalize an event_time datetime to the canonical UTC ISO string the
        # storage layer persists (so timeline ordering stays consistent).
        if isinstance(updates.get("event_time"), datetime):
            updates["event_time"] = to_utc_iso(updates["event_time"])

        updated = await self.storage.update_memory(
            workspace_id=workspace_id,
            memory_id=memory_id,
            **updates,
        )

        # Content or aliases (metadata) feed the full-text index, which the
        # update path does not touch inline — reconcile it write-behind.
        if updated is not None and ("content" in updates or "metadata" in updates):
            await self._reconcile_fts_index(workspace_id, memory_id)

        return updated

    async def get(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> Memory | None:
        """Get a single memory by ID within a workspace."""
        self.logger.debug("Getting memory: %s in workspace: %s", memory_id, workspace_id)
        return await self.storage.get_memory(workspace_id, memory_id)

    async def get_by_id(
        self,
        memory_id: str,
        *,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Memory | None:
        """Get a single memory by ID without workspace filter. Memory IDs are globally unique."""
        self.logger.debug("Getting memory by ID: %s", memory_id)
        return await self.storage.get_memory_by_id(
            memory_id,
            track_access=track_access,
            include_deleted=include_deleted,
        )

    async def list_memories(
        self,
        workspace_id: str,
        *,
        types: list[MemoryType] | None = None,
        subtypes: list[str] | None = None,
        tags: list[str] | None = None,
        context_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """List/browse memories ordered by recency (created_at desc) without a query.

        Thin delegate over ``storage.search_memories_by_filter`` (no embeddings).
        The storage filter natively supports subtype/tag/context/pagination; the
        ``types`` (cognitive type) filter is applied in Python because the
        filtered-search storage seam does not key on the ``type`` column.
        """
        self.logger.debug(
            "Listing memories in workspace: %s (types=%s, subtypes=%s, tags=%s, context_id=%s, limit=%d, offset=%d)",
            workspace_id,
            types,
            subtypes,
            tags,
            context_id,
            limit,
            offset,
        )
        memories = await self.storage.search_memories_by_filter(
            workspace_id,
            subtypes=subtypes or None,
            tags=tags or None,
            context_id=context_id,
            limit=limit,
            offset=offset,
        )
        if types:
            type_set = {t.value if isinstance(t, MemoryType) else str(t) for t in types}
            memories = [m for m in memories if (m.type.value if isinstance(m.type, MemoryType) else str(m.type)) in type_set]
        return memories

    async def increment_access(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> None:
        """Increment access count and update last_accessed_at."""
        try:
            memory = await self.storage.get_memory(workspace_id, memory_id)
            if memory:
                importance = await self.decay_service.calculate_access_boost(memory)
                await self.storage.update_memory(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    access_count=memory.access_count + 1,
                    last_accessed_at=datetime.now(UTC),
                    importance=importance,
                )
        except Exception as e:
            self.logger.warning("Failed to increment access for %s: %s", memory_id, e)

    def _apply_detail_level(self, memories: list[Memory], detail_level: DetailLevel) -> list[Memory]:
        """
        Apply detail level filtering to memories.

        Args:
            memories: List of Memory objects to filter
            detail_level: Level of detail to return

        Returns:
            List of filtered memories
        """
        if detail_level == DetailLevel.FULL:
            # No filtering needed for full detail
            return memories

        filtered_memories = []

        for memory in memories:
            # Create a copy of the memory with modified content
            memory_dict = memory.model_dump()

            if detail_level == DetailLevel.ABSTRACT:
                # Use abstract field if available, else truncate to ~100 chars
                if memory.abstract:
                    memory_dict["content"] = memory.abstract
                else:
                    memory_dict["content"] = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content

            elif detail_level == DetailLevel.OVERVIEW:
                # Use overview field if available, else truncate to ~500 chars
                if memory.overview:
                    memory_dict["content"] = memory.overview
                else:
                    memory_dict["content"] = memory.content[:500] + "..." if len(memory.content) > 500 else memory.content

            filtered_memories.append(Memory(**memory_dict))

        return filtered_memories

    async def _apply_consensus_boost(
        self,
        workspace_id: str,
        memories: list[Memory],
        weight: float,
    ) -> list[Memory]:
        """A2: use the association graph as a ranking-only consensus signal.

        Unlike graph expansion, this injects NO new candidates. Each already-
        retrieved memory is boosted by how strongly it is connected to OTHER
        retrieved memories — the intuition being that a memory corroborated by
        its retrieved neighbours is more likely to be on-topic. Because the
        candidate set is unchanged, this can only reorder results (and truncate
        to the same limit later); it can never displace gold with injected noise.

        boosted_score *= 1 + weight * log1p(sum_of_edge_strengths_to_other_hits)

        Args:
            workspace_id: Workspace boundary
            memories: Retrieved memories (the fixed candidate pool)
            weight: Boost magnitude; 0 would be a no-op (caller gates on > 0)

        Returns:
            The same memories, re-scored and sorted by boosted_score descending.
        """
        if len(memories) < 2 or weight <= 0.0:
            return memories

        retrieved_ids = {m.id for m in memories}

        async def _consensus_strength(memory_id: str) -> float:
            try:
                associations = await self.storage.get_associations(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    direction="both",
                )
            except Exception as e:
                self.logger.warning("Consensus boost: get_associations(%s) failed: %s", memory_id, e)
                return 0.0
            total = 0.0
            for assoc in associations:
                other_id = assoc.target_id if assoc.source_id == memory_id else assoc.source_id
                if other_id != memory_id and other_id in retrieved_ids:
                    total += assoc.strength or 0.0
            return total

        strengths = await asyncio.gather(*[_consensus_strength(m.id) for m in memories])

        for memory, strength in zip(memories, strengths):
            if strength <= 0.0:
                continue
            base = getattr(memory, "boosted_score", None)
            if base is None:
                base = getattr(memory, "relevance_score", 0.0) or 0.0
            boosted = base * (1.0 + weight * math.log1p(strength))
            memory.boosted_score = boosted
            # Record the consensus signal so results can explain the re-rank.
            memory.match_signals = list(getattr(memory, "match_signals", None) or []) + ["consensus"]

        memories.sort(
            key=lambda m: getattr(m, "boosted_score", 0.0) or 0.0,
            reverse=True,
        )
        return memories

    async def _expand_with_associations(
        self,
        workspace_id: str,
        memories: list[Memory],
        traverse_depth: int,
        include_associations: bool,
        max_expansion: int = 50,
        query_embedding: list[float] | None = None,
    ) -> list[Memory]:
        """Expand recall results by traversing association graph.

        BFS-traverses associations up to traverse_depth hops from recalled memories.

        Two scoring modes for discovered memories:
          - Legacy (default): parent_score * association_strength * (0.8 ^ depth).
            Purely graph-topological — a well-connected neighbor can outrank a
            direct hit regardless of whether it answers the query.
          - Query-aware (``query_embedding`` provided and ``assoc_query_aware``):
            score = cosine(query, neighbor). Neighbors below ``assoc_query_floor``
            are dropped. Expansion can then only surface neighbors that genuinely
            match the query — it adds evidence the vector arm missed but never
            displaces direct hits with topically-similar-but-off-query noise.

        Args:
            workspace_id: Workspace boundary
            memories: Initially recalled memories
            traverse_depth: Maximum BFS depth (0 = no expansion)
            include_associations: Whether to include associated memories
            max_expansion: Maximum number of graph-discovered memories to add
            query_embedding: Query vector for query-aware neighbor scoring; when
                None (or the feature is off) the legacy topological score is used

        Returns:
            Combined list of original + discovered memories, sorted by score descending
        """
        if not include_associations and traverse_depth <= 0:
            return memories

        effective_depth = max(traverse_depth, 1) if include_associations else traverse_depth

        # Query-aware scoring: pre-normalize the query vector once so each
        # neighbor's score is a plain dot product. Falls back to legacy scoring
        # if the query vector is absent or degenerate.
        query_aware = bool(self.assoc_query_aware and query_embedding is not None)
        q_unit = None
        if query_aware:
            q_arr = np.asarray(query_embedding, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_arr))
            if q_norm == 0.0:
                query_aware = False
            else:
                q_unit = q_arr / q_norm

        # Phase 4: Check association expansion cache. The cache key is derived
        # only from seed IDs + depth, so it is NOT safe for query-aware scoring
        # (results then depend on the query). Skip the cache entirely in that
        # mode — leaving assoc_cache_key None also short-circuits the write below.
        assoc_cache_key = None
        if self.cache and memories and not query_aware:
            # Cache key based on seed memory IDs and depth
            seed_ids = sorted(m.id for m in memories)
            assoc_cache_key = f"assoc:{workspace_id}:{compute_content_hash(':'.join(seed_ids))[:16]}:{effective_depth}"
            try:
                cached = await self.cache.get(assoc_cache_key)
                if cached is not None:
                    self.logger.debug("Association expansion cache hit")
                    return [Memory(**m) for m in cached]
            except Exception as e:
                self.logger.debug("Association cache get failed: %s", e)

        # Track already-seen memory IDs to avoid duplicates
        seen_ids: set[str] = {m.id for m in memories}
        discovered: list[tuple[Memory, float]] = []

        # BFS queue: (memory_id, parent_score, current_depth)
        queue: list[tuple[str, float, int]] = []
        for memory in memories:
            parent_score = getattr(memory, "boosted_score", None) or getattr(memory, "relevance_score", 0.5)
            queue.append((memory.id, parent_score, 0))

        while queue:
            if len(discovered) >= max_expansion:
                self.logger.debug(
                    "Graph expansion cap reached (%d memories), stopping BFS",
                    max_expansion,
                )
                break

            memory_id, parent_score, depth = queue.pop(0)

            if depth >= effective_depth:
                continue

            try:
                associations = await self.storage.get_associations(
                    workspace_id=workspace_id,
                    memory_id=memory_id,
                    direction="both",
                )
            except Exception as e:
                self.logger.warning("Failed to get associations for %s: %s", memory_id, e)
                continue

            for assoc in associations:
                # Determine the other end of the association
                target_id = assoc.target_id if assoc.source_id == memory_id else assoc.source_id

                if target_id in seen_ids:
                    continue
                seen_ids.add(target_id)

                try:
                    target_memory = await self.storage.get_memory(workspace_id, target_id)
                except Exception:
                    continue

                if target_memory is None:
                    continue

                # Skip non-active memories
                if hasattr(target_memory, "status") and target_memory.status != MemoryStatus.ACTIVE:
                    continue

                if query_aware:
                    # Query-aware: score the neighbor by its similarity to the
                    # query, not by graph topology. Drop neighbors we cannot
                    # score (no embedding) or that fall below the floor so
                    # off-query memories are never injected into the pool.
                    neighbor_embedding = getattr(target_memory, "embedding", None)
                    if neighbor_embedding is None:
                        continue
                    n_arr = np.asarray(neighbor_embedding, dtype=np.float32)
                    n_norm = float(np.linalg.norm(n_arr))
                    if n_norm == 0.0:
                        continue
                    score = float(np.dot(q_unit, n_arr / n_norm))
                    if score < self.assoc_query_floor:
                        # Still traverse through this node (a weak neighbor can
                        # bridge to a strong one) but do not add it to results.
                        queue.append((target_id, score, depth + 1))
                        continue
                else:
                    # Legacy: parent_score * strength * decay_per_hop
                    hop_decay = 0.8 ** (depth + 1)
                    score = parent_score * assoc.strength * hop_decay

                # Attach score metadata
                memory_dict = target_memory.model_dump()
                memory_dict["relevance_score"] = score
                memory_dict["boosted_score"] = score
                memory_dict["source_scope"] = "association"
                scored_memory = Memory(**memory_dict)

                discovered.append((scored_memory, score))

                # Continue BFS from this memory
                queue.append((target_id, score, depth + 1))

        if not discovered:
            return memories

        # Combine original memories with discovered ones
        combined = list(memories)
        combined.extend([m for m, _ in discovered])

        # Sort by boosted_score descending
        combined.sort(
            key=lambda m: getattr(m, "boosted_score", 0.0) or 0.0,
            reverse=True,
        )

        # Phase 4: Cache the expanded result
        if self.cache and assoc_cache_key:
            try:
                await self.cache.set(
                    assoc_cache_key,
                    [m.model_dump() for m in combined],
                    ttl_seconds=600,
                )
            except Exception as e:
                self.logger.debug("Association cache set failed: %s", e)

        return combined

    def _should_decompose(
        self,
        content: str,
        memory_type: MemoryType | None,
        force: bool = False,
    ) -> bool:
        """Determine whether a memory should be decomposed into atomic facts.

        Criteria:
        - Fact decomposition must be enabled
        - Content length must exceed the configured minimum
        - Content must contain multiple sentences (>1 period or semicolon)
        - Memory must not be a working type (working memories are transient)

        ``force`` bypasses ONLY the content-shape checks (length + multiple
        sentences), not the enabled flag or the WORKING-type exclusion. It is set
        for memories whose real content is not their stored text — e.g. an
        OCR-free document-page memory whose ``content`` is a short visual
        placeholder while the substance lives in the page image (read at
        decomposition time by a vision model). Without it those memories fail the
        text heuristic and never get decomposed.

        Args:
            content: The memory content
            memory_type: The memory type (if known)
            force: Skip the content-length/multi-sentence heuristic.

        Returns:
            True if the memory should be decomposed
        """
        if not self.fact_decomposition_enabled:
            return False

        if memory_type == MemoryType.WORKING:
            return False

        if force:
            return True

        if len(content) < self.fact_decomposition_min_length:
            return False

        # Check for multiple sentences (periods, semicolons, or question marks followed by space/end)
        sentence_terminators = re.findall(r"[.;?!]\s", content)
        # Also check for a terminator at the very end of the string
        if content and content[-1] in ".;?!":
            sentence_terminators.append(content[-1])
        if len(sentence_terminators) <= 1:
            return False

        return True

    async def _classify_memory_type(self, content: str) -> MemoryType:
        """Backward-compatible wrapper around the pure rule classifier."""
        return deterministic_classify_content(content).memory_type

    def _get_relevance_threshold(self, tolerance: SearchTolerance, min_relevance: float | None) -> float:
        """
        Calculate effective relevance threshold.

        Priority:
        1. min_relevance is None: use tolerance-based floor (server default)
        2. min_relevance <= 0.0: bypass all thresholds (testing mode)
        3. Explicit value: respect caller's choice, applying tolerance floor as minimum
        """
        # Tolerance floors are server-configurable (see __init__ /
        # MEMORYLAYER_TOLERANCE_FLOOR_*). Defaults match the historical
        # hardcoded values so behavior is unchanged unless overridden.
        floor = self.tolerance_floors.get(tolerance, self.tolerance_floors[SearchTolerance.MODERATE])

        # No explicit value: use tolerance-based floor as the server default
        if min_relevance is None:
            return floor

        # Testing mode: bypass all thresholds
        if min_relevance <= 0.0:
            return min_relevance

        # Explicit value: respect caller's choice, but enforce tolerance floor
        return max(min_relevance, floor)

    def apply_scope_boosts(
        self, memories: list, query_context_id: str, query_workspace_id: str, boosts: ScopeBoosts | None = None
    ) -> list[Memory]:
        """
        Apply locality-based score boosts to recalled memories.

        Args:
            memories: List of (memory, score) tuples from storage
            query_context_id: The context the query originated from
            query_workspace_id: The workspace the query originated from
            boosts: ScopeBoosts configuration (uses defaults if None)

        Returns:
            List of Memory objects sorted by boosted score with source_scope added
        """
        if boosts is None:
            boosts = getattr(self, "default_scope_boosts", ScopeBoosts())

        boosted_memories = []

        for memory, base_score in memories:
            # Determine scope and boost
            memory_context_id = memory.context_id if memory.context_id else DEFAULT_CONTEXT_ID
            memory_workspace_id = memory.workspace_id

            if memory_context_id == query_context_id:
                source_scope = "same_context"
                boost = boosts.same_context
            elif memory_workspace_id == query_workspace_id:
                source_scope = "same_workspace"
                boost = boosts.same_workspace
            elif memory_workspace_id == GLOBAL_WORKSPACE_ID:
                source_scope = "global_workspace"
                boost = boosts.global_workspace
            elif memory_workspace_id == GLOBAL_USER_WORKSPACE_ID:
                source_scope = "global_user_workspace"
                boost = boosts.global_workspace
            else:
                source_scope = "other"
                boost = 1.0

            boosted_score = base_score * boost

            # Create new Memory object with ranking metadata
            memory_dict = memory.model_dump()
            memory_dict["source_scope"] = source_scope
            memory_dict["relevance_score"] = base_score
            memory_dict["boosted_score"] = boosted_score

            boosted_memory = Memory(**memory_dict)
            boosted_memories.append((boosted_memory, boosted_score))

        # Sort by boosted score descending
        boosted_memories.sort(key=lambda x: x[1], reverse=True)

        return [m for m, _ in boosted_memories]

    def apply_recency_boost(
        self,
        memories: list[Memory],
        recency_weight: float,
        half_life_hours: float = DEFAULT_RECENCY_HALF_LIFE_HOURS,
    ) -> list[Memory]:
        """
        Apply time-based recency boost to recalled memories.

        Uses exponential decay based on the memory's effective event time
        (event_time if set, else created_at) — NOT updated_at. increment_access
        bumps updated_at on every recall, so keying recency on updated_at created
        a feedback loop where recalling a memory made it look "newer" and boosted
        it further. event_time/created_at are stable and match the temporal
        timeline ordering convention used elsewhere.
        Recent memories get higher scores; old memories decay toward
        (1 - recency_weight) of their boosted score.

        Args:
            memories: List of Memory objects with boosted_score already set
            recency_weight: How much recency affects ranking (0.0-1.0)
            half_life_hours: Hours until recency factor reaches 0.5

        Returns:
            List of Memory objects re-sorted by recency-adjusted boosted_score
        """
        if recency_weight <= 0.0 or not memories:
            return memories

        now = datetime.now(UTC)

        for memory in memories:
            # Effective event time = event_time if set, else created_at (stable,
            # not bumped by recall-time access writes; see docstring).
            effective_time = memory.event_time if memory.event_time is not None else memory.created_at
            age_hours = max(0.0, (now - effective_time).total_seconds() / 3600.0)
            recency_factor = self._exponential_freshness(age_hours, half_life_hours)
            # Blend: at weight=0 no effect, at weight=1 full decay
            adjusted_score = memory.boosted_score * (1.0 - recency_weight + recency_weight * recency_factor)
            memory.boosted_score = adjusted_score

        # Re-sort by adjusted boosted_score
        memories.sort(key=lambda m: m.boosted_score, reverse=True)
        return memories

    async def apply_supersession(
        self,
        workspace_id: str,
        memories: list[Memory],
        signals: dict[str, set[str]] | None = None,
    ) -> list[Memory]:
        """Demote or drop memories that a later memory has superseded.

        This is the read half of contradiction detection. Detection has always run on
        every store and recorded which of the two memories is current; recall simply never
        looked. A fact positively identified as stale was still returned at full score.

        Unlike the association boosts around it, this does not add candidates — it removes
        or down-weights wrong ones, which is a different lever from the one our moat eval
        tested when it found association edges a wash.

        Applied over the full candidate pool before truncation so a demoted memory can
        actually lose its slot. Degrades to a no-op on lookup failure: a stale result is
        worse than a fresh one, but losing recall entirely is worse than both.
        """
        if self._supersession_mode == SUPERSESSION_MODE_OFF or not memories:
            return memories

        ids_by_workspace: dict[str, list[str]] = {}
        for memory in memories:
            ids_by_workspace.setdefault(memory.workspace_id, []).append(memory.id)

        superseded: set[str] = set()
        try:
            for ws, ids in ids_by_workspace.items():
                superseded |= await self.storage.get_superseded_memory_ids(ws, ids)
        except Exception as e:
            self.logger.debug("Supersession skipped (lookup failed): %s", e)
            return memories

        if not superseded:
            return memories

        if self._supersession_mode == SUPERSESSION_MODE_EXCLUDE:
            kept = [m for m in memories if m.id not in superseded]
            self.logger.debug("Supersession: excluded %d of %d candidates", len(memories) - len(kept), len(memories))
            return kept

        for memory in memories:
            if memory.id in superseded:
                memory.boosted_score = (memory.boosted_score or 0.0) * self._supersession_penalty
                if signals is not None:
                    signals.setdefault(memory.id, set()).add("superseded")
        self.logger.debug("Supersession: demoted %d of %d candidates", len(superseded), len(memories))
        return memories

    async def apply_backlink_boost(
        self,
        workspace_id: str,
        memories: list[Memory],
        weight: float,
        signals: dict[str, set[str]] | None = None,
    ) -> list[Memory]:
        """Boost recall scores by graph in-degree (incoming-association count).

        Mirrors gbrain's backlink boost: a memory referenced by many others is a
        hub and likely more salient, so it earns a log-scale multiplier on its
        boosted_score. In-degree is fetched per workspace scope so cross-scope
        candidates (e.g. _global) are counted against their own graph. When a
        ``signals`` map is given, boosted ids are recorded for the recall
        evidence contract. Degrades to a no-op when the graph has no incoming
        edges or on fetch failure.
        """
        if weight <= 0.0 or not memories:
            return memories

        # Group candidate ids by their owning workspace so in-degree is counted
        # within the correct association graph.
        ids_by_workspace: dict[str, list[str]] = {}
        for memory in memories:
            ids_by_workspace.setdefault(memory.workspace_id, []).append(memory.id)

        indegree: dict[str, int] = {}
        try:
            for ws, ids in ids_by_workspace.items():
                associations = await self.storage.get_associations_batch(ws, ids, direction="incoming")
                for assoc in associations:
                    # Exclude similar_to edges: on a KNN-similarity graph, in-degree
                    # correlates with genericness, not relevance (anti-signal).
                    if assoc.relationship == "similar_to":
                        continue
                    indegree[assoc.target_id] = indegree.get(assoc.target_id, 0) + 1
        except Exception as e:
            self.logger.debug("Backlink boost skipped (association fetch failed): %s", e)
            return memories

        if not indegree:
            return memories

        for memory in memories:
            degree = indegree.get(memory.id, 0)
            if degree and memory.boosted_score is not None:
                memory.boosted_score = memory.boosted_score * (1.0 + weight * math.log1p(degree))
                if signals is not None:
                    signals.setdefault(memory.id, set()).add("backlink")

        memories.sort(key=lambda m: m.boosted_score if m.boosted_score is not None else 0.0, reverse=True)
        return memories

    def apply_alias_boost(
        self,
        query: str,
        memories: list[Memory],
        weight: float,
        signals: dict[str, set[str]] | None = None,
    ) -> list[Memory]:
        """Boost memories whose metadata aliases share a term with the query.

        Aliases live in ``metadata['aliases']`` and are folded into the full-text
        index at write time, so alias-only queries already retrieve the canonical
        memory; this adds a small multiplicative nudge (and records the "alias"
        evidence signal) when the match was via an alias. No-op when weight is
        non-positive or the query has no usable terms.
        """
        if weight <= 0.0 or not memories:
            return memories

        query_terms = set(re.findall(r"[0-9a-z]+", query.lower()))
        if not query_terms:
            return memories

        changed = False
        for memory in memories:
            aliases = (memory.metadata or {}).get("aliases")
            if not isinstance(aliases, list):
                continue
            matched = any(query_terms & set(re.findall(r"[0-9a-z]+", str(alias).lower())) for alias in aliases)
            if matched:
                if memory.boosted_score is not None:
                    memory.boosted_score = memory.boosted_score * (1.0 + weight)
                if signals is not None:
                    signals.setdefault(memory.id, set()).add("alias")
                changed = True

        if changed:
            memories.sort(key=lambda m: m.boosted_score if m.boosted_score is not None else 0.0, reverse=True)
        return memories

    def _effective_time(self, memory: Memory) -> datetime:
        """Timeline position of a memory: explicit event_time, else created_at (UTC-aware)."""
        et = memory.event_time or memory.created_at
        return et.replace(tzinfo=UTC) if et is not None and et.tzinfo is None else et

    def _apply_temporal_filter_order(self, memories: list[Memory], input: RecallInput) -> list[Memory]:
        """Filter recalled memories by effective event-time window and/or order by time."""
        after = input.event_after
        before = input.event_before
        if after is not None and after.tzinfo is None:
            after = after.replace(tzinfo=UTC)
        if before is not None and before.tzinfo is None:
            before = before.replace(tzinfo=UTC)

        filtered: list[Memory] = []
        for memory in memories:
            eff = self._effective_time(memory)
            if after is not None and (eff is None or eff < after):
                continue
            if before is not None and (eff is None or eff > before):
                continue
            filtered.append(memory)

        if input.time_order in ("asc", "desc"):
            epoch = datetime.min.replace(tzinfo=UTC)
            filtered.sort(key=lambda m: self._effective_time(m) or epoch, reverse=(input.time_order == "desc"))
        return filtered

    async def get_timeline(
        self,
        workspace_id: str,
        event_after: datetime | None = None,
        event_before: datetime | None = None,
        ascending: bool = True,
        limit: int = 50,
        offset: int = 0,
        types: list[MemoryType] | None = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Browse memories ordered by effective event time (event_time or created_at)."""
        return await self.storage.get_timeline(
            workspace_id,
            event_after=to_utc_iso(event_after),
            event_before=to_utc_iso(event_before),
            ascending=ascending,
            limit=limit,
            offset=offset,
            types=[t.value for t in types] if types else None,
            include_archived=include_archived,
        )

    async def temporal_neighbors(
        self,
        workspace_id: str,
        memory_id: str,
        limit: int = 5,
        types: list[MemoryType] | None = None,
        include_archived: bool = False,
    ) -> dict:
        """Return the memories immediately before and after a memory on the timeline.

        Anchors on the memory's effective event time (event_time or created_at) and
        returns the closest earlier ("before", newest-first) and later ("after",
        oldest-first) memories. Surfaces temporal relationships without requiring
        explicit precedes/follows edges (which, when present, recall's association
        expansion already traverses).
        """
        anchor = await self.storage.get_memory(workspace_id, memory_id, track_access=False)
        if anchor is None:
            return {"anchor": None, "before": [], "after": []}

        anchor_iso = to_utc_iso(self._effective_time(anchor))
        type_values = [t.value for t in types] if types else None

        # Fetch one extra to drop the anchor itself (timeline bounds are inclusive).
        before = await self.storage.get_timeline(
            workspace_id, event_before=anchor_iso, ascending=False, limit=limit + 1, types=type_values, include_archived=include_archived
        )
        after = await self.storage.get_timeline(
            workspace_id, event_after=anchor_iso, ascending=True, limit=limit + 1, types=type_values, include_archived=include_archived
        )
        return {
            "anchor": anchor,
            "before": [m for m in before if m.id != memory_id][:limit],
            "after": [m for m in after if m.id != memory_id][:limit],
        }

    def _annotate_freshness(
        self,
        memories: list[Memory],
        half_life_days: float | None = None,
    ) -> list[Memory]:
        """
        Annotate recalled memories with freshness scores and staleness warnings.

        Uses exponential decay: score = exp(-age_days / half_life_days)

        Staleness tiers:
          - none:     age < 1 day
          - mild:     1 <= age < 7 days
          - moderate: 7 <= age < 30 days
          - severe:   age >= 30 days

        Also applies a small access-recency bonus: memories accessed within
        the last 24 hours get a +0.05 bonus (capped at 1.0).

        Args:
            memories: List of Memory objects to annotate (mutated in place)
            half_life_days: Days until freshness score reaches 0.5 (None uses instance default)

        Returns:
            The same list of Memory objects with freshness_score, staleness_warning,
            and age_days populated.
        """
        if not memories:
            return memories

        if half_life_days is None:
            half_life_days = getattr(self, "freshness_half_life_days", DEFAULT_MEMORYLAYER_FRESHNESS_HALF_LIFE_DAYS)

        now = datetime.now(UTC)

        for memory in memories:
            age_days = max(0.0, (now - memory.created_at).total_seconds() / 86400.0)
            freshness_score = self._exponential_freshness(age_days, half_life_days)

            # Access recency bonus: memories accessed in the last 24h feel fresher
            if memory.last_accessed_at is not None:
                hours_since_access = (now - memory.last_accessed_at).total_seconds() / 3600.0
                if hours_since_access < 24.0:
                    freshness_score = min(1.0, freshness_score + 0.05)

            # Staleness tier
            if age_days < 1.0:
                staleness_warning = "none"
            elif age_days < 7.0:
                staleness_warning = "mild"
            elif age_days < 30.0:
                staleness_warning = "moderate"
            else:
                staleness_warning = "severe"

            memory.freshness_score = round(freshness_score, 4)
            memory.staleness_warning = staleness_warning
            memory.age_days = round(age_days, 2)

        return memories

    async def recall_with_global(
        self, workspace_id: str, context_id: str, query: str, include_global: bool = True, boosts: ScopeBoosts | None = None, **kwargs
    ) -> list[Memory]:
        """
        Recall memories from workspace and optionally _global.

        Args:
            workspace_id: The workspace to search
            context_id: The context the query is from (for boosting)
            query: The search query
            include_global: Whether to include _global workspace
            boosts: ScopeBoosts configuration
            **kwargs: Additional recall parameters (limit, types, etc.)

        Returns:
            Combined and ranked memories with locality boosts applied
        """
        # Build RecallInput from query and kwargs
        recall_input = RecallInput(
            query=query,
            context_id=context_id,
            limit=kwargs.get("limit", 10),
            types=kwargs.get("types", []),
            subtypes=kwargs.get("subtypes", []),
            tags=kwargs.get("tags", []),
            mode=kwargs.get("mode", RecallMode.RAG),
            tolerance=kwargs.get("tolerance", SearchTolerance.MODERATE),
            min_relevance=kwargs.get("min_relevance"),
        )

        # Generate query embedding once
        query_embedding = await self.embedding.embed(query)

        entity_filters = {}
        if kwargs.get("observer_id") is not None:
            entity_filters["observer_id"] = kwargs["observer_id"]
        if kwargs.get("subject_id") is not None:
            entity_filters["subject_id"] = kwargs["subject_id"]

        # Get memories from current workspace
        workspace_results = await self.storage.search_memories(
            workspace_id=workspace_id,
            query_embedding=query_embedding,
            limit=recall_input.limit,
            min_relevance=self._get_relevance_threshold(recall_input.tolerance, recall_input.min_relevance),
            types=[t.value for t in recall_input.types] if recall_input.types else None,
            subtypes=list(recall_input.subtypes) if recall_input.subtypes else None,
            tags=recall_input.tags if recall_input.tags else None,
            **entity_filters,
        )

        # Get memories from _global if enabled
        global_results = []
        if include_global and workspace_id != GLOBAL_WORKSPACE_ID:
            global_results = await self.storage.search_memories(
                workspace_id=GLOBAL_WORKSPACE_ID,
                query_embedding=query_embedding,
                limit=recall_input.limit,
                min_relevance=self._get_relevance_threshold(recall_input.tolerance, recall_input.min_relevance),
                types=[t.value for t in recall_input.types] if recall_input.types else None,
                subtypes=list(recall_input.subtypes) if recall_input.subtypes else None,
                tags=recall_input.tags if recall_input.tags else None,
                **entity_filters,
            )

        # Combine results
        all_memories = workspace_results + global_results

        # Apply scope boosts and return sorted
        ranked = self.apply_scope_boosts(all_memories, query_context_id=context_id, query_workspace_id=workspace_id, boosts=boosts)

        # Apply recency boost
        effective_recency_weight = kwargs.get("recency_weight", DEFAULT_RECENCY_WEIGHT)
        ranked = self.apply_recency_boost(
            ranked,
            recency_weight=effective_recency_weight,
        )

        return ranked


class DefaultMemoryServicePlugin(MemoryServicePluginBase):
    """Default memory service plugin."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> MemoryService:
        cache = self.get_extension(EXT_CACHE_SERVICE, v)
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        embedding: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)
        deduplication: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        association_service: AssociationService = self.get_extension(EXT_ASSOCIATION_SERVICE, v)
        tier_generation_service: SemanticTieringService = self.get_extension(EXT_SEMANTIC_TIERING_SERVICE, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        reranker_service: RerankerService = self.get_extension(EXT_RERANKER_SERVICE, v)
        decay_service: DecayService = self.get_extension(EXT_DECAY_SERVICE, v)
        contradiction_service: ContradictionService = self.get_extension(EXT_CONTRADICTION_SERVICE, v)
        extraction_service: ExtractionService = self.get_extension(EXT_EXTRACTION_SERVICE, v)

        # TaskService is optional -- auto-association works inline without it
        task_service: TaskService | None = None
        try:
            task_service = self.get_extension(EXT_TASK_SERVICE, v)
        except Exception:
            logger.debug("TaskService not available, auto-association will run inline")

        # EntityRegistryService is optional and ships DARK -- accretion only runs
        # when MEMORYLAYER_ENTITY_REGISTRY_ENABLED is on AND the service resolves.
        entity_registry_service = None
        try:
            entity_registry_service = self.get_extension(EXT_ENTITY_REGISTRY_SERVICE, v)
        except Exception:
            logger.debug("EntityRegistryService not available, entity accretion disabled")

        # GraphQueryService is optional and ships DARK -- the graph-traversal recall
        # channel (P4 channel G) only runs when MEMORYLAYER_GRAPH_RECALL_ENABLED is
        # on AND the service resolves. A missing service -> None -> the arm no-ops.
        graph_query_service = None
        try:
            graph_query_service = self.get_extension(EXT_GRAPH_QUERY_SERVICE, v)
        except Exception:
            logger.debug("GraphQueryService not available, graph-traversal recall channel disabled")

        return MemoryService(
            storage=storage,
            embedding_service=embedding,
            deduplication_service=deduplication,
            association_service=association_service,
            cache=cache,
            tier_generation_service=tier_generation_service,
            llm_service=llm_service,
            reranker_service=reranker_service,
            decay_service=decay_service,
            contradiction_service=contradiction_service,
            task_service=task_service,
            extraction_service=extraction_service,
            entity_registry_service=entity_registry_service,
            graph_query_service=graph_query_service,
            v=v,
        )
