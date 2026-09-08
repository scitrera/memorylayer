from ...config import DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE, MEMORYLAYER_ONTOLOGY_SERVICE
from .._constants import EXT_ONTOLOGY_SERVICE
from .._plugin_factory import make_service_plugin_base


class FeatureRequiresUpgradeError(Exception):
    """Raised when a feature requires enterprise upgrade."""

    def __init__(self, feature: str):
        self.feature = feature
        super().__init__(f"Feature '{feature}' requires MemoryLayer Enterprise. Visit https://memorylayer.ai/enterprise to upgrade.")


# Unified ontology.  The compact knowledge-work application profile below is
# informed by PROV-O, DCMI Terms, Schema.org Action/Role, and W3C ORG; it keeps
# connector metadata interoperable without importing their full RDF models.
BASE_ONTOLOGY = {
    # --- Hierarchical relationships ---
    "parent_of": {
        "description": "Parent-child hierarchy",
        "symmetric": False,
        "transitive": True,
        "inverse": "child_of",
        "category": "hierarchical",
    },
    "child_of": {
        "description": "Child-parent hierarchy",
        "symmetric": False,
        "transitive": True,
        "inverse": "parent_of",
        "category": "hierarchical",
    },
    "part_of": {
        "description": "Component of a whole",
        "symmetric": False,
        "transitive": True,
        "inverse": "has_part",
        "category": "hierarchical",
    },
    "has_part": {
        "description": "Whole contains part",
        "symmetric": False,
        "transitive": True,
        "inverse": "part_of",
        "category": "hierarchical",
    },
    "instance_of": {
        "description": "Instance of a type/class",
        "symmetric": False,
        "transitive": True,
        "inverse": "type_of",
        "category": "hierarchical",
    },
    "type_of": {
        "description": "Type/class of instances",
        "symmetric": False,
        "transitive": True,
        "inverse": "instance_of",
        "category": "hierarchical",
    },
    # --- Causal relationships ---
    "causes": {
        "description": "Direct causation",
        "symmetric": False,
        "transitive": True,
        "inverse": "caused_by",
        "category": "causal",
    },
    "caused_by": {
        "description": "Caused by another event",
        "symmetric": False,
        "transitive": True,
        "inverse": "causes",
        "category": "causal",
    },
    "enables": {
        "description": "Makes possible or facilitates",
        "symmetric": False,
        "transitive": False,
        "inverse": "enabled_by",
        "category": "causal",
    },
    "enabled_by": {
        "description": "Made possible by",
        "symmetric": False,
        "transitive": False,
        "inverse": "enables",
        "category": "causal",
    },
    "triggers": {
        "description": "A triggers B",
        "symmetric": False,
        "transitive": False,
        "inverse": "triggered_by",
        "category": "causal",
    },
    "triggered_by": {
        "description": "Triggered by another event",
        "symmetric": False,
        "transitive": False,
        "inverse": "triggers",
        "category": "causal",
    },
    "leads_to": {
        "description": "A leads to B",
        "symmetric": False,
        "transitive": True,
        "inverse": "led_to_by",
        "category": "causal",
    },
    "led_to_by": {
        "description": "Led to by another event",
        "symmetric": False,
        "transitive": True,
        "inverse": "leads_to",
        "category": "causal",
    },
    "prevents": {
        "description": "A prevents B",
        "symmetric": False,
        "transitive": False,
        "inverse": "prevented_by",
        "category": "causal",
    },
    "prevented_by": {
        "description": "Prevented by another event",
        "symmetric": False,
        "transitive": False,
        "inverse": "prevents",
        "category": "causal",
    },
    # --- Temporal relationships ---
    "before": {
        "description": "Occurs before in time",
        "symmetric": False,
        "transitive": True,
        "inverse": "after",
        "category": "temporal",
    },
    "after": {
        "description": "Occurs after in time",
        "symmetric": False,
        "transitive": True,
        "inverse": "before",
        "category": "temporal",
    },
    "during": {
        "description": "Occurs during timespan",
        "symmetric": False,
        "transitive": False,
        "inverse": None,
        "category": "temporal",
    },
    # --- Similarity relationships ---
    "similar_to": {
        "description": "Similar content or meaning",
        "symmetric": True,
        "transitive": False,
        "inverse": "similar_to",
        "category": "similarity",
    },
    "duplicate_of": {
        "description": "Exact or near duplicate",
        "symmetric": True,
        "transitive": True,
        "inverse": "duplicate_of",
        "category": "similarity",
    },
    "related_to": {
        "description": "Generic related relationship",
        "symmetric": True,
        "transitive": False,
        "inverse": "related_to",
        "category": "similarity",
    },
    "variant_of": {
        "description": "A is a variant of B",
        "symmetric": True,
        "transitive": False,
        "inverse": "variant_of",
        "category": "similarity",
    },
    # --- Learning relationships (formerly "logical") ---
    "contradicts": {
        "description": "Logically contradicts",
        "symmetric": True,
        "transitive": False,
        "inverse": "contradicts",
        "category": "learning",
    },
    "supports": {
        "description": "Provides evidence for",
        "symmetric": False,
        "transitive": False,
        "inverse": "supported_by",
        "category": "learning",
    },
    "supported_by": {
        "description": "Evidence provided by",
        "symmetric": False,
        "transitive": False,
        "inverse": "supports",
        "category": "learning",
    },
    "builds_on": {
        "description": "A builds on knowledge in B",
        "symmetric": False,
        "transitive": True,
        "inverse": "built_upon_by",
        "category": "learning",
    },
    "built_upon_by": {
        "description": "Knowledge built upon by another",
        "symmetric": False,
        "transitive": True,
        "inverse": "builds_on",
        "category": "learning",
    },
    "confirms": {
        "description": "A confirms or validates B",
        "symmetric": True,
        "transitive": False,
        "inverse": "confirms",
        "category": "learning",
    },
    "supersedes": {
        "description": "A supersedes B with newer information",
        "symmetric": False,
        "transitive": True,
        "inverse": "superseded_by",
        "category": "learning",
    },
    "superseded_by": {
        "description": "Superseded by newer information",
        "symmetric": False,
        "transitive": True,
        "inverse": "supersedes",
        "category": "learning",
    },
    # --- Refinement relationships ---
    "refines": {
        "description": "Refines or elaborates on",
        "symmetric": False,
        "transitive": False,
        "inverse": "refined_by",
        "category": "refinement",
    },
    "refined_by": {
        "description": "Refined or elaborated by",
        "symmetric": False,
        "transitive": False,
        "inverse": "refines",
        "category": "refinement",
    },
    "replaces": {
        "description": "Supersedes or replaces",
        "symmetric": False,
        "transitive": False,
        "inverse": "replaced_by",
        "category": "refinement",
    },
    "replaced_by": {
        "description": "Superseded by",
        "symmetric": False,
        "transitive": False,
        "inverse": "replaces",
        "category": "refinement",
    },
    # --- Knowledge-work relationships ---
    "owns": {
        "description": "Agent owns or is accountable for a work resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "owned_by",
        "category": "knowledge_work",
    },
    "owned_by": {
        "description": "Work resource is owned by an agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "owns",
        "category": "knowledge_work",
    },
    "responsible_for": {
        "description": "Agent is assigned or responsible for a work resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "assigned_to",
        "category": "knowledge_work",
    },
    "assigned_to": {
        "description": "Work resource is assigned to a responsible agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "responsible_for",
        "category": "knowledge_work",
    },
    "authored": {
        "description": "Agent created or authored an artifact",
        "symmetric": False,
        "transitive": False,
        "inverse": "authored_by",
        "category": "knowledge_work",
    },
    "authored_by": {
        "description": "Artifact was created or authored by an agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "authored",
        "category": "knowledge_work",
    },
    "contributed_to": {
        "description": "Agent made a contribution to a work resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "has_contributor",
        "category": "knowledge_work",
    },
    "has_contributor": {
        "description": "Work resource has a contributing agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "contributed_to",
        "category": "knowledge_work",
    },
    "reviewed": {
        "description": "Agent reviewed a work resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "reviewed_by",
        "category": "knowledge_work",
    },
    "reviewed_by": {
        "description": "Work resource was reviewed by an agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "reviewed",
        "category": "knowledge_work",
    },
    "approved": {
        "description": "Agent approved a work resource or decision",
        "symmetric": False,
        "transitive": False,
        "inverse": "approved_by",
        "category": "knowledge_work",
    },
    "approved_by": {
        "description": "Work resource or decision was approved by an agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "approved",
        "category": "knowledge_work",
    },
    "decided": {
        "description": "Agent made or authorized a decision",
        "symmetric": False,
        "transitive": False,
        "inverse": "decided_by",
        "category": "knowledge_work",
    },
    "decided_by": {
        "description": "Decision was made or authorized by an agent",
        "symmetric": False,
        "transitive": False,
        "inverse": "decided",
        "category": "knowledge_work",
    },
    "based_on": {
        "description": "Decision or artifact is based on source evidence",
        "symmetric": False,
        "transitive": False,
        "inverse": "basis_for",
        "category": "knowledge_work",
    },
    "basis_for": {
        "description": "Evidence or artifact provides the basis for another resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "based_on",
        "category": "knowledge_work",
    },
    "affects": {
        "description": "Decision, event, or work item affects another resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "affected_by",
        "category": "knowledge_work",
    },
    "affected_by": {
        "description": "Resource is affected by a decision, event, or work item",
        "symmetric": False,
        "transitive": False,
        "inverse": "affects",
        "category": "knowledge_work",
    },
    "about": {
        "description": "Artifact or work resource is primarily about a topic",
        "symmetric": False,
        "transitive": False,
        "inverse": "subject_of",
        "category": "knowledge_work",
    },
    "subject_of": {
        "description": "Topic is the subject of an artifact or work resource",
        "symmetric": False,
        "transitive": False,
        "inverse": "about",
        "category": "knowledge_work",
    },
    "member_of": {
        "description": "Agent is a member of an organization or team",
        "symmetric": False,
        "transitive": False,
        "inverse": "has_member",
        "category": "knowledge_work",
    },
    "has_member": {
        "description": "Organization or team has an agent as a member",
        "symmetric": False,
        "transitive": False,
        "inverse": "member_of",
        "category": "knowledge_work",
    },
    # --- Reference relationships ---
    "references": {
        "description": "References or cites",
        "symmetric": False,
        "transitive": False,
        "inverse": "referenced_by",
        "category": "reference",
    },
    "referenced_by": {
        "description": "Referenced or cited by",
        "symmetric": False,
        "transitive": False,
        "inverse": "references",
        "category": "reference",
    },
    # --- Solution relationships ---
    "solves": {
        "description": "A solves problem B",
        "symmetric": False,
        "transitive": False,
        "inverse": "solved_by",
        "category": "solution",
    },
    "solved_by": {
        "description": "Problem solved by A",
        "symmetric": False,
        "transitive": False,
        "inverse": "solves",
        "category": "solution",
    },
    "addresses": {
        "description": "A addresses issue B",
        "symmetric": False,
        "transitive": False,
        "inverse": "addressed_by",
        "category": "solution",
    },
    "addressed_by": {
        "description": "Issue addressed by A",
        "symmetric": False,
        "transitive": False,
        "inverse": "addresses",
        "category": "solution",
    },
    "alternative_to": {
        "description": "A is an alternative to B",
        "symmetric": True,
        "transitive": False,
        "inverse": "alternative_to",
        "category": "solution",
    },
    "improves": {
        "description": "A improves B",
        "symmetric": False,
        "transitive": False,
        "inverse": "improved_by",
        "category": "solution",
    },
    "improved_by": {
        "description": "Improved by A",
        "symmetric": False,
        "transitive": False,
        "inverse": "improves",
        "category": "solution",
    },
    # --- Context relationships ---
    "occurs_in": {
        "description": "A occurs in context B",
        "symmetric": False,
        "transitive": False,
        "inverse": "contains_occurrence",
        "category": "context",
    },
    "contains_occurrence": {
        "description": "Context B contains occurrence of A",
        "symmetric": False,
        "transitive": False,
        "inverse": "occurs_in",
        "category": "context",
    },
    "applies_to": {
        "description": "A applies to B",
        "symmetric": False,
        "transitive": False,
        "inverse": "has_applicable",
        "category": "context",
    },
    "has_applicable": {
        "description": "B has applicable A",
        "symmetric": False,
        "transitive": False,
        "inverse": "applies_to",
        "category": "context",
    },
    "works_with": {
        "description": "A works with B",
        "symmetric": True,
        "transitive": False,
        "inverse": "works_with",
        "category": "context",
    },
    "requires": {
        "description": "A requires B",
        "symmetric": False,
        "transitive": True,
        "inverse": "required_by",
        "category": "context",
    },
    "required_by": {
        "description": "Required by A",
        "symmetric": False,
        "transitive": True,
        "inverse": "requires",
        "category": "context",
    },
    # --- Workflow relationships ---
    "follows": {
        "description": "A follows B in sequence",
        "symmetric": False,
        "transitive": True,
        "inverse": "followed_by",
        "category": "workflow",
    },
    "followed_by": {
        "description": "Followed by A in sequence",
        "symmetric": False,
        "transitive": True,
        "inverse": "follows",
        "category": "workflow",
    },
    "depends_on": {
        "description": "A depends on B",
        "symmetric": False,
        "transitive": True,
        "inverse": "depended_on_by",
        "category": "workflow",
    },
    "depended_on_by": {
        "description": "Depended on by A",
        "symmetric": False,
        "transitive": True,
        "inverse": "depends_on",
        "category": "workflow",
    },
    "blocks": {
        "description": "A blocks B",
        "symmetric": False,
        "transitive": False,
        "inverse": "blocked_by",
        "category": "workflow",
    },
    "blocked_by": {
        "description": "Blocked by A",
        "symmetric": False,
        "transitive": False,
        "inverse": "blocks",
        "category": "workflow",
    },
    # --- Quality relationships ---
    "effective_for": {
        "description": "A is effective for B",
        "symmetric": False,
        "transitive": False,
        "inverse": "has_effective",
        "category": "quality",
    },
    "has_effective": {
        "description": "B has effective A",
        "symmetric": False,
        "transitive": False,
        "inverse": "effective_for",
        "category": "quality",
    },
    "preferred_over": {
        "description": "A is preferred over B",
        "symmetric": False,
        "transitive": True,
        "inverse": "less_preferred_than",
        "category": "quality",
    },
    "less_preferred_than": {
        "description": "A is less preferred than B",
        "symmetric": False,
        "transitive": True,
        "inverse": "preferred_over",
        "category": "quality",
    },
    "deprecated_by": {
        "description": "A is deprecated by B",
        "symmetric": False,
        "transitive": False,
        "inverse": "deprecates",
        "category": "quality",
    },
    "deprecates": {
        "description": "A deprecates B",
        "symmetric": False,
        "transitive": False,
        "inverse": "deprecated_by",
        "category": "quality",
    },
}


# ---------------------------------------------------------------------------
# Entity-type vocabulary (the ontology's third dimension, alongside relationship
# types and memory subtypes). This is the CANONICAL home for the set of entity
# types and their NER labels — the GLiNER2 extractor derives its label set from
# here rather than from a hard-coded enum, so a deployment/domain can add types
# (e.g. equipment, chemical, standard) via env config or an OntologyContributor
# without touching code. ``EntityType`` (models.entity_registry) stays as the
# small CORE/reserved set with special resolution semantics (person promotion,
# concept catch-all); everything here is a superset of it.
#
# Each entry: {ner_label: str | None, description: str, core: bool}. ``ner_label``
# is the label sent to the zero-shot NER model; ``None`` means the type exists but
# is NOT extracted by NER (e.g. ``concept`` is the catch-all fallback — as a NER
# label it would tag common words).
# ---------------------------------------------------------------------------
BASE_ENTITY_TYPES: dict[str, dict] = {
    "person": {"ner_label": "person", "description": "An individual person.", "core": True},
    "org": {"ner_label": "organization", "description": "A company, institution, or group.", "core": True},
    "project": {"ner_label": "project", "description": "A named project, product, method, or system.", "core": True},
    "place": {"ner_label": "location", "description": "A geographic or physical location.", "core": True},
    "event": {"ner_label": "event", "description": "A named event or occurrence.", "core": True},
    "artifact": {
        "ner_label": None,
        "description": "A document, dataset, model, design, message, or other knowledge-work output.",
        "core": False,
    },
    "work_item": {
        "ner_label": None,
        "description": "A task, issue, requirement, deliverable, or unit of planned work.",
        "core": False,
    },
    "decision": {
        "ner_label": None,
        "description": "A named choice, approval, or policy decision.",
        "core": False,
    },
    "topic": {
        "ner_label": None,
        "description": "A named subject area used to organize knowledge resources.",
        "core": False,
    },
    "concept": {"ner_label": None, "description": "Catch-all for any other named concept.", "core": True},
}

_REQUIRED_ENTITY_TYPE_META_FIELDS = ("ner_label", "description")


from abc import ABC, abstractmethod


class OntologyService(ABC):
    """Interface for ontology service."""

    # -- Entity-type vocabulary (base + contributions + tenant/workspace) -----
    @abstractmethod
    def list_entity_types(self, tenant_id: str = "_default", workspace_id: str | None = None) -> list[str]:
        """List all entity types in the merged vocabulary (base + contributed)."""
        ...

    @abstractmethod
    def validate_entity_type(
        self, entity_type: str, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> bool:
        """Return True if ``entity_type`` is in the merged entity-type vocabulary."""
        ...

    @abstractmethod
    def get_entity_type_info(
        self, entity_type: str, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> dict | None:
        """Return metadata (ner_label, description, core) for an entity type, or None."""
        ...

    @abstractmethod
    def get_ner_labels(self, tenant_id: str = "_default", workspace_id: str | None = None) -> list[str]:
        """The NER label set for extraction — every entity type with a non-null ner_label."""
        ...

    @abstractmethod
    def ner_label_to_entity_type(
        self, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> dict[str, str]:
        """Reverse map ``ner_label -> entity_type`` for classifying NER output."""
        ...

    @abstractmethod
    def extend_entity_types(
        self, entity_types: dict[str, dict] | None = None, *, source: str = "runtime"
    ) -> None:
        """Push entity types into the vocabulary after init (mirrors extend_subtypes).

        ``entity_types`` maps ``type_name -> {ner_label, description}``. Use this
        (push) path for runtime/config contributions; a plugin's
        ``OntologyContributorPlugin.get_entity_types()`` is the pull equivalent.
        """
        ...

    @abstractmethod
    def get_merged_ontology(self, tenant_id: str, workspace_id: str | None = None) -> dict:
        """Get merged ontology (base + custom for enterprise)."""
        pass

    @abstractmethod
    def validate_relationship(self, relationship_type: str, tenant_id: str, workspace_id: str | None = None) -> bool:
        """Validate that a relationship type exists in the ontology."""
        pass

    @abstractmethod
    def get_relationship_info(self, relationship_type: str, tenant_id: str, workspace_id: str | None = None) -> dict:
        """Get metadata about a relationship type."""
        pass

    @abstractmethod
    def create_ontology(self, tenant_id: str, name: str, relationships: dict, workspace_id: str | None = None) -> dict:
        """Create a custom ontology (Enterprise only)."""
        pass

    @abstractmethod
    def list_relationship_types(self, tenant_id: str, workspace_id: str | None = None) -> list[str]:
        """List all available relationship types."""
        pass

    @abstractmethod
    async def classify_relationship(
        self,
        content_a: str,
        content_b: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> str:
        """Use LLM to classify the relationship between two memory contents.

        Returns a relationship type string from the ontology.
        Falls back to related_to if classification fails.
        """
        pass

    async def classify_relationships_batch(
        self,
        content_a: str,
        candidates: list[tuple[str, str]],
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        """Classify ``content_a``'s relationship to each candidate.

        ``candidates`` is a list of ``(candidate_id, candidate_content)``.
        Returns ``{candidate_id: relationship_type}``.

        This concrete default loops :meth:`classify_relationship` (one LLM
        call per candidate) so any subclass works out of the box.
        :class:`DefaultOntologyService` overrides it to do the whole batch in
        a SINGLE LLM call, which is the cost-reduction path.
        """
        results: dict[str, str] = {}
        for cand_id, cand_content in candidates:
            results[cand_id] = await self.classify_relationship(
                content_a=content_a,
                content_b=cand_content,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
        return results

    @abstractmethod
    def get_relationships_by_category(
        self,
        category: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> list[str]:
        """Get all relationship types in a category."""
        pass

    @abstractmethod
    def extend_ontology(
        self,
        relationship_types: dict[str, dict] | None = None,
        *,
        source: str = "runtime",
    ) -> None:
        """Push relationship types into the ontology after init.

        Use cases: test fixtures, runtime feature toggles, plugins that
        compute contributions dynamically. For static contributions, prefer
        subclassing :class:`OntologyContributorPlugin` instead.

        Collisions with the base ontology or with previously contributed
        types from a different source are logged at WARNING level and the
        new metadata replaces the old. The ``source`` argument is recorded
        for diagnostics and is exposed via :meth:`list_contributors`.
        """
        pass

    @abstractmethod
    def list_categories(self, tenant_id: str, workspace_id: str | None = None) -> list[str]:
        """List relationship categories present in the merged ontology."""
        pass

    @abstractmethod
    def list_contributors(self) -> list[dict]:
        """List relationship type and subtype contributions and their sources.

        Returns a list of dicts. Each entry has a ``kind`` field
        (``"relationship"`` or ``"subtype"``) plus kind-specific fields:

        - relationship: ``{"type_name": str, "kind": "relationship", "source": str}``
        - subtype: ``{"memory_type": str, "subtype": str, "kind": "subtype", "source": str}``
        """
        pass

    @abstractmethod
    def extend_subtypes(
        self,
        subtypes: dict[str, set[str]] | None = None,
        *,
        source: str = "runtime",
    ) -> None:
        """Push memory subtypes into the ontology after init.

        Mirrors :meth:`extend_ontology` for memory subtypes. The
        ``subtypes`` argument maps memory_type -> set of subtype strings.
        Use ``"*"`` as the memory_type key to contribute subtypes that
        apply to any memory type.

        Collisions with the OSS-known subtype set are logged at WARNING
        level (the new entry is recorded for diagnostics regardless).
        The ``source`` argument is recorded and exposed via
        :meth:`list_contributors`.
        """
        pass

    @abstractmethod
    def list_subtypes(
        self,
        memory_type: str | None = None,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> list[str]:
        """List valid subtypes for the given memory type.

        Returns the union of OSS-known subtypes and contributed
        subtypes. If ``memory_type`` is provided, returns only the
        subtypes that apply to that memory type (subtypes contributed
        with the ``"*"`` key always apply). If ``memory_type`` is None,
        returns the union across all memory types.
        """
        pass

    @abstractmethod
    def validate_subtype(
        self,
        memory_type: str,
        subtype: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> bool:
        """Return True if ``subtype`` is valid for ``memory_type``."""
        pass


# noinspection PyAbstractClass
OntologyServicePluginBase = make_service_plugin_base(
    ext_name=EXT_ONTOLOGY_SERVICE,
    config_key=MEMORYLAYER_ONTOLOGY_SERVICE,
    default_value=DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE,
)
