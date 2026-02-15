from scitrera_app_framework.api import Plugin, Variables, enabled_option_pattern

from ...config import MEMORYLAYER_ONTOLOGY_SERVICE, DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE

from .._constants import EXT_ONTOLOGY_SERVICE

# All valid relationship categories
RELATIONSHIP_CATEGORIES = {
    "hierarchical", "causal", "temporal", "similarity",
    "learning", "refinement", "reference",
    "solution", "context", "workflow", "quality",
}


class FeatureRequiresUpgradeError(Exception):
    """Raised when a feature requires enterprise upgrade."""

    def __init__(self, feature: str):
        self.feature = feature
        super().__init__(
            f"Feature '{feature}' requires MemoryLayer Enterprise. "
            "Visit https://memorylayer.ai/enterprise to upgrade."
        )


# Unified ontology with 45 relationship types across 11 categories
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


from abc import ABC, abstractmethod
from typing import Optional


class OntologyService(ABC):
    """Interface for ontology service."""

    @abstractmethod
    def get_merged_ontology(
        self,
        tenant_id: str,
        workspace_id: Optional[str] = None
    ) -> dict:
        """Get merged ontology (base + custom for enterprise)."""
        pass

    @abstractmethod
    def validate_relationship(
        self,
        relationship_type: str,
        tenant_id: str,
        workspace_id: Optional[str] = None
    ) -> bool:
        """Validate that a relationship type exists in the ontology."""
        pass

    @abstractmethod
    def get_relationship_info(
        self,
        relationship_type: str,
        tenant_id: str,
        workspace_id: Optional[str] = None
    ) -> dict:
        """Get metadata about a relationship type."""
        pass

    @abstractmethod
    def create_ontology(
        self,
        tenant_id: str,
        name: str,
        relationships: dict,
        workspace_id: Optional[str] = None
    ) -> dict:
        """Create a custom ontology (Enterprise only)."""
        pass

    @abstractmethod
    def list_relationship_types(
        self,
        tenant_id: str,
        workspace_id: Optional[str] = None
    ) -> list[str]:
        """List all available relationship types."""
        pass

    @abstractmethod
    async def classify_relationship(
        self,
        content_a: str,
        content_b: str,
        tenant_id: str = "_default",
        workspace_id: Optional[str] = None,
    ) -> str:
        """Use LLM to classify the relationship between two memory contents.

        Returns a relationship type string from the ontology.
        Falls back to related_to if classification fails.
        """
        pass

    @abstractmethod
    def get_relationships_by_category(
        self,
        category: str,
        tenant_id: str = "_default",
        workspace_id: Optional[str] = None,
    ) -> list[str]:
        """Get all relationship types in a category."""
        pass


# noinspection PyAbstractClass
class OntologyServicePluginBase(Plugin):
    """Base plugin for ontology service."""
    PROVIDER_NAME: str = None

    def name(self) -> str:
        return f"{EXT_ONTOLOGY_SERVICE}|{self.PROVIDER_NAME}"

    def extension_point_name(self, v: Variables) -> str:
        return EXT_ONTOLOGY_SERVICE

    def is_enabled(self, v: Variables) -> bool:
        return enabled_option_pattern(self, v, MEMORYLAYER_ONTOLOGY_SERVICE, self_attr='PROVIDER_NAME')

    def on_registration(self, v: Variables) -> None:
        v.set_default_value(MEMORYLAYER_ONTOLOGY_SERVICE, DEFAULT_MEMORYLAYER_ONTOLOGY_SERVICE)
