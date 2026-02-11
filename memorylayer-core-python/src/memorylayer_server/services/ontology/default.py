"""
Default Ontology Service implementation.

Provides relationship type definitions and validation.
OSS version includes unified ontology with 65 relationship types across 11 categories.
"""
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import (
    OntologyService,
    OntologyServicePluginBase,
    FeatureRequiresUpgradeError,
    BASE_ONTOLOGY,
    RELATIONSHIP_CATEGORIES,
)


class DefaultOntologyService(OntologyService):
    """Default ontology service implementation for OSS."""

    def __init__(self, v: Variables = None, llm_service=None):
        """Initialize ontology service with base ontology.

        Args:
            v: Application variables for configuration.
            llm_service: Optional LLM service for relationship classification.
        """
        self.base_ontology = BASE_ONTOLOGY
        self.llm_service = llm_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized DefaultOntologyService with %s relationship types across %s categories",
            len(BASE_ONTOLOGY),
            len(RELATIONSHIP_CATEGORIES),
        )

    def get_merged_ontology(
            self,
            tenant_id: str,
            workspace_id: Optional[str] = None
    ) -> dict:
        """
        Get merged ontology (base + custom for enterprise).

        OSS version returns base ontology only.

        Args:
            tenant_id: Tenant ID
            workspace_id: Optional workspace ID for workspace-level ontologies

        Returns:
            Merged ontology dictionary
        """
        # OSS: Return base ontology only
        return self.base_ontology.copy()

    def validate_relationship(
            self,
            relationship_type: str,
            tenant_id: str,
            workspace_id: Optional[str] = None
    ) -> bool:
        """
        Validate that a relationship type exists in the ontology.

        Args:
            relationship_type: Relationship type to validate
            tenant_id: Tenant ID
            workspace_id: Optional workspace ID

        Returns:
            True if relationship type is valid

        Raises:
            ValueError: If relationship type is invalid
        """
        ontology = self.get_merged_ontology(tenant_id, workspace_id)

        if relationship_type not in ontology:
            valid_types = ", ".join(sorted(ontology.keys()))
            raise ValueError(
                f"Invalid relationship type: {relationship_type}. "
                f"Valid types: {valid_types}"
            )

        return True

    def get_relationship_info(
            self,
            relationship_type: str,
            tenant_id: str,
            workspace_id: Optional[str] = None
    ) -> dict:
        """
        Get metadata about a relationship type.

        Args:
            relationship_type: Relationship type
            tenant_id: Tenant ID
            workspace_id: Optional workspace ID

        Returns:
            Relationship metadata (description, symmetric, transitive, inverse, category)

        Raises:
            ValueError: If relationship type is invalid
        """
        self.validate_relationship(relationship_type, tenant_id, workspace_id)
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return ontology[relationship_type].copy()

    def create_ontology(
            self,
            tenant_id: str,
            name: str,
            relationships: dict,
            workspace_id: Optional[str] = None
    ) -> dict:
        """
        Create a custom ontology.

        This feature requires MemoryLayer Enterprise.

        Args:
            tenant_id: Tenant ID
            name: Ontology name
            relationships: Custom relationship definitions
            workspace_id: Optional workspace ID for workspace-level ontology

        Raises:
            FeatureRequiresUpgradeError: Always (OSS limitation)
        """
        raise FeatureRequiresUpgradeError("custom_ontologies")

    def list_relationship_types(
            self,
            tenant_id: str,
            workspace_id: Optional[str] = None
    ) -> list[str]:
        """
        List all available relationship types.

        Args:
            tenant_id: Tenant ID
            workspace_id: Optional workspace ID

        Returns:
            List of relationship type names
        """
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return sorted(ontology.keys())

    async def classify_relationship(
            self,
            content_a: str,
            content_b: str,
            tenant_id: str = "_default",
            workspace_id: Optional[str] = None,
    ) -> str:
        """Use LLM to classify the relationship between two memory contents.

        Builds a prompt listing all relationship types with descriptions,
        asks the LLM to pick the best one.

        Args:
            content_a: Content of the first memory.
            content_b: Content of the second memory.
            tenant_id: Tenant ID for ontology lookup.
            workspace_id: Optional workspace ID.

        Returns:
            A relationship type string from the ontology.
            Falls back to related_to if LLM is unavailable or classification fails.
        """
        if self.llm_service is None:
            self.logger.debug("LLM service not available, falling back to related_to")
            return "related_to"

        ontology = self.get_merged_ontology(tenant_id, workspace_id)

        # Build the type listing for the prompt
        type_lines = []
        for rel_type, info in sorted(ontology.items()):
            type_lines.append(f"  {rel_type}: {info['description']}")
        types_list = "\n".join(type_lines)

        prompt = (
            "Given two pieces of content, classify the relationship between them.\n"
            "\n"
            f"Content A: {content_a}\n"
            "\n"
            f"Content B: {content_b}\n"
            "\n"
            "Available relationship types (A -> B):\n"
            f"{types_list}\n"
            "\n"
            'Respond with ONLY the relationship type name (e.g., "causes", "similar_to").\n'
            'If unsure, respond with "related_to".'
        )

        try:
            from ...models.llm import LLMRequest, LLMMessage, LLMRole

            request = LLMRequest(
                messages=[
                    LLMMessage(role=LLMRole.USER, content=prompt),
                ],
                temperature_factor=0.15,
                max_tokens=250,
            )

            response = await self.llm_service.complete(request, profile="ontology")
            result = response.content.strip().lower().replace('"', '').replace("'", '').rstrip('.')

            # Validate the LLM response against the ontology
            if result in ontology:
                self.logger.debug("LLM classified relationship as %s", result)
                return result

            # Try prefix matching for truncated LLM responses
            # (e.g. 'built_' -> 'built_upon_by', 'referenced_' -> 'referenced_by')
            if result:
                prefix_matches = [t for t in ontology if t.startswith(result)]
                if len(prefix_matches) == 1:
                    matched = prefix_matches[0]
                    self.logger.debug(
                        "Prefix-matched truncated relationship '%s' to '%s'",
                        result, matched,
                    )
                    return matched

            self.logger.warning(
                "LLM returned invalid relationship type '%s', falling back to related_to",
                result,
            )
            return "related_to"

        except Exception:
            self.logger.exception("Failed to classify relationship via LLM, falling back to related_to")
            return "related_to"

    def get_relationships_by_category(
            self,
            category: str,
            tenant_id: str = "_default",
            workspace_id: Optional[str] = None,
    ) -> list[str]:
        """Get all relationship types in a category.

        Args:
            category: Category name (e.g., "causal", "solution").
            tenant_id: Tenant ID for ontology lookup.
            workspace_id: Optional workspace ID.

        Returns:
            Sorted list of relationship type names in the given category.

        Raises:
            ValueError: If the category is not recognized.
        """
        if category not in RELATIONSHIP_CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. "
                f"Valid categories: {', '.join(sorted(RELATIONSHIP_CATEGORIES))}"
            )

        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return sorted(
            rel_type
            for rel_type, info in ontology.items()
            if info.get("category") == category
        )


class DefaultOntologyServicePlugin(OntologyServicePluginBase):
    """Default ontology service plugin."""
    PROVIDER_NAME = 'default'

    def get_dependencies(self, v: Variables):
        return ()  # LLM is optional, don't require it

    def initialize(self, v: Variables, logger) -> OntologyService:
        # Try to get LLM service, but don't fail if unavailable
        llm_service = None
        try:
            from ..llm import EXT_LLM_SERVICE
            llm_service = self.get_extension(EXT_LLM_SERVICE, v)
        except Exception:
            logger.debug("LLM service not available for ontology classification")
        return DefaultOntologyService(v=v, llm_service=llm_service)
