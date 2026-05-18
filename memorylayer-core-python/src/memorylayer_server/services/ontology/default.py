"""
Default Ontology Service implementation.

Provides relationship type definitions, validation, and an extensible
contribution mechanism (pull via OntologyContributorPlugin, push via
extend_ontology).
"""

from logging import Logger

from scitrera_app_framework import get_extensions, get_logger
from scitrera_app_framework.api import Variables

from ...models.memory import OSS_KNOWN_SUBTYPES
from .._constants import EXT_MULTI_ONTOLOGY_CONTRIBUTORS
from .base import (
    BASE_ONTOLOGY,
    OntologyService,
    OntologyServicePluginBase,
)

_REQUIRED_META_FIELDS = ("description", "symmetric", "transitive", "inverse", "category")


class DefaultOntologyService(OntologyService):
    """Default ontology service implementation for OSS."""

    # Class-level flag so the in-memory persistence warning fires only once
    # per process regardless of how many service instances exist.
    _persistence_warning_emitted: bool = False

    def __init__(self, v: Variables = None, llm_service=None):
        """Initialize ontology service with base ontology.

        Args:
            v: Application variables for configuration.
            llm_service: Optional LLM service for relationship classification.
        """
        self.base_ontology = BASE_ONTOLOGY
        self.llm_service = llm_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        # Contributed types (pull via OntologyContributorPlugin or push via
        # extend_ontology). Both paths funnel into the same dict.
        self._contributions: dict[str, dict] = {}
        self._contribution_sources: dict[str, str] = {}
        # Contributed subtypes (pull via OntologyContributorPlugin or push
        # via extend_subtypes). Mirrors the relationship-type contribution
        # storage. Keyed by memory_type ("*" means "any memory type").
        self._contributed_subtypes: dict[str, set[str]] = {}
        # source name keyed by (memory_type, subtype) for diagnostics.
        self._subtype_sources: dict[tuple[str, str], str] = {}
        # Tenant/workspace-scoped persisted custom ontologies. In-memory
        # only for this PR; the seam exists for a SQL follow-up.
        self._persistent: dict[tuple[str, str | None], dict[str, dict]] = {}
        base_categories = len({v["category"] for v in BASE_ONTOLOGY.values()})
        self.logger.info(
            "Initialized DefaultOntologyService with %s base relationship types across %s categories",
            len(BASE_ONTOLOGY),
            base_categories,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_meta(type_name: str, meta) -> None:
        if not isinstance(meta, dict):
            raise ValueError(f"Relationship type '{type_name}' metadata must be a dict, got {type(meta).__name__}")
        missing = [f for f in _REQUIRED_META_FIELDS if f not in meta]
        if missing:
            raise ValueError(f"Relationship type '{type_name}' is missing required metadata field(s): {', '.join(missing)}")

    def _load_persistent(self) -> None:
        """Persistence load seam for a future SQL implementation."""
        return None

    def _save_persistent(self, tenant_id: str, workspace_id: str | None) -> None:
        """Persistence save seam for a future SQL implementation."""
        return None

    # ------------------------------------------------------------------
    # Merged ontology
    # ------------------------------------------------------------------

    def get_merged_ontology(self, tenant_id: str, workspace_id: str | None = None) -> dict:
        """Return the layered merged ontology for ``(tenant_id, workspace_id)``.

        Layer order (later overrides earlier):
            1. ``BASE_ONTOLOGY``
            2. ``self._contributions`` (pull + push)
            3. tenant-level persisted custom ontology
            4. workspace-level persisted custom ontology
        """
        merged: dict[str, dict] = {}
        merged.update(BASE_ONTOLOGY)
        merged.update(self._contributions)
        merged.update(self._persistent.get((tenant_id, None), {}))
        if workspace_id is not None:
            merged.update(self._persistent.get((tenant_id, workspace_id), {}))
        return merged

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def validate_relationship(self, relationship_type: str, tenant_id: str, workspace_id: str | None = None) -> bool:
        ontology = self.get_merged_ontology(tenant_id, workspace_id)

        if relationship_type not in ontology:
            valid_types = ", ".join(sorted(ontology.keys()))
            raise ValueError(f"Invalid relationship type: {relationship_type}. Valid types: {valid_types}")

        return True

    def get_relationship_info(self, relationship_type: str, tenant_id: str, workspace_id: str | None = None) -> dict:
        self.validate_relationship(relationship_type, tenant_id, workspace_id)
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return ontology[relationship_type].copy()

    def list_relationship_types(self, tenant_id: str, workspace_id: str | None = None) -> list[str]:
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return sorted(ontology.keys())

    def get_relationships_by_category(
        self,
        category: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> list[str]:
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        known_categories = {info.get("category") for info in ontology.values()}
        if category not in known_categories:
            raise ValueError(f"Invalid category: {category}. Valid categories: {', '.join(sorted(c for c in known_categories if c))}")
        return sorted(rel_type for rel_type, info in ontology.items() if info.get("category") == category)

    def list_categories(self, tenant_id: str, workspace_id: str | None = None) -> list[str]:
        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        return sorted({info["category"] for info in ontology.values() if info.get("category")})

    def list_contributors(self) -> list[dict]:
        entries: list[dict] = [{"type_name": k, "kind": "relationship", "source": v} for k, v in self._contribution_sources.items()]
        for (memory_type, subtype), source in self._subtype_sources.items():
            entries.append(
                {
                    "memory_type": memory_type,
                    "subtype": subtype,
                    "kind": "subtype",
                    "source": source,
                }
            )
        return entries

    # ------------------------------------------------------------------
    # Subtypes (push, list, validate)
    # ------------------------------------------------------------------

    @staticmethod
    def _oss_known_subtypes_for(memory_type: str | None) -> set[str]:
        """Return OSS-known subtypes applicable to ``memory_type``.

        OSS-known subtypes registered under the ``"*"`` key apply to
        every memory type. Subtypes registered under a specific memory
        type apply only to that one. If ``memory_type`` is None, the
        union across all memory types is returned.
        """
        if memory_type is None:
            result: set[str] = set()
            for values in OSS_KNOWN_SUBTYPES.values():
                result |= values
            return result
        result = set(OSS_KNOWN_SUBTYPES.get("*", set()))
        result |= OSS_KNOWN_SUBTYPES.get(memory_type, set())
        return result

    def _contributed_subtypes_for(self, memory_type: str | None) -> set[str]:
        if memory_type is None:
            result: set[str] = set()
            for values in self._contributed_subtypes.values():
                result |= values
            return result
        result = set(self._contributed_subtypes.get("*", set()))
        result |= self._contributed_subtypes.get(memory_type, set())
        return result

    def extend_subtypes(
        self,
        subtypes: dict[str, set[str]] | None = None,
        *,
        source: str = "runtime",
    ) -> None:
        if not subtypes:
            return

        for memory_type, values in subtypes.items():
            if not isinstance(memory_type, str) or not memory_type:
                raise ValueError(f"Subtype contribution memory_type must be a non-empty string, got {memory_type!r}")
            if not isinstance(values, (set, frozenset, list, tuple)):
                raise ValueError(f"Subtype contribution for memory_type '{memory_type}' must be an iterable of strings")
            normalized = set()
            for value in values:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Subtype contribution for memory_type '{memory_type}' must contain non-empty strings")
                normalized.add(value)

            oss_known = self._oss_known_subtypes_for(memory_type)
            for value in normalized:
                if value in oss_known:
                    self.logger.warning(
                        "Subtype contribution from '%s' overrides OSS-known subtype '%s' for memory_type '%s'",
                        source,
                        value,
                        memory_type,
                    )
                existing_source = self._subtype_sources.get((memory_type, value))
                if existing_source is not None and existing_source != source:
                    self.logger.warning(
                        "Subtype contribution from '%s' overrides previous contribution of '%s' (memory_type '%s') from '%s'",
                        source,
                        value,
                        memory_type,
                        existing_source,
                    )
                self._subtype_sources[(memory_type, value)] = source

            bucket = self._contributed_subtypes.setdefault(memory_type, set())
            bucket |= normalized

    def list_subtypes(
        self,
        memory_type: str | None = None,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> list[str]:
        merged = self._oss_known_subtypes_for(memory_type) | self._contributed_subtypes_for(memory_type)
        return sorted(merged)

    def validate_subtype(
        self,
        memory_type: str,
        subtype: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> bool:
        return subtype in self._oss_known_subtypes_for(memory_type) or subtype in self._contributed_subtypes_for(memory_type)

    # ------------------------------------------------------------------
    # Push (extend) and create (persisted) APIs
    # ------------------------------------------------------------------

    def extend_ontology(
        self,
        relationship_types: dict[str, dict] | None = None,
        *,
        source: str = "runtime",
    ) -> None:
        if not relationship_types:
            return

        for type_name, meta in relationship_types.items():
            self._validate_meta(type_name, meta)

            if type_name in BASE_ONTOLOGY:
                self.logger.warning(
                    "Ontology contribution from '%s' overrides base relationship type '%s'",
                    source,
                    type_name,
                )

            existing_source = self._contribution_sources.get(type_name)
            if existing_source is not None and existing_source != source:
                self.logger.warning(
                    "Ontology contribution from '%s' overrides previous contribution of '%s' from '%s'",
                    source,
                    type_name,
                    existing_source,
                )

            self._contributions[type_name] = dict(meta)
            self._contribution_sources[type_name] = source

    def create_ontology(self, tenant_id: str, name: str, relationships: dict, workspace_id: str | None = None) -> dict:
        """Create or extend a tenant/workspace-scoped persisted custom ontology.

        Persistence is in-memory only in this PR; SQL persistence is a
        follow-up. A WARNING is logged on first use to make this clear.
        """
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Ontology name must be a non-empty string")
        if not relationships:
            raise ValueError("Ontology must contain at least one relationship type")

        # Validate every entry up-front so partial writes don't happen.
        for type_name, meta in relationships.items():
            self._validate_meta(type_name, meta)

        if not DefaultOntologyService._persistence_warning_emitted:
            self.logger.warning("Custom ontology persistence is in-memory; data will be lost on restart. SQL persistence is a follow-up.")
            DefaultOntologyService._persistence_warning_emitted = True

        key = (tenant_id, workspace_id)
        scoped = self._persistent.setdefault(key, {})
        for type_name, meta in relationships.items():
            scoped[type_name] = dict(meta)

        self._save_persistent(tenant_id, workspace_id)

        return {
            "name": name,
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "relationship_count": len(relationships),
        }

    # ------------------------------------------------------------------
    # LLM-backed classification
    # ------------------------------------------------------------------

    async def classify_relationship(
        self,
        content_a: str,
        content_b: str,
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> str:
        """Use LLM to classify the relationship between two memory contents.

        Builds a prompt listing all relationship types from the merged
        ontology with their descriptions, asks the LLM to pick the best one.
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
            from ...models.llm import LLMMessage, LLMRequest, LLMRole

            request = LLMRequest(
                messages=[
                    LLMMessage(role=LLMRole.USER, content=prompt),
                ],
                temperature_factor=0.15,
                max_tokens=250,
            )

            response = await self.llm_service.complete(request, profile="ontology")
            result = response.content.strip().lower().replace('"', "").replace("'", "").rstrip(".")

            if result in ontology:
                self.logger.debug("LLM classified relationship as %s", result)
                return result

            # Try prefix matching for truncated LLM responses
            if result:
                prefix_matches = [t for t in ontology if t.startswith(result)]
                if len(prefix_matches) == 1:
                    matched = prefix_matches[0]
                    self.logger.debug(
                        "Prefix-matched truncated relationship '%s' to '%s'",
                        result,
                        matched,
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


class DefaultOntologyServicePlugin(OntologyServicePluginBase):
    """Default ontology service plugin."""

    PROVIDER_NAME = "default"

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

    async def async_ready(self, v: Variables, logger: Logger, value: OntologyService) -> None:
        """Collect ontology contributors and merge them into the live service."""
        contributors = get_extensions(EXT_MULTI_ONTOLOGY_CONTRIBUTORS, v) or {}
        for c in contributors.values():
            name = getattr(c, "name", lambda: c.__class__.__name__)()
            try:
                types = c.get_relationship_types()
                value.extend_ontology(types, source=name)
            except Exception:
                logger.exception("Ontology contributor %s failed (relationship types)", name)
            try:
                subtypes = c.get_subtypes()
                if subtypes:
                    value.extend_subtypes(subtypes, source=name)
            except Exception:
                logger.exception("Ontology contributor %s failed (subtypes)", name)
        return None
