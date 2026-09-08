"""
Default Ontology Service implementation.

Provides relationship type definitions, validation, and an extensible
contribution mechanism (pull via OntologyContributorPlugin, push via
extend_ontology).
"""

import re
from logging import Logger

from scitrera_app_framework import get_extensions, get_logger
from scitrera_app_framework.api import Variables

from ...models.memory import OSS_KNOWN_SUBTYPES
from ...models.generation import GenerationActivity
from .._constants import EXT_MULTI_ONTOLOGY_CONTRIBUTORS
from ...config import (
    DEFAULT_MEMORYLAYER_ONTOLOGY_MAX_TOKENS,
    MEMORYLAYER_ENTITY_TYPES,
    MEMORYLAYER_ONTOLOGY_MAX_TOKENS,
)
from .base import (
    _REQUIRED_ENTITY_TYPE_META_FIELDS,
    BASE_ENTITY_TYPES,
    BASE_ONTOLOGY,
    OntologyService,
    OntologyServicePluginBase,
)

_REQUIRED_META_FIELDS = ("description", "symmetric", "transitive", "inverse", "category")

#: Relationship classification asks for ONE label, so reasoning is pure cost and
#: pure risk: a thinking model spends its budget deliberating and then emits the
#: deliberation as the answer.
DEFAULT_CLASSIFY_REASONING_EFFORT = "none"

#: Matches a <think>...</think> block, including an unclosed one (a reply
#: truncated mid-thought leaves no closing tag).
_THINK_BLOCK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def _clean_label(content: str | None) -> str:
    """Reduce a model reply to a bare lowercase label.

    Defence in depth behind ``reasoning_effort``: a model that reasons anyway
    (or a profile pointed at a model that cannot disable it) would otherwise
    have its whole chain of thought treated as the label, matching nothing and
    silently downgrading every edge to ``related_to``. Taking the LAST non-empty
    line recovers the verdict, since reasoning precedes the answer.
    """
    if not content:
        return ""
    text = _THINK_BLOCK_RE.sub("", content).strip()
    if not text:
        return ""
    last_line = text.splitlines()[-1].strip()
    return last_line.lower().replace('"', "").replace("'", "").rstrip(".").strip()


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
        # Completion cap for relationship classification (env-tunable).
        self.ontology_max_tokens = (
            v.get(MEMORYLAYER_ONTOLOGY_MAX_TOKENS,
                  DEFAULT_MEMORYLAYER_ONTOLOGY_MAX_TOKENS)
            if v is not None else DEFAULT_MEMORYLAYER_ONTOLOGY_MAX_TOKENS
        )
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
        # Contributed entity types (pull via OntologyContributorPlugin.get_entity_types
        # or push via extend_entity_types). Mirrors the subtype contribution storage.
        self._contributed_entity_types: dict[str, dict] = {}
        self._entity_type_sources: dict[str, str] = {}
        # Deployment-level entity-type extensions from env (per-tenant, since ML is
        # deployed per tenant): MEMORYLAYER_ENTITY_TYPES = CSV of "type" or
        # "type:ner_label" (ner_label defaults to the type name; use "type:" to add a
        # non-NER type). E.g. "equipment,chemical:chemical,standard:standard".
        self._load_entity_types_from_env(v)
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
    # Entity-type vocabulary (base + contributions)
    # ------------------------------------------------------------------

    def _load_entity_types_from_env(self, v: Variables | None) -> None:
        """Parse MEMORYLAYER_ENTITY_TYPES (CSV of ``type`` / ``type:ner_label``)."""
        if v is None:
            return
        raw = v.environ(MEMORYLAYER_ENTITY_TYPES, default="")
        if not raw:
            return
        contributed: dict[str, dict] = {}
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if ":" in token:
                name, _, label = token.partition(":")
                name = name.strip()
                label = label.strip() or None  # "type:" -> non-NER type
            else:
                name = token
                label = token  # bare "type" -> ner_label defaults to the type name
            if name:
                contributed[name] = {"ner_label": label, "description": f"Domain entity type '{name}'."}
        if contributed:
            self.extend_entity_types(contributed, source="env:MEMORYLAYER_ENTITY_TYPES")

    def _merged_entity_types(self) -> dict[str, dict]:
        """base ∪ contributed (contributions win on collision)."""
        merged = dict(BASE_ENTITY_TYPES)
        merged.update(self._contributed_entity_types)
        return merged

    def extend_entity_types(
        self, entity_types: dict[str, dict] | None = None, *, source: str = "runtime"
    ) -> None:
        if not entity_types:
            return
        for name, meta in entity_types.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Entity type name must be a non-empty string, got {name!r}")
            if not isinstance(meta, dict):
                raise ValueError(f"Entity type '{name}' metadata must be a dict, got {type(meta).__name__}")
            missing = [f for f in _REQUIRED_ENTITY_TYPE_META_FIELDS if f not in meta]
            if missing:
                raise ValueError(
                    f"Entity type '{name}' is missing required metadata field(s): {', '.join(missing)}"
                )
            if name in BASE_ENTITY_TYPES:
                self.logger.warning(
                    "Entity-type contribution from '%s' overrides base entity type '%s'", source, name
                )
            prior = self._entity_type_sources.get(name)
            if prior is not None and prior != source:
                self.logger.warning(
                    "Entity-type contribution from '%s' overrides previous '%s' from '%s'", source, name, prior
                )
            self._contributed_entity_types[name] = dict(meta)
            self._entity_type_sources[name] = source

    def list_entity_types(self, tenant_id: str = "_default", workspace_id: str | None = None) -> list[str]:
        return sorted(self._merged_entity_types().keys())

    def validate_entity_type(
        self, entity_type: str, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> bool:
        return entity_type in self._merged_entity_types()

    def get_entity_type_info(
        self, entity_type: str, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> dict | None:
        info = self._merged_entity_types().get(entity_type)
        return dict(info) if info is not None else None

    def get_ner_labels(self, tenant_id: str = "_default", workspace_id: str | None = None) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for meta in self._merged_entity_types().values():
            label = meta.get("ner_label")
            if label and label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def ner_label_to_entity_type(
        self, tenant_id: str = "_default", workspace_id: str | None = None
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for name, meta in self._merged_entity_types().items():
            label = meta.get("ner_label")
            if label:
                mapping.setdefault(label, name)  # first (base before contributed) wins
        return mapping

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
                max_tokens=self.ontology_max_tokens,
                # This is a single-label classification, not a reasoning task.
                # A thinking model emits its chain of thought into `content`,
                # which matches no relationship type and silently degrades EVERY
                # edge to related_to -- visible only as a log warning.
                reasoning_effort=DEFAULT_CLASSIFY_REASONING_EFFORT,
            )

            response = await self.llm_service.complete(
                request,
                profile="ontology",
                activity=GenerationActivity.RELATIONSHIP_CLASSIFICATION,
            )
            result = _clean_label(response.content)

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

    async def classify_relationships_batch(
        self,
        content_a: str,
        candidates: list[tuple[str, str]],
        tenant_id: str = "_default",
        workspace_id: str | None = None,
    ) -> dict[str, str]:
        """Classify ``content_a`` against every candidate in a SINGLE LLM call.

        The ontology type menu is listed once; each candidate is numbered and
        the model returns one ``index: relationship`` line per candidate. This
        collapses what used to be N per-pair calls into one call per new
        memory. Missing/invalid lines fall back to ``related_to``.
        """
        if not candidates:
            return {}

        # No LLM -> uniform related_to (matches single-call fallback contract).
        if self.llm_service is None:
            return {cand_id: "related_to" for cand_id, _ in candidates}

        # A single candidate isn't worth the batch framing/parse overhead.
        if len(candidates) == 1:
            cand_id, cand_content = candidates[0]
            return {
                cand_id: await self.classify_relationship(
                    content_a=content_a,
                    content_b=cand_content,
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                )
            }

        ontology = self.get_merged_ontology(tenant_id, workspace_id)
        types_list = "\n".join(f"  {rel_type}: {info['description']}" for rel_type, info in sorted(ontology.items()))

        candidate_lines = "\n".join(f"[{i}] {cand_content}" for i, (_, cand_content) in enumerate(candidates))

        prompt = (
            "Classify the relationship from the ANCHOR content to each numbered CANDIDATE.\n"
            "\n"
            f"ANCHOR (A): {content_a}\n"
            "\n"
            "CANDIDATES (B):\n"
            f"{candidate_lines}\n"
            "\n"
            "Available relationship types (A -> B):\n"
            f"{types_list}\n"
            "\n"
            "Respond with ONE line per candidate in the exact form `<index>: <relationship>`\n"
            '(e.g. "0: causes"). Use only relationship type names from the list above.\n'
            'If unsure for a candidate, use "related_to".'
        )

        try:
            from ...models.llm import LLMMessage, LLMRequest, LLMRole

            # Scale the completion budget with the candidate count so long
            # batches aren't truncated mid-list.
            max_tokens = min(self.ontology_max_tokens * max(1, len(candidates)), self.ontology_max_tokens * 8)
            request = LLMRequest(
                messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
                temperature_factor=0.15,
                max_tokens=max_tokens,
                # Same reasoning as the single-pair path: labels, not analysis.
                # More acute here — one leaked chain of thought costs the WHOLE
                # batch, not one edge.
                reasoning_effort=DEFAULT_CLASSIFY_REASONING_EFFORT,
            )
            response = await self.llm_service.complete(
                request,
                profile="ontology",
                activity=GenerationActivity.RELATIONSHIP_CLASSIFICATION,
            )
            parsed = self._parse_batch_response(response.content, ontology, len(candidates))
        except Exception:
            self.logger.exception("Batch relationship classification failed; falling back to related_to")
            parsed = {}

        # Map parsed index -> candidate_id, defaulting anything unparsed.
        results: dict[str, str] = {}
        for i, (cand_id, _) in enumerate(candidates):
            results[cand_id] = parsed.get(i, "related_to")
        return results

    def _parse_batch_response(self, raw: str, ontology: dict, n: int) -> dict[int, str]:
        """Parse ``<index>: <relationship>`` lines into ``{index: rel_type}``.

        Tolerates surrounding prose, quoting, trailing punctuation, and
        truncated type names (single-prefix match), mirroring the single-call
        parser. Out-of-range indices are dropped.

        Reasoning is stripped before scanning rather than merely tolerated: a
        chain of thought discussing the candidates contains lines that LOOK like
        ``<index>: <text>`` and would be parsed as verdicts, so leaving it in
        risks a confidently wrong answer instead of an honest fallback.
        """
        parsed: dict[int, str] = {}
        if not raw:
            return parsed
        for line in _THINK_BLOCK_RE.sub("", raw).splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            idx_part, _, rel_part = line.partition(":")
            idx_part = idx_part.strip().lstrip("[").rstrip("]").strip()
            if not idx_part.isdigit():
                continue
            idx = int(idx_part)
            if idx < 0 or idx >= n:
                continue
            rel = rel_part.strip().lower().replace('"', "").replace("'", "").rstrip(".").strip()
            if rel in ontology:
                parsed[idx] = rel
                continue
            if rel:
                prefix_matches = [t for t in ontology if t.startswith(rel)]
                if len(prefix_matches) == 1:
                    parsed[idx] = prefix_matches[0]
        return parsed


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
            try:
                entity_types = c.get_entity_types()
                if entity_types:
                    value.extend_entity_types(entity_types, source=name)
            except Exception:
                logger.exception("Ontology contributor %s failed (entity types)", name)
        return None
