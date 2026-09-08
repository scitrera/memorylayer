"""Fused duplicate + contradiction detection in a single LLM call per stored memory.

The deterministic provider (`default.py`) uses regex negation pairs. It costs nothing and
catches almost nothing: measured strictly against LongMemEval `knowledge-update` — the
benchmark that asks for a fact's *current* value after it changed — it detects **0.0%** of
supersessions. Relocations, job changes, count changes, date changes and value updates are
all invisible to it, because natural updates rarely contain a literal "is"/"is not" pair.

This provider replaces the pairwise regex test with **one** LLM call per stored memory that
judges the whole candidate neighbourhood at once, returning duplicates *and* contradictions
together (the shape Graphiti uses in `resolve_edge`). One call per memory, not per pair.

Measured on the same benchmark and candidate sets:

    regex   @ floor 0.7 (previous default)    0.0%
    regex   @ floor 0.5                       3.8%
    fused   @ floor 0.5, cheap model         ~49%      (ceiling at that floor: 93.6%)

Both halves were required. Raising the floor alone is nearly worthless; the detector is the
dominant term — but ~49% sits *above* the 51.3% ceiling that floor 0.7 imposes, so the
detector could not have got there without the floor moving too. See
`docs/DESIGN_graphiti_adoption.md`.

Selection:
    MEMORYLAYER_CONTRADICTION_PROVIDER=llm

Routing:
    Uses the ``contradiction`` LLM activity. Assign it to a cheap text-only profile —
    it runs on every stored memory:

        MEMORYLAYER_LLM_ASSIGN_CONTRADICTION=cheap

    A cheaper model measured ~49% against ~55% for a model 5.7x its price, and errs
    conservative (fewer flags, higher precision), which is the right failure direction
    for a detector whose false positives would demote correct memories.
"""

from logging import Logger

from ...models.generation import GenerationActivity

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...config import (
    DEFAULT_MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT,
    DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
    DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MIN_RELEVANCE,
    MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT,
    MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
    MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE,
)
from ...models.llm import LLMMessage, LLMRequest, LLMRole
from ..llm import EXT_LLM_SERVICE, LLMNotConfiguredError, LLMService
from ..storage import EXT_STORAGE_BACKEND
from ..storage.base import StorageBackend
from .base import (
    CONTRADICTION_TYPE_TEMPORAL_SUPERSESSION,
    ContradictionRecord,
    ContradictionService,
    ContradictionServicePluginBase,
)
from .default import DefaultContradictionService

LLM_PROFILE_CONTRADICTION = "contradiction"
DETECTION_METHOD = "llm_fused"

SYSTEM_PROMPT = (
    "You decide how a NEW FACT relates to EXISTING FACTS from the same user's memory.\n"
    "For each existing fact, decide if it is:\n"
    "  - a DUPLICATE: states the same thing as the new fact\n"
    "  - CONTRADICTED: the new fact makes it false or out of date, because the underlying "
    "value, status, preference, count, date, or location has CHANGED.\n"
    "Treat a superseding update as contradiction: if the user previously reported one value "
    "and the new fact reports a different value for the same thing, the old fact is contradicted.\n"
    "Elaborations, related topics, and facts about different things are NEITHER.\n"
    'Reply ONLY with JSON: {"duplicate_facts": [<indices>], "contradicted_facts": [<indices>]}'
)


class LLMContradictionService(DefaultContradictionService):
    """Contradiction detection via one fused LLM call over the candidate neighbourhood.

    Subclasses the deterministic service so resolution, ``get_unresolved``,
    ``scan_workspace`` and the candidate seam are inherited unchanged — this swaps the
    *detector*, not the contradiction authority or its storage contract.
    """

    def __init__(
        self,
        storage: StorageBackend,
        llm_service: LLMService,
        v: Variables = None,
        candidate_limit: int = DEFAULT_MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT,
        min_relevance: float = DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MIN_RELEVANCE,
        max_tokens: int = DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
    ):
        super().__init__(storage=storage, v=v, candidate_limit=candidate_limit, min_relevance=min_relevance)
        self.llm_service = llm_service
        self.max_tokens = max_tokens
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info(
            "Initialized LLMContradictionService (candidates=%d, min_relevance=%.2f, max_tokens=%d)",
            self.candidate_limit,
            self.min_relevance,
            self.max_tokens,
        )

    async def check_new_memory(self, workspace_id: str, memory_id: str) -> list[ContradictionRecord]:
        new_memory = await self._storage.get_memory(workspace_id, memory_id, track_access=False)
        if not new_memory:
            self.logger.warning("Memory %s not found in workspace %s", memory_id, workspace_id)
            return []
        if not new_memory.embedding:
            self.logger.debug("Memory %s has no embedding, skipping contradiction check", memory_id)
            return []

        candidates = [(mem, rel) for mem, rel in await self.fetch_candidates(workspace_id, new_memory.embedding) if mem.id != memory_id]
        if not candidates:
            return []

        # Exact-restatement short-circuit, as Graphiti does before its resolve call: a
        # verbatim repeat is a duplicate, never a contradiction, and needs no inference.
        normalized_new = " ".join(new_memory.content.lower().split())
        candidates = [(mem, rel) for mem, rel in candidates if " ".join(mem.content.lower().split()) != normalized_new]
        if not candidates:
            self.logger.debug("Memory %s: all candidates are exact restatements, no LLM call", memory_id)
            return []

        contradicted = await self._classify(new_memory.content, [mem.content for mem, _ in candidates])
        if not contradicted:
            return []

        contradictions: list[ContradictionRecord] = []
        for idx in contradicted:
            if not (0 <= idx < len(candidates)):
                self.logger.warning("LLM returned out-of-range index %s (have %d candidates)", idx, len(candidates))
                continue
            existing_memory, relevance = candidates[idx]
            record = ContradictionRecord(
                workspace_id=workspace_id,
                memory_a_id=memory_id,
                memory_b_id=existing_memory.id,
                contradiction_type=CONTRADICTION_TYPE_TEMPORAL_SUPERSESSION,
                confidence=relevance,
                detection_method=DETECTION_METHOD,
                newer_memory_id=self._determine_newer_memory(new_memory, existing_memory),
            )
            contradictions.append(await self._storage.create_contradiction(record))
            self.logger.info(
                "Contradiction detected between %s and %s (relevance=%.2f, method=%s)",
                memory_id,
                existing_memory.id,
                relevance,
                DETECTION_METHOD,
            )
        return contradictions

    async def _classify(self, new_content: str, candidate_contents: list[str]) -> list[int]:
        """Return indices of candidates the new fact contradicts. Empty on any failure.

        Failure must degrade to "detected nothing" rather than raising: contradiction
        detection runs inside the post-store pipeline and must never block ingest.
        """
        listing = "\n".join(f"{i}: {text}" for i, text in enumerate(candidate_contents))
        request = LLMRequest(
            messages=[
                LLMMessage(role=LLMRole.SYSTEM, content=SYSTEM_PROMPT),
                LLMMessage(role=LLMRole.USER, content=f"NEW FACT:\n{new_content}\n\nEXISTING FACTS:\n{listing}"),
            ],
            max_tokens=self.max_tokens,
            temperature_factor=0.0,
        )

        try:
            response = await self.llm_service.complete(
                request,
                profile=LLM_PROFILE_CONTRADICTION,
                activity=GenerationActivity.CONTRADICTION_DETECTION,
            )
        except LLMNotConfiguredError as e:
            # Expected on a server with no LLM; the registry announces it once at startup.
            self.logger.debug("Skipping contradiction classification: %s", e)
            return []
        except Exception as e:
            self.logger.warning("Contradiction classification failed: %s", e)
            return []

        # A truncated reply contains no JSON and would parse as "no contradictions" — a
        # failure indistinguishable from a clean negative. Reasoning models emit a long
        # chain before the JSON, so this is the expected shape of a too-small budget.
        if response.finish_reason == "length":
            self.logger.warning(
                "Contradiction classification truncated at %d tokens; treating as no result. Raise %s if this recurs.",
                self.max_tokens,
                MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
            )
            return []

        return _parse_contradicted(response.content)


def _parse_contradicted(content: str) -> list[int]:
    """Extract ``contradicted_facts`` indices, tolerating fenced or prose-prefixed JSON."""
    import json

    text = (content or "").strip()
    if "```" in text:
        fenced = [part for part in text.split("```") if "{" in part]
        if fenced:
            text = fenced[0]
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    # `duplicate_facts` is returned by the same call but deliberately not acted on here:
    # writing duplicate edges is the association service's job, and adding a second writer
    # is the exact duplication DESIGN_contradiction_association_unification.md exists to
    # remove. It stays in the response so a future unified path can consume it.
    out: list[int] = []
    for value in payload.get("contradicted_facts") or []:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


class LLMContradictionServicePlugin(ContradictionServicePluginBase):
    """Plugin that creates the fused-LLM contradiction service."""

    PROVIDER_NAME = "llm"

    def initialize(self, v: Variables, logger: Logger) -> ContradictionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        return LLMContradictionService(
            storage=storage,
            llm_service=llm_service,
            v=v,
            candidate_limit=v.environ(
                MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT,
                default=DEFAULT_MEMORYLAYER_CONTRADICTION_CANDIDATE_LIMIT,
                type_fn=int,
            ),
            min_relevance=v.environ(
                MEMORYLAYER_CONTRADICTION_MIN_RELEVANCE,
                default=DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MIN_RELEVANCE,
                type_fn=float,
            ),
            max_tokens=v.environ(
                MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
                default=DEFAULT_MEMORYLAYER_CONTRADICTION_LLM_MAX_TOKENS,
                type_fn=int,
            ),
        )
