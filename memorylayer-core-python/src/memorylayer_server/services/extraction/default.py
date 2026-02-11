"""
Default Extraction Service implementation.

Extracts memories from session content using LLM-based classification.
Uses the OpenViking-inspired 6-category taxonomy:
- PROFILE: User identity, background
- PREFERENCES: Choices, settings
- ENTITIES: Projects, people, concepts
- EVENTS: Decisions, milestones
- CASES: Problems with solutions
- PATTERNS: Reusable processes
"""
import json
import re
import time
from datetime import datetime, timezone
from typing import Optional

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ...models.memory import Memory, MemoryType, MemorySubtype
from ...models.llm import LLMMessage, LLMRequest, LLMRole
from ...utils import compute_content_hash, generate_id
from ..storage import EXT_STORAGE_BACKEND, StorageBackend
from ..llm import EXT_LLM_SERVICE, LLMService
from ..embedding import EXT_EMBEDDING_SERVICE, EmbeddingService
from ..deduplication import EXT_DEDUPLICATION_SERVICE, DeduplicationService, DeduplicationAction
from ...config import DEFAULT_TENANT_ID
from .base import (
    ExtractionService,
    ExtractionServicePluginBase,
    ExtractionCategory,
    ExtractionOptions,
    ExtractedMemory,
    ExtractionResult,
    CATEGORY_MAPPING,
)

# System prompt for LLM extraction
EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Your task is to analyze conversation/session content and extract distinct, reusable memories that would be valuable for future reference.

## Memory Categories

Extract memories into these 6 categories:

1. **profile** - User identity and background information
   - Name, role, expertise, location, organization
   - Example: "User is a senior Python developer at TechCorp"

2. **preferences** - User choices, settings, and preferences
   - Coding style, tools, frameworks, communication preferences
   - Example: "User prefers pytest over unittest for testing"

3. **entities** - Important projects, people, concepts, or things
   - Project names, team members, technologies, systems
   - Example: "Project Aurora is a microservices migration initiative"

4. **events** - Significant decisions, milestones, or occurrences
   - Completed tasks, decisions made, deadlines, meetings
   - Example: "Decided to use PostgreSQL instead of MongoDB on 2026-01-15"

5. **cases** - Problems encountered with their solutions
   - Bug fixes, troubleshooting steps, workarounds
   - Example: "Fixed authentication timeout by increasing JWT expiry to 24h"

6. **patterns** - Reusable processes, workflows, or procedures
   - How-to knowledge, best practices, standard procedures
   - Example: "To deploy to production: run tests, build Docker image, push to registry, update k8s"

## Output Format

Return a JSON array of extracted memories. Each memory should have:
- "content": The memory content (clear, standalone, reusable)
- "category": One of: profile, preferences, entities, events, cases, patterns
- "importance": Float from 0.0 to 1.0 (how valuable for future reference)
- "tags": Array of relevant keywords

## Guidelines

1. Extract DISTINCT memories - don't repeat similar information
2. Make each memory STANDALONE - it should make sense without context
3. Be CONCISE but COMPLETE - include necessary details
4. Focus on REUSABLE information - things valuable for future sessions
5. Assign appropriate IMPORTANCE:
   - 0.9-1.0: Critical user identity, major decisions, key solutions
   - 0.7-0.8: Important preferences, significant events, useful patterns
   - 0.5-0.6: General entities, minor preferences, context
   - Below 0.5: Transient or low-value information (skip these)
6. If no valuable memories exist, return an empty array []

Return ONLY the JSON array, no additional text."""

# User prompt template
EXTRACTION_USER_PROMPT = """Analyze this session content and extract valuable memories:

---
{context}
---

Extract memories as a JSON array. Focus on information that would be useful in future sessions."""


class DefaultExtractionService(ExtractionService):
    """Default extraction service implementation."""

    def __init__(
            self,
            llm_service: Optional[LLMService] = None,
            storage: Optional[StorageBackend] = None,
            deduplication_service=None,
            embedding_service: Optional[EmbeddingService] = None,
            v: Variables = None
    ):
        """
        Initialize extraction service.

        Args:
            llm_service: Optional LLM service for extraction
            storage: Optional storage backend
            deduplication_service: Optional deduplication service
            embedding_service: Optional embedding service for deduplication
            v: Variables for logging context
        """
        self.llm_service = llm_service
        self.storage = storage
        self.deduplication_service = deduplication_service
        self.embedding_service = embedding_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized DefaultExtractionService")

    async def extract_from_session(
            self,
            session_id: str,
            workspace_id: str,
            context_id: str,
            session_content: str,
            working_memory: dict,
            options: ExtractionOptions
    ) -> ExtractionResult:
        """
        Extract memories from a session.

        Args:
            session_id: The session ID
            workspace_id: The workspace ID
            context_id: The context ID
            session_content: Combined session content to analyze
            working_memory: Current working memory key-value pairs
            options: Extraction options

        Returns:
            ExtractionResult with extracted memories
        """
        start_time = time.time()

        # Determine categories to extract
        categories = options.categories or list(ExtractionCategory)

        # Build session context for extraction
        context = self._build_extraction_context(session_content, working_memory)

        # Extract memories using LLM (if available)
        if self.llm_service:
            extracted = await self._llm_extraction(context, categories)
        else:
            # Fallback: simple extraction without LLM
            extracted = self._simple_extraction(context, categories, options)

        # Filter by importance
        extracted = [m for m in extracted if m.importance >= options.min_importance]

        # Limit count
        extracted = extracted[:options.max_memories]

        # Deduplicate if enabled
        memories_deduplicated = 0
        if options.deduplicate and self.deduplication_service:
            extracted, memories_deduplicated = await self._deduplicate(
                extracted, workspace_id
            )

        # Convert to Memory objects
        memories_created = []
        for em in extracted:
            memory_type, memory_subtype = CATEGORY_MAPPING.get(
                em.category,
                (MemoryType.SEMANTIC, None)
            )

            memory = Memory(
                id=generate_id("mem"),
                tenant_id=DEFAULT_TENANT_ID,
                workspace_id=workspace_id,
                context_id=context_id,
                session_id=session_id,
                content=em.content,
                content_hash=compute_content_hash(em.content),
                type=memory_type,
                subtype=memory_subtype,
                category=em.category.value,
                importance=em.importance,
                tags=em.tags,
                metadata=em.metadata,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            memories_created.append(memory)

        # Calculate breakdown
        breakdown = {}
        for m in memories_created:
            cat = m.category or "unknown"
            breakdown[cat] = breakdown.get(cat, 0) + 1

        elapsed_ms = int((time.time() - start_time) * 1000)

        self.logger.info(
            "Extracted %s memories from session %s in %s ms",
            len(memories_created), session_id, elapsed_ms
        )

        return ExtractionResult(
            session_id=session_id,
            memories_extracted=len(extracted) + memories_deduplicated,
            memories_deduplicated=memories_deduplicated,
            memories_created=memories_created,
            breakdown=breakdown,
            extraction_time_ms=elapsed_ms
        )

    async def decompose_to_facts(self, content: str) -> list[dict]:
        """Decompose composite content into atomic facts using LLM.

        Falls back to returning the original content as a single fact
        if no LLM is available or on failure.

        Args:
            content: Composite memory content to decompose

        Returns:
            List of dicts with keys: 'content', 'type' (optional), 'subtype' (optional)
        """
        if not self.llm_service:
            self.logger.debug("No LLM provider available, returning content as single fact")
            return [{"content": content}]

        system_prompt = (
            "You are a fact decomposition assistant. Break the following composite text "
            "into individual atomic facts. Each fact should be a single, standalone piece "
            "of information that makes sense on its own.\n\n"
            "## Output Format\n\n"
            "Return a JSON array of objects. Each object must have:\n"
            '- "content": The atomic fact (clear, standalone, concise)\n'
            '- "type": (optional) One of: episodic, semantic, procedural, working\n'
            '- "subtype": (optional) One of: solution, problem, code_pattern, fix, error, '
            "workflow, preference, decision, profile, entity, event, directive\n\n"
            "## Guidelines\n\n"
            "1. Each fact should express exactly ONE piece of information\n"
            "2. Facts must be STANDALONE - understandable without the original context\n"
            "3. Preserve important details (names, numbers, specifics)\n"
            "4. Do NOT add information not present in the original\n"
            "5. If the content is already a single atomic fact, return it as-is in the array\n\n"
            "Return ONLY the JSON array, no additional text."
        )

        user_prompt = (
            "Decompose this content into atomic facts:\n\n"
            f"---\n{content}\n---"
        )

        try:
            messages = [
                LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
                LLMMessage(role=LLMRole.USER, content=user_prompt),
            ]

            request = LLMRequest(
                messages=messages,
                max_tokens=6000,
                temperature_factor=0.3,
            )

            response = await self.llm_service.complete(request, profile="extraction")
            raw = response.content.strip()

            # Strip markdown code block wrapper if present
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)

            # Extract JSON array from response (handle surrounding text)
            array_start = raw.find('[')
            if array_start > 0:
                raw = raw[array_start:]

            try:
                facts = json.loads(raw)
            except json.JSONDecodeError:
                facts = self._parse_partial_json_array(raw)
            if not isinstance(facts, list):
                self.logger.warning("Decompose response is not a JSON array, returning single fact")
                return [{"content": content}]

            # Validate each fact has at least a 'content' key
            validated = []
            for item in facts:
                if isinstance(item, dict) and "content" in item and item["content"].strip():
                    validated.append({
                        "content": item["content"].strip(),
                        "type": item.get("type"),
                        "subtype": item.get("subtype"),
                    })

            if not validated:
                self.logger.warning("No valid facts extracted, returning single fact")
                return [{"content": content}]

            self.logger.info("Decomposed content into %d atomic facts", len(validated))
            return validated

        except Exception as e:
            self.logger.warning("Fact decomposition failed: %s, returning single fact", e)
            return [{"content": content}]

    def _parse_partial_json_array(self, raw: str) -> list:
        """Recover facts from a truncated or malformed JSON array.

        Handles common LLM JSON issues: trailing commas, unterminated
        strings, and output truncated mid-object.

        Args:
            raw: Raw JSON string that failed standard parsing.

        Returns:
            Parsed list of fact dicts.

        Raises:
            json.JSONDecodeError: If the JSON cannot be recovered.
        """
        # Remove trailing commas before } or ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Truncate at the last complete JSON object and close the array
        last_brace = cleaned.rfind('}')
        if last_brace >= 0:
            candidate = cleaned[:last_brace + 1] + ']'
            first_bracket = candidate.find('[')
            if first_bracket >= 0:
                candidate = candidate[first_bracket:]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list) and result:
                        self.logger.info(
                            "Recovered %d fact(s) from truncated JSON response",
                            len(result),
                        )
                        return result
                except json.JSONDecodeError:
                    pass

        raise json.JSONDecodeError(
            "Could not recover facts from malformed JSON", raw, 0
        )

    async def classify_content(self, content: str) -> tuple[MemoryType, 'Optional[MemorySubtype]']:
        """Classify a single memory's content into a type and subtype.

        Uses LLM to determine the extraction category, then maps through
        CATEGORY_MAPPING to get (MemoryType, MemorySubtype).

        Returns (MemoryType.SEMANTIC, None) as fallback.
        """
        if not self.llm_service:
            return (MemoryType.SEMANTIC, None)

        prompt = (
            "Classify this memory into exactly one category.\n"
            "Categories: profile, preferences, entities, events, cases, patterns\n\n"
            "- profile: user identity, background, role\n"
            "- preferences: user choices, settings, likes/dislikes\n"
            "- entities: projects, people, concepts, tools, technologies\n"
            "- events: decisions, milestones, occurrences, incidents\n"
            "- cases: problems encountered with solutions or fixes\n"
            "- patterns: reusable processes, workflows, procedures, how-to\n\n"
            f'Memory: "{content}"\n\n'
            "Reply with just the category name."
        )

        try:
            messages = [
                LLMMessage(role=LLMRole.USER, content=prompt),
            ]
            request = LLMRequest(
                messages=messages,
                max_tokens=20,
                temperature=0.0,
            )
            response = await self.llm_service.complete(request, profile="extraction")
            category_str = response.content.strip().lower()

            try:
                category = ExtractionCategory(category_str)
            except ValueError:
                self.logger.debug("Unrecognized classification category: %s", category_str)
                return (MemoryType.SEMANTIC, None)

            return CATEGORY_MAPPING.get(category, (MemoryType.SEMANTIC, None))

        except Exception as e:
            self.logger.warning("Content classification failed: %s", e)
            return (MemoryType.SEMANTIC, None)

    async def _llm_extraction(
            self,
            context: str,
            categories: list[ExtractionCategory]
    ) -> list[ExtractedMemory]:
        """
        Extract memories using LLM-based classification.

        Uses the OpenViking-inspired 6-category taxonomy to classify
        session content into distinct, reusable memories.

        Args:
            context: Combined session content to analyze
            categories: Categories to extract (for filtering)

        Returns:
            List of extracted memories
        """
        try:
            # Build the LLM request
            messages = [
                LLMMessage(role=LLMRole.SYSTEM, content=EXTRACTION_SYSTEM_PROMPT),
                LLMMessage(role=LLMRole.USER, content=EXTRACTION_USER_PROMPT.format(context=context)),
            ]

            request = LLMRequest(
                messages=messages,
                max_tokens=4000,  # Allow room for multiple memories
                temperature_factor=0.4,  # Lower temperature for more consistent extraction
            )

            self.logger.debug("Sending extraction request to LLM")
            response = await self.llm_service.complete(request, profile="extraction")

            # Parse the JSON response
            extracted = self._parse_llm_response(response.content, categories)

            self.logger.info(
                "LLM extraction completed: %d memories extracted (tokens: %d prompt, %d completion)",
                len(extracted), response.prompt_tokens, (response.completion_tokens or -1),
            )

            return extracted

        except Exception as e:
            self.logger.warning(
                "LLM extraction failed, falling back to simple extraction: %s", str(e)
            )
            return self._simple_extraction(context, categories, ExtractionOptions())

    def _parse_llm_response(
            self,
            response_content: str,
            categories: list[ExtractionCategory]
    ) -> list[ExtractedMemory]:
        """
        Parse LLM response into ExtractedMemory objects.

        Args:
            response_content: Raw LLM response (expected JSON array)
            categories: Categories to filter by

        Returns:
            List of extracted memories
        """
        # Clean up the response - handle markdown code blocks
        content = response_content.strip()
        if content.startswith("```"):
            # Remove markdown code block wrapper
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            memories_data = json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.warning("Failed to parse LLM response as JSON: %s", str(e))
            return []

        if not isinstance(memories_data, list):
            self.logger.warning("LLM response is not a JSON array")
            return []

        # Convert category filter to set of values for efficient lookup
        allowed_categories = {c.value for c in categories}

        extracted = []
        for item in memories_data:
            try:
                # Validate required fields
                if not isinstance(item, dict):
                    continue
                if "content" not in item or "category" not in item:
                    continue

                category_str = item.get("category", "").lower()

                # Skip if category not in allowed list
                if category_str not in allowed_categories:
                    continue

                # Map category string to enum
                try:
                    category = ExtractionCategory(category_str)
                except ValueError:
                    self.logger.debug("Unknown category: %s", category_str)
                    continue

                # Extract importance (default to 0.6 if not provided)
                importance = float(item.get("importance", 0.6))  # TODO: configurable default value
                importance = max(0.0, min(1.0, importance))  # Clamp to [0, 1]

                # Extract tags (default to empty list)
                tags = item.get("tags", [])
                if not isinstance(tags, list):
                    tags = []
                tags = [str(t) for t in tags]  # Ensure strings

                extracted.append(ExtractedMemory(
                    content=str(item["content"]),
                    category=category,
                    importance=importance,
                    tags=tags,
                    metadata={"extraction_method": "llm"}
                ))

            except (KeyError, ValueError, TypeError) as e:
                self.logger.debug("Skipping invalid memory item: %s", str(e))
                continue

        return extracted

    def _build_extraction_context(
            self,
            session_content: str,
            working_memory: dict
    ) -> str:
        """Build context string for extraction."""
        parts = [session_content]

        if working_memory:
            wm_str = "\n".join(f"- {k}: {v}" for k, v in working_memory.items())
            parts.append(f"\nWorking Memory:\n{wm_str}")

        return "\n".join(parts)

    def _simple_extraction(
            self,
            context: str,
            categories: list[ExtractionCategory],
            options: ExtractionOptions
    ) -> list[ExtractedMemory]:
        """Simple extraction without LLM - returns the full context as one memory."""
        # Without LLM, we can't do sophisticated extraction
        # Just create a single memory from the context
        if not context.strip():
            return []

        return [
            ExtractedMemory(
                content=context[:1000],  # Limit length
                category=ExtractionCategory.CASES,
                importance=0.6,  # TODO: configurable default (same as value at LLM extraction)
                tags=["auto-extracted"],
                metadata={"extraction_method": "simple"}
            )
        ]

    async def _deduplicate(
            self,
            extracted: list[ExtractedMemory],
            workspace_id: str
    ) -> tuple[list[ExtractedMemory], int]:
        """
        Deduplicate extracted memories against existing memories.

        Uses content hash for exact matches and embedding similarity
        for semantic deduplication.

        Args:
            extracted: List of extracted memories to deduplicate
            workspace_id: Workspace to check against

        Returns:
            Tuple of (deduplicated memories, count of duplicates removed)
        """
        if not extracted:
            return extracted, 0

        try:
            # Generate embeddings for all extracted memories
            contents = [em.content for em in extracted]
            embeddings = await self.embedding_service.embed_batch(contents)

            # Build candidates for deduplication
            # Format: list of (content, content_hash, embedding)
            candidates = []
            for em, embedding in zip(extracted, embeddings):
                content_hash = compute_content_hash(em.content)
                candidates.append((em.content, content_hash, embedding))

            # Run batch deduplication
            results = await self.deduplication_service.deduplicate_batch(
                candidates, workspace_id
            )

            # Filter extracted memories based on deduplication results
            deduplicated = []
            duplicates_count = 0

            for em, result in zip(extracted, results):
                if result.action == DeduplicationAction.SKIP:
                    # Exact duplicate - skip entirely
                    duplicates_count += 1
                    self.logger.debug(
                        "Skipping duplicate memory (exact match): %s",
                        em.content[:50]
                    )
                elif result.action == DeduplicationAction.UPDATE:
                    # Semantic duplicate - could update existing, but for extraction
                    # we'll skip to avoid redundancy (existing memory is sufficient)
                    duplicates_count += 1
                    self.logger.debug(
                        "Skipping duplicate memory (semantic match %.3f): %s",
                        result.similarity_score or 0, em.content[:50]
                    )
                elif result.action == DeduplicationAction.MERGE:
                    # Merge candidate - include but flag for potential merge
                    em.metadata["merge_candidate"] = True
                    em.metadata["similar_memory_id"] = result.existing_memory_id
                    em.metadata["similarity_score"] = result.similarity_score
                    deduplicated.append(em)
                else:
                    # CREATE - new unique memory
                    deduplicated.append(em)

            self.logger.info(
                "Deduplication complete: %d memories in, %d out, %d duplicates removed",
                len(extracted), len(deduplicated), duplicates_count
            )

            return deduplicated, duplicates_count

        except Exception as e:
            self.logger.warning(
                "Deduplication failed, returning all extracted memories: %s", str(e)
            )
            return extracted, 0


class DefaultExtractionServicePlugin(ExtractionServicePluginBase):
    """Default extraction service plugin."""
    PROVIDER_NAME = 'default'

    def initialize(self, v: Variables, logger) -> ExtractionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        llm_service: LLMService = self.get_extension(EXT_LLM_SERVICE, v)
        deduplication_service: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        embedding_service: EmbeddingService = self.get_extension(EXT_EMBEDDING_SERVICE, v)

        return DefaultExtractionService(
            llm_service=llm_service,
            storage=storage,
            deduplication_service=deduplication_service,
            embedding_service=embedding_service,
            v=v
        )
