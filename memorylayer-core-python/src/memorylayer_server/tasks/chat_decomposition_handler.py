"""
Chat Decomposition Task Handler.

Background task that extracts memories from unprocessed chat messages in a thread.

When triggered:
1. Loads thread metadata (watermark, entity attribution)
2. Fetches unprocessed messages (after last_decomposed_index)
3. Batches messages into chunks (with overlap for context continuity)
4. For each chunk, uses LLM to extract structured memories
5. Routes each memory through MemoryService.remember() for full pipeline
6. Updates thread decomposition watermark
"""
import json
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from scitrera_app_framework import get_logger, Variables

from ..models.memory import RememberInput
from ..services.storage import StorageBackend, EXT_STORAGE_BACKEND
from ..services.tasks import TaskHandlerPlugin, TaskSchedule
from ..services.memory import MemoryService, EXT_MEMORY_SERVICE
from ..services.llm import EXT_LLM_SERVICE
from ..services._constants import EXT_CHAT_SERVICE
from ..config import (
    MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE,
    DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE,
    MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP,
    DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP,
)

CHAT_DECOMPOSITION_TASK = "chat_decomposition"

DECOMPOSITION_SYSTEM_PROMPT = """You are a memory extraction specialist. Given a conversation excerpt, extract distinct memories that would be valuable to remember long-term.

For each memory, provide:
- content: A clear, self-contained statement of the fact/preference/decision/event
- type: One of: episodic, semantic, procedural
- subtype: One of: solution, problem, code_pattern, fix, error, workflow, preference, decision, profile, entity, event, directive
- importance: Float 0.0-1.0 (how important is this to remember?)
- tags: List of relevant tags

Focus on extracting:
- User preferences and habits
- Decisions made and their reasoning
- Facts about entities (people, projects, tools)
- Problems encountered and solutions found
- Workflows and procedures described
- Directives or instructions given

Do NOT extract:
- Trivial pleasantries or greetings
- Redundant information already covered by another extracted memory
- Speculative or hypothetical statements (unless they reveal preferences)

Return a JSON array of memory objects. If no meaningful memories can be extracted, return an empty array []."""

DECOMPOSITION_USER_TEMPLATE = """Extract memories from this conversation excerpt:

{conversation}

Return a JSON array of memory objects with fields: content, type, subtype, importance, tags"""


class ChatDecompositionTaskHandler(TaskHandlerPlugin):
    """On-demand chat decomposition task handler."""

    def get_task_type(self) -> str:
        return CHAT_DECOMPOSITION_TASK

    def get_schedule(self, v: Variables) -> Optional[TaskSchedule]:
        return None  # On-demand only

    async def handle(self, v: Variables, payload: dict) -> None:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get("workspace_id")
        thread_id = payload.get("thread_id")

        if not workspace_id or not thread_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, thread_id=%s",
                workspace_id, thread_id,
            )
            return

        # 1. Load thread
        thread = await storage.get_thread(workspace_id, thread_id)
        if not thread:
            logger.warning("Thread %s not found in workspace %s", thread_id, workspace_id)
            return

        if thread.unprocessed_count == 0:
            logger.debug("Thread %s has no unprocessed messages, skipping", thread_id)
            return

        # 2. Fetch unprocessed messages
        messages = await storage.get_messages(
            workspace_id=workspace_id,
            thread_id=thread_id,
            after_index=thread.last_decomposed_index - 1 if thread.last_decomposed_index > 0 else None,
            limit=10000,  # Get all unprocessed
            order="asc",
        )

        if not messages:
            logger.debug("No messages to decompose for thread %s", thread_id)
            return

        logger.info(
            "Decomposing %d messages from thread %s (index %d to %d)",
            len(messages), thread_id,
            thread.last_decomposed_index, thread.message_count,
        )

        # 3. Chunk messages
        chunk_size = v.get(
            MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE,
            default=DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_CHUNK_SIZE,
        )
        overlap = v.get(
            MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP,
            default=DEFAULT_MEMORYLAYER_CHAT_DECOMPOSE_OVERLAP,
        )

        chunks = self._chunk_messages(messages, chunk_size, overlap)
        total_memories_created = 0

        # 4. Process each chunk
        for chunk in chunks:
            try:
                memories_created = await self._decompose_chunk(
                    v=v,
                    logger=logger,
                    storage=storage,
                    memory_service=memory_service,
                    workspace_id=workspace_id,
                    thread=thread,
                    messages=chunk,
                )
                total_memories_created += memories_created
            except Exception as e:
                logger.error(
                    "Failed to decompose chunk for thread %s: %s",
                    thread_id, e, exc_info=True,
                )

        # 5. Update watermark
        max_index = max(m.message_index for m in messages) + 1
        now = datetime.now(timezone.utc)
        await storage.update_thread(
            workspace_id,
            thread_id,
            last_decomposed_index=max_index,
            last_decomposed_at=now,
        )

        logger.info(
            "Decomposed thread %s: %d memories created from %d messages",
            thread_id, total_memories_created, len(messages),
        )

    def _chunk_messages(self, messages: list, chunk_size: int, overlap: int) -> list[list]:
        if len(messages) <= chunk_size:
            return [messages]

        chunks = []
        step = max(1, chunk_size - overlap)
        for i in range(0, len(messages), step):
            chunk = messages[i:i + chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(messages):
                break
        return chunks

    async def _decompose_chunk(
            self,
            v: Variables,
            logger: Logger,
            storage: StorageBackend,
            memory_service: MemoryService,
            workspace_id: str,
            thread,
            messages: list,
    ) -> int:
        """Decompose a chunk of messages into memories via LLM."""
        # Format conversation for LLM
        conversation_lines = []
        for msg in messages:
            content = msg.content
            if not isinstance(content, str):
                # Structured content — extract text parts
                parts = []
                for block in content:
                    if hasattr(block, "text") and block.text:
                        parts.append(block.text)
                    elif hasattr(block, "type"):
                        parts.append(f"[{block.type}]")
                content = " ".join(parts) if parts else "[structured content]"
            conversation_lines.append(f"{msg.role}: {content}")

        conversation_text = "\n".join(conversation_lines)

        # Try LLM extraction
        try:
            llm_service = self.get_extension(EXT_LLM_SERVICE, v)
        except Exception:
            logger.warning("LLM service not available, skipping decomposition")
            return 0

        prompt = DECOMPOSITION_USER_TEMPLATE.format(conversation=conversation_text)

        try:
            response = await llm_service.generate(
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        except Exception as e:
            logger.error("LLM generation failed during decomposition: %s", e)
            return 0

        # Parse extracted memories
        try:
            extracted = self._parse_llm_response(response, logger)
        except Exception as e:
            logger.warning("Failed to parse LLM decomposition response: %s", e)
            return 0

        if not extracted:
            return 0

        # Route each memory through the remember pipeline
        memories_created = 0
        msg_range_start = messages[0].message_index
        msg_range_end = messages[-1].message_index

        for mem_data in extracted:
            try:
                remember_input = RememberInput(
                    content=mem_data.get("content", ""),
                    type=mem_data.get("type"),
                    subtype=mem_data.get("subtype"),
                    importance=mem_data.get("importance", 0.5),
                    tags=mem_data.get("tags", []),
                    metadata={
                        "source": "chat_history",
                        "thread_id": thread.id,
                        "message_range": [msg_range_start, msg_range_end],
                    },
                    context_id=thread.context_id,
                    observer_id=thread.observer_id,
                    subject_id=thread.subject_id,
                )

                await memory_service.remember(workspace_id, remember_input)
                memories_created += 1

            except Exception as e:
                logger.warning(
                    "Failed to store decomposed memory from thread %s: %s",
                    thread.id, e,
                )

        return memories_created

    def _parse_llm_response(self, response: str, logger: Logger) -> list[dict]:
        """Parse LLM response into a list of memory dicts."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [m for m in result if isinstance(m, dict) and m.get("content")]
            return []
        except json.JSONDecodeError:
            logger.debug("LLM response was not valid JSON: %s", text[:200])
            return []
