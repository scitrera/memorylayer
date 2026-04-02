"""Session extraction task handler for on-demand token-budget-triggered extraction."""

import json
from logging import Logger

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from ..models import MemoryType, RememberInput
from ..services._constants import EXT_MEMORY_SERVICE, EXT_SESSION_SERVICE
from ..services.tasks import TaskHandlerPlugin, TaskSchedule


class SessionExtractionTaskHandler(TaskHandlerPlugin):
    """
    On-demand session extraction task handler.

    Triggered when cumulative working memory token usage exceeds configured
    thresholds in touch_session(). Extracts working memory content and
    persists it to long-term memory via the memory service.

    No recurring schedule -- runs only when explicitly submitted via the
    task service.
    """

    def get_task_type(self) -> str:
        return "session_extraction"

    def get_schedule(self, v: Variables) -> TaskSchedule | None:
        # No recurring schedule - triggered on-demand by token budget logic
        return None

    async def handle(self, v: Variables, payload: dict) -> None:
        logger: Logger = get_logger(v, name=self.get_task_type())

        workspace_id = payload.get("workspace_id")
        session_id = payload.get("session_id")
        context_id = payload.get("context_id")

        if not workspace_id or not session_id:
            logger.warning(
                "Missing required payload fields: workspace_id=%s, session_id=%s",
                workspace_id,
                session_id,
            )
            return

        from ..config import (
            DEFAULT_MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL,
            MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL,
        )
        from ..services.memory import MemoryService
        from ..services.session import SessionService

        session_service: SessionService = self.get_extension(EXT_SESSION_SERVICE, v)
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)

        token_budget: int = v.environ(
            MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL,
            default=DEFAULT_MEMORYLAYER_SESSION_TOKEN_BUDGET_TOTAL,
            type_fn=int,
        )

        # Fetch all working memory entries for the session
        try:
            working_memory_entries = await session_service.get_all_working_memory(workspace_id, session_id)
        except Exception as e:
            logger.warning(
                "Failed to fetch working memory for session %s in workspace %s: %s",
                session_id,
                workspace_id,
                e,
            )
            return

        if not working_memory_entries:
            logger.debug("No working memory entries found for session %s", session_id)
            return

        logger.info(
            "Extracting %d working memory entries for session %s (token budget: %d)",
            len(working_memory_entries),
            session_id,
            token_budget,
        )

        # Extract entries respecting the token budget
        tokens_used = 0
        extracted_count = 0

        for wm in working_memory_entries:
            content_str = wm.value if isinstance(wm.value, str) else json.dumps(wm.value, default=str)
            entry_tokens = len(content_str) // 4

            if tokens_used + entry_tokens > token_budget:
                logger.debug(
                    "Token budget (%d) reached after %d entries, stopping extraction",
                    token_budget,
                    extracted_count,
                )
                break

            remember_input = RememberInput(
                content=content_str,
                type=MemoryType.WORKING,
                importance=0.5,
                metadata={
                    "session_id": session_id,
                    "working_memory_key": wm.key,
                    "extraction_trigger": "token_budget",
                },
                context_id=context_id,
            )

            try:
                memory = await memory_service.remember(
                    workspace_id=workspace_id,
                    input=remember_input,
                )
                tokens_used += entry_tokens
                extracted_count += 1
                logger.debug(
                    "Extracted working memory key '%s' as memory %s",
                    wm.key,
                    memory.id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to extract working memory key '%s' for session %s: %s",
                    wm.key,
                    session_id,
                    e,
                )

        logger.info(
            "Session extraction complete for %s: %d/%d entries extracted (%d tokens)",
            session_id,
            extracted_count,
            len(working_memory_entries),
            tokens_used,
        )
