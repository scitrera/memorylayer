"""Persistent session service using storage backend."""
from datetime import datetime, timezone, timedelta
from logging import Logger
from typing import Optional, Any, List, TYPE_CHECKING

from scitrera_app_framework import get_logger
from scitrera_app_framework.api import Variables

from .base import SessionServicePluginBase, SessionService, CommitResult, CommitOptions
from ..storage import StorageBackend, EXT_STORAGE_BACKEND
from ..extraction import ExtractionService, EXT_EXTRACTION_SERVICE, ExtractionOptions
from ..deduplication import DeduplicationService, EXT_DEDUPLICATION_SERVICE
from ..memory import MemoryService, EXT_MEMORY_SERVICE
from ..contradiction import ContradictionService, EXT_CONTRADICTION_SERVICE
from ...models import Session, WorkingMemory
from ...models.session import SessionBriefing


class PersistentSessionService(SessionService):
    """Storage-backed session service.

    Sessions persist across server restarts using the configured
    StorageBackend (SQLite, PostgreSQL, etc.).

    This is the recommended session service for production deployments
    where session data should survive server restarts.
    """

    def __init__(
            self,
            storage: StorageBackend,
            v: Variables = None,
            extraction_service: Optional[ExtractionService] = None,
            deduplication_service: Optional[DeduplicationService] = None,
            memory_service: Optional[MemoryService] = None,
            contradiction_service: Optional[ContradictionService] = None,
    ):
        self.storage = storage
        self.extraction_service = extraction_service
        self.deduplication_service = deduplication_service
        self._memory_service = memory_service
        self.contradiction_service = contradiction_service
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.logger.info("Initialized PersistentSessionService with storage backend")

    async def create_session(
            self,
            workspace_id: str,
            session: Session,
            context_id: Optional[str] = None
    ) -> Session:
        """Store a new session in storage backend."""
        return await self.storage.create_session(workspace_id, session)

    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """Retrieve session from storage if not expired."""
        return await self.storage.get_session(workspace_id, session_id)

    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID without workspace filter."""
        return await self.storage.get_session_by_id(session_id)

    async def delete_session(self, workspace_id: str, session_id: str, skip_auto_commit: bool = False) -> bool:
        """Delete session and all context from storage.

        If the session has auto_commit=True, working memory is committed
        to long-term storage before deletion.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to delete
            skip_auto_commit: If True, skip auto-commit even if session has auto_commit=True

        Returns:
            True if session was deleted, False if not found
        """
        # Get session to check auto_commit flag
        session = await self.storage.get_session(workspace_id, session_id)

        if session is None:
            return False

        # Auto-commit if enabled and not already committed
        if not skip_auto_commit and session.auto_commit and session.committed_at is None:
            try:
                self.logger.info(
                    "Auto-committing session %s before deletion (auto_commit=True)",
                    session_id
                )
                await self.commit_session(workspace_id, session_id)
            except Exception as e:
                self.logger.warning(
                    "Auto-commit failed for session %s, proceeding with deletion: %s",
                    session_id,
                    e
                )

        return await self.storage.delete_session(workspace_id, session_id)

    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """Set working memory in storage backend."""
        # Verify session exists
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        return await self.storage.set_working_memory(
            workspace_id, session_id, key, value, ttl_seconds
        )

    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional[WorkingMemory]:
        """Get working memory from storage backend."""
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return None

        return await self.storage.get_working_memory(workspace_id, session_id, key)

    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> List[WorkingMemory]:
        """Get all working memory from storage backend."""
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            return []

        return await self.storage.get_all_working_memory(workspace_id, session_id)

    async def cleanup_expired(self, workspace_id: str) -> int:
        """Cleanup expired sessions. Should be called periodically."""
        count = await self.storage.cleanup_expired_sessions(workspace_id)
        if count > 0:
            self.logger.info("Cleaned up %d expired sessions in workspace %s", count, workspace_id)
        return count

    async def get_briefing(
            self,
            workspace_id: str,
            lookback_minutes: int = 60,
            detail_level: str = "abstract",
            limit: int = 10,
            include_memories: bool = True,
            include_contradictions: bool = True,
    ) -> SessionBriefing:
        """
        Generate a session briefing with workspace summary and recent activity.

        Uses storage backend to provide comprehensive statistics about memories,
        associations, and recent activity. Advanced features like contradiction
        detection and LLM-enhanced summaries require custom implementations.

        Args:
            workspace_id: Workspace identifier
            lookback_minutes: Time window for recent memories (default 60)
            detail_level: Memory detail level - abstract, overview, or full
            limit: Maximum memories to include
            include_memories: Whether to include memory content
            include_contradictions: Whether to detect contradictions

        Returns:
            SessionBriefing with workspace summary and activity
        """
        # Get workspace statistics from storage
        stats = await self.storage.get_workspace_stats(workspace_id)

        # Build workspace summary with real data from storage
        workspace_summary = {
            "total_memories": stats.get("total_memories", 0),
            "recent_memories": 0,  # Will calculate below
            "active_topics": [],
            "total_categories": stats.get("total_categories", 0),
            "total_associations": stats.get("total_associations", 0),
            "memory_types": stats.get("memory_types", {}),
        }

        # Get recent memories if requested
        memories = []
        if include_memories:
            now = datetime.now(timezone.utc)
            created_after = now - timedelta(minutes=lookback_minutes)
            memories = await self.storage.get_recent_memories(
                workspace_id, created_after=created_after,
                limit=limit, detail_level=detail_level
            )

            # Update workspace summary with recent count
            workspace_summary["recent_memories"] = len(memories)

        # Build recent activity list
        recent_activity = []
        # Note: Storage backend doesn't track detailed session activity
        # Custom implementations can enhance this with actual activity tracking
        recent_activity.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": f"Workspace stats: {workspace_summary['total_memories']} total memories",
            "memories_created": 0,
            "key_decisions": [],
        })

        self.logger.debug(
            "Generated briefing for workspace %s: %d memories, %d associations, %d recent memories",
            workspace_id,
            workspace_summary["total_memories"],
            workspace_summary["total_associations"],
            workspace_summary["recent_memories"]
        )

        # Get unresolved contradictions
        contradictions_detected = []
        if include_contradictions and self.contradiction_service:
            try:
                records = await self.contradiction_service.get_unresolved(workspace_id, limit=3)
                for record in records:
                    contradictions_detected.append({
                        "id": record.id,
                        "memory_a_id": record.memory_a_id,
                        "memory_b_id": record.memory_b_id,
                        "type": record.contradiction_type,
                        "confidence": record.confidence,
                    })
            except Exception as e:
                self.logger.warning("Failed to get contradictions for briefing: %s", e)

        return SessionBriefing(
            workspace_summary=workspace_summary,
            recent_activity=recent_activity,
            open_threads=[],  # Advanced feature - empty for OSS
            contradictions_detected=contradictions_detected,
            memories=memories,
        )

    async def commit_session(
            self,
            workspace_id: str,
            session_id: str,
            options: Optional[CommitOptions] = None
    ) -> CommitResult:
        """
        Commit session working memory to long-term storage.

        Uses extraction service to extract memories from working memory,
        then stores them in the memory backend.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to commit
            options: Optional commit configuration

        Returns:
            CommitResult with extraction statistics

        Raises:
            ValueError: If session not found or already expired
        """
        # Verify session exists
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        # Get working memory entries
        working_memory_list = await self.storage.get_all_working_memory(workspace_id, session_id)

        self.logger.info(
            "Committing session %s: %d working memory entries",
            session_id,
            len(working_memory_list)
        )

        # If no extraction service, return basic result
        if not self.extraction_service:
            self.logger.warning("No extraction service configured, skipping memory extraction")
            return CommitResult(
                session_id=session_id,
                committed_at=datetime.now(timezone.utc),
                memories_committed=0,
                associations_committed=0,
                success=True
            )

        # Convert CommitOptions to ExtractionOptions
        from ..extraction import ExtractionOptions, ExtractionCategory

        # Build extraction options from commit options
        min_importance = options.importance_threshold if options and options.importance_threshold is not None else 0.5
        extraction_opts = ExtractionOptions(
            min_importance=min_importance,
            deduplicate=True,  # Always deduplicate
            categories=None,  # Extract all categories
            max_memories=50
        )

        # Build session content from working memory
        working_memory_dict = {wm.key: wm.value for wm in working_memory_list}
        session_content = "\n".join(
            f"{wm.key}: {wm.value}" for wm in working_memory_list
        )

        # Extract memories using extraction service
        extraction_result = await self.extraction_service.extract_from_session(
            session_id=session_id,
            workspace_id=workspace_id,
            context_id=session.context_id,
            session_content=session_content,
            working_memory=working_memory_dict,
            options=extraction_opts
        )

        # Store extracted memories via memory service
        from ...models import RememberInput
        from ..extraction import CATEGORY_MAPPING

        if not self._memory_service:
            self.logger.warning("No memory service available, cannot store extracted memories")
            return CommitResult(
                session_id=session_id,
                committed_at=datetime.now(timezone.utc),
                memories_committed=0,
                associations_committed=0,
                success=True,
                memories_extracted=extraction_result.memories_extracted,
                memories_deduplicated=extraction_result.memories_deduplicated,
                extraction_summary={
                    "breakdown": extraction_result.breakdown,
                    "extraction_time_ms": extraction_result.extraction_time_ms,
                    "stored_count": 0,
                    "failed_count": len(extraction_result.memories_created),
                }
            )

        memory_service = self._memory_service
        stored_memories = []

        for extracted_memory in extraction_result.memories_created:
            # Convert ExtractedMemory to RememberInput
            memory_type, memory_subtype = CATEGORY_MAPPING[extracted_memory.category]

            remember_input = RememberInput(
                content=extracted_memory.content,
                type=memory_type,
                subtype=memory_subtype,
                importance=extracted_memory.importance,
                tags=extracted_memory.tags,
                metadata=extracted_memory.metadata,
                context_id=session.context_id,
                user_id=None,  # Will be inferred from workspace
            )

            # Store memory
            try:
                memory = await memory_service.remember(
                    workspace_id=workspace_id,
                    input=remember_input
                )
                stored_memories.append(memory)
                self.logger.debug(
                    "Stored extracted memory: %s (category: %s, importance: %.2f)",
                    memory.id,
                    extracted_memory.category,
                    extracted_memory.importance
                )
            except Exception as e:
                self.logger.error(
                    "Failed to store extracted memory (category: %s): %s",
                    extracted_memory.category,
                    e
                )

        # Update session committed_at timestamp
        committed_at = datetime.now(timezone.utc)
        session.committed_at = committed_at

        # Persist the committed_at timestamp
        try:
            await self.storage.update_session(
                workspace_id,
                session_id,
                committed_at=committed_at
            )
        except Exception as e:
            self.logger.warning(
                "Failed to persist committed_at for session %s: %s",
                session_id,
                e
            )

        self.logger.info(
            "Committed session %s: extracted %d memories, stored %d, deduplicated %d, breakdown: %s",
            session_id,
            len(extraction_result.memories_created),
            len(stored_memories),
            extraction_result.memories_deduplicated,
            extraction_result.breakdown
        )

        # Build extraction summary for API response
        extraction_summary = {
            "breakdown": extraction_result.breakdown,
            "extraction_time_ms": extraction_result.extraction_time_ms,
            "stored_count": len(stored_memories),
            "failed_count": len(extraction_result.memories_created) - len(stored_memories),
        }

        # Create CommitResult with extraction statistics and actual stored memories
        commit_result = CommitResult(
            session_id=session_id,
            committed_at=datetime.now(timezone.utc),
            memories_committed=len(stored_memories),  # Use actual stored count
            associations_committed=0,  # Not yet implemented
            success=True,
            memories_extracted=extraction_result.memories_extracted,
            memories_deduplicated=extraction_result.memories_deduplicated,
            extraction_summary=extraction_summary
        )

        return commit_result

    async def touch_session(
            self,
            workspace_id: str,
            session_id: str,
            extend_seconds: Optional[int] = None
    ) -> Session:
        """
        Extend session TTL.

        Args:
            workspace_id: Workspace boundary
            session_id: Session to extend
            extend_seconds: Additional seconds to add (uses 3600 if None)

        Returns:
            Updated session with new expiration time

        Raises:
            ValueError: If session not found or already expired
        """
        session = await self.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found or expired")

        # Extend TTL
        extend_by = extend_seconds or 3600
        session.expires_at = session.expires_at + timedelta(seconds=extend_by)

        # TODO: THIS IS UNFINISHED!!
        # Note: storage.update_session() not yet implemented
        # For now, just return the modified session object
        self.logger.warning(
            "Session TTL extension not persisted - storage.update_session() not implemented"
        )

        self.logger.info(
            "Extended session %s TTL by %d seconds, new expiration: %s",
            session_id,
            extend_by,
            session.expires_at.isoformat()
        )

        return session

    async def list_sessions(
            self,
            workspace_id: str,
            context_id: Optional[str] = None,
            include_expired: bool = False
    ) -> List[Session]:
        """
        List sessions in a workspace.

        Args:
            workspace_id: Workspace to list sessions for
            context_id: Optional filter by context
            include_expired: Whether to include expired sessions

        Returns:
            List of sessions matching criteria
        """
        # Note: storage.list_sessions() not yet implemented
        self.logger.warning("list_sessions not implemented in storage backend")
        return []


class PersistentSessionServicePlugin(SessionServicePluginBase):
    """Plugin for persistent session service."""
    PROVIDER_NAME = 'default'

    def get_dependencies(self, v: Variables):
        return (EXT_STORAGE_BACKEND, EXT_EXTRACTION_SERVICE, EXT_DEDUPLICATION_SERVICE, EXT_MEMORY_SERVICE, EXT_CONTRADICTION_SERVICE)

    def initialize(self, v: Variables, logger: Logger) -> SessionService:
        storage: StorageBackend = self.get_extension(EXT_STORAGE_BACKEND, v)
        extraction_service: ExtractionService = self.get_extension(EXT_EXTRACTION_SERVICE, v)
        deduplication_service: DeduplicationService = self.get_extension(EXT_DEDUPLICATION_SERVICE, v)
        memory_service: MemoryService = self.get_extension(EXT_MEMORY_SERVICE, v)
        contradiction_service: ContradictionService = self.get_extension(EXT_CONTRADICTION_SERVICE, v)

        return PersistentSessionService(
            storage=storage,
            extraction_service=extraction_service,
            deduplication_service=deduplication_service,
            memory_service=memory_service,
            contradiction_service=contradiction_service,
            v=v
        )
