"""Raw-first checkpoint capture and deterministic post-durability indexing."""

from __future__ import annotations

import hashlib
import re

from ...models.context_pack import CheckpointWorkStatus, SessionCheckpoint, SessionCheckpointInput
from ..memory import MemoryService
from ..storage import StorageBackend

_SOURCE_KIND = re.compile(r"[^a-z0-9_-]+")


class SessionCheckpointService:
    def __init__(
        self,
        storage: StorageBackend,
        memory_service: MemoryService,
        *,
        max_bytes: int,
    ):
        self.storage = storage
        self.memory_service = memory_service
        self.max_bytes = max_bytes

    async def capture(
        self,
        workspace_id: str,
        session_id: str,
        input: SessionCheckpointInput,
    ) -> SessionCheckpoint:
        encoded = input.transcript_segment.encode("utf-8", errors="strict")
        if len(encoded) > self.max_bytes:
            raise ValueError(f"checkpoint exceeds configured byte limit ({self.max_bytes})")
        actual_hash = hashlib.sha256(encoded).hexdigest()
        if actual_hash != input.content_hash:
            raise ValueError("content_hash does not match transcript_segment")
        sanitized_kind = _SOURCE_KIND.sub("_", input.source_kind.strip().casefold()).strip("_")
        if not sanitized_kind:
            sanitized_kind = "transcript"
        normalized = input.model_copy(update={"source_kind": sanitized_kind[:64]})
        checkpoint, _replayed = await self.storage.create_session_checkpoint(
            workspace_id,
            session_id,
            normalized,
        )
        if checkpoint.index_status == CheckpointWorkStatus.COMPLETE:
            return checkpoint
        try:
            raw = await self.storage.get_memory(
                workspace_id,
                checkpoint.raw_memory_id,
                track_access=False,
            )
            if raw is None:
                raise RuntimeError("durable checkpoint raw memory is unavailable")
            await self.memory_service._store_deterministic_segments(workspace_id, raw)
            checkpoint = await self.storage.update_session_checkpoint_status(
                workspace_id,
                session_id,
                checkpoint.id,
                index_status=CheckpointWorkStatus.COMPLETE.value,
            )
        except Exception:
            checkpoint = await self.storage.update_session_checkpoint_status(
                workspace_id,
                session_id,
                checkpoint.id,
                index_status=CheckpointWorkStatus.FAILED.value,
            )
        return checkpoint
