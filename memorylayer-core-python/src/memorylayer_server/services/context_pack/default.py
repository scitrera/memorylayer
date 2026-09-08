"""Stable deterministic context-pack rendering and cursor-based deltas."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Any

from ...models.context_pack import (
    BudgetSummary,
    ContextDelta,
    ContextDeltaInput,
    ContextEventKind,
    ContextPack,
    ContextPackInput,
    ContextPackItem,
    CursorExpiredError,
    SessionContextEvent,
)
from ...models.generation import EnrichmentPolicy, GenerationSummary
from ...models.memory import MemorySubtype, RecallInput, RecallMode
from ..entity_registry._normalize import normalize_entity_name
from ..memory import MemoryService
from ..memory.budget import ESTIMATOR_NAME, estimate_tokens
from ..storage import StorageBackend


class ContextPackService:
    """Assemble injectable context without synthesis or broad graph expansion."""

    CURSOR_VERSION = 1
    _KIND_PRIORITY = {
        "directives": 0,
        "working_memory": 1,
        "unresolved_contradictions": 2,
        "entity_matches": 3,
        "topic_matches": 3,
        "recent_activity": 4,
        "checkpoint_recovery": 5,
        "sandbox_summary": 6,
    }
    _ALL_HISTORY = datetime(1970, 1, 1, tzinfo=UTC)

    def __init__(
        self,
        storage: StorageBackend,
        memory_service: MemoryService,
        *,
        cursor_secret: str,
        policy: EnrichmentPolicy = EnrichmentPolicy.DETERMINISTIC,
        context_environment=None,
    ):
        self.storage = storage
        self.memory_service = memory_service
        self._secret = cursor_secret.encode("utf-8")
        self.policy = policy
        self.context_environment = context_environment

    @staticmethod
    def _scope_fingerprint(workspace_id: str, session_id: str) -> str:
        return hashlib.sha256(f"{workspace_id}\0{session_id}".encode()).hexdigest()[:24]

    def _encode_cursor(self, workspace_id: str, session_id: str, sequence: int) -> str:
        payload = json.dumps(
            {
                "v": self.CURSOR_VERSION,
                "seq": sequence,
                "scope": self._scope_fingerprint(workspace_id, session_id),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        signature = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(payload + signature).decode().rstrip("=")

    def _decode_cursor(self, cursor: str, workspace_id: str, session_id: str) -> int:
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            raw = base64.urlsafe_b64decode(padded.encode())
            payload, signature = raw[:-32], raw[-32:]
            expected = hmac.new(self._secret, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("invalid signature")
            value = json.loads(payload)
        except Exception as exc:
            raise ValueError("invalid context cursor") from exc
        if value.get("v") != self.CURSOR_VERSION:
            raise ValueError("unsupported context cursor version")
        if value.get("scope") != self._scope_fingerprint(workspace_id, session_id):
            raise ValueError("context cursor does not match request scope")
        return int(value["seq"])

    async def _recent_memories(self, workspace_id: str, limit: int):
        """Normalize the legacy recent-memory projection into domain models."""
        recent = await self.storage.get_recent_memories(
            workspace_id,
            self._ALL_HISTORY,
            limit=limit,
            detail_level="full",
        )
        if not recent or not isinstance(recent[0], dict):
            return recent
        return await self.storage.get_memories_by_ids(
            workspace_id,
            [row["id"] for row in recent],
        )

    @staticmethod
    def _memory_item(memory, kind: str = "memory", signals: list[str] | None = None) -> ContextPackItem:
        alternates = []
        for value in (memory.abstract, memory.overview):
            if value and value != memory.content and value not in alternates:
                alternates.append(value)
        return ContextPackItem(
            id=memory.id,
            kind=kind,
            content=memory.content,
            importance=memory.importance,
            event_time=memory.event_time or memory.created_at,
            match_signals=signals or list(memory.match_signals or []),
            source_references=[memory.source_memory_id] if memory.source_memory_id else [],
            alternate_contents=alternates,
        )

    @staticmethod
    def _deduplicate(items: list[ContextPackItem]) -> list[ContextPackItem]:
        seen_ids: set[str] = set()
        seen_content: set[str] = set()
        result: list[ContextPackItem] = []
        for index, item in enumerate(items):
            normalized = " ".join(item.content.casefold().split())
            if item.id in seen_ids or (normalized and normalized in seen_content):
                continue
            seen_ids.add(item.id)
            if normalized:
                seen_content.add(normalized)
            result.append(item)
        return result

    @classmethod
    def _rank_items(cls, items: list[ContextPackItem]) -> list[ContextPackItem]:
        """Apply stable section priority and deterministic within-section rank."""
        return sorted(
            items,
            key=lambda item: (
                cls._KIND_PRIORITY.get(item.kind, 99),
                -item.importance,
                -(item.event_time.timestamp() if item.event_time else 0),
                item.id,
            ),
        )

    @staticmethod
    def _render_and_pack(
        items: list[ContextPackItem],
        budget_tokens: int,
        *,
        stop_on_omit: bool = False,
    ) -> tuple[str, list[ContextPackItem], BudgetSummary]:
        rendered_parts: list[str] = []
        packed: list[ContextPackItem] = []
        used = 0
        truncated = 0
        omitted = 0
        current_kind = None
        for index, item in enumerate(items):
            heading = "" if item.kind == current_kind else f"## {item.kind.replace('_', ' ').title()}\n"
            prefix = "- [deleted] " if item.tombstone else "- "
            available = budget_tokens - used - estimate_tokens(heading + prefix)
            if available <= 0:
                if stop_on_omit:
                    omitted += len(items) - index
                    break
                omitted += 1
                continue
            content = item.content
            if estimate_tokens(content) > available:
                content = next(
                    (alternative for alternative in item.alternate_contents if estimate_tokens(alternative) <= available),
                    content[: available * 3].rstrip(),
                )
                while content and estimate_tokens(content) > available:
                    content = content[:-1]
                if not content:
                    if stop_on_omit:
                        omitted += len(items) - index
                        break
                    omitted += 1
                    continue
                truncated += 1
                item = item.model_copy(update={"content": content})
            part = f"{heading}{prefix}{content}\n"
            cost = estimate_tokens(part)
            if used + cost > budget_tokens:
                if stop_on_omit:
                    omitted += len(items) - index
                    break
                omitted += 1
                continue
            rendered_parts.append(part)
            packed.append(item)
            used += cost
            current_kind = item.kind
        return (
            "".join(rendered_parts).rstrip(),
            packed,
            BudgetSummary(
                requested=budget_tokens,
                used=used,
                estimator=ESTIMATOR_NAME,
                truncated_items=truncated,
                omitted_items=omitted,
            ),
        )

    async def build_pack(
        self,
        workspace_id: str,
        session_id: str,
        input: ContextPackInput,
    ) -> ContextPack:
        session = await self.storage.get_session(workspace_id, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        items: list[ContextPackItem] = []
        degradation_notices: list[str] = []
        open_threads: list[dict[str, Any]] = []
        limits = input.section_limits
        if input.include_directives:
            directives = await self._recent_memories(
                workspace_id,
                max(limits.directives * 4, limits.directives),
            )
            directives = [
                memory
                for memory in directives
                if str(memory.subtype)
                in {
                    MemorySubtype.DIRECTIVE.value,
                    MemorySubtype.PREFERENCE.value,
                    str(MemorySubtype.DIRECTIVE),
                    str(MemorySubtype.PREFERENCE),
                }
                or memory.pinned
            ][: limits.directives]
            items.extend(self._memory_item(memory, "directives", ["explicit"]) for memory in directives)
        if input.include_working_memory:
            working = await self.storage.get_all_working_memory(workspace_id, session_id)
            working.sort(key=lambda value: (value.key, value.updated_at))
            for value in working[: limits.working_memory]:
                content = value.value if isinstance(value.value, str) else json.dumps(value.value, sort_keys=True, default=str)
                if value.key.casefold() in {"open_threads", "open_items", "next_steps", "blocked"}:
                    open_threads.append({"key": value.key, "value": value.value})
                items.append(
                    ContextPackItem(
                        id=f"working:{value.key}",
                        kind="working_memory",
                        content=f"{value.key}: {content}",
                        importance=1.0,
                        event_time=value.updated_at,
                        match_signals=["working"],
                    )
                )
        entity_ids = list(dict.fromkeys(input.entity_ids))[:8]
        try:
            for name in input.entity_names[:8]:
                normalized = normalize_entity_name(name)
                exact = await self.storage.find_entities_by_normalized_name_any_type(
                    workspace_id,
                    normalized,
                )
                aliases = (
                    []
                    if exact
                    else await self.storage.find_entities_by_normalized_alias(
                        workspace_id,
                        normalized,
                    )
                )
                for entity in [*exact, *aliases]:
                    if entity["id"] not in entity_ids:
                        entity_ids.append(entity["id"])
            member_ids: list[str] = []
            for entity_id in entity_ids[:8]:
                members = await self.storage.list_entity_members(
                    workspace_id,
                    entity_id,
                    limit=limits.recent_activity,
                )
                member_ids.extend(member["memory_id"] for member in members)
            entity_memories = await self.storage.get_memories_by_ids(
                workspace_id,
                list(dict.fromkeys(member_ids))[: limits.recent_activity],
            )
            items.extend(self._memory_item(memory, "entity_matches", ["exact_entity"]) for memory in entity_memories)
        except (NotImplementedError, AttributeError):
            if input.entity_ids or input.entity_names:
                degradation_notices.append("entity_resolution_unavailable")
        topic = " ".join(part.strip() for part in [input.topic or "", *input.entity_names] if part and part.strip())
        if topic and input.include_recent_activity:
            try:
                recalled = await self.memory_service.recall(
                    workspace_id,
                    RecallInput(
                        query=topic,
                        mode=RecallMode.RAG,
                        limit=limits.recent_activity,
                        include_associations=False,
                        include_confidence=False,
                        budget_tokens=None,
                    ),
                )
                items.extend(self._memory_item(memory, "topic_matches") for memory in recalled.memories)
            except Exception:
                degradation_notices.append("topic_retrieval_unavailable")
                recent = await self._recent_memories(workspace_id, limits.recent_activity)
                items.extend(self._memory_item(memory, "recent_activity", ["recent"]) for memory in recent)
        elif input.include_recent_activity:
            recent = await self._recent_memories(workspace_id, limits.recent_activity)
            recent.sort(key=lambda memory: (-memory.importance, -(memory.event_time or memory.created_at).timestamp(), memory.id))
            items.extend(self._memory_item(memory, "recent_activity", ["recent"]) for memory in recent)
        contradictions: list[dict[str, Any]] = []
        if input.include_contradictions:
            records = await self.storage.get_unresolved_contradictions(
                workspace_id,
                limit=limits.contradictions,
            )
            for record in records:
                contradictions.append(
                    {
                        "id": record.id,
                        "memory_a_id": record.memory_a_id,
                        "memory_b_id": record.memory_b_id,
                        "type": record.contradiction_type,
                    }
                )
                items.append(
                    ContextPackItem(
                        id=record.id,
                        kind="unresolved_contradictions",
                        content=f"{record.memory_a_id} conflicts with {record.memory_b_id} ({record.contradiction_type})",
                        importance=0.9,
                        event_time=record.detected_at,
                        source_references=[record.memory_a_id, record.memory_b_id],
                    )
                )
        if input.include_sandbox_summary:
            if self.context_environment is None:
                degradation_notices.append("sandbox_summary_unavailable")
            else:
                try:
                    sandbox = await self.context_environment.inspect(
                        session_id,
                        preview_chars=160,
                    )
                    if sandbox.get("variable_count", 0) > 0:
                        items.append(
                            ContextPackItem(
                                id=f"sandbox:{session_id}",
                                kind="sandbox_summary",
                                content=json.dumps(sandbox, sort_keys=True, default=str),
                                importance=1.0,
                                match_signals=["sandbox"],
                            )
                        )
                except Exception:
                    degradation_notices.append("sandbox_summary_unavailable")
        if input.include_checkpoint_recovery:
            checkpoints = await self.storage.list_session_checkpoints(
                workspace_id,
                session_id,
                limit=limits.checkpoints,
            )
            for checkpoint in checkpoints:
                raw = await self.storage.get_memory(
                    workspace_id,
                    checkpoint.raw_memory_id,
                    track_access=False,
                )
                if raw:
                    items.append(self._memory_item(raw, "checkpoint_recovery", ["checkpoint"]))
        items = self._rank_items(self._deduplicate(items))
        rendered, packed, summary = self._render_and_pack(items, input.budget_tokens)
        _minimum, maximum = await self.storage.get_context_event_bounds(workspace_id, session_id)
        return ContextPack(
            rendered=rendered,
            items=packed,
            open_threads=open_threads,
            unresolved_contradictions=contradictions,
            budget_summary=summary,
            generation_summary=GenerationSummary(policy=self.policy),
            cursor=self._encode_cursor(workspace_id, session_id, maximum),
            degradation_notices=degradation_notices,
        )

    async def _event_item(self, workspace_id: str, session_id: str, event: SessionContextEvent) -> ContextPackItem:
        if event.event_kind in {ContextEventKind.MEMORY_DELETE, ContextEventKind.WORKING_DELETE}:
            return ContextPackItem(
                id=event.subject_id,
                kind=event.subject_kind,
                content=event.subject_id,
                event_time=event.event_time,
                tombstone=True,
            )
        if event.subject_kind == "memory":
            memory = await self.storage.get_memory(workspace_id, event.subject_id, track_access=False)
            if memory:
                return self._memory_item(memory, "memory_changes", ["delta"])
        if event.subject_kind == "working_memory":
            value = await self.storage.get_working_memory(workspace_id, session_id, event.subject_id)
            if value:
                content = value.value if isinstance(value.value, str) else json.dumps(value.value, sort_keys=True, default=str)
                return ContextPackItem(
                    id=f"working:{value.key}",
                    kind="working_memory_changes",
                    content=f"{value.key}: {content}",
                    importance=1.0,
                    event_time=value.updated_at,
                )
        if event.subject_kind == "checkpoint":
            checkpoint = await self.storage.get_session_checkpoint(
                workspace_id,
                session_id,
                event.subject_id,
            )
            if checkpoint:
                raw = await self.storage.get_memory(workspace_id, checkpoint.raw_memory_id, track_access=False)
                if raw:
                    return self._memory_item(raw, "checkpoint_changes", ["checkpoint", "delta"])
        return ContextPackItem(
            id=event.subject_id,
            kind=f"{event.subject_kind}_changes",
            content=json.dumps(event.metadata, sort_keys=True) if event.metadata else event.event_kind.value,
            event_time=event.event_time,
        )

    async def build_delta(
        self,
        workspace_id: str,
        session_id: str,
        input: ContextDeltaInput,
    ) -> ContextDelta:
        after = self._decode_cursor(input.cursor, workspace_id, session_id)
        minimum, _maximum = await self.storage.get_context_event_bounds(workspace_id, session_id)
        if after > 0 and minimum > after + 1:
            raise CursorExpiredError()
        events = await self.storage.list_context_events(
            workspace_id,
            session_id,
            after_sequence=after,
            limit=501,
        )
        latest: dict[tuple[str, str], SessionContextEvent] = {}
        for event in events[:500]:
            latest[(event.subject_kind, event.subject_id)] = event
        collapsed = sorted(latest.values(), key=lambda event: event.sequence)
        event_items: list[tuple[int, ContextPackItem]] = []
        for event in collapsed:
            event_items.append((event.sequence, await self._event_item(workspace_id, session_id, event)))
        rendered, packed, summary = self._render_and_pack(
            [item for _sequence, item in event_items],
            input.budget_tokens,
            stop_on_omit=True,
        )
        included_ids = {(item.kind, item.id) for item in packed}
        delivered_sequences = [sequence for sequence, item in event_items if (item.kind, item.id) in included_ids]
        delivered = max(delivered_sequences, default=after)
        has_more = len(events) > 500 or delivered < max((event.sequence for event in collapsed), default=after)
        return ContextDelta(
            rendered=rendered,
            items=packed,
            budget_summary=summary,
            generation_summary=GenerationSummary(policy=self.policy),
            cursor=self._encode_cursor(workspace_id, session_id, delivered),
            has_more=has_more,
        )
