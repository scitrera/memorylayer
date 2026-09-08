"""Email -> memory normalize adapter.

Cross-source retrieval means one entity (a project, a person) recurs across chat,
email, AND document sources, with a leakage-safe per-observer perspective. Chat
and document producers already land on the SAME post-store pipeline via
``MemoryService.enqueue_post_store`` (decompose + enrich + entity accretion);
email was the one first-class source missing.

This module is a PURE MAPPING: it turns an inbound email into a single
``RememberInput`` that, when handed to ``MemoryService.remember()``, rides the
identical shared pipeline (no parallel enrichment fork). The mapping mirrors the
document/fact path's two structural decisions:

  * **observer_id = sender.** Emails are PREFIX-LESS like documents — there is no
    ``[ts] Speaker:`` dialogue prefix in an email body, so the speaker cannot be
    parsed from the content regex (``_seed_perspective_ids``). The authoritative
    perspective anchor is the ``observer_id`` FIELD set from the sender, exactly
    the doc-path pattern (cf. ``oss@d25f3a3``). This is what makes the email show
    up in perspective-scoped recall / ``get_representation`` for that sender.
  * **grounded content.** The raw body is prefixed with a ``[event_date] Subject:``
    line for temporal + topical grounding (like the doc/fact path grounds page
    text), so retrieval has the subject and date in the content even though the
    structured fields also carry them.

``source=SourceType.EMAIL`` is stamped into ``metadata['source']`` (the
benchmark/coverage scorers read the producing source straight off the memory),
``sender`` is stamped for attribution. Recipients remain in metadata and also
produce explicit ``sent_to`` adapter relations; relation endpoints resolve only
against exact canonical names or aliases and are otherwise reported unresolved.
The thread/message id maps to ``source_thread_id`` for thread linkage.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ...models.entity_relation import EntityRelationInput, RelationEvidenceKind
from ...models.memory import MemoryType, RememberInput, SourceType
from .knowledge_work import normalize_connector_metadata


def _format_event_date(event_time: datetime | None) -> str | None:
    """Render the grounding date prefix component (YYYY-MM-DD) from event_time."""
    if event_time is None:
        return None
    return event_time.date().isoformat()


def _ground_content(subject: str | None, body: str, event_time: datetime | None) -> str:
    """Prefix the email body with a ``[date] Subject:`` grounding line.

    Mirrors the doc/fact grounding: the subject + date are folded into the
    content text so retrieval surfaces the topic/time even when only the body
    text is matched. Components that are missing are simply omitted (never a
    bare/empty prefix). Returns the body unchanged when there is nothing to
    ground with.
    """
    date_part = _format_event_date(event_time)
    subject_part = subject.strip() if subject and subject.strip() else None

    prefix_bits: list[str] = []
    if date_part:
        prefix_bits.append(f"[{date_part}]")
    if subject_part:
        prefix_bits.append(f"{subject_part}:")

    if not prefix_bits:
        return body
    return f"{' '.join(prefix_bits)} {body}"


def email_to_remember_input(
    *,
    sender: str,
    body: str,
    to: Sequence[str] | None = None,
    subject: str | None = None,
    timestamp: datetime | None = None,
    thread_id: str | None = None,
    message_id: str | None = None,
    importance: float = 0.5,
    context_id: str | None = None,
    extra_metadata: dict | None = None,
    relations: Sequence[EntityRelationInput] | None = None,
) -> RememberInput:
    """Normalize an inbound email into a ``RememberInput`` for the shared pipeline.

    The returned input is meant to be passed straight to
    ``MemoryService.remember()`` so the email gets the identical
    decompose + enrich + entity-accretion lifecycle chat/doc memories get (via
    ``enqueue_post_store``). This function performs NO storage and NO enrichment
    itself — it is a deterministic mapping only.

    Mapping:
      * ``observer_id = sender`` — the perspective anchor (emails are prefix-less,
        so the sender FIELD is authoritative, like the doc path).
      * ``content`` — the body grounded with a ``[date] Subject:`` prefix.
      * ``source_thread_id`` — ``thread_id`` if present, else ``message_id`` (so a
        standalone message still links to its own thread).
      * ``event_time`` — the email ``timestamp`` (temporal grounding).
      * ``metadata`` — ``source=SourceType.EMAIL.value`` (attribution/coverage read
        this), ``sender``, ``recipients``,
        ``subject``, and ``message_id`` when present. Caller ``extra_metadata`` is
        merged in first so the canonical keys always win.
      * ``relations`` — caller relations plus an adapter-evidenced ``sent_to``
        relation for each sender/recipient pair.

    Args:
        sender: The email's from/sender — becomes ``observer_id``.
        body: Raw email body text.
        to: Recipients (carried in metadata for subject/mention extraction).
        subject: Email subject line (folded into the grounded content + metadata).
        timestamp: When the email was sent — becomes ``event_time``.
        thread_id: Conversation/thread id — preferred ``source_thread_id``.
        message_id: Message id — fallback ``source_thread_id`` + stamped in metadata.
        importance: Memory importance (default 0.5, matching the doc path).
        context_id: Optional target context.
        extra_metadata: Optional caller metadata merged under the canonical keys.
        relations: Optional explicit typed relations to preserve alongside the
            adapter-derived sender/recipient relations.

    Returns:
        A ``RememberInput`` ready for ``MemoryService.remember()``.

    Raises:
        ValueError: if ``sender`` or ``body`` is empty (the two required anchors).
    """
    if not sender or not sender.strip():
        raise ValueError("email_to_remember_input requires a non-empty sender (the observer anchor)")
    if not body or not body.strip():
        raise ValueError("email_to_remember_input requires a non-empty body")

    sender = sender.strip()
    recipients = [r.strip() for r in (to or []) if r and r.strip()]

    metadata: dict = dict(extra_metadata or {})
    # Canonical keys always win over caller-supplied extra_metadata.
    metadata["source"] = SourceType.EMAIL.value
    metadata["sender"] = sender
    if recipients:
        metadata["recipients"] = recipients
    if subject and subject.strip():
        metadata["subject"] = subject.strip()
    if message_id:
        metadata["message_id"] = message_id

    # Reuse the connector-neutral metadata normalizer.  This adds an artifact
    # subject, sender authorship, and registry-backed sent_to assertions while
    # preserving an explicit caller-authored knowledge_work profile verbatim.
    metadata = normalize_connector_metadata(
        metadata,
        connector_type=SourceType.EMAIL.value,
        subject_name=subject,
        subject_type="artifact",
        record_id=message_id or thread_id,
    ).metadata

    relation_inputs = list(relations or [])
    relation_inputs.extend(
        EntityRelationInput(
            source_entity_name=sender,
            target_entity_name=recipient,
            relationship="sent_to",
            confidence=1.0,
            evidence_kind=RelationEvidenceKind.ADAPTER,
            extraction_method="email_sender_recipient",
        )
        for recipient in recipients
        if recipient != sender
    )

    return RememberInput(
        content=_ground_content(subject, body.strip(), timestamp),
        type=MemoryType.SEMANTIC,
        importance=importance,
        metadata=metadata,
        relations=relation_inputs,
        context_id=context_id,
        # Perspective anchor: emails are prefix-less, so observer_id is set from
        # the sender FIELD (not parsed from content). This is the doc-path pattern.
        observer_id=sender,
        source_thread_id=thread_id or message_id,
        event_time=timestamp,
    )
