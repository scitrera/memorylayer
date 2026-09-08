"""Unit tests for the email -> memory normalize adapter (cross-source ingestion).

The adapter is a PURE MAPPING (email payload -> RememberInput) — no storage, no
enrichment. These tests pin the critical mapping decisions:

  * observer_id = sender (the perspective anchor; emails are prefix-less like docs)
  * content is grounded with a ``[date] Subject:`` prefix (temporal + topical)
  * source = SourceType.EMAIL (read by coverage/attribution scorers)
  * recipients carried in metadata (for subject/mention extraction)
  * thread/message id maps to source_thread_id (thread linkage)
"""

from datetime import UTC, datetime

import pytest

from memorylayer_server.models.memory import (
    KNOWN_SOURCE_TYPES,
    MemoryType,
    SourceType,
)
from memorylayer_server.services.ingest import email_to_remember_input

_TS = datetime(2026, 1, 8, 9, 30, tzinfo=UTC)


def test_source_type_email_is_known():
    """SourceType.EMAIL exists and is registered in KNOWN_SOURCE_TYPES."""
    assert SourceType.EMAIL.value == "email"
    assert SourceType.EMAIL.value in KNOWN_SOURCE_TYPES


def test_observer_id_is_sender():
    """The sender becomes the observer_id perspective anchor."""
    ri = email_to_remember_input(sender="Alice", body="Phoenix ships March 14.")
    assert ri.observer_id == "Alice"


def test_content_is_grounded_with_subject_and_date():
    """Body is prefixed with ``[date] Subject:`` for temporal + topical grounding."""
    ri = email_to_remember_input(
        sender="Alice",
        body="Phoenix ships March 14.",
        subject="Launch date",
        timestamp=_TS,
    )
    assert ri.content == "[2026-01-08] Launch date: Phoenix ships March 14."
    assert ri.event_time == _TS


def test_source_and_attribution_metadata():
    """source=EMAIL, sender, recipients, subject, message_id land in metadata."""
    ri = email_to_remember_input(
        sender="Alice",
        body="Status update.",
        to=["Bob", "Carol"],
        subject="Update",
        message_id="m-123",
    )
    assert ri.metadata["source"] == SourceType.EMAIL.value
    assert ri.metadata["sender"] == "Alice"
    assert ri.metadata["recipients"] == ["Bob", "Carol"]
    assert ri.metadata["subject"] == "Update"
    assert ri.metadata["message_id"] == "m-123"
    assert [(relation.source_entity_name, relation.target_entity_name, relation.relationship) for relation in ri.relations] == [
        ("Alice", "Bob", "sent_to"),
        ("Alice", "Carol", "sent_to"),
    ]


def test_thread_linkage_prefers_thread_id_then_message_id():
    """source_thread_id = thread_id when present, else falls back to message_id."""
    with_thread = email_to_remember_input(sender="A", body="x", thread_id="t1", message_id="m1")
    assert with_thread.source_thread_id == "t1"
    only_message = email_to_remember_input(sender="A", body="x", message_id="m1")
    assert only_message.source_thread_id == "m1"


def test_defaults_semantic_type_and_no_bare_prefix():
    """No subject/timestamp -> body unchanged (no empty prefix); type is SEMANTIC."""
    ri = email_to_remember_input(sender="Alice", body="Just the body.")
    assert ri.content == "Just the body."
    assert ri.type == MemoryType.SEMANTIC


def test_canonical_metadata_wins_over_extra():
    """Caller extra_metadata is merged but canonical keys (source/sender) win."""
    ri = email_to_remember_input(
        sender="Alice",
        body="x",
        extra_metadata={"source": "spoofed", "custom": "kept"},
    )
    assert ri.metadata["source"] == SourceType.EMAIL.value
    assert ri.metadata["custom"] == "kept"


def test_empty_sender_or_body_raises():
    """sender and body are the required anchors."""
    with pytest.raises(ValueError):
        email_to_remember_input(sender="", body="x")
    with pytest.raises(ValueError):
        email_to_remember_input(sender="Alice", body="   ")
