"""Contracts for deterministic connector -> knowledge_work normalization."""

from memorylayer_server.services.entity_relation.metadata import extract_metadata_relations
from memorylayer_server.services.ingest import (
    KNOWLEDGE_WORK_NORMALIZATION_KEY,
    connector_to_remember_input,
    email_to_remember_input,
    normalize_connector_metadata,
)


def _triples(metadata: dict) -> set[tuple[str, str, str]]:
    extraction = extract_metadata_relations(metadata)
    assert extraction.errors == []
    return {
        (
            candidate.source.name or candidate.source.entity_id or "",
            candidate.target.name or candidate.target.entity_id or "",
            candidate.relationship,
        )
        for candidate in extraction.candidates
    }


def test_github_pull_request_maps_nested_identities_and_repository() -> None:
    result = normalize_connector_metadata(
        {
            "connector_type": "github",
            "github_type": "pr",
            "github_title": "Make token refresh atomic",
            "github_number": 418,
            "github_repo": "acme/platform",
            "user": {"name": "Maya Ortiz", "login": "mortiz", "id": 17},
            "github_assignees": [{"displayName": "Theo Park", "accountId": "u-9"}],
            "requested_reviewers": [{"login": "rchen", "email": "rchen@example.test"}],
            "github_labels": ["security", "identity"],
        }
    )

    profile = result.profile
    assert profile is not None
    assert profile["subject"] == {
        "name": "Make token refresh atomic",
        "type": "work_item",
        "aliases": ["418", "acme/platform#418"],
        "external_ids": {"github": "418"},
    }
    assert profile["author"]["name"] == "Maya Ortiz"
    assert profile["author"]["aliases"] == ["mortiz"]
    assert profile["assignee"]["external_ids"] == {"github": "u-9"}
    assert profile["reviewer"]["name"] == "rchen"
    assert profile["reviewer"]["aliases"] == ["rchen@example.test"]
    assert _triples(result.metadata) == {
        ("Maya Ortiz", "Make token refresh atomic", "authored"),
        ("Theo Park", "Make token refresh atomic", "responsible_for"),
        ("rchen", "Make token refresh atomic", "reviewed"),
        ("Make token refresh atomic", "acme/platform", "part_of"),
        ("Make token refresh atomic", "security", "about"),
        ("Make token refresh atomic", "identity", "about"),
    }
    assert result.metadata[KNOWLEDGE_WORK_NORMALIZATION_KEY]["record_kind"] == "code_review"


def test_nested_work_item_record_maps_camel_case_people_and_workflow() -> None:
    result = normalize_connector_metadata(
        {
            "connector_type": "jira",
            "connector_record": {
                "record_type": "issue",
                "key": "OPS-91",
                "fields": {
                    "summary": "Rotate production credentials",
                    "assignee": {"displayName": "Ari Singh", "accountId": "ari-1"},
                    "reporter": {"displayName": "Jo Ellis", "emailAddress": "jo@example.test"},
                    "project": {"name": "Operations Program", "key": "OPS"},
                    "components": [{"name": "Secrets Service"}],
                    "blockers": [{"name": "Security Exception", "type": "work_item"}],
                    "status": {"name": "In Progress"},
                },
            },
        }
    )

    assert result.profile is not None
    assert result.profile["subject"]["aliases"] == ["OPS-91"]
    assert result.profile["status"] == {"name": "In Progress"}
    assert _triples(result.metadata) == {
        ("Ari Singh", "Rotate production credentials", "responsible_for"),
        ("Jo Ellis", "Rotate production credentials", "authored"),
        ("Rotate production credentials", "Operations Program", "part_of"),
        ("Rotate production credentials", "Secrets Service", "affects"),
        ("Security Exception", "Rotate production credentials", "blocks"),
    }


def test_explicit_profile_values_win_while_missing_fields_are_filled() -> None:
    explicit_subject = {"name": "Canonical Policy", "type": "artifact"}
    result = normalize_connector_metadata(
        {
            "connector_type": "policy_system",
            "title": "Ignored source title",
            "owner": "Ignored source owner",
            "reviewer": "Source Reviewer",
            "knowledge_work": {
                "subject": explicit_subject,
                "owner": "Explicit Owner",
                "valid_from": "2026-01-01",
            },
        }
    )

    assert result.profile is not None
    assert result.profile["subject"] == explicit_subject
    assert result.profile["owner"] == "Explicit Owner"
    assert result.profile["reviewer"]["name"] == "Source Reviewer"
    assert result.profile["valid_from"] == "2026-01-01"


def test_malformed_explicit_profile_is_preserved_and_not_guessed_over() -> None:
    metadata = {"connector_type": "jira", "title": "A task", "assignee": "Alice", "knowledge_work": "bad"}
    result = normalize_connector_metadata(metadata)
    assert result.metadata["knowledge_work"] == "bad"
    assert result.profile is None
    assert result.mapped_fields == ()
    assert "normalization skipped" in result.warnings[0]


def test_subject_only_and_arbitrary_metadata_do_not_activate_relation_profile() -> None:
    connector = normalize_connector_metadata({"connector_type": "local_fs", "filename": "notes.md"})
    arbitrary = normalize_connector_metadata({"filename": "notes.md", "owner": "Alice"})
    assert connector.profile is None
    assert "knowledge_work" not in connector.metadata
    assert arbitrary.profile is None
    assert arbitrary.metadata == {"filename": "notes.md", "owner": "Alice"}


def test_email_reuses_normalizer_for_authorship_and_registry_backed_recipients() -> None:
    result = email_to_remember_input(
        sender="Alice Chen",
        to=["Bob Singh", "Carol Jones"],
        subject="Phoenix decision",
        body="The decision is ready for review.",
        message_id="msg-41",
    )
    assert _triples(result.metadata) == {
        ("Alice Chen", "Phoenix decision", "authored"),
        ("Alice Chen", "Bob Singh", "sent_to"),
        ("Alice Chen", "Carol Jones", "sent_to"),
    }
    # Preserve the legacy explicit adapter assertions for API compatibility.
    assert len(result.relations) == 2


def test_generic_connector_adapter_builds_ready_to_store_memory() -> None:
    result = connector_to_remember_input(
        connector_type="linear",
        content="Search indexing cannot ship until the privacy review closes.",
        subject_name="Search Indexing",
        subject_type="work_item",
        record_id="ENG-72",
        source_metadata={
            "assignee": {"displayName": "Nora Bell", "id": "usr-8"},
            "blocked_by": "Privacy Review",
            "project": "Search Platform",
        },
    )
    assert result.content.startswith("Search indexing")
    assert result.metadata["source"] == "linear"
    assert result.metadata["source_record_id"] == "ENG-72"
    assert _triples(result.metadata) == {
        ("Nora Bell", "Search Indexing", "responsible_for"),
        ("Privacy Review", "Search Indexing", "blocks"),
        ("Search Indexing", "Search Platform", "part_of"),
    }
