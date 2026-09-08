"""Focused invariants for deterministic memory processing and retrieval."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from memorylayer_server.config import DEFAULT_MEMORYLAYER_RELATIONAL_RECALL_ENABLED
from memorylayer_server.models.context_pack import ContextPackItem
from memorylayer_server.models.memory import DetailLevel, Memory, MemoryType
from memorylayer_server.services.context_pack import ContextPackService
from memorylayer_server.services.entity_relation import EntityRelationService
from memorylayer_server.services.entity_relation.metadata import extract_metadata_relations
from memorylayer_server.services.extraction.deterministic import (
    classify_content,
    extract_marker_candidates,
    extractive_tiers,
    segment_content,
)
from memorylayer_server.services.memory.budget import estimate_tokens, pack_recall_memories
from memorylayer_server.services.memory.confidence import retrieval_confidence
from memorylayer_server.services.memory.relation_intent import classify_relation_intent


def _memory(memory_id: str, content: str, **updates) -> Memory:
    values = {
        "id": memory_id,
        "workspace_id": "ws_test",
        "tenant_id": "tenant_test",
        "content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "type": MemoryType.EPISODIC,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    values.update(updates)
    return Memory(**values)


def test_relational_recall_default_tracks_the_passing_quality_gate():
    assert DEFAULT_MEMORYLAYER_RELATIONAL_RECALL_ENABLED is True


def test_explicit_classification_wins_and_neutral_fallback_is_non_generative():
    explicit = classify_content(
        "currently blocked on deployment",
        explicit_type=MemoryType.SEMANTIC,
        explicit_subtype="decision",
    )
    assert explicit.memory_type == MemoryType.SEMANTIC
    assert explicit.subtype == "decision"
    assert explicit.explicitly_classified is True
    assert explicit.matched_rules == ("explicit_type",)

    fallback = classify_content("An otherwise unclassified statement.")
    assert fallback.memory_type == MemoryType.EPISODIC
    assert fallback.confidence == "weak"
    assert fallback.matched_rules == ("neutral_episodic_fallback",)


def test_segmentation_is_stable_source_linkable_and_never_invents_facts():
    content = "User: First message.\nAssistant: Second message.\nTool: changed src/app.py"
    first = segment_content(content, max_chars=30)
    second = segment_content(content, max_chars=30)

    assert first == second
    assert first
    for segment in first:
        assert content[segment.source_span_start : segment.source_span_end] == segment.content
        assert segment.memory_type == MemoryType.EPISODIC
        assert segment.method in {"chat_boundary", "sentence_group", "bounded_character_split"}


def test_marker_extraction_is_bounded_to_explicit_text():
    content = "DECISION: Keep the raw event.\nWe must never discard the source transcript.\nA speculative rationale is absent."
    candidates = extract_marker_candidates(content)

    assert [(item.kind, item.content) for item in candidates] == [
        ("decision", "Keep the raw event."),
        ("directive", "We must never discard the source transcript."),
    ]
    for item in candidates:
        assert content[item.source_span_start : item.source_span_end] == item.content


def test_extractive_tiers_only_contain_source_spans():
    content = (
        "The first sentence establishes the durable source. "
        "The second sentence describes deterministic indexing. "
        "The final sentence confirms that generation is optional."
    )
    abstract, overview = extractive_tiers(content, abstract_chars=70, overview_chars=130)

    for tier in (abstract, overview):
        assert tier.content
        assert tier.method in {"extractive", "source_reuse", "bounded_source_excerpt"}
        for start, end in tier.source_spans:
            assert content[start:end] in tier.content


def test_recall_budget_is_hard_and_uses_smaller_honest_representation():
    memories = [
        _memory(
            "mem_1",
            "x" * 600,
            overview="Overview from the source.",
            abstract="Short source abstract.",
        ),
        _memory("mem_2", "another memory that should not overrun the budget"),
    ]
    packed, summary = pack_recall_memories(memories, 12, DetailLevel.FULL)

    assert summary.used <= 12
    assert sum(estimate_tokens(memory.content) for memory in packed) <= 12
    assert packed[0].content in {"Overview from the source.", "Short source abstract."}


def test_context_rendering_is_stable_and_never_exceeds_budget():
    items = [
        ContextPackItem(id="a", kind="directives", content="Always preserve the raw source.", importance=1),
        ContextPackItem(id="b", kind="recent_activity", content="x" * 500, importance=0.5),
    ]
    first = ContextPackService._render_and_pack(items, 24)
    second = ContextPackService._render_and_pack(items, 24)

    assert first == second
    rendered, _packed, summary = first
    assert estimate_tokens(rendered) <= 24
    assert summary.used <= 24


def test_confidence_describes_retrieval_evidence_not_truth():
    strong = _memory("mem_strong", "matched", match_signals=["keyword", "vector"])
    confidence, reasons = retrieval_confidence("query", [strong])
    assert confidence == "strong"
    assert reasons == ["multiple_independent_retrieval_signals"]

    weak = _memory("mem_weak", "uncalibrated", match_signals=["vector"])
    confidence, reasons = retrieval_confidence("query", [weak], unresolved_entity=True)
    assert confidence == "weak"
    assert "uncalibrated_vector_only" in reasons
    assert "unresolved_query_entity" in reasons


def test_relation_intent_only_routes_bounded_relation_queries():
    relation = classify_relation_intent("Who works for Acme Corp?")
    assert relation.is_relation_query is True
    assert relation.max_hops == 1
    assert "employed_by" in relation.relationship_types
    assert "Acme Corp" in relation.seed_phrases

    topical = classify_relation_intent("Summarize Acme Corp product plans")
    assert topical.is_relation_query is False


def test_relation_intent_preserves_exact_seed_for_common_structural_forms():
    cases = {
        "Who works at Anchor - Data Infrastructure Startup?": ("employed_by", "Anchor - Data Infrastructure Startup"),
        "Who advises Orbit Labs?": ("advises", "Orbit Labs"),
        "Who attended Acme Board Meeting Q1 2025?": ("attended", "Acme Board Meeting Q1 2025"),
        "Who founded Cedar Labs?": ("founded", "Cedar Labs"),
    }
    for query, (relationship, seed) in cases.items():
        intent = classify_relation_intent(query)
        assert intent.is_relation_query is True
        assert relationship in intent.relationship_types
        assert intent.seed_phrases[0] == seed


def test_relation_intent_covers_general_knowledge_work_questions():
    cases = {
        "Who is assigned to Customer Backfill?": ("responsible_for", "Customer Backfill"),
        "Who reviewed Data Retention Policy?": ("reviewed", "Data Retention Policy"),
        "What project is Interview Synthesis part of?": ("part_of", "Interview Synthesis"),
        "What does Production Cutover depend on?": ("depends_on", "Production Cutover"),
        "What is Pricing Analysis based on?": ("based_on", "Pricing Analysis"),
        "What supports Capacity Decision?": ("supports", "Capacity Decision"),
        "What superseded Data Retention Policy 2024?": ("supersedes", "Data Retention Policy 2024"),
        "What is Compliance Memo about?": ("about", "Compliance Memo"),
    }
    for query, (relationship, seed) in cases.items():
        intent = classify_relation_intent(query)
        assert intent.is_relation_query is True
        assert relationship in intent.relationship_types
        assert intent.seed_phrases[0] == seed


def test_high_precision_relation_patterns_cover_common_structural_phrasing():
    content = (
        "Alice Davis works at Acme Corp. "
        "Bob Smith is an advisor to Orbit Labs. "
        "Carol Jones invested in Beacon Labs. "
        "Dana Price founded Cedar Labs. "
        "Evan Reed attended Summit Meeting."
    )
    relations = EntityRelationService.pattern_relations(content)
    assert [(item.source_entity_name, item.target_entity_name, item.relationship) for item in relations] == [
        ("Alice Davis", "Acme Corp", "employed_by"),
        ("Bob Smith", "Orbit Labs", "advises"),
        ("Carol Jones", "Beacon Labs", "invested_in"),
        ("Dana Price", "Cedar Labs", "founded"),
        ("Evan Reed", "Summit Meeting", "attended"),
    ]


def test_knowledge_work_metadata_maps_to_normalized_evidence_candidates():
    extraction = extract_metadata_relations(
        {
            "knowledge_work": {
                "subject": {
                    "name": "Atlas Cutover",
                    "type": "work_item",
                    "aliases": ["CUT-42"],
                },
                "owner": {"name": "Alice Chen", "type": "person"},
                "assignees": ["Bob Singh"],
                "project": "Atlas Program",
                "depends_on": {"name": "Security Sign-off", "type": "work_item"},
                "blocked_by": "Legal Review",
                "based_on": "Migration Design",
                "about": "Customer Identity",
            }
        }
    )

    assert extraction.errors == []
    assert [
        (item.source.name, item.target.name, item.relationship)
        for item in extraction.candidates
    ] == [
        ("Alice Chen", "Atlas Cutover", "owns"),
        ("Bob Singh", "Atlas Cutover", "responsible_for"),
        ("Atlas Cutover", "Atlas Program", "part_of"),
        ("Atlas Cutover", "Security Sign-off", "depends_on"),
        ("Atlas Cutover", "Migration Design", "based_on"),
        ("Atlas Cutover", "Customer Identity", "about"),
        ("Legal Review", "Atlas Cutover", "blocks"),
    ]


def test_knowledge_work_metadata_is_reserved_tolerant_and_deduplicated():
    absent = extract_metadata_relations({"owner": "Alice"})
    assert absent.candidates == []
    assert absent.errors == []

    malformed = extract_metadata_relations(
        {
            "knowledge_work": {
                "subject": "Launch Brief",
                "author": "Alice",
                "authors": ["Alice"],
                "reviewers": [42, "Bob"],
            }
        }
    )
    assert [(item.source.name, item.relationship) for item in malformed.candidates] == [
        ("Alice", "authored"),
        ("Bob", "reviewed"),
    ]
    assert malformed.errors == ["knowledge_work.reviewers[0]: entity must be a string or object"]


def test_checked_in_knowledge_work_benchmark_has_exact_acquisition_qrels():
    path = Path(__file__).parents[2] / "benchmarks" / "datasets" / "knowledge_work_relations_v1.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert len(dataset["records"]) >= 20
    assert len(dataset["queries"]) >= 35
    for record in dataset["records"]:
        extraction = extract_metadata_relations(record.get("metadata"))
        predicted = {
            (
                item.source.name or item.source.entity_id,
                item.target.name or item.target.entity_id,
                item.relationship,
            )
            for item in extraction.candidates
        }
        expected = {tuple(item) for item in record.get("gold_relations") or []}
        assert extraction.errors == [], record["id"]
        assert predicted == expected, record["id"]


@pytest.mark.asyncio
async def test_relation_recall_is_a_noop_for_unsupported_storage():
    class UnsupportedStorage:
        @staticmethod
        def supports_capability(capability: str) -> bool:
            assert capability == "entity_relations"
            return False

        async def find_entities_by_normalized_name_any_type(self, *_args, **_kwargs):
            raise AssertionError("unsupported storage must not be queried")

    result = await EntityRelationService(UnsupportedStorage()).recall(
        "ws_test",
        "Who works at Acme Corp?",
    )

    assert result.memories == []
    assert result.paths == []
