"""Unit tests for the KB citation lint + grounding-coverage metric (numbered refs)."""

from memorylayer_server.services.knowledgebase.citations import (
    CitationReport,
    audit_citations,
    summary_snippet,
)


def test_empty_summary_is_zeroed():
    r = audit_citations("", 5)
    assert r.total == 0 and r.valid == 0 and r.invalid == 0
    assert r.distinct_members == 0 and r.coverage == 0.0
    assert r.clean is True  # nothing invented
    assert r.as_metadata() == {
        "citation_coverage": 0.0,
        "cited_member_count": 0,
        "invalid_citation_count": 0,
    }


def test_all_valid_refs_full_coverage():
    summary = "Acme ships widgets [m1]. It grew last year [m2]."
    r = audit_citations(summary, 3)
    assert r.total == 2 and r.valid == 2 and r.invalid == 0
    assert r.distinct_members == 2
    assert r.sentences == 2 and r.sentences_cited == 2
    assert r.coverage == 1.0 and r.clean is True


def test_out_of_range_ref_flagged():
    # only 3 members shown; [m9] is invented.
    summary = "Acme is real [m1]. Nonsense claim [m9]."
    r = audit_citations(summary, 3)
    assert r.total == 2 and r.valid == 1 and r.invalid == 1
    assert r.invalid_ids == ["m9"]
    assert r.clean is False
    assert r.as_metadata()["invalid_citation_count"] == 1


def test_does_not_match_domain_bracket_refs():
    # bare [12] (a domain "Reference [12]") is NOT a member citation — only [m<n>] is.
    r = audit_citations("Reference [12] was cited [m1].", 5)
    assert r.total == 1 and r.valid == 1 and r.invalid == 0


def test_partial_coverage():
    summary = "First uncited claim. Second cited [m1]. Third uncited."
    r = audit_citations(summary, 3)
    assert r.sentences == 3 and r.sentences_cited == 1
    assert round(r.coverage, 3) == 0.333


def test_distinct_members_dedupes_repeated_ref():
    summary = "A [m1]. B [m1]. C [m2]."
    r = audit_citations(summary, 5)
    assert r.total == 3 and r.valid == 3
    assert r.distinct_members == 2  # m1 counted once


def test_report_is_dataclass_default():
    r = CitationReport()
    assert r.coverage == 0.0 and r.clean is True and r.invalid_ids == []


def test_summary_snippet_strips_refs_and_takes_first_sentence():
    assert (
        summary_snippet("Acme makes widgets [m1]. It grew [m2].")
        == "Acme makes widgets."
    )


def test_summary_snippet_no_trailing_space_before_punctuation():
    assert summary_snippet("A fact [m3], really.") == "A fact, really."


def test_summary_snippet_empty_and_truncation():
    assert summary_snippet("") == "" and summary_snippet("   ") == ""
    long = summary_snippet("y" * 400)
    assert long.endswith("…") and len(long) == 200
