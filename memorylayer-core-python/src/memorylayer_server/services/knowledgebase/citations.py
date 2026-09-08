"""Citation lint + grounding-coverage metric for KB article summaries.

Community and entity summaries are prompted to ground each claim with NUMBERED
member references — ``[m1]``, ``[m2]`` … — that map to the numbered member block
fed to the model (see ``_build_member_block`` / the summary prompts). Numbered
references are used instead of raw ``[memory:<id>]`` because LLMs copy small
integers reliably but drop/mangle long opaque ids (and the ``m`` prefix keeps them
distinct from the domain's own ``[12]``-style references in the source text).

This module MEASURES how well a summary honored that contract against the number
of members shown — it does NOT rewrite the summary. The report is folded into
article metadata so the frontend / quality dashboards can surface grounding, and a
summary that cites out-of-range (invented) numbers is logged for follow-up.

Pure and I/O-free; parsing only.
"""

import re
from dataclasses import dataclass, field

# Numbered member reference: [m1], [m12] … The ``m`` prefix distinguishes a member
# citation from the domain's own bracketed references (e.g. "Reference [12]").
_CITATION_RE = re.compile(r"\[m(\d+)\]")

# Coarse sentence split for the coverage denominator. Good enough for a metric —
# we only need a stable count of claim-bearing sentences, not perfect segmentation.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class CitationReport:
    """How well a summary grounded its claims in numbered member references."""

    total: int = 0                       # every [m<n>] token emitted
    valid: int = 0                       # tokens whose number is in [1, member_count]
    invalid: int = 0                     # tokens whose number is out of range (invented)
    distinct_members: int = 0            # distinct in-range member numbers cited
    sentences: int = 0                   # claim-bearing sentences
    sentences_cited: int = 0             # sentences carrying >= 1 citation
    invalid_ids: list[str] = field(default_factory=list)  # deduped out-of-range refs, for logging

    @property
    def coverage(self) -> float:
        """Fraction of sentences that carry at least one citation (0.0-1.0)."""
        return (self.sentences_cited / self.sentences) if self.sentences else 0.0

    @property
    def clean(self) -> bool:
        """True when every citation is an in-range member number (nothing invented)."""
        return self.invalid == 0

    def as_metadata(self) -> dict:
        """Compact, JSON-safe fields for article metadata."""
        return {
            "citation_coverage": round(self.coverage, 3),
            "cited_member_count": self.distinct_members,
            "invalid_citation_count": self.invalid,
        }


def summary_snippet(text: str, max_len: int = 200) -> str:
    """First sentence of a summary as a clean one-line description/preview.

    Inline ``[m<n>]`` references are removed and whitespace collapsed — this derives
    a preview field (OKF ``description``) from already-good content; it is NOT
    model-output cleanup. Truncated with an ellipsis past ``max_len``. Never raises;
    empty input -> "".
    """
    if not text:
        return ""
    # Drop references AND any whitespace immediately preceding them so "x [m1]."
    # collapses to "x." rather than "x .".
    without_citations = re.sub(r"\s*\[m\d+\]", "", text)
    cleaned = re.sub(r"\s+", " ", without_citations).strip()
    if not cleaned:
        return ""
    first = _SENTENCE_SPLIT_RE.split(cleaned)[0].strip() or cleaned
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "…"
    return first


def audit_citations(summary: str, member_count: int) -> CitationReport:
    """Measure the grounding of ``summary`` against ``member_count`` numbered members.

    Args:
        summary: the generated summary/description text.
        member_count: how many members were shown to the model (the valid range is
            ``[m1] .. [m{member_count}]``); a reference outside it was invented.

    Returns:
        A ``CitationReport`` (never raises; empty/None summary -> zeroed report).
    """
    report = CitationReport()
    if not summary:
        return report

    invalid_seen: set[int] = set()
    distinct_valid: set[int] = set()
    for tok in _CITATION_RE.findall(summary):
        n = int(tok)
        report.total += 1
        if 1 <= n <= member_count:
            report.valid += 1
            distinct_valid.add(n)
        else:
            report.invalid += 1
            invalid_seen.add(n)
    report.distinct_members = len(distinct_valid)
    report.invalid_ids = [f"m{n}" for n in sorted(invalid_seen)]

    # Coverage: count claim-bearing sentences and how many carry a reference.
    for sentence in _SENTENCE_SPLIT_RE.split(summary.strip()):
        s = sentence.strip()
        if not s:
            continue
        report.sentences += 1
        if _CITATION_RE.search(s):
            report.sentences_cited += 1

    return report
