"""member_citation_ids must stay aligned with the [m1..mN] numbering.

The UI maps a citation [m{n}] to member_citation_ids(members)[n-1] to link it to a
memory, so the id order must match _build_member_block's numbering exactly —
including the skip of empty-content members and the budget cut.
"""

from memorylayer_server.services.knowledgebase.default import DefaultKnowledgebaseService


def _svc(budget: int = 10000) -> DefaultKnowledgebaseService:
    svc = DefaultKnowledgebaseService.__new__(DefaultKnowledgebaseService)
    svc.summary_budget_chars = budget
    return svc


def test_ids_skip_empty_content_and_align_with_block():
    svc = _svc()
    members = [
        {"id": "a", "content": "alpha"},
        {"id": "b", "content": "   "},   # blank → not numbered, not cited
        {"id": "c", "content": "gamma"},
    ]
    ids = svc.member_citation_ids(members)
    block, _snips, count = svc._build_member_block(members)

    assert ids == ["a", "c"]          # [m1]=a, [m2]=c (b skipped)
    assert count == 2
    assert "[m1] alpha" in block and "[m2] gamma" in block


def test_budget_cut_truncates_ids_the_same_way():
    # Tiny budget: only the first member survives the cut.
    svc = _svc(budget=1)
    members = [
        {"id": "a", "content": "alpha"},
        {"id": "b", "content": "beta"},
    ]
    ids = svc.member_citation_ids(members)
    _block, _snips, count = svc._build_member_block(members)
    assert ids == ["a"]
    assert count == 1


def test_missing_id_keeps_slot_for_alignment():
    svc = _svc()
    members = [{"content": "no id here"}, {"id": "b", "content": "beta"}]
    # The id-less member keeps its citation slot as "" so [m2] still maps to "b"
    # (a citation to the empty slot simply won't resolve — safe).
    assert svc.member_citation_ids(members) == ["", "b"]
