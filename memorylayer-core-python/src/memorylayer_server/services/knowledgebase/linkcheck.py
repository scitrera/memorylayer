"""Wikilink integrity for generated KB articles.

Articles cross-reference each other with Obsidian ``[[wikilink]]`` syntax. Those
links are built from graph structure (a bridge to community N, an association to
memory M), but the FILE a link must point at is named after the article that was
actually produced -- and an article id carries a disambiguation suffix the graph
edge knows nothing about. Nothing previously reconciled the two, so links were
emitted for targets that no article was ever written for.

Two responsibilities, both deterministic, I/O-free and testable in isolation:

1. **Link construction** (:class:`LinkIndex`). The set of vault paths a run will
   actually produce, keyed by the identifier the *renderer* has on hand (a
   community id, a target memory/entity id). The renderer asks the index for a
   path and links only when one comes back, so an unresolvable reference degrades
   to plain text instead of a dangling link. This is the fix, not just the check:
   a link can only be emitted for an article that exists.

2. **Link validation** (:func:`validate_articles`). A post-generation sweep over
   the rendered markdown that re-derives every ``[[target]]`` and asserts it
   resolves against the produced article set. This is the safety net for the
   paths construction cannot cover -- notably articles reused verbatim from a
   prior run (see ``_maybe_reuse_article``), which can outlive the article they
   link to when GC reclaims it.

``RENDER_FORMAT_VERSION`` is folded into the article content hashes so that a
change to link rendering busts the reuse cache exactly once; without it every
unchanged article would be reused verbatim and keep its stale links forever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bumped when the rendered output format changes in a way that must invalidate
# content-hash reuse.
#   "2" = links resolved through LinkIndex (previously every entity cross-link was
#         emitted without the article id's disambiguation suffix, so none resolved)
#   "3" = OKF front matter on god-node articles; mermaid topology diagram on
#         community articles
RENDER_FORMAT_VERSION = "3"

# [[target]] or [[target|display]]. The target is everything up to an optional
# pipe; a display segment may itself be empty.
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+?)(?:\|([^\]]*))?\]\]")

__all__ = [
    "RENDER_FORMAT_VERSION",
    "BrokenLink",
    "LinkIndex",
    "LinkReport",
    "article_vault_path",
    "extract_wikilinks",
    "validate_articles",
]


def article_vault_path(article_type: str, article_id: str) -> str:
    """Vault-relative path for an article, WITHOUT the ``.md`` extension.

    The single source of truth for article placement: link targets and export
    paths are both derived from it, so they cannot drift apart (they did before —
    the exporter suffixed entity ids while the renderer's links did not).
    """
    if article_type == "index":
        return "index"
    if article_type == "community":
        return f"communities/{article_id}"
    if article_type == "entity":
        return f"entities/{article_id.removeprefix('entity-')}"
    return article_id


@dataclass(frozen=True)
class BrokenLink:
    """One wikilink whose target resolves to no produced article."""

    source_path: str  # vault path of the article containing the link
    target: str       # link target exactly as written
    line: int         # 1-based line number within the source article


@dataclass
class LinkReport:
    """Outcome of one validation sweep over a generated vault."""

    total: int = 0
    broken: list[BrokenLink] = field(default_factory=list)

    @property
    def broken_count(self) -> int:
        return len(self.broken)

    @property
    def ok(self) -> bool:
        return not self.broken

    def as_metadata(self) -> dict:
        """Compact form for article metadata / quality dashboards."""
        return {
            "links_total": self.total,
            "links_broken": self.broken_count,
            # Sample only: a systemic break would otherwise write thousands of
            # rows into the index article's metadata blob.
            "links_broken_sample": [
                f"{b.source_path} -> {b.target}" for b in self.broken[:10]
            ],
        }


class LinkIndex:
    """Vault paths of the articles a generation run will actually produce.

    Populated BEFORE any article is rendered, so the renderer can resolve a
    cross-reference at the moment it emits it. Keys are the identifiers the
    renderer holds: community ids (int) and entity keys (the ``target_id`` on a
    connection — a memory id in OSS, a registry entity id in enterprise).
    """

    def __init__(self) -> None:
        self._entities: dict[str, str] = {}
        self._communities: dict[int, str] = {}

    def add_entity(self, key: str, article_id: str) -> None:
        """Register the entity article produced for ``key``."""
        if key:
            self._entities[key] = article_vault_path("entity", article_id)

    def add_community(self, community_id: int, article_id: str) -> None:
        """Register the community article produced for ``community_id``."""
        self._communities[community_id] = article_vault_path("community", article_id)

    def entity_path(self, key: str | None) -> str | None:
        """Vault path of the entity article for ``key``, or None if none exists."""
        if not key:
            return None
        return self._entities.get(key)

    def community_path(self, community_id: int | None) -> str | None:
        """Vault path of the community article for ``community_id``, or None."""
        if community_id is None:
            return None
        return self._communities.get(community_id)

    def known_targets(self) -> set[str]:
        """Every path this index can resolve to (plus the always-present index)."""
        return {"index", *self._entities.values(), *self._communities.values()}

    def __len__(self) -> int:
        return len(self._entities) + len(self._communities)


def extract_wikilinks(markdown: str) -> list[tuple[str, int]]:
    """Every ``[[target]]`` in ``markdown`` as ``(target, 1-based line)``."""
    found: list[tuple[str, int]] = []
    for lineno, line in enumerate(markdown.splitlines(), start=1):
        for match in _WIKILINK_RE.finditer(line):
            target = (match.group(1) or "").strip()
            if target:
                found.append((target, lineno))
    return found


def validate_articles(
    articles: dict[str, str],
    known_targets: set[str],
) -> LinkReport:
    """Check every wikilink in ``articles`` against the produced article set.

    Args:
        articles: Map of vault path (no ``.md``) -> rendered markdown.
        known_targets: Vault paths that exist. A link to anything else is broken.

    Returns:
        A :class:`LinkReport`; ``ok`` is True when no link dangles.
    """
    report = LinkReport()
    for source_path, markdown in sorted(articles.items()):
        for target, line in extract_wikilinks(markdown or ""):
            report.total += 1
            if target not in known_targets:
                report.broken.append(
                    BrokenLink(source_path=source_path, target=target, line=line)
                )
    return report
