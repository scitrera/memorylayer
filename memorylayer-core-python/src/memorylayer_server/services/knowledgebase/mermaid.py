"""Mermaid diagrams for KB articles, generated through escaping and a validation gate.

A community's bridges are already listed as bullets, but a list does not show local
topology -- which neighbours a cluster touches, and how strongly. A small flowchart
does, at no LLM cost, from graph data already in hand.

The engineering here is not the drawing, it is the refusal to ship a broken one. An
INVALID diagram is worse than none: Obsidian renders a fenced error block where the
picture should be, so a stray quote in a community label would visibly corrupt the
page. Labels therefore go through :func:`escape_label`, and every assembled block is
checked by :func:`validate_flowchart` before a caller embeds it -- the check is what
makes the escaping load-bearing rather than merely hopeful.

Deliberately NOT a mermaid parser: it verifies the structure this module actually
emits (a header, at least one node, well-formed lines, no unescaped delimiters).
Being narrow is what makes it deterministic and dependency-free; a mermaid renderer
in Python would be neither.
"""

from __future__ import annotations

import re

__all__ = [
    "MERMAID_FENCE",
    "build_community_flowchart",
    "escape_label",
    "fenced",
    "node_id",
    "validate_flowchart",
]

MERMAID_FENCE = "```mermaid"

# Mermaid node ids must not carry syntax characters; anything else is folded to "_".
_ID_UNSAFE = re.compile(r"[^A-Za-z0-9_]")

# Characters that terminate or redirect a mermaid node/edge label. Quotes end the
# quoted label, brackets/braces/parens open shape syntax, and the rest are edge or
# direction operators.
_LABEL_UNSAFE = re.compile(r'["\[\]{}()<>|;`\\]')

# A rendered label is a caption, not a paragraph.
_MAX_LABEL_CHARS = 48

_VALID_LINE = re.compile(
    r"""^(
        flowchart\s+(TD|LR|TB|RL|BT)      # header
      | [A-Za-z0-9_]+\["[^"]*"\]          # node:  id["label"]
      | [A-Za-z0-9_]+\s-->\s(\|"[^"]*"\|\s)?[A-Za-z0-9_]+   # edge, optional label
    )$""",
    re.VERBOSE,
)


def node_id(raw: str | int, prefix: str = "n") -> str:
    """A mermaid-safe node id derived from ``raw``.

    Prefixed because a bare number is not a usable mermaid identifier.
    """
    return f"{prefix}{_ID_UNSAFE.sub('_', str(raw))}"


def escape_label(text: str, max_chars: int = _MAX_LABEL_CHARS) -> str:
    """Make ``text`` safe to place inside a quoted mermaid label.

    Collapses whitespace (a newline would end the statement), drops the characters
    that would escape the quoted label, and truncates to caption length. Never
    returns an empty string -- an empty label renders as an anonymous box.
    """
    collapsed = " ".join(str(text or "").split())
    cleaned = _LABEL_UNSAFE.sub("", collapsed).strip()
    if not cleaned:
        return "untitled"
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def fenced(block: str) -> str:
    """Wrap a validated diagram in a ```mermaid fence."""
    return f"{MERMAID_FENCE}\n{block}\n```"


def validate_flowchart(block: str) -> list[str]:
    """Structural problems with an assembled flowchart; empty means safe to embed.

    Checks what this module emits rather than the whole mermaid grammar: a leading
    ``flowchart`` header, at least one node so the diagram is not blank, and every
    line matching the node/edge forms built above (which is where an escaping miss
    would surface).
    """
    problems: list[str] = []
    lines = [ln.strip() for ln in (block or "").strip().splitlines() if ln.strip()]

    if not lines:
        return ["empty diagram"]
    if not lines[0].startswith("flowchart "):
        problems.append(f"first line is not a flowchart header: {lines[0]!r}")
    if not any(_VALID_LINE.match(ln) and "[" in ln for ln in lines):
        problems.append("diagram declares no nodes")
    problems.extend(f"malformed line: {ln!r}" for ln in lines if not _VALID_LINE.match(ln))
    return problems


def build_community_flowchart(
    community_id: int,
    community_label: str,
    neighbours: list[tuple[int, str, str, float]],
    max_neighbours: int = 12,
) -> str | None:
    """A flowchart of one community and the communities it bridges to.

    Args:
        community_id: The community this article is about (rendered as the centre).
        community_label: Its display label.
        neighbours: ``(other_id, other_label, relationship, strength)`` per bridge,
            already deduplicated by the caller.
        max_neighbours: Drawn-neighbour cap. A hub community can bridge to dozens of
            others and the picture stops being readable long before that. Truncating
            here loses nothing: the article's bullet list is the complete record, and
            this diagram is the overview of it.

    Returns:
        A fenced, validated mermaid block, or None when there is nothing worth
        drawing (no neighbours) or the assembled diagram fails validation -- callers
        omit the section rather than embed something that will not render.
    """
    if not neighbours:
        return None

    centre = node_id(community_id, prefix="c")
    lines = ["flowchart LR", f'{centre}["{escape_label(community_label)}"]']
    for other_id, other_label, relationship, strength in neighbours[:max_neighbours]:
        other = node_id(other_id, prefix="c")
        lines.append(f'{other}["{escape_label(other_label)}"]')
        edge_label = escape_label(f"{relationship} {strength:.2f}", max_chars=32)
        lines.append(f'{centre} --> |"{edge_label}"| {other}')

    block = "\n".join(lines)
    if validate_flowchart(block):
        return None
    return fenced(block)
