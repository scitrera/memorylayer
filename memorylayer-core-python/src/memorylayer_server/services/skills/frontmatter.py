"""SKILL.md frontmatter parsing and rendering utilities.

Parses the YAML-ish frontmatter block (between ``---`` fences) from a
SKILL.md file and renders it back with stable key ordering. Uses only
stdlib — no PyYAML dependency required.
"""
from __future__ import annotations

import re

# Canonical key order for rendered frontmatter (agentskills spec fields first)
_CANONICAL_KEY_ORDER = [
    "name",
    "description",
    "version",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
]


def parse_skill_md(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter dict and body from SKILL.md text.

    Supports the agentskills spec ``---`` fence format:
    ``---\\n<yaml>\\n---\\n<body>``.

    Returns a (frontmatter_dict, body) tuple. If no frontmatter is found,
    returns ({}, original text).
    """
    text = text or ""
    # Match opening --- fence at very start, optional trailing newline
    if not text.startswith("---"):
        return {}, text

    # Find closing ---
    rest = text[3:]
    # strip a leading newline after opening ---
    if rest.startswith("\n"):
        rest = rest[1:]

    close = re.search(r"^---[ \t]*$", rest, re.MULTILINE)
    if close is None:
        return {}, text

    fm_text = rest[: close.start()]
    body = rest[close.end() :]
    # strip a single leading newline from the body
    if body.startswith("\n"):
        body = body[1:]

    frontmatter = _parse_simple_yaml(fm_text)
    return frontmatter, body


def render_skill_md(frontmatter: dict, body: str) -> str:
    """Render a SKILL.md string from a frontmatter dict and body.

    Keys are emitted in canonical order (agentskills spec fields first),
    with any extra keys appended alphabetically.
    """
    lines = ["---"]
    ordered_keys = _CANONICAL_KEY_ORDER + sorted(
        k for k in frontmatter if k not in _CANONICAL_KEY_ORDER
    )
    for key in ordered_keys:
        if key not in frontmatter:
            continue
        value = frontmatter[key]
        if value is None:
            continue
        lines.append(f"{key}: {_yaml_scalar(value)}")
    lines.append("---")
    if body:
        lines.append(body if body.startswith("\n") else "\n" + body)
    else:
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Minimal YAML subset parser (key: value, no nesting, no lists)
# ---------------------------------------------------------------------------

def _parse_simple_yaml(text: str) -> dict:
    """Parse a flat YAML block (key: value pairs only).

    Handles:
    - bare scalars:   name: pdf-processing
    - quoted strings: description: "Extract tables"
    - multi-line values are not supported (single-line values only)
    """
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        result[key] = _unquote(raw_value)
    return result


def _unquote(value: str) -> str:
    """Strip surrounding quotes from a YAML scalar value."""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _yaml_scalar(value: object) -> str:
    """Format a scalar value for YAML emission."""
    s = str(value)
    # Quote if it contains YAML-special characters
    if any(c in s for c in (':', '#', '"', "'", '\n', '[')):
        escaped = s.replace('"', '\\"')
        return f'"{escaped}"'
    return s
