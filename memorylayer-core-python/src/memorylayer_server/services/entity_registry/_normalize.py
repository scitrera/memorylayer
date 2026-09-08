"""Deterministic entity-name normalization (parity contract).

``normalize_entity_name`` is the single source of truth for the normalized key
used by exact and alias resolution. The OSS relational backend and the
enterprise Postgres backend BOTH call this function so they agree byte-for-byte
on what "the same entity" means — normalization is done in Python (not in SQL)
specifically so the two backends cannot drift.

Normalization steps (order matters):
  1. Unicode NFKD decomposition, then strip combining marks (diacritics).
  2. casefold() for aggressive, locale-independent lowercasing.
  3. Strip possessive suffixes (``'s`` / ``’s`` and trailing bare ``'``).
  4. Replace any remaining punctuation/symbols with a space.
  5. Collapse all whitespace runs to a single space and strip ends.

The result is a compact, case/diacritic/punctuation-insensitive key. It is NOT
meant to be displayed — callers keep the original surface form as
``canonical_name`` / ``alias`` and use this only as the match key.
"""

import unicodedata

__all__ = ("normalize_entity_name",)

# Possessive suffixes to strip BEFORE punctuation removal (so "Alice's" and
# "Alice" collapse to the same key). Both straight and typographic apostrophes.
_POSSESSIVE_SUFFIXES = ("'s", "’s")


def normalize_entity_name(s: str) -> str:
    """Return the deterministic normalized match key for an entity name.

    Casefolds, strips diacritics (NFKD), removes possessives, replaces
    punctuation with whitespace, and collapses whitespace. Returns an empty
    string for falsy / whitespace-only input.

    This is the parity contract: any backend that resolves entities MUST use
    this exact function so normalized keys are identical across backends.
    """
    if not s:
        return ""

    # 1. NFKD decompose then drop combining marks (diacritics).
    decomposed = unicodedata.normalize("NFKD", s)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))

    # 2. Aggressive lowercase, then trim surrounding whitespace so the
    #    possessive-suffix check below sees the true trailing characters.
    folded = without_marks.casefold().strip()

    # 3. Strip possessive suffixes (single pass; longest match wins).
    for suffix in _POSSESSIVE_SUFFIXES:
        if folded.endswith(suffix):
            folded = folded[: -len(suffix)]
            break
    # Trailing bare apostrophe possessive (e.g. "James'").
    folded = folded.rstrip("'’")

    # 4. Replace any non-alphanumeric / non-space char with a space. This drops
    #    punctuation and symbols (hyphens, periods, commas, remaining quotes)
    #    deterministically regardless of script.
    cleaned = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in folded)

    # 5. Collapse whitespace runs and strip.
    return " ".join(cleaned.split())
