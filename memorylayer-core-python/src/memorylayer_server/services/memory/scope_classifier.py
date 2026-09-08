"""Preference-vs-episodic scope classifier (Slice 2 of cross-workspace user scope).

This module decides whether an incoming memory whose caller gave NO explicit
``scope`` is a *durable user preference / personality trait* (route to USER
scope so it follows the user across workspaces) or an *episodic / workspace
fact* (stay workspace-scoped). It feeds ``MemoryService._route_user_scope``;
the routing seam, not the classifier, owns the user_id guard and the actual
workspace rewrite.

Two tiers ship:

* OSS (this file): :class:`HeuristicScopeClassifier` — a conservative,
  dependency-free, deterministic heuristic. It only fires on first/second
  person *durable* preference patterns ("I prefer ...", "I always ...",
  persona/settings statements) and is intentionally LOW-RECALL: a false
  negative is cheap (the memory just stays workspace-scoped), a false positive
  is expensive (it pollutes the user's cross-workspace profile), so the
  heuristic errs hard toward *episodic*.
* Enterprise: an LLM-backed classifier that subclasses ``MemoryService`` and
  overrides ``_classify_user_scope`` (prompt-injection-hardened, temp 0, JSON
  out, bounded tokens). See ``memorylayer_saas`` enterprise memory service.

CARDINAL invariants every classifier must honour (the routing seam enforces the
fail-safe by treating ``None``/low-confidence as "not a user preference"):

* When uncertain, classify NOT a user preference (episodic by default).
* Never raise; failure modes degrade to "not a user preference".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ScopeClassification:
    """Result of a preference-vs-episodic classification.

    Attributes:
        is_user_preference: True iff the content reads as a durable user
            preference / personality trait that should follow the user.
        confidence: 0.0-1.0 confidence in ``is_user_preference``. The routing
            seam only promotes to USER scope when this meets the configured
            threshold, so a low-confidence ``True`` will NOT reroute.
        reason: Short human-readable rationale (for logs / debugging).
    """

    is_user_preference: bool
    confidence: float
    reason: str = ""


@runtime_checkable
class ScopeClassifier(Protocol):
    """Structural type for a scope classifier.

    Implementations MUST be conservative and MUST NOT raise: on any uncertainty
    or failure return ``ScopeClassification(is_user_preference=False, ...)``.
    """

    async def classify(self, content: str) -> ScopeClassification:
        ...


# Durable first/second-person preference / persona patterns. These are
# deliberately narrow: each requires a personal subject (I / my / me / you /
# your) bound to a durable preference/identity verb. Episodic phrasing
# ("we decided", "the build uses", "in project X we ...") is NOT matched.
#
# Matching is case-insensitive on a whitespace-normalised copy of the content.
_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "I prefer ...", "I like ...", "I love ...", "I hate ...", "I enjoy ..."
    re.compile(r"\bi (?:prefer|like|love|hate|dislike|enjoy|favou?r)\b", re.IGNORECASE),
    # "I always ...", "I never ...", "I usually ..." (durable habit)
    re.compile(r"\bi (?:always|never|usually|tend to|generally)\b", re.IGNORECASE),
    # "I don't like ...", "I do not want ..."
    re.compile(r"\bi (?:do not|don't|do n't) (?:like|want|use)\b", re.IGNORECASE),
    # "my preference is ...", "my favorite ...", "my preferred ..."
    re.compile(r"\bmy (?:preference|favou?rite|preferred|style|persona)\b", re.IGNORECASE),
    # "please always ...", "always respond ...", "never use ..." (standing directive)
    re.compile(r"\b(?:please )?(?:always|never) (?:respond|reply|use|answer|address|call|format)\b", re.IGNORECASE),
    # "call me ...", "address me as ..." (persona / how-to-address)
    re.compile(r"\b(?:call me|address me as|refer to me as)\b", re.IGNORECASE),
)

# Episodic / project-fact veto patterns. If any of these match, the heuristic
# refuses to classify as a user preference even if a preference cue is present
# — these denote a decision/fact about a project or a group, not a durable
# personal trait. This is the false-positive guard.
_EPISODIC_VETO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwe (?:decided|chose|agreed|will use|are using|switched|migrated)\b", re.IGNORECASE),
    re.compile(r"\b(?:the|this|our) (?:project|build|team|repo|service|app|system)\b", re.IGNORECASE),
    re.compile(r"\bin project\b", re.IGNORECASE),
)


class HeuristicScopeClassifier:
    """Conservative deterministic OSS classifier (no LLM, no network).

    Fires ONLY on narrow first/second-person durable-preference patterns and
    vetoes anything that looks episodic/project-scoped. Designed for low recall
    and ~zero false positives: when in doubt it returns ``is_user_preference``
    False so the memory stays workspace-scoped.

    A positive match is reported with a fixed, moderate confidence
    (``_MATCH_CONFIDENCE``). Operators who want the heuristic to actually
    promote memories must set the autoclassify threshold knob at or below that
    value (the default 0.85 threshold sits ABOVE it, so out of the box the OSS
    heuristic *proves the seam* without ever rerouting — enterprise's LLM
    classifier, returning calibrated higher confidences, makes it good).
    """

    # A heuristic pattern hit is suggestive, not authoritative; cap its
    # confidence below the conservative default threshold so OSS does not
    # silently reroute on a regex alone. Operators opt in by lowering the
    # threshold knob.
    _MATCH_CONFIDENCE = 0.80

    async def classify(self, content: str) -> ScopeClassification:
        """Return a conservative preference-vs-episodic verdict.

        Never raises: any unexpected input degrades to a non-preference verdict.
        """
        try:
            if not content or not content.strip():
                return ScopeClassification(False, 0.0, "empty content")

            # Normalise whitespace so multi-line content matches single-line
            # patterns and so crafted newlines cannot dodge the veto.
            normalised = re.sub(r"\s+", " ", content).strip()

            # False-positive guard first: episodic/project phrasing wins.
            for veto in _EPISODIC_VETO_PATTERNS:
                if veto.search(normalised):
                    return ScopeClassification(
                        False, 0.0, "episodic/project phrasing (veto)"
                    )

            for pat in _PREFERENCE_PATTERNS:
                if pat.search(normalised):
                    return ScopeClassification(
                        True, self._MATCH_CONFIDENCE, "durable preference pattern matched"
                    )

            return ScopeClassification(False, 0.0, "no durable preference pattern")
        except Exception:  # noqa: BLE001 - classification must never raise
            return ScopeClassification(False, 0.0, "classifier error (fail-safe)")


class NoopScopeClassifier:
    """A classifier that classifies nothing (always episodic).

    Useful as an explicit "off" implementation or for deployments that want the
    seam present but no heuristic behavior. Returns a non-preference verdict for
    all input so it can never reroute a memory.
    """

    async def classify(self, content: str) -> ScopeClassification:  # noqa: ARG002
        return ScopeClassification(False, 0.0, "noop classifier")
