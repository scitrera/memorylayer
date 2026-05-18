"""SkillsResolutionService: precedence-based skill lookup.

Implements the USER > WORKSPACE > TENANT > GLOBAL scope ordering with
source_mode tie-breaking (server > mirrored > filesystem) described in
the agentskills plan.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...models.skill import Skill
    from ..storage import StorageBackend

# Workspace IDs with special scope meanings
_GLOBAL_WORKSPACE_ID = "_global"

_SCOPE_RANK: dict[str, int] = {
    "user": 0,
    "workspace": 1,
    "tenant": 2,
    "global": 3,
}

_MODE_RANK: dict[str, int] = {
    "server": 0,
    "mirrored": 1,
    "filesystem": 2,
}


def _scope_rank(skill: "Skill", ctx_workspace_id: str, ctx_user_id: Optional[str]) -> int:
    """Return the scope rank of a skill given the request context."""
    if skill.user_id and skill.user_id == ctx_user_id:
        return _SCOPE_RANK["user"]
    if skill.workspace_id == _GLOBAL_WORKSPACE_ID:
        return _SCOPE_RANK["global"]
    if skill.workspace_id == ctx_workspace_id:
        return _SCOPE_RANK["workspace"]
    return _SCOPE_RANK["tenant"]


def _scope_name(skill: "Skill", ctx_workspace_id: str, ctx_user_id: Optional[str]) -> str:
    """Return the string scope name of a skill given the request context."""
    if skill.user_id and skill.user_id == ctx_user_id:
        return "user"
    if skill.workspace_id == _GLOBAL_WORKSPACE_ID:
        return "global"
    if skill.workspace_id == ctx_workspace_id:
        return "workspace"
    return "tenant"


def _mode_rank(skill: "Skill") -> int:
    return _MODE_RANK.get(skill.source_mode, 99)


class RequestContext:
    """Lightweight context object carrying workspace/user identity for skill resolution.

    In the FastAPI layer this is built from auth headers; in unit tests
    it can be constructed directly.
    """

    __slots__ = ("workspace_id", "user_id", "tenant_id")

    def __init__(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        tenant_id: str = "",
    ) -> None:
        self.workspace_id = workspace_id
        self.user_id = user_id
        self.tenant_id = tenant_id


class SkillsResolutionService:
    """Resolves skills by name with deterministic scope precedence.

    Precedence: user > workspace > tenant > global.
    Within scope: server > mirrored > filesystem, then most-recent updated_at.

    Enterprise can subclass and override ``visible_scopes_for`` to inject
    RBAC-filtered visibility without changing resolution logic.
    """

    def __init__(self, storage: "StorageBackend") -> None:
        self._storage = storage

    def visible_scopes_for(self, ctx: RequestContext) -> list[dict]:
        """Return the ordered list of scope dicts to search for a given context.

        Each dict contains ``workspace_id`` and optional ``user_id``.
        """
        scopes: list[dict] = []
        if ctx.user_id:
            scopes.append({"workspace_id": ctx.workspace_id, "user_id": ctx.user_id})
        scopes.append({"workspace_id": ctx.workspace_id})
        scopes.append({"workspace_id": _GLOBAL_WORKSPACE_ID})
        return scopes

    async def resolve(
        self,
        name: str,
        ctx: RequestContext,
        scope_hint: Optional[str] = None,
    ) -> "Optional[Skill]":
        """Return the precedence-winning skill for the given name + context.

        Args:
            name: Exact skill name to resolve.
            ctx: Request context carrying workspace/user identity.
            scope_hint: Optional scope to restrict candidates to before precedence
                sort.  Accepted values: ``"user"``, ``"workspace"``, ``"global"``.
                Unknown values are silently ignored so forward-compatible callers
                (e.g. Enterprise with extra scopes) still work.
        """
        scopes = self.visible_scopes_for(ctx)
        candidates = await self._storage.find_skills_by_name(name, scopes)
        if not candidates:
            return None
        if scope_hint:
            candidates = self._filter_by_scope(candidates, ctx, scope_hint)
            if not candidates:
                return None
        return self._rank(candidates, ctx)[0]

    def apply_shadowing(
        self,
        skills: "list[Skill]",
        ctx: RequestContext,
        scope_hint: Optional[str] = None,
    ) -> "list[Skill]":
        """Given a list of skills, return only the precedence winner per name.

        Used by GET /v1/skills when ``include_shadowed=false`` (default).

        Args:
            skills: Full candidate list (possibly from multiple scopes).
            ctx: Request context carrying workspace/user identity.
            scope_hint: If provided, restrict each name's candidates to this
                scope before selecting the winner.  Same values as
                :meth:`resolve`.
        """
        by_name: dict[str, list["Skill"]] = {}
        for s in skills:
            by_name.setdefault(s.name, []).append(s)

        result = []
        for name_skills in by_name.values():
            filtered = self._filter_by_scope(name_skills, ctx, scope_hint) if scope_hint else name_skills
            if filtered:
                result.append(self._rank(filtered, ctx)[0])
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_scope(
        self,
        candidates: "list[Skill]",
        ctx: RequestContext,
        scope_hint: str,
    ) -> "list[Skill]":
        """Return only candidates whose effective scope matches ``scope_hint``."""
        target = scope_hint.lower()
        return [s for s in candidates if _scope_name(s, ctx.workspace_id, ctx.user_id) == target]

    def _rank(
        self,
        candidates: "list[Skill]",
        ctx: RequestContext,
    ) -> "list[Skill]":
        """Sort candidates by (scope_rank, mode_rank, -updated_at) ascending."""
        return sorted(
            candidates,
            key=lambda s: (
                _scope_rank(s, ctx.workspace_id, ctx.user_id),
                _mode_rank(s),
                -s.updated_at.timestamp(),
            ),
        )
