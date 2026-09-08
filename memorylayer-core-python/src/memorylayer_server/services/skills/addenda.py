"""Skill addenda composition helpers.

Accepted ``skill_addendum`` memories let real-use corrections ride alongside a
canonical skill without mutating the skill manifest itself.
"""

from collections.abc import Sequence

from ...models.memory import Memory
from ...models.skill import Skill
from ..storage import StorageBackend

SKILL_ADDENDUM_SUBTYPE = "skill_addendum"
SKILL_ADDENDUM_ACCEPTED_STATUS = "accepted"
SKILL_ADDENDUM_SECTION_TITLE = "## Learned Addenda"
SKILL_ADDENDUM_LIMIT = 25


async def load_accepted_addenda(
    storage: StorageBackend,
    workspace_id: str,
    skill_id: str,
    *,
    limit: int = SKILL_ADDENDUM_LIMIT,
) -> list[Memory]:
    """Load accepted addenda memories for one skill, oldest first."""
    memories = await storage.search_memories_by_filter(
        workspace_id=workspace_id,
        subtypes=[SKILL_ADDENDUM_SUBTYPE],
        metadata_filter={"skill_id": skill_id, "status": SKILL_ADDENDUM_ACCEPTED_STATUS},
        limit=limit,
    )
    return sorted(memories, key=lambda memory: (memory.created_at, memory.id))


def compose_body_with_addenda(body: str, addenda: Sequence[Memory]) -> str:
    """Append accepted addenda to a skill body as a deterministic markdown section."""
    items = [_format_addendum(memory.content) for memory in addenda]
    items = [item for item in items if item]
    if not items:
        return body

    section = f"{SKILL_ADDENDUM_SECTION_TITLE}\n\n" + "\n".join(items)
    body_text = body.rstrip()
    if not body_text:
        return section
    return f"{body_text}\n\n{section}"


async def compose_skill_with_addenda(storage: StorageBackend, skill: Skill) -> Skill:
    """Return a copy of ``skill`` whose body includes accepted addenda."""
    addenda = await load_accepted_addenda(storage, skill.workspace_id, skill.id)
    body = compose_body_with_addenda(skill.body, addenda)
    if body == skill.body:
        return skill
    return skill.model_copy(update={"body": body})


async def compose_skills_with_addenda(storage: StorageBackend, skills: Sequence[Skill]) -> list[Skill]:
    """Return copies of skills with accepted addenda composed into each body."""
    return [await compose_skill_with_addenda(storage, skill) for skill in skills]


def _format_addendum(content: str) -> str:
    lines = [line.rstrip() for line in content.strip().splitlines()]
    if not lines:
        return ""
    first, *rest = lines
    formatted = [f"- {first}"]
    formatted.extend(f"  {line}" if line else "" for line in rest)
    return "\n".join(formatted)
