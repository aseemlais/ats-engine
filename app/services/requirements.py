"""Requirement slots: one slot per required skill, or ONE slot for a group of accepted alternatives."""
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from app.models.job import JobDescription


@dataclass(frozen=True)
class Slot:
    members: Tuple[str, ...]          # a single skill, or accepted alternatives (any one satisfies the slot)

    @property
    def label(self) -> str:
        return self.members[0] if len(self.members) == 1 else " / ".join(self.members) + " (any one)"

    def satisfied_by(self, skills: Set[str]) -> List[str]:
        return [m for m in self.members if m in skills]


def build_slots(skills: Iterable[str], jd: JobDescription) -> List[Slot]:
    """Slots for `skills`; groups from jd.skill_alternatives with >= 2 members in `skills` collapse to one slot."""
    pool = set(skills)
    slots: List[Slot] = []
    used: Set[str] = set()
    for g in jd.skill_alternatives:
        members = tuple(sorted(m for m in g if m in pool))
        if len(members) >= 2 and not (set(members) & used):
            slots.append(Slot(members)); used.update(members)
    slots.extend(Slot((s,)) for s in sorted(pool - used))
    return slots
