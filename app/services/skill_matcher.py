from typing import Dict, List, Set, Tuple

from app.config import config
from app.models.job import JobDescription
from app.models.resume import ParsedResume
from app.services.requirements import build_slots

# Relative tier weights inside the skill score; tiers that are empty for a JD are dropped and the rest renormalised.
TIER_WEIGHTS = {"required": 0.70, "preferred": 0.20, "supporting": 0.10}


class SkillMatcherService:
    def is_applicable(self, jd: JobDescription) -> bool:
        """False when the JD yielded no recognisable skills at all (the score would be uninformative)."""
        return bool(jd.required_skills or jd.preferred_skills or jd.supporting_skills)

    def evaluate_detailed(self, resume: ParsedResume, jd: JobDescription) -> Dict:
        """
        Skill score = sum(tier_weight * tier_coverage) / sum(tier_weight of non-empty tiers).
          required   0.70  - every required skill is one slot; accepted alternatives ("SQL or MySQL") are ONE slot,
                             satisfied by any member; a skill absent from the resume earns nothing.
          preferred  0.20  - good-to-have skills (same slot logic)
          supporting 0.10  - trainable / concept-level / responsibility-only skills (lower weight, never eligibility)
        Skills the JD says are NOT mandatory never enter the required tier (see JDParserService).
        """
        have: Set[str] = set(resume.skills)
        tiers = {"required": jd.required_skills, "preferred": jd.preferred_skills, "supporting": jd.supporting_skills}
        coverage: Dict[str, float] = {}
        matched: Dict[str, List[str]] = {}
        missing: Dict[str, List[str]] = {}
        alt_used: List[str] = []
        for tier, skills in tiers.items():
            slots = build_slots(skills, jd)
            matched[tier], missing[tier] = [], []
            if not slots:
                continue
            met = 0
            for slot in slots:
                hit = slot.satisfied_by(have)
                if hit:
                    met += 1
                    matched[tier].extend(hit)
                    if len(slot.members) > 1:
                        alt_used.append(f"{slot.label}: satisfied by {', '.join(hit)}")
                else:
                    missing[tier].append(slot.label)
            coverage[tier] = met / len(slots)
        if config.na_policy == "legacy" and "preferred" not in coverage:
            coverage["preferred"] = 1.0 if not jd.required_skills else 0.5
        total_w = sum(TIER_WEIGHTS[t] for t in coverage)
        score = sum(TIER_WEIGHTS[t] * c for t, c in coverage.items()) / total_w if total_w else 0.0
        return {
            "score": min(max(score, 0.0), 1.0), "coverage": coverage,
            "matched_required": sorted(set(matched["required"])), "missing_required": missing["required"],
            "matched_preferred": sorted(set(matched["preferred"])), "missing_preferred": missing["preferred"],
            "matched_supporting": sorted(set(matched["supporting"])), "missing_supporting": missing["supporting"],
            "alternatives_satisfied": alt_used,
        }

    def evaluate_skills(self, resume: ParsedResume, jd: JobDescription) -> Tuple[float, List[str], List[str], List[str]]:
        """Backward-compatible tuple: (skill_score, matched_required, missing_required, matched_preferred)."""
        d = self.evaluate_detailed(resume, jd)
        return d["score"], d["matched_required"], d["missing_required"], d["matched_preferred"] + d["matched_supporting"]
