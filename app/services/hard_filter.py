from typing import Dict, List, Tuple

from app.config import config
from app.models.job import JobDescription
from app.models.resume import ParsedResume
from app.services.requirements import build_slots
from app.utils.degree_normalizer import degree_rank


class HardFilterService:
    """Mandatory eligibility only. Preferred, supporting, trainable and explicitly non-mandatory skills never
    reject a candidate, and experience is checked only when the JD establishes an experience requirement."""

    def __init__(self, enabled: bool | None = None, strict: bool | None = None):
        self.enabled = config.hard_filter_enabled if enabled is None else enabled
        self.strict = config.strict_hard_filter if strict is None else strict

    def evaluate(self, resume: ParsedResume, jd: JobDescription) -> Dict:
        """Returns {eligible, reasons, missing_mandatory}. Eligibility is independent of the fit score."""
        if not self.enabled:
            return {"eligible": True, "reasons": [], "missing_mandatory": [], "filter_enabled": False}
        return self.assess(resume, jd)

    def assess(self, resume: ParsedResume, jd: JobDescription) -> Dict:
        """Eligibility report that ignores the `enabled` switch (used for reporting even when filtering is off).
        {eligible, reasons, missing_mandatory (ALL unmet mandatory slots, also in lenient mode),
         mandatory_requirements_met, filter_enabled}. In lenient mode (STRICT_HARD_FILTER=false) a candidate is only
        rejected when NONE of the mandatory slots is met, but unmet slots are still reported so a good fit score
        can never hide a missing mandatory skill."""
        have = set(resume.skills)
        reasons: List[str] = []
        missing: List[str] = []

        mandatory_slots = build_slots(jd.mandatory_skills, jd)
        if mandatory_slots:
            unmet = [s for s in mandatory_slots if not s.satisfied_by(have)]
            missing = [s.label for s in unmet]
            if self.strict and unmet:
                reasons.append(f"Missing mandatory skills: {', '.join(missing)}")
            elif not self.strict and len(unmet) == len(mandatory_slots):
                reasons.append("Candidate possesses none of the required technical skills")

        if jd.has_experience_requirement and jd.minimum_experience_years > 0:
            threshold = jd.minimum_experience_years if self.strict else max(0.0, jd.minimum_experience_years * 0.4)
            if resume.experience_years < threshold:
                reasons.append(f"Experience {resume.experience_years:.1f} yrs below minimum threshold {threshold:.1f} yrs")

        if self.strict and jd.education_requirements:
            req_rank = min(degree_rank(d) for d in jd.education_requirements)   # any listed degree is accepted
            cand_rank = max([degree_rank(d) for d in resume.degrees], default=0)
            if cand_rank < req_rank:
                reasons.append(f"Education tier {cand_rank} below required {req_rank}")

        return {"eligible": not reasons, "reasons": reasons, "missing_mandatory": missing,
                "mandatory_requirements_met": not missing and not reasons, "filter_enabled": True}

    def passes_filter(self, resume: ParsedResume, jd: JobDescription) -> Tuple[bool, str]:
        """Backward-compatible: (passes, rejection_reason)."""
        if not self.enabled:
            return True, "Filter disabled"
        r = self.evaluate(resume, jd)
        return r["eligible"], ("; ".join(r["reasons"]) if r["reasons"] else "Passed")

    def filter_candidates(self, resumes: List[ParsedResume], jd: JobDescription) -> Tuple[List[ParsedResume], List[Tuple[str, str]]]:
        passed: List[ParsedResume] = []
        rejected: List[Tuple[str, str]] = []
        for r in resumes:
            ok, reason = self.passes_filter(r, jd)
            if ok:
                passed.append(r)
            else:
                rejected.append((r.candidate_id, reason))
        return passed, rejected
