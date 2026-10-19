import re
from typing import Dict, Optional

from app.models.job import JobDescription
from app.models.resume import ParsedResume

INTERN_WEIGHT = 0.4          # internship years count at 40% of full-time years (documented policy)
_GENERIC_TITLE_WORDS = ("senior", "junior", "lead", "associate", "role", "position", "trainee", "intern")


class ExperienceMatcherService:
    """
    Experience is scored ONLY when the JD establishes a minimum-experience requirement; otherwise the
    component is NOT APPLICABLE (is_applicable() is False) and the ranker excludes it and renormalises the
    remaining weights. No default/fabricated score is ever produced for a JD without an experience requirement.

    When applicable:
      confirmed_years = full-time work years + INTERN_WEIGHT * dated internship years
        - internships without any dates are UNVERIFIED and add nothing
        - projects, training and education never count as professional experience
        - overlapping employment periods are merged by the extractor (no period counted twice)
      base  = min(confirmed_years / minimum, 1.0)
      range = if the JD states a maximum and confirmed_years exceeds it: base *= max(0.5, maximum / confirmed_years)
      role  = +0.15 (or +0.08) when previous work/internship titles match the JD title; only with confirmed_years > 0
      score = min(base + role, 1.0)
    """

    def is_applicable(self, jd: JobDescription) -> bool:
        return bool(jd.has_experience_requirement and jd.minimum_experience_years > 0)

    @staticmethod
    def confirmed_years(resume: ParsedResume) -> float:
        work = max(getattr(resume, "work_experience_years", 0.0) or 0.0, 0.0)
        intern = getattr(resume, "internship_years", 0.0) or 0.0
        intern = intern if intern > 0 else 0.0           # -0.01 marker = undated internship = unverified
        if work == 0.0 and intern == 0.0:
            return max(resume.experience_years or 0.0, 0.0)   # legacy records that only stored a total
        return work + INTERN_WEIGHT * intern

    def evaluate_detailed(self, resume: ParsedResume, jd: JobDescription) -> Dict:
        if not self.is_applicable(jd):
            return {"applicable": False, "score": None, "confirmed_years": self.confirmed_years(resume),
                    "reason": "JD does not establish a minimum experience requirement"}
        req = jd.minimum_experience_years
        total = self.confirmed_years(resume)
        base = min(total / req, 1.0)
        mx: Optional[float] = jd.maximum_experience_years
        over_factor = 1.0
        if mx and total > mx:
            over_factor = max(0.5, mx / total)
        base *= over_factor

        bonus = 0.0
        jd_title = (jd.title or "").lower()
        if total > 0 and jd_title:
            kws = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", jd_title) if w not in _GENERIC_TITLE_WORDS]
            exp_text = (resume.section_texts.get("experience", "") + " " + resume.section_texts.get("internship", "")).lower()
            titles = [t.lower() for t in resume.job_titles]
            hit = sum(1 for w in kws if w in exp_text or any(w in t for t in titles))
            if kws and hit / len(kws) >= 0.5:
                bonus = 0.15
            elif kws and any(w in exp_text for w in kws):
                bonus = 0.08
        score = round(min(base + bonus, 1.0), 4)
        return {"applicable": True, "score": score, "confirmed_years": round(total, 2), "required_years": req,
                "maximum_years": mx, "over_maximum_factor": round(over_factor, 3), "role_bonus": bonus}

    def evaluate_experience(self, resume: ParsedResume, jd: JobDescription) -> float:
        """Float in [0,1]. Returns 0.0 when not applicable - callers must check is_applicable() first."""
        d = self.evaluate_detailed(resume, jd)
        return float(d["score"]) if d["applicable"] else 0.0
