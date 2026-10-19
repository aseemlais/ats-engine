import re
from typing import Dict, List, Optional

from app.models.resume import ParsedResume
from app.models.job import JobDescription
from app.config import config
from app.utils.skill_normalizer import extract_skills_from_text

_CERT_RE = re.compile(r"\b(?:certif|licen[cs]e|credential|accredit)\w*", re.I)
_PREF_MARK_RE = re.compile(r"\b(?:nice\s+to\s+have|preferred|bonus|plus|desirable|good\s+to\s+have|advantage|optional)\b", re.I)
_REQ_MARK_RE = re.compile(r"\b(?:required|must|mandatory|essential|need(?:ed)?)\b", re.I)
_REQ_HDR_RE = re.compile(r"(?:requirements?|must\s+have|qualifications|required)", re.I)
_SKIP_HDR_RE = re.compile(r"(?:responsibilit|what\s+you.?ll\s+do|benefits|about|perks|what\s+we\s+offer)", re.I)
_CERT_NEG_RE = re.compile(r"\b(?:no|not|without|neither)\b[^.\n]{0,40}?(?:certif|licen[cs]e|credential)|(?:certif\w*|licen[cs]e\w*|credential\w*)[^.\n]{0,30}\b(?:not|isn't|is\s+not)\s+(?:required|mandatory|necessary|needed)", re.I)
_BENEFIT_RE = re.compile(r"\b(?:reimburs|stipend|sponsor|fund|support|paid\s+for)\w*", re.I)


def certification_requirement_level(jd_text: str) -> str:
    """'required' | 'preferred' | 'none' for certifications, from the JD text (deterministic, line/section based).
    Lines under responsibilities/benefits/about headers and benefit-style lines ("certification
    reimbursement") are ignored. A cert line is 'preferred' under a nice-to-have header or with a
    preferred marker, otherwise 'required' only if it carries a requirement marker or sits under a
    requirements header; anything else mentioning certification is treated as 'preferred' (soft)."""
    level, section = "none", "neutral"
    for raw in (jd_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        is_header = (len(line.split()) <= 6 and (line.endswith(":") or line.isupper())) and not _CERT_RE.search(line)
        if is_header:
            if _PREF_MARK_RE.search(line):
                section = "preferred"
            elif _SKIP_HDR_RE.search(line):
                section = "skip"
            elif _REQ_HDR_RE.search(line):
                section = "required"
            else:
                section = "neutral"
            continue
        if section == "skip" or not _CERT_RE.search(line) or _BENEFIT_RE.search(line):
            continue
        if _CERT_NEG_RE.search(line):
            continue  # "no certification required", "certification is not mandatory" -> not a requirement
        if _PREF_MARK_RE.search(line) or section == "preferred":
            this = "preferred"
        elif _REQ_MARK_RE.search(line) or section == "required":
            this = "required"
        else:
            this = "preferred"
        if this == "required":
            return "required"
        level = "preferred"
    return level


_DOMAIN_TERMS = ("data", "analytics", "analysis", "machine learning", "cloud", "database", "sql", "python", "statistics",
                 "engineering", "ai", "bi", "business intelligence", "big data")


class CertificationMatcherService:
    """
    Certification score in [0, 1] from RELEVANCE, not from the mere presence of certificates.
      per certificate: 1.0 if it names a skill the JD asks for (any tier, aliases/implications resolved),
                       0.5 if it only shares a domain keyword with the JD, 0.0 if unrelated
      score = min(1.0, 0.5 * sum(relevance))   -> two directly relevant certificates saturate the component
    Whether the score matters is decided by the JD (requirement_level()): 'none' -> not applicable,
    'preferred' -> a held relevant certificate can only help, 'required' -> absence scores 0.0.
    The embedding model is no longer used here (deterministic and explainable); embedding_service is accepted
    only for backward compatibility.
    """

    def __init__(self, embedding_service=None):
        self.embedding_service = embedding_service

    def requirement_level(self, jd: JobDescription) -> str:
        return certification_requirement_level(jd.raw_text)

    @staticmethod
    def _cert_items(resume: ParsedResume) -> List[str]:
        items = [c for c in (resume.certifications or []) if c and c.strip()]
        sec = resume.section_texts.get("certifications", "").strip()
        if not items and sec:
            items = [l.strip("-• ").strip() for l in sec.splitlines() if len(l.strip()) > 3]
        if not items:  # no section: look for explicit certificate lines in the text (conservative)
            items = [l.strip() for l in resume.cleaned_text.splitlines()
                     if re.search(r"\b(?:certified|certification|certificate)\b", l, re.I) and len(l) < 160][:5]
        return items[:10]

    def evaluate_detailed(self, resume: ParsedResume, jd: JobDescription) -> Dict:
        level = self.requirement_level(jd)
        items = self._cert_items(resume)
        jd_skills = set(jd.required_skills) | set(jd.preferred_skills) | set(jd.supporting_skills)
        jd_low = (jd.raw_text or "").lower() + " " + (jd.title or "").lower()
        relevant, partial, unrelated = [], [], []
        total = 0.0
        for it in items:
            skills = set(extract_skills_from_text(it, expand=True))
            if skills & jd_skills:
                total += 1.0; relevant.append(it)
            elif any(t in it.lower() and t in jd_low for t in _DOMAIN_TERMS):
                total += 0.5; partial.append(it)
            else:
                unrelated.append(it)
        score = round(min(1.0, 0.5 * total), 4)
        return {"level": level, "applicable": level != "none", "score": score, "has_certifications": bool(items),
                "relevant": relevant, "partially_relevant": partial, "unrelated": unrelated}

    def evaluate_certifications(self, resume: ParsedResume, jd: JobDescription) -> float:
        """Relevance score in [0,1]. Legacy policy (NA_POLICY=legacy) restores the 0.60 no-evidence baseline."""
        d = self.evaluate_detailed(resume, jd)
        if config.na_policy == "legacy" and not d["has_certifications"]:
            has_req = bool(re.search(r"\b(?:certif|license|credential|accredit)\w*", jd.raw_text, re.I))
            return 0.60 if not has_req else 0.0
        return d["score"]
