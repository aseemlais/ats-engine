"""
Structured technical BM25 query (Stage 1 / P1).

The lexical component should measure technical/role relevance, so its query is built from the PARSED job
description - mandatory, preferred and supporting skills plus technical responsibility text - and never from
salary, shift, interview-process, location or similar administrative wording. The full JD text stays
available to eligibility, semantic matching and every other component; only the BM25 query is narrowed.

Query modes (config BM25_QUERY_MODE, default "structured" = the baseline, unchanged behaviour):
  structured  parsed skills + administrative-filtered responsibility lines (baseline)
  filtered    same, but clearly generic boilerplate words (soft-skill / HR / corporate filler such as "stakeholders",
              "fast-paced", "motivated", "ability") are also removed from the FREE responsibility text. Parsed skill terms and
              any taxonomy skill mention are never removed. The filter is a fixed list: deterministic, independent of the
              corpus and of the batch being scored.
(the full raw JD is not a production mode; scripts/bm25_query_strategy_compare.py uses it as a comparison baseline.)

Fallback: when the structured query is empty or shorter than MIN_QUERY_TOKENS, the full JD text is used with
administrative lines removed, and `fallback_reason` says why. Nothing here is specific to a technology or role.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from app.config import config
from app.models.job import JobDescription
from app.utils.skill_normalizer import extract_skill_mentions
from app.utils.text_cleaner import tokenize_for_bm25

MIN_QUERY_TOKENS = 4
QUERY_MODES = ("structured", "filtered")

# Clearly generic JD / HR / corporate filler: carries no information about whether a candidate can do the job in ANY domain.
# Deliberately conservative: technical-activity verbs (build, design, optimise, deploy, test, monitor, analyse), domain nouns
# (data, patient, campaign, ledger, ...) and anything in the skill taxonomy are NOT here. Extend only with words that are
# plainly non-discriminative; every removal is reported in BM25Query.removed_terms.
GENERIC_TERMS = frozenset("""
ability able abilities accountable adaptability adaptable attitude attributes
collaborate collaborates collaborating collaboration collaborative commitment committed communicate communication communicator
contribute contributes contributing contribution cross-functional culture
demonstrate demonstrated demonstrates diverse eager effective effectively energetic enthusiasm enthusiastic excellent exceptional
exposure fast-paced flexible good great highly ideal impactful independent independently individual initiative innovative leadership
long-term motivated motivation multitask multitasking organisation organization ownership passion passionate positive proactive
proficient proven quick responsible responsibility responsibilities results-oriented role roles self-motivated self-starter
stakeholder stakeholders strong successful supportive team teams teamwork thrive together trusted well willing willingness
""".split())
# Deliberately NOT listed because they carry domain meaning somewhere: performance, delivery, growth, environment, worker,
# company, candidate(s), opportunity, dynamic, driven, efficient, reliable, competitive, awareness, professional, smart, solid,
# work/working/workplace, career, best. Audit with scripts/bm25_query_strategy_compare.py (removed terms are listed per JD).


def _protected_tokens(line: str, skills: List[str]) -> set:
    """Tokens that must survive filtering: parsed skill terms and any taxonomy skill mention in the line."""
    prot = set()
    for sk in skills:
        prot.update(tokenize_for_bm25(sk, query=True))
    for start, end, _canon in extract_skill_mentions(line):
        prot.update(tokenize_for_bm25(line[start:end], query=True))
    return prot


def filter_generic(line: str, skills: List[str]) -> Tuple[str, List[str]]:
    """(filtered_line, removed_terms). Only GENERIC_TERMS not protected by the skill list/taxonomy are removed."""
    prot = _protected_tokens(line, skills)
    kept, removed = [], []
    for tok in tokenize_for_bm25(line, query=True):
        if tok in GENERIC_TERMS and tok not in prot:
            removed.append(tok)
        else:
            kept.append(tok)
    return " ".join(kept), removed

ADMIN_RE = re.compile(
    r"\b(?:salary|ctc|compensation|lpa|per\s+annum|stipend|pay\s*scale|shifts?|rotational|interviews?|interview\s+rounds?|"
    r"hiring\s+process|selection\s+process|assessment\s+round|notice\s+period|relocat\w*|locations?|work\s+from\s+(?:home|office)|"
    r"wfh|hybrid|remote|on-?site|office\s+based|visa|work\s+authori[sz]ation|benefits?|insurance|leaves?|holidays?|"
    r"apply|application\s+deadline|equal\s+opportunity|joining|bond|walk-?in|working\s+hours|perks|reporting\s+to)\b", re.I)


@dataclass
class BM25Query:
    text: str
    method: str                       # "structured" | "fallback_full_jd"
    fallback_reason: str = ""
    skill_terms: List[str] = field(default_factory=list)
    responsibility_lines_used: int = 0
    lines_excluded_as_administrative: int = 0
    mode: str = "structured"                                  # "structured" | "filtered" (the mode that built this query)
    removed_terms: List[str] = field(default_factory=list)   # generic terms dropped by the filtered mode (empty for baseline)

    @property
    def tokens(self) -> List[str]:
        return tokenize_for_bm25(self.text, query=True)


def _clean_lines(text: str):
    """(kept_lines, n_excluded): drops administrative lines."""
    kept, dropped = [], 0
    for line in (text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if ADMIN_RE.search(s):
            dropped += 1
        else:
            kept.append(s)
    return kept, dropped


def build_bm25_query(jd: JobDescription, mode: Optional[str] = None) -> BM25Query:
    """mode=None reads config.bm25_query_mode (default "structured" = baseline, unchanged)."""
    mode = (mode or config.bm25_query_mode)
    if mode not in QUERY_MODES:
        raise ValueError(f"Unknown BM25 query mode '{mode}'; choose one of {QUERY_MODES}")
    skills: List[str] = []
    for s in list(jd.required_skills) + list(jd.preferred_skills) + list(jd.supporting_skills):
        if s not in skills:
            skills.append(s)
    resp_lines, dropped = _clean_lines(jd.responsibilities)
    removed: List[str] = []
    if mode == "filtered":
        filtered_lines = []
        for ln in resp_lines:
            txt, rem = filter_generic(ln, skills)
            removed += rem
            if txt:
                filtered_lines.append(txt)
        resp_lines = filtered_lines
    structured = "\n".join([" ".join(skills)] + resp_lines).strip()
    if len(tokenize_for_bm25(structured, query=True)) >= MIN_QUERY_TOKENS:
        return BM25Query(structured, "structured", "", skills, len(resp_lines), dropped, mode, removed)

    reason = "no parsed technical skills or responsibilities" if not structured else \
        f"structured query shorter than {MIN_QUERY_TOKENS} tokens"
    kept, dropped_raw = _clean_lines(jd.raw_text)
    if mode == "filtered":
        removed, fk = [], []
        for ln in kept:
            txt, rem = filter_generic(ln, skills)
            removed += rem
            if txt:
                fk.append(txt)
        kept = fk
    return BM25Query("\n".join(kept), "fallback_full_jd", reason, skills, len(resp_lines), dropped_raw, mode, removed)
