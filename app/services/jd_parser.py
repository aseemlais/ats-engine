"""
Deterministic Job Description parser (no LLM).

Pipeline: split into sections by header lexicon -> classify every line (mandatory / preferred / supporting,
negation, trainable, soft wording) -> extract skills, accepted-alternative groups, experience range,
education and title. Rules are generic; nothing is specific to one job description.

Skill tiers
  required   (= mandatory)  skills a candidate must show: requirement-section lines, or lines with must/mandatory/required
  preferred  good-to-have / nice-to-have / bonus / preferred lines
  supporting trainable skills, concept-level wording ("understanding of", "exposure to"), explicit negations
             ("X is not mandatory", "training will be provided") and responsibility-only mentions
A skill takes the STRONGEST tier it is mentioned in, but a negated/trainable mention can never make it required.
"""
import re
from typing import Dict, List, Optional, Set, Tuple

from app.models.job import JobDescription
from app.utils.degree_normalizer import normalize_degree
from app.utils.skill_normalizer import SKILL_CATEGORY, extract_skill_mentions

BULLET_RE = re.compile(r"^\s*(?:[-•*–·▪●]|\d+[.)])\s+")

HEADER_RULES: List[Tuple[str, re.Pattern]] = [
    ("preferred", re.compile(r"good\s+to\s+have|nice\s+to\s+have|preferred|desirable|bonus|plus\s+points?|added\s+advantage|advantage|optional|would\s+be\s+a\s+plus", re.I)),
    ("skip", re.compile(r"^(?:about|benefits|perks|what\s+we\s+offer|why\s+join|how\s+to\s+apply|equal\s+opportunity|company|location|compensation|salary|stipend|night\s+shift|shift|selection\s+process|interview|commitment|career\s+opportunity|career\s+path)\b", re.I)),
    ("responsibilities", re.compile(r"responsibilit|what\s+you(?:'ll|\s+will)\s+do|duties|your\s+role|the\s+role\b|day[\s-]to[\s-]day|key\s+tasks", re.I)),
    ("attributes", re.compile(r"what\s+we\s+value|soft\s+skills|attributes|personal\s+qualities|who\s+you\s+are|culture|values", re.I)),
    ("education", re.compile(r"eligibility|education|academic", re.I)),
    ("supporting", re.compile(r"^(?:supporting|secondary|additional|other|related)\s+(?:skills?|technologies|tools)|\bfamiliar(?:ity)?\s+with\b|\bexposure\s+to\b|trainable\s+skills?", re.I)),
    ("required", re.compile(r"requirements?|must[\s-]*haves?|required|mandatory|qualifications?|what\s+you(?:'ll)?\s+(?:need|bring)|skills?\s+(?:required|needed)|key\s+skills|technical\s+skills|essential|you\s+have", re.I)),
]

NEG_RE = re.compile(
    r"\b(?:not|isn'?t|aren'?t|no\s+longer|never|n'?t)\s+(?:be\s+)?(?:a\s+)?(?:mandatory|required|necessary|essential|needed|compulsory|must|requirement|prerequisite)\b"
    r"|\b(?:no|without)\s+(?:prior\s+|previous\s+)?(?:\w+\s+){0,3}?(?:experience\s+)?(?:is\s+|are\s+)?(?:required|needed|necessary|mandatory)\b"
    r"|\boptional\b|\bnot\s+a\s+(?:must|requirement)\b", re.I)
TRAIN_RE = re.compile(
    r"training\s+(?:will\s+be\s+|is\s+)?provided|(?:we|company)\s+(?:will|'ll)\s+(?:train|teach)|will\s+be\s+trained|train\s+you|can\s+be\s+(?:learned|learnt|taught)"
    r"|learn\s+on\s+the\s+job|on[\s-]the[\s-]job|(?:willing(?:ness)?|eager(?:ness)?|ready|keen)\s+to\s+learn|trainable", re.I)
SOFT_RE = re.compile(r"\b(?:understanding|familiar(?:ity)?|exposure|awareness|aware\s+of|basic|basics|introductory|foundational)\b", re.I)
# Wording that shows a REAL requirement even if the clause also says "fundamentals"/"concepts" ("strong SQL fundamentals",
# "working knowledge of Python fundamentals"). Inside a requirements section this beats the soft-wording downgrade.
STRONG_RE = re.compile(r"\b(?:strong|hands[\s-]on|proficien\w*|working\s+knowledge|solid|expert\w*|good\s+command)\b", re.I)
# A section header that itself says the items are mandatory ("Mandatory Requirements", "Must have", "Essential skills")
HARD_HEADER_RE = re.compile(r"mandatory|must[\s-]*haves?|essential|compulsory", re.I)
HARD_RE = re.compile(r"\bmust\b|\bmandatory\b|\bessential\b|\bcompulsory\b|\(\s*required\s*\)|\brequired\s*[:)]|\bis\s+required\b|\bare\s+required\b|\brequired\b", re.I)
PREF_RE = re.compile(r"good\s+to\s+have|nice\s+to\s+have|\bpreferred\b|\bdesirable\b|\bbonus\b|\bplus\b|added\s+advantage|\ban\s+advantage\b|\bbenefici", re.I)
ANY_OF_RE = re.compile(r"\b(?:any\s+of|one\s+of|either|at\s+least\s+one|or\s+similar|or\s+equivalent|or\s+comparable|or\s+related)\b", re.I)
CERT_WORD_RE = re.compile(r"\b(?:certif|licen[cs]e|credential|accredit)\w*", re.I)
EXPERIENCE_WORD_RE = re.compile(r"\b(?:experience|proficien\w*|knowledge|hands-on|expertise|skills?)\b", re.I)

TIER_RANK = {"required": 3, "preferred": 2, "supporting": 1}

# ---------------------------------------------------------------- experience
_NUM = r"(\d+(?:\.\d+)?)"
EXP_CTX = re.compile(r"\b(?:experience|exp)\b", re.I)
EXP_RANGE_RE = re.compile(rf"{_NUM}\s*(?:-|–|—|to)\s*{_NUM}\s*\+?\s*(?:years?|yrs?)", re.I)
EXP_MIN_WORDS_RE = re.compile(rf"(?:minimum|min\.?|at\s+least|atleast|over|more\s+than)\s*(?:of\s+)?{_NUM}\s*\+?\s*(?:years?|yrs?)", re.I)
EXP_PLUS_RE = re.compile(rf"{_NUM}\s*\+\s*(?:years?|yrs?)", re.I)
EXP_YEARS_EXP_RE = re.compile(rf"{_NUM}\s*(?:years?|yrs?)\s*(?:of\s+)?(?:[a-z-]+\s+){{0,2}}?(?:experience|exp)\b", re.I)
EXP_MAX_RE = re.compile(rf"(?:up\s*to|upto|maximum\s+of|max\.?(?:imum)?|not\s+more\s+than|no\s+more\s+than|at\s+most|less\s+than|under|below)\s*{_NUM}\s*(?:years?|yrs?)", re.I)
NO_EXP_RE = re.compile(
    r"\bfreshers?\b|\bfresh\s+graduates?\b|\bentry[\s-]level\b|\bno\s+(?:prior\s+|previous\s+|professional\s+|work\s+)?experience(?:\s+(?:is\s+)?(?:required|needed|necessary))?\b"
    r"|\bexperience\s+(?:is\s+)?not\s+(?:required|necessary|needed|mandatory)\b|\b0\s*(?:years?|yrs?)\b|\bgraduates?\s+welcome\b", re.I)

# ---------------------------------------------------------------- education
DEGREE_WORD_RE = re.compile(r"\b(?:ph\.?d|doctorate|masters?(?:'s|’s)?|bachelors?(?:'s|’s)?|b\.?\s?tech|m\.?\s?tech|b\.?\s?sc|m\.?\s?sc|mba|mca|bca|bba|diploma|undergraduate|postgraduate)\b", re.I)
DEGREE_ABBR_RE = re.compile(r"\b(?:B\.?E\.?|M\.?E\.?|B\.?S\.?|M\.?S\.?)(?=[\s,/)]|$)")
DEGREE_CTX_RE = re.compile(r"\b(?:degree|graduate|graduation|diploma|qualification|b\.?tech|m\.?tech|educat|bachelor|master|ph\.?d)\w*", re.I)

# ---------------------------------------------------------------- title
TITLE_KEY_RE = re.compile(r"^\s*(?:job\s+title|title|role|position|designation|opening)\s*[:\-–]\s*(.+?)\s*$", re.I)
ROLE_WORD_RE = re.compile(
    r"\b(?:engineer\w*|developer|analyst|scientist|trainee|intern(?:ship)?|manager|consultant|architect|lead|specialist|associate|administrator|designer|executive|officer|technician|researcher|apprentice|programmer|tester|qa|devops|coordinator|director|assistant|fresher|writer|marketer|accountant|recruiter)\b", re.I)
HIRING_RE = re.compile(r"(?:hiring|looking\s+for|seeking|recruiting)\s+(?:an?\s+|the\s+)?([A-Z][\w/&+\- ]{3,60}?(?:engineer\w*|developer|analyst|scientist|trainee|intern|manager|consultant|architect|specialist|associate|administrator|designer|executive))", re.I)

_ABBREV_END_RE = re.compile(r"(?:\be\.g|\bi\.e|\betc|\bvs|\bapprox|\bincl|\bno|\bsr|\bjr|\bdr)\.$", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|(?<=;)\s+")
_CONTRAST_RE = re.compile(r"\s*(?:,\s*)?\b(?:but|however|although|though|whereas)\b\s*,?\s*", re.I)


def _clauses(line: str) -> List[str]:
    """Split one JD line into independently-classified clauses so that negation / hard / preferred / trainable
    wording only applies to the skills in ITS OWN sentence ("Python is required. Spark is not mandatory." must not
    make Python non-mandatory). Splits on sentence ends and ';', and on a contrast word (but/however/although/
    though/whereas) only when BOTH sides name a skill, so "X is not mandatory but a plus" stays one clause.
    Abbreviations (e.g., i.e., etc.) never split. Lines with no sentence boundary are returned unchanged."""
    pieces: List[str] = []
    for part in _SENT_SPLIT_RE.split(line):
        part = part.strip()
        if not part:
            continue
        if pieces and _ABBREV_END_RE.search(pieces[-1]):
            pieces[-1] = pieces[-1] + " " + part
        else:
            pieces.append(part)
    out: List[str] = []
    for piece in pieces:
        cur_start = 0
        for m in _CONTRAST_RE.finditer(piece):
            left, right = piece[cur_start:m.start()], piece[m.end():]
            if left.strip() and extract_skill_mentions(left) and extract_skill_mentions(right):
                out.append(left.strip())
                cur_start = m.end()
        tail = piece[cur_start:].strip()
        if tail:
            out.append(tail)
    return out or [line]


def _strip(line: str) -> str:
    return BULLET_RE.sub("", line).strip()


def _header_category(text: str) -> Optional[str]:
    t = text.strip().rstrip(":").strip()
    if not t or len(t.split()) > 7 or t.endswith((".", ",", ";")):
        return None
    for cat, rx in HEADER_RULES:
        if rx.search(t):
            return cat
    return None


def _split_header(line: str) -> Tuple[Optional[str], str]:
    """(category, remainder). Handles 'Requirements:' / 'Good to have' / 'Requirements: Python, SQL'."""
    raw = line.strip()
    if BULLET_RE.match(raw):
        return None, line
    if ":" in raw:
        left, right = raw.split(":", 1)
        cat = _header_category(left) if len(left.split()) <= 5 else None
        if cat:
            return cat, right.strip()
    cat = _header_category(raw)
    return (cat, "") if cat else (None, line)


def _is_cert_only(line: str) -> bool:
    return bool(CERT_WORD_RE.search(line)) and not EXPERIENCE_WORD_RE.search(line)


def _line_groups(text: str, mentions: List[tuple]) -> List[List[str]]:
    """Accepted-alternative groups from the ORDER of skill mentions and the text between them."""
    groups: List[List[str]] = []
    n = len(mentions)
    i = 0
    while i < n:
        chain = [mentions[i]]
        links: List[str] = []
        j = i
        while j + 1 < n:
            gap = text[mentions[j][1]:mentions[j + 1][0]]
            if re.fullmatch(r"\s*(?:,\s*)?(?:or|/|\||and\s*/\s*or)\s*", gap, re.I):
                link = "or"
            elif re.fullmatch(r"\s*,\s*", gap):
                link = "comma"
            elif re.fullmatch(r"\s*(?:,\s*)?(?:and|&|\+)\s*", gap, re.I):
                link = "and"
            else:
                break
            links.append(link); chain.append(mentions[j + 1]); j += 1
        tail = text[chain[-1][1]: chain[-1][1] + 22]
        trailing_or = bool(re.match(r"\s*,?\s*(?:or\s+(?:similar|equivalent|comparable|related)|etc)", tail, re.I))
        canon = []
        for m in chain:
            if m[2] not in canon:
                canon.append(m[2])
        if len(canon) >= 2 and "and" not in links:
            if "or" in links or trailing_or:
                if "comma" not in links:
                    groups.append(sorted(canon))                      # A or B / A / B
                else:
                    same_cat = len({SKILL_CATEGORY.get(c, c) for c in canon}) == 1 and canon[0] in SKILL_CATEGORY
                    if same_cat or ANY_OF_RE.search(text) or trailing_or:
                        groups.append(sorted(canon))                  # "AWS, Azure or GCP", "(A, B, or similar)"
                    else:                                             # "Python, SQL or MySQL": only the pair around 'or'
                        k = max(idx for idx, l in enumerate(links) if l == "or")
                        pair = sorted({chain[k][2], chain[k + 1][2]})
                        if len(pair) == 2:
                            groups.append(pair)
        elif len(canon) >= 2 and "and" in links and "or" in links and False:
            pass
        i = j + 1
    return groups


class JDParserService:
    def parse_job_description(self, text: str, job_id: str = "JD_001") -> JobDescription:
        lines = [l.rstrip() for l in (text or "").splitlines()]
        non_empty = [l.strip() for l in lines if l.strip()]

        title = self._extract_title(non_empty)

        # ---- pass 1: section + line classification
        section = "general"
        section_hard = False
        saw_required_header = False
        skill_tier: Dict[str, str] = {}
        nonmandatory: Set[str] = set()
        group_candidates: List[Tuple[List[str], str]] = []
        resp_lines: List[str] = []
        edu_lines: List[str] = []
        exp_lines: List[str] = []
        records = []
        for raw in lines:
            if not raw.strip():
                continue
            cat, rest = _split_header(raw)
            if cat:
                section = cat
                section_hard = cat == "required" and bool(HARD_HEADER_RE.search(raw.split(":", 1)[0]))
                saw_required_header = saw_required_header or cat == "required"
                if not rest.strip():
                    continue
                raw = rest
            line = _strip(raw)
            if not line:
                continue
            records.append((section, line, section_hard))
        default_required = not saw_required_header  # headerless JD: general lines count as requirements

        for section, line, sec_hard in records:
            sec = "required" if (section == "general" and default_required) else section
            if sec == "responsibilities":
                resp_lines.append(line)
            if sec in ("education", "required", "general", "preferred"):
                edu_lines.append(line)
            if sec != "skip":
                exp_lines.append(line)
            if sec in ("skip", "attributes", "education"):
                continue  # degree/eligibility and culture lines carry no skill requirements
            for clause in _clauses(line):
                if _is_cert_only(clause):
                    continue  # certification-only clauses carry no skill requirements
                mentions = extract_skill_mentions(clause)
                if not mentions:
                    continue
                neg = bool(NEG_RE.search(clause))
                trainable = bool(TRAIN_RE.search(clause))
                hard = bool(HARD_RE.search(clause)) and not neg
                pref = bool(PREF_RE.search(clause)) or sec == "preferred"
                soft = bool(SOFT_RE.search(clause))
                if soft and sec == "required" and (sec_hard or STRONG_RE.search(clause)):
                    soft = False   # a "Mandatory ..." section or strong wording outweighs "fundamentals"
                if trainable:
                    tier = "supporting"
                elif neg:
                    tier = "preferred" if pref else "supporting"   # "not mandatory but a plus" stays a preference
                elif pref and not hard:
                    tier = "preferred"
                elif hard:
                    tier = "required"
                elif sec == "required":
                    tier = "supporting" if soft else "required"
                elif sec == "supporting":
                    tier = "supporting"
                else:  # responsibilities / general with explicit header structure
                    tier = "supporting"
                for _, _, canon in mentions:
                    if neg:
                        nonmandatory.add(canon)
                        skill_tier[canon] = "preferred" if pref else "supporting"
                    elif canon not in nonmandatory:
                        if TIER_RANK[tier] > TIER_RANK.get(skill_tier.get(canon, ""), 0):
                            skill_tier[canon] = tier
                for g in _line_groups(clause, mentions):
                    group_candidates.append((g, tier))

        # Invariant: A negated/non-mandatory skill can never be required
        for k in nonmandatory:
            if skill_tier.get(k) == "required":
                skill_tier[k] = "supporting"

        required = sorted(k for k, v in skill_tier.items() if v == "required" and k not in nonmandatory)
        preferred = sorted(k for k, v in skill_tier.items() if v == "preferred")
        supporting = sorted(k for k, v in skill_tier.items() if v == "supporting")
        nonmandatory = sorted(nonmandatory)

        # alternative groups: keep groups whose members all ended in the same tier (>= 2 members)
        alternatives: List[List[str]] = []
        for g, _tier in group_candidates:
            tiers = {skill_tier.get(m) for m in g}
            if len(tiers) == 1 and None not in tiers and sorted(g) not in alternatives:
                alternatives.append(sorted(g))

        min_exp, max_exp, exp_applicable = self._extract_experience(exp_lines)
        education = self._extract_education(edu_lines)

        return JobDescription(
            job_id=job_id,
            title=title,
            raw_text=text,
            required_skills=required,
            preferred_skills=preferred,
            supporting_skills=supporting,
            nonmandatory_skills=nonmandatory,
            minimum_experience_years=min_exp,
            maximum_experience_years=max_exp,
            experience_applicable=exp_applicable,
            education_requirements=education,
            mandatory_skills=list(required),
            skill_alternatives=alternatives,
            responsibilities="\n".join(resp_lines),
        )

    # ------------------------------------------------------------------ title
    @staticmethod
    def _extract_title(non_empty: List[str]) -> str:
        head = non_empty[:10]
        for line in head:
            m = TITLE_KEY_RE.match(line)
            if m:
                return m.group(1).strip(" -–|")
        for line in head:
            cand = re.sub(r"^\s*(?:job\s+description|jd)\s*[:\-–]?\s*", "", line, flags=re.I).strip()
            if (cand and len(cand.split()) <= 12 and not cand.endswith(".") and ROLE_WORD_RE.search(cand)
                    and not re.match(r"(?:about|we\s|our\s|the\s|you\s|location|company)", cand, re.I)):
                return cand.strip(" -–|")
        m = HIRING_RE.search(" ".join(non_empty[:12]))
        if m:
            return m.group(1).strip().title()
        return "Untitled Role"

    # ------------------------------------------------------------- experience
    @staticmethod
    def _extract_experience(lines: List[str]) -> Tuple[float, Optional[float], Optional[bool]]:
        min_exp: Optional[float] = None
        max_exp: Optional[float] = None
        no_exp = False
        for line in lines:
            if NO_EXP_RE.search(line):
                no_exp = True
            if not EXP_CTX.search(line):
                continue
            if _is_degree_duration(line):
                continue
            m = EXP_RANGE_RE.search(line)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                if lo <= hi and min_exp is None:
                    min_exp, max_exp = lo, hi
                continue
            m = EXP_MIN_WORDS_RE.search(line) or EXP_PLUS_RE.search(line) or EXP_YEARS_EXP_RE.search(line)
            if m and min_exp is None:
                min_exp = float(m.group(1))
            mx = EXP_MAX_RE.search(line)
            if mx and max_exp is None:
                max_exp = float(mx.group(1))
        value = min_exp or 0.0
        if value > 0:
            applicable: Optional[bool] = True
        elif no_exp or min_exp == 0.0:
            applicable = False
        else:
            applicable = False       # nothing stated: experience is NOT part of this JD's requirements
        if max_exp is not None and value > 0 and max_exp < value:
            max_exp = None
        return value, max_exp, applicable

    # -------------------------------------------------------------- education
    @staticmethod
    def _extract_education(lines: List[str]) -> List[str]:
        tiers: List[str] = []
        for line in lines:
            if not (DEGREE_CTX_RE.search(line) or DEGREE_WORD_RE.search(line)):
                continue
            hits = [m.group(0) for m in DEGREE_WORD_RE.finditer(line)] + [m.group(0) for m in DEGREE_ABBR_RE.finditer(line)]
            for h in hits:
                tier = normalize_degree(h)
                if tier != "none" and tier not in tiers:
                    tiers.append(tier)
        return tiers


def _is_degree_duration(line: str) -> bool:
    """'3-year bachelor's degree' style durations must not be read as experience requirements."""
    return bool(re.search(r"\b\d+\s*-?\s*years?\s*(?:\w+\s+){0,2}(?:degree|bachelor|master|diploma|course|program)", line, re.I))
