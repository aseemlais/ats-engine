"""
Isolated Education Eligibility Component for Local ATS Scoring Experiment.
Batch 2 Implementation.

Rules & Guarantees:
1. Education is strictly an eligibility filter, NOT a relevance scoring component.
2. Supports alternative qualifications (OR logic: satisfying ANY one alternative makes candidate ELIGIBLE).
3. Evaluates specific degrees, field of study constraints, and completion status (completed vs in-progress).
4. Handles missing education headings by searching full resume text before returning UNKNOWN or INELIGIBLE.
5. Employs 4 deterministic statuses:
   - ELIGIBLE: sufficient evidence satisfies explicit mandatory criteria.
   - INELIGIBLE: reliable evidence establishes candidate fails explicit mandatory criteria.
   - UNKNOWN: requirement or candidate evidence cannot be determined reliably (flagged for verification).
   - NOT_REQUIRED: JD does not specify a mandatory education requirement.
"""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.models.job import JobDescription
from app.utils.degree_normalizer import DEGREE_LEVELS, degree_rank, normalize_degree
from experiments.spacy_section_detector import DocumentExtractionResult


class EducationEligibilityStatus(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"
    NOT_REQUIRED = "NOT_REQUIRED"


@dataclass
class DegreeSpec:
    """Specification of an accepted degree qualification."""
    raw_text: str
    tier: str                      # doctorate, master, bachelor, associate, none
    degree_type: str               # e.g. btech, be, bs, bsc, mtech, ms, msc, mca, phd, bachelor, master
    field_of_study: Optional[str] = None
    requires_stem_related: bool = False

    def matches_tier_or_higher(self, candidate_tier: str) -> bool:
        """Checks if candidate degree tier meets or exceeds required tier."""
        req_rank = DEGREE_LEVELS.get(self.tier, 0)
        cand_rank = DEGREE_LEVELS.get(candidate_tier, 0)
        return cand_rank >= req_rank and cand_rank > 0


@dataclass
class EducationRequirement:
    """Extracted education requirement from Job Description."""
    is_mandatory: bool
    raw_requirement_text: str
    accepted_alternatives: List[DegreeSpec] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    allows_related_fields: bool = True
    requires_completed: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class CandidateEducationEvidence:
    """Extracted education evidence from candidate resume."""
    raw_text: str
    degree_name: str
    tier: str
    degree_type: str
    field_of_study: Optional[str] = None
    is_in_progress: bool = False
    institution: Optional[str] = None
    year: Optional[str] = None
    source_section: str = "education"  # 'education', 'summary', 'full_text'


@dataclass
class EducationEligibilityResult:
    """Structured decision output of education eligibility assessment."""
    status: EducationEligibilityStatus
    is_eligible: bool                     # True for ELIGIBLE and NOT_REQUIRED; False for INELIGIBLE
    matched_degree: Optional[str] = None
    matched_field: Optional[str] = None
    candidate_degrees: List[str] = field(default_factory=list)
    jd_requirement_summary: str = ""
    decision_reason: str = ""
    requires_manual_verification: bool = False
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lexicons and Patterns for Extraction & Normalization
# ---------------------------------------------------------------------------

DEGREE_TYPE_MAP: Dict[str, Tuple[str, str]] = {
    # alias: (canonical_degree_type, tier)
    "phd": ("phd", "doctorate"),
    "ph.d": ("phd", "doctorate"),
    "doctorate": ("doctorate", "doctorate"),
    "doctor of philosophy": ("phd", "doctorate"),
    "m.tech": ("mtech", "master"),
    "mtech": ("mtech", "master"),
    "m.e": ("me", "master"),
    "me": ("me", "master"),
    "m.sc": ("msc", "master"),
    "msc": ("msc", "master"),
    "m.s": ("ms", "master"),
    "ms": ("ms", "master"),
    "master": ("master", "master"),
    "masters": ("master", "master"),
    "master's": ("master", "master"),
    "mca": ("mca", "master"),
    "mba": ("mba", "master"),
    "m.phil": ("mphil", "master"),
    "m.eng": ("meng", "master"),
    "b.tech": ("btech", "bachelor"),
    "btech": ("btech", "bachelor"),
    "b.e": ("be", "bachelor"),
    "be": ("be", "bachelor"),
    "b.s": ("bs", "bachelor"),
    "bs": ("bs", "bachelor"),
    "b.sc": ("bsc", "bachelor"),
    "bsc": ("bsc", "bachelor"),
    "bachelor": ("bachelor", "bachelor"),
    "bachelors": ("bachelor", "bachelor"),
    "bachelor's": ("bachelor", "bachelor"),
    "bca": ("bca", "bachelor"),
    "bba": ("bba", "bachelor"),
    "b.com": ("bcom", "bachelor"),
    "bcom": ("bcom", "bachelor"),
    "b.eng": ("beng", "bachelor"),
    "associate": ("associate", "associate"),
    "associate's": ("associate", "associate"),
    "diploma": ("diploma", "associate"),
}

# Regex to detect in-progress or pursuing education status
IN_PROGRESS_PATTERN = re.compile(
    r"\b(?:pursuing|in[\s-]progress|currently\s+(?:enrolled|studying|pursuing)|"
    r"candidate\s+for|expected(?:\s+(?:graduation|completion))?\s+(?:in\s+)?20\d\d|"
    r"expected\s+20\d\d|ongoing|present|enrolled)\b",
    re.IGNORECASE,
)

# Regex to detect optional or preferred education phrasing in JD
NON_MANDATORY_PATTERN = re.compile(
    r"\b(?:preferred|good\s+to\s+have|nice\s+to\s+have|plus|bonus|optional|"
    r"desirable|advantage|not\s+mandatory|not\s+required|not\s+necessary)\b",
    re.IGNORECASE,
)

# Common fields of study keywords
FIELD_KEYWORDS = [
    ("computer science and engineering", "computer science"),
    ("computer science", "computer science"),
    ("information technology", "information technology"),
    ("information science", "information technology"),
    ("software engineering", "software engineering"),
    ("data science", "data science"),
    ("artificial intelligence", "artificial intelligence"),
    ("machine learning", "machine learning"),
    ("electrical engineering", "electrical engineering"),
    ("electronics and communication", "electronics"),
    ("electronics", "electronics"),
    ("mechanical engineering", "mechanical engineering"),
    ("civil engineering", "civil engineering"),
    ("business administration", "business"),
    ("business", "business"),
    ("finance", "finance"),
    ("mathematics", "mathematics"),
    ("statistics", "statistics"),
    ("physics", "physics"),
    ("engineering", "engineering"),
]


class EducationEligibilityService:
    """Isolated, deterministic service evaluating education requirements as a hard filter."""

    @staticmethod
    def _extract_field_of_study(text: str) -> Optional[str]:
        """Extracts field of study from a degree mention line or clause."""
        text_lower = text.lower()
        for pattern, canonical in FIELD_KEYWORDS:
            if pattern in text_lower:
                return canonical
        return None

    @staticmethod
    def _is_related_field(candidate_field: Optional[str], required_field: Optional[str]) -> bool:
        """Determines if a candidate's field is matching or sufficiently related (e.g. STEM/computing)."""
        if not required_field:
            return True
        if not candidate_field:
            return False

        c_field = candidate_field.lower()
        r_field = required_field.lower()

        if c_field == r_field:
            return True

        # Computing and software related group
        computing_group = {
            "computer science", "information technology", "software engineering",
            "data science", "artificial intelligence", "machine learning", "engineering"
        }
        if r_field in computing_group and c_field in computing_group:
            return True

        # Engineering related group
        engineering_group = {
            "engineering", "electrical engineering", "electronics",
            "mechanical engineering", "civil engineering", "computer science"
        }
        if r_field in engineering_group and c_field in engineering_group:
            return True

        return False

    def parse_jd_requirements(self, jd: JobDescription) -> EducationRequirement:
        """
        Extracts and structures education requirements from a Job Description.
        Supports alternative qualifications separated by OR / or / equivalent / commas.
        Distinguishes mandatory requirements from preferred/optional mentions.
        """
        # 1. Determine candidate text lines describing education in JD
        edu_lines: List[str] = []
        raw_lines = jd.raw_text.splitlines() if jd.raw_text else []
        in_edu_section = False
        in_pref_section = False

        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Detect section transitions
            line_lower = line_str.lower()
            if re.match(r"^(?:education|qualifications?|academic\s+requirements?|eligibility)\b", line_lower):
                in_edu_section = True
                in_pref_section = False
                continue
            elif re.match(r"^(?:good\s+to\s+have|preferred|nice\s+to\s+have|bonus)\b", line_lower):
                in_pref_section = True
                in_edu_section = False
                continue
            elif re.match(r"^(?:responsibilities|skills?|experience|about|benefits)\b", line_lower):
                in_edu_section = False
                in_pref_section = False

            # Collect education mentions
            if in_edu_section and not in_pref_section:
                edu_lines.append(line_str)
            elif not in_pref_section:
                if re.search(r"\b(?:bachelor|master|b\.?tech|m\.?tech|b\.?sc|m\.?sc|ph\.?d|degree|graduate)\b", line_lower):
                    # Check if line itself contains non-mandatory words
                    if not NON_MANDATORY_PATTERN.search(line_lower):
                        edu_lines.append(line_str)

        # Also inspect pre-parsed jd.education_requirements if present
        if jd.education_requirements and not edu_lines:
            edu_lines.extend(jd.education_requirements)

        if not edu_lines:
            return EducationRequirement(
                is_mandatory=False,
                raw_requirement_text="",
                accepted_alternatives=[],
                notes=["No mandatory education requirement found in JD."]
            )

        combined_req_text = " ; ".join(edu_lines)

        # Check if the entire requirement is explicitly optional/preferred
        if NON_MANDATORY_PATTERN.search(combined_req_text) and not re.search(r"\b(?:must|mandatory|required|essential|minimum)\b", combined_req_text, re.I):
            # If all detected lines are preferred/optional, it's not a mandatory hard filter
            return EducationRequirement(
                is_mandatory=False,
                raw_requirement_text=combined_req_text,
                accepted_alternatives=[],
                notes=["Education mentioned as preferred/optional only; not mandatory."]
            )

        # 2. Extract alternative qualifications using OR / / / commas
        accepted_alternatives: List[DegreeSpec] = []
        required_fields: List[str] = []
        allows_related = bool(re.search(r"\b(?:or\s+related|or\s+equivalent|related\s+field|similar\s+field)\b", combined_req_text, re.I))

        # Check for field of study across requirement text
        found_field = self._extract_field_of_study(combined_req_text)
        if found_field:
            required_fields.append(found_field)

        # Split on OR / comma / slash / semicolons to find degree alternatives
        tokens = re.split(r"\b(?:or|OR|equivalent|/|\|)\b|,|;", combined_req_text)
        for token in tokens:
            t = token.strip()
            if not t:
                continue
            t_lower = t.lower()

            # Find degree mentions in this alternative token
            matched_spec = None
            for alias, (deg_type, tier) in DEGREE_TYPE_MAP.items():
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, t_lower):
                    token_field = self._extract_field_of_study(t) or found_field
                    matched_spec = DegreeSpec(
                        raw_text=t,
                        tier=tier,
                        degree_type=deg_type,
                        field_of_study=token_field,
                        requires_stem_related=allows_related or (token_field is not None),
                    )
                    break

            if matched_spec:
                # Deduplicate by (tier, degree_type)
                if not any(a.degree_type == matched_spec.degree_type for a in accepted_alternatives):
                    accepted_alternatives.append(matched_spec)

        # Fallback if no specific tokens matched via DEGREE_TYPE_MAP
        if not accepted_alternatives:
            for item in jd.education_requirements:
                tier = normalize_degree(item)
                if tier != "none":
                    accepted_alternatives.append(
                        DegreeSpec(
                            raw_text=item,
                            tier=tier,
                            degree_type=tier,
                            field_of_study=found_field,
                        )
                    )

        if not accepted_alternatives:
            # Contains words like "degree" without specific level -> accept any graduate degree (bachelor+)
            if "degree" in combined_req_text.lower() or "graduate" in combined_req_text.lower():
                accepted_alternatives.append(
                    DegreeSpec(
                        raw_text=combined_req_text,
                        tier="bachelor",
                        degree_type="bachelor",
                        field_of_study=found_field,
                    )
                )

        if not accepted_alternatives:
            return EducationRequirement(
                is_mandatory=False,
                raw_requirement_text=combined_req_text,
                accepted_alternatives=[],
                notes=["Could not identify specific degree tier requirement; treating as not mandatory."]
            )

        return EducationRequirement(
            is_mandatory=True,
            raw_requirement_text=combined_req_text,
            accepted_alternatives=accepted_alternatives,
            required_fields=required_fields,
            allows_related_fields=allows_related,
            requires_completed=not bool(re.search(r"\b(?:pursuing|final\s+year|interns?)\s+eligible\b", combined_req_text, re.I)),
        )

    def extract_candidate_education(
        self,
        extraction: DocumentExtractionResult,
    ) -> List[CandidateEducationEvidence]:
        """
        Extracts candidate's education evidence.
        1. Checks already detected resume sections (category == 'education').
        2. Fallback: if heading is missing or empty, searches full extracted text.
        3. Identifies degree names, canonical tiers, field of study, and completion/in-progress status.
        """
        evidence_list: List[CandidateEducationEvidence] = []
        edu_texts: List[Tuple[str, str]] = []

        # 1. Check detected education section
        for sec in extraction.sections:
            if sec.category == "education" and sec.text.strip():
                edu_texts.append((sec.text, "education"))

        # 2. Fallback to full text if education section missing
        if not edu_texts:
            full_text = extraction.cleaned_text or extraction.raw_text
            if full_text.strip():
                edu_texts.append((full_text, "full_text"))

        # Parse degree mentions from gathered text
        seen_degree_types: Set[str] = set()

        for block_text, source_tag in edu_texts:
            lines = block_text.splitlines()
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                line_lower = line_str.lower()

                # Check for each degree alias
                for alias, (deg_type, tier) in DEGREE_TYPE_MAP.items():
                    pattern = rf"\b{re.escape(alias)}\b"
                    if re.search(pattern, line_lower):
                        # Determine if in-progress
                        is_in_progress = bool(IN_PROGRESS_PATTERN.search(line_lower))

                        # If not explicit on this line, check surrounding 80 characters
                        field_found = self._extract_field_of_study(line_str)

                        # Extract possible year (e.g. 2020, 2018-2022)
                        year_match = re.search(r"\b(20\d\d|19\d\d)\b", line_str)
                        year = year_match.group(1) if year_match else None

                        deg_key = f"{deg_type}_{is_in_progress}"
                        if deg_key not in seen_degree_types:
                            seen_degree_types.add(deg_key)
                            evidence_list.append(
                                CandidateEducationEvidence(
                                    raw_text=line_str,
                                    degree_name=alias.upper(),
                                    tier=tier,
                                    degree_type=deg_type,
                                    field_of_study=field_found,
                                    is_in_progress=is_in_progress,
                                    year=year,
                                    source_section=source_tag,
                                )
                            )

        return evidence_list

    def evaluate_eligibility(
        self,
        jd: JobDescription,
        extraction: DocumentExtractionResult,
    ) -> EducationEligibilityResult:
        """
        Determines whether the candidate satisfies the education criteria.
        Returns one of four deterministic statuses:
        - ELIGIBLE
        - INELIGIBLE
        - UNKNOWN
        - NOT_REQUIRED
        """
        # Step 1: Parse JD education requirement
        req = self.parse_jd_requirements(jd)
        if not req.is_mandatory or not req.accepted_alternatives:
            return EducationEligibilityResult(
                status=EducationEligibilityStatus.NOT_REQUIRED,
                is_eligible=True,
                jd_requirement_summary=req.raw_requirement_text or "None specified",
                decision_reason="No mandatory education requirement specified in JD.",
                requires_manual_verification=False,
            )

        # Step 2: Extract candidate education evidence
        evidence = self.extract_candidate_education(extraction)
        req_summary = " OR ".join(
            f"{a.degree_type.upper()}" + (f" in {a.field_of_study}" if a.field_of_study else "")
            for a in req.accepted_alternatives
        )

        # If no education evidence can be extracted anywhere
        if not evidence:
            return EducationEligibilityResult(
                status=EducationEligibilityStatus.UNKNOWN,
                is_eligible=False,
                jd_requirement_summary=req_summary,
                decision_reason="JD mandates education qualification, but no verifiable education credentials were found in candidate resume.",
                requires_manual_verification=True,
                notes=["Education heading and degree keywords missing from entire resume text."],
            )

        candidate_deg_names = [f"{e.degree_name}{' (in-progress)' if e.is_in_progress else ''}" for e in evidence]
        evidence_dicts = [
            {
                "degree": e.degree_name,
                "tier": e.tier,
                "type": e.degree_type,
                "field": e.field_of_study,
                "in_progress": e.is_in_progress,
                "source": e.source_section,
                "raw_text": e.raw_text,
            }
            for e in evidence
        ]

        # Step 3: Evaluate alternatives using OR logic
        # If candidate satisfies ANY one accepted alternative -> ELIGIBLE
        in_progress_matches = []
        field_mismatch_matches = []

        for alt in req.accepted_alternatives:
            for ev in evidence:
                # Check degree tier / equivalence
                tier_satisfied = alt.matches_tier_or_higher(ev.tier)
                type_satisfied = (alt.degree_type == ev.degree_type) or tier_satisfied

                if not type_satisfied:
                    continue

                # Check completion requirement
                if req.requires_completed and ev.is_in_progress:
                    in_progress_matches.append(ev)
                    continue

                # Check field of study constraint
                if req.required_fields and alt.field_of_study:
                    if not self._is_related_field(ev.field_of_study, alt.field_of_study):
                        field_mismatch_matches.append((ev, alt))
                        continue

                # If we reached here, this alternative is FULLY satisfied!
                return EducationEligibilityResult(
                    status=EducationEligibilityStatus.ELIGIBLE,
                    is_eligible=True,
                    matched_degree=ev.degree_name,
                    matched_field=ev.field_of_study or alt.field_of_study,
                    candidate_degrees=candidate_deg_names,
                    jd_requirement_summary=req_summary,
                    decision_reason=f"Candidate satisfies education requirement via alternative '{alt.degree_type.upper()}' with degree '{ev.degree_name}'" + (f" in {ev.field_of_study}." if ev.field_of_study else "."),
                    requires_manual_verification=False,
                    evidence_items=evidence_dicts,
                )

        # Step 4: If no alternative was satisfied, determine if demonstrably INELIGIBLE or UNKNOWN
        # Check if candidate failed due to degree in-progress
        if in_progress_matches and not any(not ev.is_in_progress for ev in evidence if alt.matches_tier_or_higher(ev.tier)):
            in_prog_names = ", ".join(m.degree_name for m in in_progress_matches)
            return EducationEligibilityResult(
                status=EducationEligibilityStatus.INELIGIBLE,
                is_eligible=False,
                candidate_degrees=candidate_deg_names,
                jd_requirement_summary=req_summary,
                decision_reason=f"Mandatory degree required is completed, but candidate qualification is currently in-progress ({in_prog_names}).",
                requires_manual_verification=False,
                evidence_items=evidence_dicts,
                notes=["Candidate is still pursuing or has not completed required degree."],
            )

        # Check if candidate failed due to field mismatch when strict field required
        if field_mismatch_matches and not any(ev.field_of_study for ev in evidence):
            # Candidate has the degree level, but field of study is unmentioned/uncertain
            return EducationEligibilityResult(
                status=EducationEligibilityStatus.UNKNOWN,
                is_eligible=False,
                candidate_degrees=candidate_deg_names,
                jd_requirement_summary=req_summary,
                decision_reason=f"Candidate holds required degree tier ({evidence[0].degree_name}), but field of study could not be verified against required '{req.required_fields}'.",
                requires_manual_verification=True,
                evidence_items=evidence_dicts,
            )

        # Highest degree tier rank of candidate
        cand_highest_rank = max((DEGREE_LEVELS.get(ev.tier, 0) for ev in evidence), default=0)
        min_required_rank = min((DEGREE_LEVELS.get(alt.tier, 0) for alt in req.accepted_alternatives), default=0)

        if cand_highest_rank < min_required_rank:
            cand_highest_tier = max(evidence, key=lambda e: DEGREE_LEVELS.get(e.tier, 0)).degree_name
            return EducationEligibilityResult(
                status=EducationEligibilityStatus.INELIGIBLE,
                is_eligible=False,
                candidate_degrees=candidate_deg_names,
                jd_requirement_summary=req_summary,
                decision_reason=f"Candidate's highest degree tier ({cand_highest_tier}) is below mandatory minimum required ({req_summary}).",
                requires_manual_verification=False,
                evidence_items=evidence_dicts,
            )

        # Inconclusive/uncertain
        return EducationEligibilityResult(
            status=EducationEligibilityStatus.UNKNOWN,
            is_eligible=False,
            candidate_degrees=candidate_deg_names,
            jd_requirement_summary=req_summary,
            decision_reason=f"Candidate credentials ({', '.join(candidate_deg_names)}) do not unambiguously satisfy mandatory requirement ({req_summary}).",
            requires_manual_verification=True,
            evidence_items=evidence_dicts,
        )
