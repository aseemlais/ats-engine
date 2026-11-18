"""
Isolated Experience & Internship Scoring Component for Local ATS Scoring Experiment.
Batch 3 Implementation.

Rules & Guarantees:
1. Distinguishes experience eligibility/requirements from experience relevance scoring.
2. Extracts structured experience records (role/title, organization, type, dates, responsibilities).
3. Distinguishes employment & internships from academic/personal projects (projects are never employment).
4. Merges overlapping employment and internship intervals so concurrent periods are never double-counted.
5. Calculates relevance of actual duties via section/record-level BM25 and semantic similarity against JD responsibilities. Job title alone does not determine relevance.
6. Applies configurable type multipliers: Professional (1.0), Internship (0.4), Other (0.7), Unknown (0.5).
7. Formula: Duration Ratio x Experience-Type Multiplier x Duties Relevance.
8. If JD does not require experience: Status is NOT_REQUIRED; experience is treated neutrally so freshers are not penalized, and optional capped fresher bonus is reported separately.
9. Deterministic statuses: MEETS_REQUIREMENT, BELOW_REQUIREMENT, NOT_REQUIRED, UNKNOWN.
"""
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from app.models.job import JobDescription
from experiments.experiment_config import ExperienceScoringConfig, SectionDetectionConfig
from experiments.spacy_section_detector import DetectedSection, DocumentExtractionResult


class ExperienceType(str, Enum):
    PROFESSIONAL = "PROFESSIONAL"
    INTERNSHIP = "INTERNSHIP"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ExperienceRequirementStatus(str, Enum):
    MEETS_REQUIREMENT = "MEETS_REQUIREMENT"
    BELOW_REQUIREMENT = "BELOW_REQUIREMENT"
    NOT_REQUIRED = "NOT_REQUIRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class ExperienceRecord:
    """Individual extracted employment or internship record."""
    role_title: str = "Unknown"
    organization: str = "Unknown"
    exp_type: ExperienceType = ExperienceType.UNKNOWN
    start_date_str: Optional[str] = None
    end_date_str: Optional[str] = None
    is_current: bool = False
    duration_months: float = 0.0
    duration_years: float = 0.0
    intervals: List[Tuple[int, int]] = field(default_factory=list)  # (start_month_idx, end_month_idx)
    responsibilities_text: str = ""
    source_section: str = "experience"
    
    # Duties relevance metrics
    bm25_raw: float = 0.0
    bm25_normalized: float = 0.0
    semantic_raw_cosine: float = 0.0
    semantic_calibrated: float = 0.0
    duties_relevance: float = 0.0  # Blended lexical + semantic relevance of actual duties
    
    # Type multiplier & relevance-adjusted duration
    type_multiplier: float = 1.0
    relevance_adjusted_years: float = 0.0
    
    extraction_notes: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class JDExperienceRequirement:
    """Extracted experience requirement from Job Description."""
    is_mandatory: bool
    status: ExperienceRequirementStatus
    minimum_years: float
    raw_text: str
    lower_bound_years: float
    upper_bound_years: Optional[float] = None
    is_range: bool = False
    units_used: str = "years"
    parsing_notes: List[str] = field(default_factory=list)
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass
class ExperienceEvaluationResult:
    """Detailed evaluation result of candidate experience."""
    status: ExperienceRequirementStatus
    required_years: float
    verified_total_years: float      # Non-overlapping union of employment/internship intervals
    relevance_adjusted_years: float  # Non-overlapping intervals weighted by type & duties relevance
    duration_ratio: float            # min(verified / required, 1.0)
    effective_multiplier: float      # Duration-weighted average type multiplier
    duties_relevance: float          # Duration-weighted average duties relevance
    final_experience_score: float    # In [0.0, 1.0] to supply core ATS experience dimension
    fresher_bonus: float = 0.0       # Separately reported bonus when JD experience is NOT_REQUIRED
    records: List[ExperienceRecord] = field(default_factory=list)
    overlapping_intervals_handled: bool = False
    decision_reason: str = ""
    uncertainty_notes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Date and Regex Patterns for Experience Extraction
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "sept": 9,
}

# Date ranges: "Jan 2021 - Mar 2023", "May 2022 to Present", "06/2021 - 08/2022"
DATE_WITH_MONTH_RE = re.compile(
    r"\b(?:(\d{1,2})\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:(?:(\d{1,2})\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})|(present|current|now|ongoing))\b",
    re.I
)

DATE_SLASH_RE = re.compile(
    r"\b(0?[1-9]|1[0-2])[\/\.-]((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(?:(0?[1-9]|1[0-2])[\/\.-]((?:19|20)\d{2})|(present|current|now|ongoing))\b",
    re.I
)

DATE_DOT_RE = re.compile(
    r"\b(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-]((?:19|20)\d{2})\s*(?:-|–|—|to)\s*(\d{1,2})[\.\/\-](\d{1,2})[\.\/\-]((?:19|20)\d{2})\b",
    re.I
)

YEAR_RANGE_RE = re.compile(
    r"\b((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now|ongoing)\b",
    re.I
)

# Explicit durations
EXPLICIT_MONTHS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:-|–|—)?\s*(?:months?|mos?)\b",
    re.I
)

EXPLICIT_YEARS_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp|work)?\b",
    re.I
)

# Internship keywords
INTERNSHIP_KEYWORDS = re.compile(
    r"\b(?:intern|internship|trainee|apprentice|fellow|summer\s+intern|student\s+researcher)\b",
    re.I
)

# Professional employment keywords
PROFESSIONAL_KEYWORDS = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|consultant|architect|manager|lead|"
    r"specialist|director|administrator|associate|executive|officer|programmer|technician)\b",
    re.I
)

# Other employment keywords
OTHER_KEYWORDS = re.compile(
    r"\b(?:freelance|contractor|volunteer|part-time|self-employed|contract)\b",
    re.I
)

# JD Experience Patterns
JD_NO_EXP_RE = re.compile(
    r"\b(?:freshers?(?:\s+welcome)?|fresh\s+graduates?|entry[\s-]level|"
    r"no\s+(?:prior\s+|previous\s+|professional\s+|work\s+)?experience(?:\s+(?:is\s+)?(?:required|needed|necessary))?|"
    r"experience\s+(?:is\s+)?not\s+(?:required|necessary|needed|mandatory)|"
    r"0\s*(?:years?|yrs?)|0\s*-\s*1\s*(?:years?|yrs?)|graduates?\s+welcome)\b",
    re.I
)

JD_EXP_PREF_RE = re.compile(
    r"\b(?:experience\s+(?:is\s+)?(?:preferred|a\s+plus|desirable|optional|good\s+to\s+have|nice\s+to\s+have)|"
    r"prior\s+experience\s+(?:is\s+)?(?:a\s+plus|preferred)|"
    r"preferred\s*:\s*\d+.*(?:years?|yrs?)|"
    r"would\s+be\s+a\s+plus)\b",
    re.I
)

JD_MONTHS_REQ_RE = re.compile(
    r"(?:minimum|at\s+least|atleast|min\.?)\s*(?:of\s+)?(\d+)\s*\+?\s*(?:months?|mos?)(?:\s+of)?\s+(?:experience|work|professional\s+experience)?"
    r"|(\d+)\s*(?:-|–|—|to)\s*(\d+)\s*(?:months?|mos?)(?:\s+of)?\s+(?:experience|work)?"
    r"|(\d+)\+?\s*(?:months?|mos?)\s+(?:of\s+)?(?:relevant\s+|professional\s+)?experience",
    re.I
)

JD_YEARS_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)",
    re.I
)

JD_YEARS_MIN_RE = re.compile(
    r"(?:minimum|min\.?|at\s+least|atleast|over|more\s+than)\s*(?:of\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)"
    r"|(\d+(?:\.\d+)?)\s*\+\s*(?:years?|yrs?)"
    r"|(\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp|relevant\s+experience)",
    re.I
)

# Wording stating experience is required without numbers
JD_AMBIGUOUS_EXP_RE = re.compile(
    r"\b(?:proven\s+experience|relevant\s+experience|work\s+experience|prior\s+experience)\s+(?:is\s+)?(?:required|mandatory|essential|needed)\b",
    re.I
)


class ExperienceScorerService:
    """
    Isolated service for deterministic experience extraction, duration calculation,
    duties relevance scoring, and type-adjusted evaluation.
    """

    def __init__(
        self,
        config: Optional[ExperienceScoringConfig] = None,
        detection_config: Optional[SectionDetectionConfig] = None,
    ):
        self.config = config or ExperienceScoringConfig()
        self.detection_config = detection_config or SectionDetectionConfig()

    # -----------------------------------------------------------------------
    # 1. JD Requirement Determination
    # -----------------------------------------------------------------------
    def parse_jd_experience_requirement(self, jd: JobDescription) -> JDExperienceRequirement:
        """
        Determines the minimum required experience duration from JD.
        - Identifies years and months.
        - Uses lower bound of ranges (e.g. 1-2 years -> 1.0 year).
        - If not specified or explicitly freshers: NOT_REQUIRED (0.0 years).
        - If ambiguous without quantifiable duration: UNKNOWN.
        """
        text = jd.raw_text or ""
        lines = text.splitlines()

        # Check explicit no experience / freshers
        for line in lines:
            if JD_NO_EXP_RE.search(line):
                # Verify line is not inside a "preferred" block
                if not re.search(r"^(?:preferred|nice\s+to\s+have|plus)\b", line.strip(), re.I):
                    return JDExperienceRequirement(
                        is_mandatory=False,
                        status=ExperienceRequirementStatus.NOT_REQUIRED,
                        minimum_years=0.0,
                        raw_text=line.strip(),
                        lower_bound_years=0.0,
                        parsing_notes=["JD explicitly accepts freshers or specifies 0 experience needed."]
                    )

        # Check if experience is only mentioned as preferred/optional
        pref_mentions = []
        for line in lines:
            if JD_EXP_PREF_RE.search(line):
                pref_mentions.append(line.strip())

        # Check for month requirements first (e.g., "Minimum 6 months of professional experience")
        for line in lines:
            # Skip if explicitly in a preferred section or preferred line
            if JD_EXP_PREF_RE.search(line):
                continue
            m = JD_MONTHS_REQ_RE.search(line)
            if m:
                # Groups: min months (1), or range (2, 3), or explicit (4)
                if m.group(1):
                    months = float(m.group(1))
                elif m.group(2) and m.group(3):
                    months = float(m.group(2))  # lower bound
                elif m.group(4):
                    months = float(m.group(4))
                else:
                    continue

                years = round(months / 12.0, 3)
                return JDExperienceRequirement(
                    is_mandatory=True,
                    status=ExperienceRequirementStatus.MEETS_REQUIREMENT,
                    minimum_years=years,
                    raw_text=line.strip(),
                    lower_bound_years=years,
                    units_used="months",
                    parsing_notes=[f"Extracted requirement of {months} months ({years:.2f} years)."]
                )

        # Check for year ranges (e.g., "1–2 years of experience")
        for line in lines:
            if JD_EXP_PREF_RE.search(line):
                continue
            m = JD_YEARS_RANGE_RE.search(line)
            if m:
                lo = float(m.group(1))
                hi = float(m.group(2))
                if lo <= hi:
                    return JDExperienceRequirement(
                        is_mandatory=True,
                        status=ExperienceRequirementStatus.MEETS_REQUIREMENT,
                        minimum_years=lo,
                        raw_text=line.strip(),
                        lower_bound_years=lo,
                        upper_bound_years=hi,
                        is_range=True,
                        units_used="years",
                        parsing_notes=[f"Extracted experience range {lo}-{hi} years; lower bound {lo} years adopted as minimum."]
                    )

        # Check for minimum years ("At least 2 years of relevant experience")
        for line in lines:
            if JD_EXP_PREF_RE.search(line):
                continue
            m = JD_YEARS_MIN_RE.search(line)
            if m:
                val_str = m.group(1) or m.group(2) or m.group(3)
                if val_str:
                    val = float(val_str)
                    return JDExperienceRequirement(
                        is_mandatory=True,
                        status=ExperienceRequirementStatus.MEETS_REQUIREMENT,
                        minimum_years=val,
                        raw_text=line.strip(),
                        lower_bound_years=val,
                        units_used="years",
                        parsing_notes=[f"Extracted minimum requirement of {val} years."]
                    )

        # Check for ambiguous experience phrasing without numbers ("relevant experience is required")
        for line in lines:
            if JD_EXP_PREF_RE.search(line):
                continue
            if JD_AMBIGUOUS_EXP_RE.search(line):
                return JDExperienceRequirement(
                    is_mandatory=True,
                    status=ExperienceRequirementStatus.UNKNOWN,
                    minimum_years=0.0,
                    raw_text=line.strip(),
                    lower_bound_years=0.0,
                    uncertainty_notes=["JD states experience is required but provides no quantifiable duration."]
                )

        # If JD has pre-parsed minimum_experience_years
        if jd.minimum_experience_years > 0:
            return JDExperienceRequirement(
                is_mandatory=True,
                status=ExperienceRequirementStatus.MEETS_REQUIREMENT,
                minimum_years=jd.minimum_experience_years,
                raw_text=f"Minimum {jd.minimum_experience_years} years",
                lower_bound_years=jd.minimum_experience_years,
                upper_bound_years=jd.maximum_experience_years,
                parsing_notes=["Used pre-parsed minimum_experience_years from JobDescription."]
            )

        # Fallback: No experience requirement specified
        return JDExperienceRequirement(
            is_mandatory=False,
            status=ExperienceRequirementStatus.NOT_REQUIRED,
            minimum_years=0.0,
            raw_text="None specified",
            lower_bound_years=0.0,
            parsing_notes=["No mandatory experience requirement specified in JD."]
        )

    # -----------------------------------------------------------------------
    # 2. Date and Interval Parsing
    # -----------------------------------------------------------------------
    def parse_date_intervals(self, text: str) -> Tuple[List[Tuple[int, int]], bool, Optional[str], Optional[str]]:
        """
        Extracts start/end dates and converts them to (start_month_idx, end_month_idx) intervals.
        month_idx = year * 12 + month.
        Returns: (intervals, is_current, start_date_str, end_date_str)
        """
        intervals: List[Tuple[int, int]] = []
        is_current = False
        start_str = None
        end_str = None
        ref_idx = self.config.REFERENCE_YEAR * 12 + self.config.REFERENCE_MONTH

        # 1. Month-name ranges (e.g., 'Jan 2021 - Mar 2023', 'May 2022 - Present')
        for m in DATE_WITH_MONTH_RE.finditer(text):
            try:
                _, sm, sy, _, em, ey, pres = m.groups()
                start_m = MONTH_MAP.get(sm.lower(), 1)
                start_y = int(sy)
                st = start_y * 12 + start_m
                start_str = f"{sm.capitalize()} {sy}"

                if pres:
                    en = ref_idx
                    is_current = True
                    end_str = "Present"
                else:
                    end_m = MONTH_MAP.get(em.lower(), 1) if em else 1
                    end_y = int(ey)
                    en = end_y * 12 + end_m
                    end_str = f"{em.capitalize()} {ey}"

                if en >= st:
                    intervals.append((st, max(en, st + 1)))
            except Exception:
                pass

        # 2. Slash/Dot ranges (e.g., '06/2021 - 08/2022')
        if not intervals:
            for m in DATE_SLASH_RE.finditer(text):
                try:
                    sm, sy, em, ey, pres = m.groups()
                    st = int(sy) * 12 + int(sm)
                    start_str = f"{sm}/{sy}"
                    if pres:
                        en = ref_idx
                        is_current = True
                        end_str = "Present"
                    else:
                        en = int(ey) * 12 + int(em)
                        end_str = f"{em}/{ey}"
                    if en >= st:
                        intervals.append((st, max(en, st + 1)))
                except Exception:
                    pass

        # 3. Year ranges (e.g., '2021 - 2023', '2022 - Present')
        if not intervals:
            for m in YEAR_RANGE_RE.finditer(text):
                try:
                    sy, ey = m.groups()
                    start_y = int(sy)
                    st = start_y * 12 + 1
                    start_str = str(sy)
                    if ey.lower() in ("present", "current", "now", "ongoing"):
                        en = ref_idx
                        is_current = True
                        end_str = "Present"
                    else:
                        end_y = int(ey)
                        en = end_y * 12 + 12
                        end_str = str(ey)
                    if en >= st:
                        intervals.append((st, max(en, st + 1)))
                except Exception:
                    pass

        # 4. Explicit duration mentions (e.g. '6 months', '1.5 years', '9 months')
        if not intervals:
            m_exp = EXPLICIT_MONTHS_RE.search(text)
            if m_exp:
                try:
                    dur_m = float(m_exp.group(1))
                    st = ref_idx - int(dur_m)
                    intervals.append((st, ref_idx))
                    start_str = f"{dur_m} mos duration"
                    end_str = "Completed"
                except Exception:
                    pass

        if not intervals:
            y_exp = EXPLICIT_YEARS_RE.search(text)
            if y_exp:
                try:
                    dur_y = float(y_exp.group(1))
                    dur_m = int(dur_y * 12)
                    st = ref_idx - dur_m
                    intervals.append((st, ref_idx))
                    start_str = f"{dur_y} yrs duration"
                    end_str = "Completed"
                except Exception:
                    pass

        return intervals, is_current, start_str, end_str

    # -----------------------------------------------------------------------
    # 3. Resume Experience Evidence Extraction
    # -----------------------------------------------------------------------
    def extract_experience_records(
        self,
        extraction: DocumentExtractionResult,
    ) -> List[ExperienceRecord]:
        """
        Extracts structured experience records from detected resume sections.
        Strict Rules:
        - Only extracts from 'experience' or 'internship' sections.
        - Excludes 'projects', 'education', 'certifications', 'summary'. Projects are never employment.
        - Classifies role type into PROFESSIONAL, INTERNSHIP, OTHER, or UNKNOWN.
        - Preserves responsibilities text separate from title/organization.
        """
        records: List[ExperienceRecord] = []

        # Find experience and internship sections
        exp_sections = [
            s for s in extraction.sections
            if s.category in ("experience", "internship") and s.text.strip()
        ]

        if not exp_sections:
            # Check if there is an unknown section whose heading indicates employment
            for s in extraction.sections:
                if s.category == "unknown" and re.search(r"\b(employment|work\s+history|career)\b", s.raw_heading, re.I):
                    exp_sections.append(s)

        for sec in exp_sections:
            sec_text = sec.text.strip()
            # Split section text into distinct entry blocks
            # Look for entries separated by blank lines or headers
            blocks = re.split(r"\n\s*\n", sec_text)
            
            for block in blocks:
                block_str = block.strip()
                if not block_str or len(block_str) < 15:
                    continue

                lines = [l.strip() for l in block_str.splitlines() if l.strip()]
                if not lines:
                    continue

                # Header line typically contains title, company, or dates
                header_line = lines[0]
                remaining_lines = lines[1:] if len(lines) > 1 else []
                resp_text = "\n".join(remaining_lines).strip()
                if not resp_text:
                    resp_text = header_line

                # Parse dates & duration
                intervals, is_curr, s_str, e_str = self.parse_date_intervals(block_str)
                dur_months = 0.0
                if intervals:
                    dur_months = sum(en - st for st, en in intervals)
                dur_years = round(dur_months / 12.0, 3)

                # Extract title and organization heuristics
                role_title = header_line
                org_name = "Unknown"
                if " at " in header_line:
                    parts = header_line.split(" at ", 1)
                    role_title = parts[0].strip()
                    org_name = parts[1].split(",")[0].split("(")[0].strip()
                elif " - " in header_line and not any(char.isdigit() for char in header_line.split(" - ")[0]):
                    parts = header_line.split(" - ", 1)
                    role_title = parts[0].strip()
                    org_name = parts[1].split(",")[0].split("(")[0].strip()

                clean_title = re.sub(r"\(.*?\)", "", role_title).strip()

                # Classify experience type
                combined_low = (clean_title + " " + sec.raw_heading).lower()
                uncertainty_notes = []

                if INTERNSHIP_KEYWORDS.search(clean_title) or (sec.category == "internship" and not PROFESSIONAL_KEYWORDS.search(clean_title)):
                    exp_type = ExperienceType.INTERNSHIP
                    type_mult = self.config.MULTIPLIER_INTERNSHIP
                elif OTHER_KEYWORDS.search(clean_title):
                    exp_type = ExperienceType.OTHER
                    type_mult = self.config.MULTIPLIER_OTHER
                elif PROFESSIONAL_KEYWORDS.search(clean_title):
                    exp_type = ExperienceType.PROFESSIONAL
                    type_mult = self.config.MULTIPLIER_PROFESSIONAL
                elif re.search(r"\bprofessional\b", sec.raw_heading, re.I) and not re.search(r"\b(collaborative|effort|contributor|volunteer|student|member)\b", clean_title, re.I):
                    exp_type = ExperienceType.PROFESSIONAL
                    type_mult = self.config.MULTIPLIER_PROFESSIONAL
                else:
                    exp_type = ExperienceType.UNKNOWN
                    type_mult = self.config.MULTIPLIER_UNKNOWN
                    uncertainty_notes.append("Uncertain employment type: applied conservative unknown multiplier.")

                if not intervals:
                    uncertainty_notes.append("Missing dates or duration: duration cannot be verified.")

                record = ExperienceRecord(
                    role_title=role_title,
                    organization=org_name,
                    exp_type=exp_type,
                    start_date_str=s_str,
                    end_date_str=e_str,
                    is_current=is_curr,
                    duration_months=dur_months,
                    duration_years=dur_years,
                    intervals=intervals,
                    responsibilities_text=resp_text,
                    source_section=sec.category,
                    type_multiplier=type_mult,
                    uncertainty_notes=uncertainty_notes,
                )
                records.append(record)

        return records

    # -----------------------------------------------------------------------
    # 4. Overlapping Intervals Handling
    # -----------------------------------------------------------------------
    @staticmethod
    def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merges overlapping or contiguous (start, end) intervals into disjoint spans."""
        if not intervals:
            return []
        sorted_ints = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_ints[0]]
        for st, en in sorted_ints[1:]:
            last_st, last_en = merged[-1]
            if st <= last_en:
                merged[-1] = (last_st, max(last_en, en))
            else:
                merged.append((st, en))
        return merged

    def compute_non_overlapping_years(self, records: List[ExperienceRecord]) -> Tuple[float, bool]:
        """
        Computes total verified non-overlapping experience years across all records.
        Returns: (total_verified_years, had_overlaps)
        """
        all_intervals: List[Tuple[int, int]] = []
        raw_sum_months = 0
        for r in records:
            all_intervals.extend(r.intervals)
            raw_sum_months += r.duration_months

        merged = self.merge_intervals(all_intervals)
        merged_months = sum(en - st for st, en in merged)
        had_overlaps = merged_months < raw_sum_months
        return round(merged_months / 12.0, 3), had_overlaps

    def compute_relevance_adjusted_years(
        self,
        records: List[ExperienceRecord],
    ) -> float:
        """
        Calculates non-overlapping relevance-adjusted experience duration.
        Uses a month-by-month credit model:
        For every month, credit = max(record.type_multiplier * record.duties_relevance for active records).
        Ensures concurrent roles do not double-count months, and higher relevance/type takes precedence.
        """
        if not records:
            return 0.0

        # Map each month index to maximum effective credit
        month_credits: Dict[int, float] = {}

        for r in records:
            eff_rate = r.type_multiplier * r.duties_relevance
            for st, en in r.intervals:
                for m_idx in range(st, en):
                    month_credits[m_idx] = max(month_credits.get(m_idx, 0.0), eff_rate)

        # Fallback if no intervals were extracted but explicit durations exist
        if not month_credits:
            total_adj = sum(r.duration_years * r.type_multiplier * r.duties_relevance for r in records)
            return round(total_adj, 3)

        total_adjusted_months = sum(month_credits.values())
        return round(total_adjusted_months / 12.0, 3)

    # -----------------------------------------------------------------------
    # 5. Evaluate Duties Relevance (BM25 + Semantic)
    # -----------------------------------------------------------------------
    def score_record_duties(
        self,
        record: ExperienceRecord,
        jd_query: str,
        jd_vector: Optional[np.ndarray],
        bm25_scorer_fn,
        semantic_scorer_fn,
    ):
        """
        Calculates lexical BM25 and Semantic similarity of a record's duties text against JD.
        Job title alone does not determine relevance; actual duties text is the primary evidence.
        """
        text = record.responsibilities_text.strip()
        if not text:
            record.bm25_normalized = 0.0
            record.bm25_raw = 0.0
            record.semantic_raw_cosine = 0.0
            record.semantic_calibrated = 0.0
            record.duties_relevance = 0.0
            record.relevance_adjusted_years = 0.0
            record.uncertainty_notes.append("No responsibilities text extracted: duties relevance is 0.0.")
            return

        # Lexical BM25 on actual duties text
        norm_bm25, raw_bm25, _ = bm25_scorer_fn(text, jd_query)
        record.bm25_normalized = round(norm_bm25, 4)
        record.bm25_raw = round(raw_bm25, 4)

        # Semantic similarity on actual duties text
        sem_cal, sem_cos = semantic_scorer_fn(text, jd_vector)
        record.semantic_calibrated = round(sem_cal, 4)
        record.semantic_raw_cosine = round(sem_cos, 4)

        # Blended duties relevance
        blended = (
            self.config.DUTIES_SEMANTIC_WEIGHT * sem_cal
            + self.config.DUTIES_LEXICAL_WEIGHT * norm_bm25
        )
        record.duties_relevance = round(min(max(blended, 0.0), 1.0), 4)
        record.relevance_adjusted_years = round(
            record.duration_years * record.type_multiplier * record.duties_relevance, 3
        )

    # -----------------------------------------------------------------------
    # 6. Full Experience Evaluation & Scoring
    # -----------------------------------------------------------------------
    def evaluate_candidate_experience(
        self,
        extraction: DocumentExtractionResult,
        jd: JobDescription,
        jd_query: str,
        jd_vector: Optional[np.ndarray],
        bm25_scorer_fn,
        semantic_scorer_fn,
    ) -> ExperienceEvaluationResult:
        """
        Complete experience evaluation pipeline:
        1. Parses JD experience requirements.
        2. Extracts candidate records.
        3. Evaluates duties relevance per record.
        4. Merges overlapping intervals.
        5. Computes duration ratio, effective multiplier, and final experience score.
        """
        # 1. JD Requirement
        jd_req = self.parse_jd_experience_requirement(jd)
        req_years = jd_req.minimum_years

        # 2. Extract Records
        records = self.extract_experience_records(extraction)

        # 3. Score Duties Relevance for each record
        for r in records:
            self.score_record_duties(r, jd_query, jd_vector, bm25_scorer_fn, semantic_scorer_fn)

        # 4. Handle Overlapping Intervals & Non-overlapping durations
        verified_total_years, had_overlaps = self.compute_non_overlapping_years(records)
        relevance_adjusted_years = self.compute_relevance_adjusted_years(records)

        uncertainty_notes = list(jd_req.uncertainty_notes)
        for r in records:
            uncertainty_notes.extend(r.uncertainty_notes)

        # 5. Determine Status & Compute Score
        notes = []
        if had_overlaps:
            notes.append("Overlapping employment/internship intervals detected and merged without double-counting.")

        # Case A: JD does NOT require experience
        if not jd_req.is_mandatory or jd_req.status == ExperienceRequirementStatus.NOT_REQUIRED:
            fresher_bonus = 0.0
            if records and relevance_adjusted_years > 0.0 and self.config.ENABLE_FRESHER_EXPERIENCE_BONUS:
                # Optional capped bonus for possessing relevant experience when not required
                fresher_bonus = min(relevance_adjusted_years * 1.0, self.config.FRESHER_EXPERIENCE_BONUS_CAP)
                fresher_bonus = round(fresher_bonus, 2)
                notes.append(f"JD does not require experience: candidate awarded +{fresher_bonus:.2f} fresher experience bonus.")

            return ExperienceEvaluationResult(
                status=ExperienceRequirementStatus.NOT_REQUIRED,
                required_years=0.0,
                verified_total_years=verified_total_years,
                relevance_adjusted_years=relevance_adjusted_years,
                duration_ratio=1.0,
                effective_multiplier=1.0,
                duties_relevance=round(sum(r.duties_relevance for r in records) / max(len(records), 1), 4) if records else 0.0,
                final_experience_score=1.0,  # Neutral full score so core score is not dragged down
                fresher_bonus=fresher_bonus,
                records=records,
                overlapping_intervals_handled=had_overlaps,
                decision_reason="No mandatory experience requirement specified in JD. Experience does not penalize score.",
                uncertainty_notes=uncertainty_notes,
                notes=notes,
            )

        # Case B: Ambiguous requirement
        if jd_req.status == ExperienceRequirementStatus.UNKNOWN:
            return ExperienceEvaluationResult(
                status=ExperienceRequirementStatus.UNKNOWN,
                required_years=0.0,
                verified_total_years=verified_total_years,
                relevance_adjusted_years=relevance_adjusted_years,
                duration_ratio=0.0,
                effective_multiplier=1.0,
                duties_relevance=round(sum(r.duties_relevance for r in records) / max(len(records), 1), 4) if records else 0.0,
                final_experience_score=0.5,
                records=records,
                overlapping_intervals_handled=had_overlaps,
                decision_reason="JD experience requirement is ambiguous; flagged for verification.",
                uncertainty_notes=uncertainty_notes,
                notes=notes,
            )

        # Case C: Mandatory Requirement Established (req_years > 0)
        duration_ratio = min(verified_total_years / req_years, self.config.MAX_DURATION_RATIO) if req_years > 0 else 1.0
        
        # Calculate duration-weighted average type multiplier and duties relevance
        if verified_total_years > 0:
            eff_mult = sum(r.duration_years * r.type_multiplier for r in records) / max(sum(r.duration_years for r in records), 0.001)
            eff_rel = sum(r.duration_years * r.duties_relevance for r in records) / max(sum(r.duration_years for r in records), 0.001)
        else:
            eff_mult = 1.0
            eff_rel = 0.0

        # Transparent Score Formula:
        # Score = (relevance_adjusted_years / req_years) capped at MAX_EXPERIENCE_SCORE
        # Equivalently: duration_ratio x eff_mult x eff_rel
        final_exp_score = min(relevance_adjusted_years / req_years, self.config.MAX_EXPERIENCE_SCORE) if req_years > 0 else 0.0
        final_exp_score = round(max(final_exp_score, 0.0), 4)

        # Determine status
        if verified_total_years >= req_years:
            status = ExperienceRequirementStatus.MEETS_REQUIREMENT
            decision_reason = (
                f"Candidate satisfies experience requirement with {verified_total_years:.2f} verified years "
                f"(relevance-adjusted: {relevance_adjusted_years:.2f} yrs vs {req_years:.2f} yrs required)."
            )
        else:
            status = ExperienceRequirementStatus.BELOW_REQUIREMENT
            decision_reason = (
                f"Candidate verified experience ({verified_total_years:.2f} yrs) is below mandatory requirement "
                f"({req_years:.2f} yrs). Duration ratio: {duration_ratio:.2f}."
            )

        # If dates were missing so duration couldn't be verified
        if not records and not verified_total_years:
            decision_reason = f"No verified employment records found; mandatory requirement is {req_years:.2f} years."
        elif any(r.uncertainty_notes for r in records) and verified_total_years < req_years:
            status = ExperienceRequirementStatus.UNKNOWN
            decision_reason = f"Candidate experience duration is uncertain due to missing dates; flagged for verification."

        return ExperienceEvaluationResult(
            status=status,
            required_years=req_years,
            verified_total_years=verified_total_years,
            relevance_adjusted_years=relevance_adjusted_years,
            duration_ratio=round(duration_ratio, 4),
            effective_multiplier=round(eff_mult, 4),
            duties_relevance=round(eff_rel, 4),
            final_experience_score=final_exp_score,
            records=records,
            overlapping_intervals_handled=had_overlaps,
            decision_reason=decision_reason,
            uncertainty_notes=uncertainty_notes,
            notes=notes,
        )
