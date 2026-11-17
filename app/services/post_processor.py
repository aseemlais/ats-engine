import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import ftfy

from app.models.resume_schema import (
    DegreeTier,
    EducationItem,
    ProcessedResume,
    RawResumeExtraction,
    WorkExperienceItem,
)

# ---------------------------------------------------------------------------
# Strict Regex & Pattern Definitions
# ---------------------------------------------------------------------------

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Matches standard international or local phone numbers with optional country code,
# parentheses, spaces, dots, or dashes. Digits count checked separately (7-15 digits).
PHONE_REGEX = re.compile(
    r"^\+?[0-9\s\-\.\(\)]+$"
)

EM_DASH_PLACEHOLDERS = {
    "—", "–", "-", "\u2014", "\u2013", "\u2012", "\u2015", "\u2212",
    "null", "none", "not available", "n/a", "undefined", "unknown", ""
}

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

PRESENT_PATTERN = re.compile(
    r"\b(present|current|now|ongoing|till\s+date|today|currently\s+working)\b",
    re.I,
)

INTERNSHIP_PATTERN = re.compile(
    r"\b(intern(?:ship)?|trainee|apprentic(?:e|eship)|co-?op)\b",
    re.I,
)

DEGREE_TIER_PATTERNS: List[Tuple[re.Pattern, DegreeTier]] = [
    (re.compile(r"\b(ph\.?d|doctorate|doctor of philosophy|d\.?phil|m\.?d)\b", re.I), DegreeTier.DOCTORATE),
    (re.compile(r"\b(master(?:'?s)?|m\.?s(?:c)?|m\.?tech|m\.?e|mca|mba|m\.?b\.?a|m\.?phil|m\.?eng|post\s*graduate)\b", re.I), DegreeTier.MASTER),
    (re.compile(r"\b(bachelor(?:'?s)?|b\.?tech|b\.?e|b\.?s(?:c)?|bca|bba|b\.?com|b\.?eng|undergraduate)\b", re.I), DegreeTier.BACHELOR),
    (re.compile(r"\b(associate(?:'?s)?|a\.?s|a\.?a)\b", re.I), DegreeTier.ASSOCIATE),
    (re.compile(r"\b(diploma|polytechnic|certificate|certification)\b", re.I), DegreeTier.DIPLOMA),
]

# Standard technology and skill alias map for canonical normalization
SKILL_ALIASES: Dict[str, str] = {
    "reactjs": "React",
    "react.js": "React",
    "react": "React",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "node": "Node.js",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "python3": "Python",
    "python 3": "Python",
    "py": "Python",
    "python": "Python",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "psql": "PostgreSQL",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "docker": "Docker",
    "aws": "AWS",
    "amazon web services": "AWS",
    "azure": "Azure",
    "microsoft azure": "Azure",
    "gcp": "Google Cloud",
    "google cloud platform": "Google Cloud",
    "google cloud": "Google Cloud",
    "graphql": "GraphQL",
    "rest api": "REST APIs",
    "restful api": "REST APIs",
    "restful apis": "REST APIs",
    "rest apis": "REST APIs",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "ci cd": "CI/CD",
    "html5": "HTML5",
    "html": "HTML5",
    "css3": "CSS3",
    "css": "CSS3",
    "tailwind": "Tailwind CSS",
    "tailwindcss": "Tailwind CSS",
    "tailwind css": "Tailwind CSS",
    "nextjs": "Next.js",
    "next.js": "Next.js",
    "vuejs": "Vue.js",
    "vue.js": "Vue.js",
    "vue": "Vue.js",
    "angularjs": "Angular",
    "angular.js": "Angular",
    "angular": "Angular",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "linux": "Linux",
    "sql": "SQL",
    "mysql": "MySQL",
    "java": "Java",
    "c++": "C++",
    "cpp": "C++",
    "c#": "C#",
    "csharp": "C#",
    "golang": "Go",
    "go": "Go",
    "express": "Express.js",
    "expressjs": "Express.js",
    "express.js": "Express.js",
    "redux": "Redux",
    "typeorm": "TypeORM",
    "jest": "Jest",
    "mocha": "Mocha",
    "chai": "Chai",
    "postman": "Postman",
    "jira": "Jira",
    "splunk": "Splunk",
    "storybook": "Storybook",
    "tdd": "TDD",
    "agile": "Agile",
    "scrum": "Scrum",
}


# ---------------------------------------------------------------------------
# Validation & Normalization Utilities
# ---------------------------------------------------------------------------

def validate_email(email_str: Optional[str]) -> Optional[str]:
    """
    Validates and normalizes candidate email addresses.
    Strictly forces em-dashes ('—', '–', '-'), nulls, or corrupted strings to None.
    Extracts valid email patterns and checks standard syntactic validity.
    """
    if not email_str:
        return None

    cleaned = ftfy.fix_text(str(email_str)).strip()
    if cleaned.lower() in EM_DASH_PLACEHOLDERS:
        return None

    # Strip accidental wrapping angles, brackets, quotes
    cleaned = re.sub(r"^[\s<(\['\"]+|[\s>)\]'\"]+$", "", cleaned).strip()

    # Exact regex match
    if EMAIL_REGEX.match(cleaned):
        return cleaned.lower()

    # If embedded inside noisy string, search for valid email substring
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", cleaned)
    if match:
        candidate = match.group(0).lower()
        if EMAIL_REGEX.match(candidate):
            return candidate

    return None


def validate_phone(phone_str: Optional[str]) -> Optional[str]:
    """
    Validates and cleans phone numbers.
    Strictly forces em-dashes or placeholder strings to None.
    Preserves parenthesized area codes (e.g. (647) 780-5215) and international prefixes.
    Ensures total digit count is between 7 and 15 digits.
    """
    if not phone_str:
        return None

    cleaned = ftfy.fix_text(str(phone_str)).strip()
    if cleaned.lower() in EM_DASH_PLACEHOLDERS:
        return None

    # Strip invalid leading/trailing non-digit characters except + or (
    cleaned = re.sub(r"^[^\d+(]+|[^\d)]+$", "", cleaned).strip()

    # Digits count validation (E.164: 7 to 15 digits)
    digits = re.sub(r"\D", "", cleaned)
    if not (7 <= len(digits) <= 15):
        return None

    # Character validity
    if not PHONE_REGEX.match(cleaned):
        return None

    return cleaned


def clean_title(title_str: Optional[str]) -> Optional[str]:
    """
    Cleans up job titles while STRICTLY PRESERVING parenthesized tech stacks.
    For example: 'Full-Stack Developer (TypeScript / Node.js / React)' is kept intact.
    Strips leading/trailing bullets, quotes, dashes, colons, and residual artifacts.
    """
    if not title_str:
        return None

    cleaned = ftfy.fix_text(str(title_str)).strip()
    if cleaned.lower() in EM_DASH_PLACEHOLDERS:
        return None

    # Strip only leading/trailing punctuation, bullets, quotes, and dashes (NEVER parens!)
    cleaned = cleaned.strip(" \t\n\r,.:;|–-—•*#\"'")

    # Remove residual zero-width or replacement artifacts and normalize non-breaking spaces
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"[\u200b\u200e\u200f\ufeff\ufffd]", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

    if not cleaned or cleaned.lower() in EM_DASH_PLACEHOLDERS:
        return None

    return cleaned


def map_degree_tier(degree_name: Optional[str]) -> DegreeTier:
    """
    Classifies a raw degree string into a canonical degree tier:
    doctorate, master, bachelor, associate, diploma, or none.
    """
    if not degree_name:
        return DegreeTier.NONE

    norm = ftfy.fix_text(str(degree_name)).strip().lower()
    for pattern, tier in DEGREE_TIER_PATTERNS:
        if pattern.search(norm):
            return tier

    return DegreeTier.NONE


def normalize_skills(skills: List[str]) -> List[str]:
    """
    Deduplicates and standardizes skills using the canonical alias taxonomy.
    Preserves skill order while ensuring consistent casing.
    """
    if not skills:
        return []

    normalized_list: List[str] = []
    seen = set()

    for raw_skill in skills:
        if not raw_skill or not isinstance(raw_skill, str):
            continue
        cleaned = ftfy.fix_text(raw_skill).strip(" \t\n\r,.:;|–-—•*\"'")
        cleaned = re.sub(r"[\u200b\u200e\u200f\ufeff\ufffd]", "", cleaned).strip()
        if not cleaned or cleaned.lower() in EM_DASH_PLACEHOLDERS:
            continue

        lookup_key = cleaned.lower()
        canonical = SKILL_ALIASES.get(lookup_key, cleaned)

        canonical_key = canonical.lower()
        if canonical_key not in seen:
            seen.add(canonical_key)
            normalized_list.append(canonical)

    return normalized_list


# ---------------------------------------------------------------------------
# Date Parsing & Experience Math
# ---------------------------------------------------------------------------

def parse_date_endpoint(
    text: str,
    is_end: bool = False,
    ref_date: Optional[datetime] = None,
) -> Tuple[Optional[datetime], bool]:
    """
    Parses a single date endpoint (start date or end date).
    Detects 'Present', 'Current', 'Ongoing', 'Now' and assigns ref_date.
    Returns: (parsed_datetime, is_current_flag)
    """
    if not text:
        return None, False

    t = ftfy.fix_text(text).strip().lower()

    # Detect current / present
    if PRESENT_PATTERN.search(t):
        current_dt = ref_date or datetime.now()
        return datetime(current_dt.year, current_dt.month, 1), True

    # Pattern 1: Month Name + Year (e.g., 'Dec 2021', 'December 2021', 'Dec. 2021')
    m_month_year = re.search(r"([a-z]+)\.?\s+((?:19|20)\d{2})", t)
    if m_month_year:
        month_word = m_month_year.group(1)[:3]
        year_num = int(m_month_year.group(2))
        month_num = MONTH_MAP.get(month_word)
        if month_num:
            return datetime(year_num, month_num, 1), False

    # Pattern 2: MM/YYYY or MM.YYYY or MM-YYYY
    m_slash = re.search(r"\b(0?[1-9]|1[0-2])[\/\.-]((?:19|20)\d{2})\b", t)
    if m_slash:
        month_num = int(m_slash.group(1))
        year_num = int(m_slash.group(2))
        return datetime(year_num, month_num, 1), False

    # Pattern 3: YYYY-MM
    m_iso = re.search(r"\b((?:19|20)\d{2})[\/\.-](0?[1-9]|1[0-2])\b", t)
    if m_iso:
        year_num = int(m_iso.group(1))
        month_num = int(m_iso.group(2))
        return datetime(year_num, month_num, 1), False

    # Pattern 4: Year only (e.g., '2021')
    m_year = re.search(r"\b((?:19|20)\d{2})\b", t)
    if m_year:
        year_num = int(m_year.group(1))
        # Start of year = Jan, End of year = Dec
        month_num = 12 if is_end else 1
        return datetime(year_num, month_num, 1), False

    return None, False


def parse_date_range(
    dates_str: Optional[str],
    ref_date: Optional[datetime] = None,
) -> Tuple[Optional[datetime], Optional[datetime], bool]:
    """
    Parses raw date ranges like:
      - 'Dec 2021 – Jun 2025'
      - '01/2020 - Present'
      - '2019 - 2023'
      - 'March 2022 to Current'
    Returns: (start_date, end_date, is_current)
    """
    if not dates_str:
        return None, None, False

    raw = ftfy.fix_text(dates_str).strip()
    if raw.lower() in EM_DASH_PLACEHOLDERS:
        return None, None, False

    # Split on standard range separators
    parts = re.split(r"\s*(?:–|—|-|to)\s*", raw, maxsplit=1)

    if len(parts) == 2:
        start_dt, _ = parse_date_endpoint(parts[0], is_end=False, ref_date=ref_date)
        end_dt, is_curr = parse_date_endpoint(parts[1], is_end=True, ref_date=ref_date)
        if start_dt and end_dt:
            # Swap if start is later than end
            if start_dt > end_dt:
                start_dt, end_dt = end_dt, start_dt
            return start_dt, end_dt, is_curr
        return start_dt, end_dt, is_curr

    # Single date given (e.g. 'Jun 2023')
    single_dt, is_curr = parse_date_endpoint(raw, is_end=False, ref_date=ref_date)
    if single_dt:
        return single_dt, single_dt, is_curr

    return None, None, False


def compute_duration_years(start_dt: datetime, end_dt: datetime) -> float:
    """
    Computes duration in decimal years between start_dt and end_dt inclusive.
    Example: Dec 2021 to Jun 2025 = 43 months -> 3.58 years.
    """
    if not start_dt or not end_dt:
        return 0.0

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    # Inclusive month calculation
    months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
    return max(0.0, round(months / 12.0, 2))


def merge_date_intervals(intervals: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """
    Merges overlapping or contiguous date intervals to avoid double-counting experience.
    Naturally preserves career gaps by keeping disjoint intervals separate.
    Example:
      [2020-01 to 2021-12] and [2020-06 to 2022-06]
      Merges to: [2020-01 to 2022-06] (30 months = 2.5 years, not 4.08 years).
    """
    if not intervals:
        return []

    # Sort intervals by start date
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = []

    for start_dt, end_dt in sorted_intervals:
        if not merged:
            merged.append((start_dt, end_dt))
            continue

        prev_start, prev_end = merged[-1]

        # Check overlap or contiguous (overlap if start_dt <= prev_end)
        if start_dt <= prev_end:
            # Merge by extending the end date if current interval ends later
            merged[-1] = (prev_start, max(prev_end, end_dt))
        else:
            merged.append((start_dt, end_dt))

    return merged


def sum_merged_intervals_years(intervals: List[Tuple[datetime, datetime]]) -> float:
    """
    Calculates total decimal years across a list of merged non-overlapping intervals.
    """
    if not intervals:
        return 0.0

    merged = merge_date_intervals(intervals)
    total_months = sum(
        (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month) + 1
        for start_dt, end_dt in merged
    )
    return max(0.0, round(total_months / 12.0, 2))


# ---------------------------------------------------------------------------
# Main Orchestrator: Post-Processor
# ---------------------------------------------------------------------------

def post_process_resume(
    raw_input: Union[RawResumeExtraction, Dict[str, Any]],
    reference_date: Optional[datetime] = None,
) -> ProcessedResume:
    """
    Deterministic post-processing pipeline for resume extraction.
    Converts raw LLM extractions into the strictly validated ProcessedResume schema.
    
    Guarantees:
    - Preserves full job titles with parentheses (e.g. 'Full-Stack Developer (TypeScript / React)').
    - Converts em-dashes and corrupted strings in email/phone/name to None.
    - Resolves dates, detects 'is_current', and computes durationYears for each role.
    - Merges overlapping jobs so experience is never double-counted.
    - Strictly excludes internships from total professional experience.
    - Maps degrees to standard canonical DegreeTier.
    - Normalizes skills against standard taxonomy.
    """
    ref_dt = reference_date or datetime.now()

    # Coerce dict to RawResumeExtraction model
    if isinstance(raw_input, dict):
        raw = RawResumeExtraction.model_validate(raw_input)
    else:
        raw = raw_input

    # 1. Candidate Name
    cand_name = None
    if raw.candidate_name:
        cand_name = ftfy.fix_text(raw.candidate_name).strip()
        cand_name = re.sub(r"[\u200b\u200e\u200f\ufeff\ufffd]", "", cand_name).strip(" \t\n\r,.:;|–-—•*#\"'")
        if cand_name.lower() in EM_DASH_PLACEHOLDERS:
            cand_name = None

    # 2. Email & Phone
    email = validate_email(raw.email)
    phone = validate_phone(raw.phone)

    # 3. Location
    location = None
    if raw.location:
        loc_cleaned = ftfy.fix_text(raw.location).strip(" \t\n\r,.:;|–-—\"'")
        if loc_cleaned.lower() not in EM_DASH_PLACEHOLDERS:
            location = loc_cleaned

    # 4. Job Title
    job_title = clean_title(raw.job_title)

    # 5. Work Experience & Date Math
    processed_experience: List[WorkExperienceItem] = []
    full_time_intervals: List[Tuple[datetime, datetime]] = []
    internship_intervals: List[Tuple[datetime, datetime]] = []

    for item in raw.work_experience:
        cleaned_role = clean_title(item.title) or "Role not identified"
        cleaned_company = None
        if item.company:
            comp_val = ftfy.fix_text(item.company).strip(" \t\n\r,.:;|–-—\"'")
            if comp_val.lower() not in EM_DASH_PLACEHOLDERS:
                cleaned_company = comp_val

        # Detect internship from title or explicit flag
        is_intern = item.is_internship or bool(INTERNSHIP_PATTERN.search(cleaned_role))

        # Parse date range
        start_dt, end_dt, is_curr = parse_date_range(item.dates, ref_date=ref_dt)

        # Compute role duration
        duration_years = 0.0
        if start_dt and end_dt:
            duration_years = compute_duration_years(start_dt, end_dt)
            if is_intern:
                internship_intervals.append((start_dt, end_dt))
            else:
                full_time_intervals.append((start_dt, end_dt))
        elif item.duration_years > 0.0:
            duration_years = round(float(item.duration_years), 2)

        processed_experience.append(
            WorkExperienceItem(
                title=cleaned_role,
                company=cleaned_company,
                dates=item.dates,
                duration_years=duration_years,
                is_current=is_curr or item.is_current,
                is_internship=is_intern,
                description=item.description,
            )
        )

    # If top-level job_title is missing, infer from most recent non-intern work experience
    if not job_title and processed_experience:
        for exp in processed_experience:
            if not exp.is_internship:
                job_title = exp.title
                break
        if not job_title:
            job_title = processed_experience[0].title

    # 6. Calculate Experience Totals (Merging Overlaps)
    # Total Professional Experience STRICTLY excludes internships
    work_exp_years = sum_merged_intervals_years(full_time_intervals)
    intern_exp_years = sum_merged_intervals_years(internship_intervals)
    total_exp_years = work_exp_years

    # 7. Education & Degree Tiers
    processed_education: List[EducationItem] = []
    degree_names: List[str] = []

    for edu in raw.education:
        if not edu.degree:
            continue
        deg_str = ftfy.fix_text(edu.degree).strip(" \t\n\r,.:;|–-—\"'")
        if deg_str.lower() in EM_DASH_PLACEHOLDERS:
            continue

        tier = map_degree_tier(deg_str) if edu.tier == DegreeTier.NONE else edu.tier
        inst_str = None
        if edu.institution:
            inst_val = ftfy.fix_text(edu.institution).strip(" \t\n\r,.:;|–-—\"'")
            if inst_val.lower() not in EM_DASH_PLACEHOLDERS:
                inst_str = inst_val

        processed_education.append(
            EducationItem(
                degree=deg_str,
                tier=tier,
                institution=inst_str,
                field_of_study=edu.field_of_study,
                year=edu.year,
            )
        )
        degree_names.append(deg_str)

    # 8. Skills Normalization
    normalized_skills = normalize_skills(raw.skills)

    # 9. Certifications & Projects
    cleaned_certs = [
        ftfy.fix_text(c).strip(" \t\n\r,.:;|–-—\"'")
        for c in raw.certifications
        if c and ftfy.fix_text(c).strip(" \t\n\r,.:;|–-—\"'").lower() not in EM_DASH_PLACEHOLDERS
    ]

    cleaned_projects = [
        ftfy.fix_text(p).strip(" \t\n\r,.:;|–-—\"'")
        for p in raw.projects
        if p and ftfy.fix_text(p).strip(" \t\n\r,.:;|–-—\"'").lower() not in EM_DASH_PLACEHOLDERS
    ]

    return ProcessedResume(
        candidate_name=cand_name,
        email=email,
        phone=phone,
        location=location,
        job_title=job_title or "Role not identified",
        total_experience_years=total_exp_years,
        work_experience_years=work_exp_years,
        internship_years=intern_exp_years,
        skills=normalized_skills,
        education=processed_education,
        degrees=degree_names,
        work_experience=processed_experience,
        certifications=cleaned_certs,
        projects=cleaned_projects,
    )
