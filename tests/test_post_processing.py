from datetime import datetime
import pytest

from app.models.resume_schema import (
    DegreeTier,
    EducationItem,
    ProcessedResume,
    RawResumeExtraction,
    WorkExperienceItem,
)
from app.services.post_processor import (
    clean_title,
    compute_duration_years,
    map_degree_tier,
    merge_date_intervals,
    normalize_skills,
    parse_date_range,
    post_process_resume,
    sum_merged_intervals_years,
    validate_email,
    validate_phone,
)


# ---------------------------------------------------------------------------
# 1. Title Preservation (Parentheses Protection)
# ---------------------------------------------------------------------------

def test_clean_title_preserves_parentheses_and_tech_stacks():
    """Job titles with parenthesized tech stacks must never be truncated."""
    title = "Full-Stack Developer (TypeScript / Node.js / React)"
    cleaned = clean_title(title)
    assert cleaned == "Full-Stack Developer (TypeScript / Node.js / React)"


def test_clean_title_preserves_nested_brackets_and_subtitles():
    title = "Senior Cloud Architect (AWS / Azure) - Enterprise Systems"
    cleaned = clean_title(title)
    assert cleaned == "Senior Cloud Architect (AWS / Azure) - Enterprise Systems"


def test_clean_title_strips_bullets_quotes_and_outer_dashes():
    title = " • \"Staff Software Engineer\" – "
    cleaned = clean_title(title)
    assert cleaned == "Staff Software Engineer"


def test_clean_title_forbids_em_dash_or_empty():
    assert clean_title("—") is None
    assert clean_title("–") is None
    assert clean_title("Not available") is None
    assert clean_title("") is None
    assert clean_title(None) is None


# ---------------------------------------------------------------------------
# 2. Email Validation & Em-dash Rejection
# ---------------------------------------------------------------------------

def test_validate_email_valid_addresses():
    assert validate_email("candidate@domain.com") == "candidate@domain.com"
    assert validate_email("shivanitbhatt@gmail.com") == "shivanitbhatt@gmail.com"
    assert validate_email("First.Last+dev@sub.company.org") == "first.last+dev@sub.company.org"


def test_validate_email_extracts_from_brackets():
    assert validate_email("<shivanitbhatt@gmail.com>") == "shivanitbhatt@gmail.com"
    assert validate_email("Email: test@example.com;") == "test@example.com"


def test_validate_email_strictly_rejects_em_dashes_and_placeholders():
    assert validate_email("—") is None
    assert validate_email("–") is None
    assert validate_email("-") is None
    assert validate_email("\u2014") is None
    assert validate_email("null") is None
    assert validate_email("None") is None
    assert validate_email("Not available") is None
    assert validate_email("N/A") is None
    assert validate_email("") is None
    assert validate_email(None) is None


def test_validate_email_rejects_syntactically_invalid_strings():
    assert validate_email("not-an-email") is None
    assert validate_email("user@") is None
    assert validate_email("@domain.com") is None
    assert validate_email("user@domain") is None


# ---------------------------------------------------------------------------
# 3. Phone Validation
# ---------------------------------------------------------------------------

def test_validate_phone_valid_formats():
    assert validate_phone("+1-647-780-5215") == "+1-647-780-5215"
    assert validate_phone("(647) 780-5215") == "(647) 780-5215"
    assert validate_phone("+91 98765 43210") == "+91 98765 43210"
    assert validate_phone("9876543210") == "9876543210"


def test_validate_phone_rejects_em_dashes_and_invalid():
    assert validate_phone("—") is None
    assert validate_phone("–") is None
    assert validate_phone("None") is None
    assert validate_phone("123") is None  # Too short
    assert validate_phone("123456789012345678") is None  # Too long (>15 digits)
    assert validate_phone("phone-abc-xyz") is None
    assert validate_phone(None) is None


# ---------------------------------------------------------------------------
# 4. Degree Tier Mapping
# ---------------------------------------------------------------------------

def test_map_degree_tier():
    assert map_degree_tier("Doctor of Philosophy in Computer Science") == DegreeTier.DOCTORATE
    assert map_degree_tier("Ph.D. in Artificial Intelligence") == DegreeTier.DOCTORATE
    assert map_degree_tier("Master of Computer Applications") == DegreeTier.MASTER
    assert map_degree_tier("M.S. in Software Engineering") == DegreeTier.MASTER
    assert map_degree_tier("MBA") == DegreeTier.MASTER
    assert map_degree_tier("Bachelor of Science in Information Technology") == DegreeTier.BACHELOR
    assert map_degree_tier("B.Tech in Computer Science and Engineering") == DegreeTier.BACHELOR
    assert map_degree_tier("BCA") == DegreeTier.BACHELOR
    assert map_degree_tier("Associate of Applied Science") == DegreeTier.ASSOCIATE
    assert map_degree_tier("Diploma in Web Technologies") == DegreeTier.DIPLOMA
    assert map_degree_tier("High School Certificate") == DegreeTier.DIPLOMA
    assert map_degree_tier("Random Course") == DegreeTier.NONE
    assert map_degree_tier(None) == DegreeTier.NONE


# ---------------------------------------------------------------------------
# 5. Skill Normalization & Deduplication
# ---------------------------------------------------------------------------

def test_normalize_skills():
    raw_skills = [
        "reactjs",
        "React.js",
        "nodejs",
        "Node.js",
        "typescript",
        "k8s",
        "kubernetes",
        "postgres",
        "PostgreSQL",
        "python3",
        "—",
        "Docker",
    ]
    normalized = normalize_skills(raw_skills)
    assert normalized == ["React", "Node.js", "TypeScript", "Kubernetes", "PostgreSQL", "Python", "Docker"]


# ---------------------------------------------------------------------------
# 6. Date Parsing & "Present" / "Current" Handling
# ---------------------------------------------------------------------------

def test_parse_date_range_month_year():
    start_dt, end_dt, is_curr = parse_date_range("Dec 2021 – Jun 2025")
    assert start_dt == datetime(2021, 12, 1)
    assert end_dt == datetime(2025, 6, 1)
    assert is_curr is False


def test_parse_date_range_present():
    ref_dt = datetime(2026, 11, 1)
    start_dt, end_dt, is_curr = parse_date_range("Jan 2024 - Present", ref_date=ref_dt)
    assert start_dt == datetime(2024, 1, 1)
    assert end_dt == datetime(2026, 11, 1)
    assert is_curr is True


def test_parse_date_range_current_ongoing():
    ref_dt = datetime(2026, 5, 1)
    start_dt, end_dt, is_curr = parse_date_range("08/2023 to Ongoing", ref_date=ref_dt)
    assert start_dt == datetime(2023, 8, 1)
    assert end_dt == datetime(2026, 5, 1)
    assert is_curr is True


# ---------------------------------------------------------------------------
# 7. Overlapping Jobs & Career Gaps Handling
# ---------------------------------------------------------------------------

def test_overlapping_jobs_merging():
    """
    Two overlapping jobs:
    Job A: Jan 2020 – Dec 2021 (24 months)
    Job B: Jun 2020 – Jun 2022 (25 months)
    Total merged interval: Jan 2020 – Jun 2022 (30 inclusive months = 2.5 years, NOT 4.08 years).
    """
    intervals = [
        (datetime(2020, 1, 1), datetime(2021, 12, 1)),
        (datetime(2020, 6, 1), datetime(2022, 6, 1)),
    ]
    merged = merge_date_intervals(intervals)
    assert len(merged) == 1
    assert merged[0] == (datetime(2020, 1, 1), datetime(2022, 6, 1))

    total_years = sum_merged_intervals_years(intervals)
    assert total_years == 2.5


def test_career_gaps_preserved_and_not_counted():
    """
    Two disjoint jobs separated by a 1-year gap:
    Job 1: Jan 2018 – Dec 2019 (24 months = 2.0 yrs)
    Gap: Jan 2020 – Dec 2020 (12 months gap)
    Job 2: Jan 2021 – Dec 2022 (24 months = 2.0 yrs)
    Total experience must be 4.0 years (the gap is excluded).
    """
    intervals = [
        (datetime(2018, 1, 1), datetime(2019, 12, 1)),
        (datetime(2021, 1, 1), datetime(2022, 12, 1)),
    ]
    merged = merge_date_intervals(intervals)
    assert len(merged) == 2
    assert sum_merged_intervals_years(intervals) == 4.0


# ---------------------------------------------------------------------------
# 8. Exclusion of Internships from Total Experience
# ---------------------------------------------------------------------------

def test_internships_excluded_from_total_experience():
    """Internships must be stored in internship_years and excluded from total_experience_years."""
    raw = RawResumeExtraction(
        candidate_name="Alex Doe",
        email="alex@example.com",
        job_title="Software Engineer",
        work_experience=[
            WorkExperienceItem(
                title="Software Engineering Intern",
                company="Startup Inc",
                dates="Jan 2020 - Jun 2020",  # 6 months = 0.5 yrs
            ),
            WorkExperienceItem(
                title="Software Engineer",
                company="Tech Corp",
                dates="Jul 2020 - Jun 2022",  # 24 months = 2.0 yrs
            ),
        ],
    )

    processed = post_process_resume(raw)
    assert processed.work_experience_years == 2.0
    assert processed.internship_years == 0.5
    assert processed.total_experience_years == 2.0  # Internships strictly excluded


# ---------------------------------------------------------------------------
# 9. Freshers with Zero Experience
# ---------------------------------------------------------------------------

def test_fresher_zero_experience():
    raw = RawResumeExtraction(
        candidate_name="Jane Fresher",
        email="jane@university.edu",
        job_title="Aspiring Web Developer",
        education=[
            EducationItem(
                degree="Bachelor of Technology in Computer Science",
                institution="State University",
                year="2024",
            )
        ],
        work_experience=[],
        projects=["E-Commerce React App"],
    )

    processed = post_process_resume(raw)
    assert processed.total_experience_years == 0.0
    assert processed.work_experience_years == 0.0
    assert processed.internship_years == 0.0
    assert processed.education[0].tier == DegreeTier.BACHELOR


# ---------------------------------------------------------------------------
# 10. End-to-End Comprehensive Resume Processing
# ---------------------------------------------------------------------------

def test_post_process_resume_full_pipeline():
    """Tests the full pipeline on a real-world multi-field resume structure."""
    raw_payload = {
        "candidateName": "Shivani Bhatt",
        "email": "shivanitbhatt@gmail.com",
        "phone": "+1-647-780-5215",
        "location": "Canada (Remote)",
        "jobTitle": "Full-Stack Developer (TypeScript / Node.js / React)",
        "skills": ["typescript", "React.js", "nodejs", "azure", "docker", "Postgres"],
        "education": [
            {
                "degree": "Master of Computer Applications",
                "institution": "CHARUSAT University",
                "year": "2016",
            },
            {
                "degree": "Bachelor of Science in Information Technology",
                "institution": "Sardar Patel University",
                "year": "2014",
            },
        ],
        "workExperience": [
            {
                "title": "Full-Stack Developer (TypeScript / Node.js / React)",
                "company": "Tata Consultancy Services (TCS)",
                "dates": "Dec 2021 – Jun 2025",
            },
            {
                "title": "React Developer",
                "company": "Trackinsight",
                "dates": "Aug 2021 – Nov 2021",
            },
            {
                "title": "Front-End Developer",
                "company": "Revo Karma Hub",
                "dates": "Apr 2017 – Nov 2018",
            },
        ],
    }

    processed = post_process_resume(raw_payload)

    assert processed.candidate_name == "Shivani Bhatt"
    assert processed.email == "shivanitbhatt@gmail.com"
    assert processed.phone == "+1-647-780-5215"
    assert processed.location == "Canada (Remote)"
    assert processed.job_title == "Full-Stack Developer (TypeScript / Node.js / React)"
    assert "TypeScript" in processed.skills
    assert "React" in processed.skills
    assert "Node.js" in processed.skills
    assert "PostgreSQL" in processed.skills

    # Degree classification
    assert processed.education[0].tier == DegreeTier.MASTER
    assert processed.education[1].tier == DegreeTier.BACHELOR

    # Experience math: 3 disjoint full-time jobs
    # Dec 2021 - Jun 2025: 43 mos (3.58 yrs)
    # Aug 2021 - Nov 2021: 4 mos (0.33 yrs)
    # Apr 2017 - Nov 2018: 20 mos (1.67 yrs)
    # Total = 67 months = 5.58 yrs
    assert processed.total_experience_years > 5.0
    assert processed.internship_years == 0.0
