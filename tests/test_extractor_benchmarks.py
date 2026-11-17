import pytest
from app.models.resume_schema import ProcessedResume, RawResumeExtraction
from app.services.post_processor import post_process_resume
from scripts.evaluate_extractor import (
    check_hallucination,
    compute_sequence_ratio,
    compute_set_metrics,
    compute_token_f1,
    is_title_truncated,
)


# ---------------------------------------------------------------------------
# 1. Regression Test: Em-dash Email ('—') Elimination
# ---------------------------------------------------------------------------

def test_regression_em_dash_email_handling():
    """
    Guarantees that em-dashes ('—', '–', '\u2014') and placeholder strings
    in raw extractions strictly produce null in the validated schema.
    """
    bad_emails = ["—", "–", "-", "\u2014", "\u2013", "\u2212", "null", "None", "not available", "N/A", ""]
    for bad_email in bad_emails:
        raw = RawResumeExtraction(
            candidate_name="Test Candidate",
            email=bad_email,
            job_title="Software Engineer",
        )
        processed = post_process_resume(raw)
        assert processed.email is None, f"Failed for email placeholder: {repr(bad_email)}"


# ---------------------------------------------------------------------------
# 2. Regression Test: Parenthesized Title Preservation
# ---------------------------------------------------------------------------

def test_regression_parenthesized_title_preservation():
    """
    Guarantees that parenthesized tech stacks in job titles are preserved
    fully without truncation at commas, slashes, or dashes.
    """
    titles = [
        "Full-Stack Developer (TypeScript / Node.js / React)",
        "Senior Cloud Architect (AWS / Azure) - Enterprise Systems",
        "Machine Learning Engineer (NLP / PyTorch / LLMs)",
        "Backend Software Engineer (Go / Kubernetes / gRPC)",
    ]
    for title in titles:
        raw = RawResumeExtraction(
            candidate_name="Alex Developer",
            job_title=title,
            work_experience=[
                {"title": title, "company": "Tech Corp", "dates": "2021 - 2024"}
            ],
        )
        processed = post_process_resume(raw)
        assert processed.job_title == title
        assert processed.work_experience[0].title == title
        assert ")" in processed.job_title


# ---------------------------------------------------------------------------
# 3. Regression Test: Missing Sections Graceful Handling
# ---------------------------------------------------------------------------

def test_regression_missing_sections_graceful_handling():
    """
    Tests resumes with missing sections (e.g. no education, no projects, or fresher)
    to ensure 100% schema compliance without throwing exceptions.
    """
    # 1. No education, no certifications
    raw_no_edu = RawResumeExtraction(
        candidate_name="Self-Taught Coder",
        email="coder@example.com",
        job_title="Frontend Developer",
        skills=["React", "JavaScript"],
        education=[],
        work_experience=[{"title": "Web Developer", "dates": "2022 - 2024"}],
    )
    processed_no_edu = post_process_resume(raw_no_edu)
    assert processed_no_edu.education == []
    assert processed_no_edu.degrees == []
    assert processed_no_edu.skills == ["React", "JavaScript"]

    # 2. Fresher with no work experience
    raw_fresher = RawResumeExtraction(
        candidate_name="Fresher Graduate",
        email="fresher@example.com",
        job_title="Junior Engineer",
        education=[{"degree": "Bachelor of Technology", "year": "2024"}],
        work_experience=[],
    )
    processed_fresher = post_process_resume(raw_fresher)
    assert processed_fresher.work_experience == []
    assert processed_fresher.total_experience_years == 0.0


# ---------------------------------------------------------------------------
# 4. Regression Test: Mojibake Encoding Resilience
# ---------------------------------------------------------------------------

def test_regression_mojibake_encoding_resilience():
    """
    Ensures that text corrupted with unicode replacement artifacts (\ufffd, non-breaking spaces)
    is cleaned without breaking schema validation.
    """
    noisy_name = "Nicole\ufffd Mclaughlin"
    noisy_title = "Data\u00a0Engineer\u200b (Python\u00a0/\u00a0Spark)"
    noisy_skill = "React\ufffd"

    raw = RawResumeExtraction(
        candidate_name=noisy_name,
        job_title=noisy_title,
        skills=[noisy_skill, "Docker"],
    )
    processed = post_process_resume(raw)

    assert "\ufffd" not in (processed.candidate_name or "")
    assert "\u200b" not in (processed.job_title or "")
    assert processed.job_title == "Data Engineer (Python / Spark)"
    assert "Docker" in processed.skills


# ---------------------------------------------------------------------------
# 5. Metric Evaluators Correctness Tests
# ---------------------------------------------------------------------------

def test_metric_is_title_truncated():
    gold = "Full-Stack Developer (TypeScript / Node.js / React)"
    truncated_pred = "Full-Stack Developer (TypeScript"
    complete_pred = "Full-Stack Developer (TypeScript / Node.js / React)"

    assert is_title_truncated(truncated_pred, gold) is True
    assert is_title_truncated(complete_pred, gold) is False


def test_metric_compute_set_metrics():
    gold = ["React", "TypeScript", "Node.js"]
    pred = ["React", "TypeScript", "Python"]

    prec, rec, f1 = compute_set_metrics(pred, gold)
    assert round(prec, 2) == 0.67
    assert round(rec, 2) == 0.67
    assert round(f1, 2) == 0.67


def test_metric_check_hallucination():
    source_text = "Experienced with React, Node.js and TypeScript on Azure."
    grounded = "TypeScript"
    hallucinated = "Ruby on Rails Cobol Mainframe"

    assert check_hallucination(grounded, source_text) is False
    assert check_hallucination(hallucinated, source_text) is True
