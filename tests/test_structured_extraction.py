import pytest
import json
from app.services.structured_extractor import (
    clean_extracted_text,
    clean_extracted_email,
    parse_llm_json_response,
    StructuredResumeExtractor,
    DATABRICKS_EXTRACTION_PROMPT,
)
from app.services.resume_parser import ResumeParserService


def test_clean_extracted_email_valid():
    """Valid email addresses are extracted and normalized."""
    assert clean_extracted_email("recruiter@datapattern.ai") == "recruiter@datapattern.ai"
    assert clean_extracted_email("  john.doe+work@example.com  ") == "john.doe+work@example.com"
    assert clean_extracted_email("<jane_smith@corp.org>") == "jane_smith@corp.org"


def test_clean_extracted_email_forbids_em_dashes_and_blanks():
    """Specifically forbids placeholder em-dashes ('—', '–', '-'), blank strings, and corrupted strings."""
    assert clean_extracted_email("—") is None
    assert clean_extracted_email("–") is None
    assert clean_extracted_email("-") is None
    assert clean_extracted_email("") is None
    assert clean_extracted_email("   ") is None
    assert clean_extracted_email("None") is None
    assert clean_extracted_email("null") is None
    assert clean_extracted_email("Not available") is None
    assert clean_extracted_email("N/A") is None
    assert clean_extracted_email("corrupted-email-without-at-sign.com") is None


def test_clean_extracted_text_removes_encoding_artifacts():
    """Cleans up candidate names and job titles, removing residual encoding artifacts and empty quotes."""
    # Empty quotes or corrupted quotes
    assert clean_extracted_text("''") is None
    assert clean_extracted_text('""') is None
    assert clean_extracted_text("''Senior Software Engineer''") == "Senior Software Engineer"
    
    # Mojibake and zero-width artifacts
    dirty_title = "Lead\u200b â€¢ Developer"
    cleaned = clean_extracted_text(dirty_title)
    assert "\u200b" not in cleaned

    # Placeholder values
    assert clean_extracted_text("—") is None
    assert clean_extracted_text("Unknown") is None
    assert clean_extracted_text("Not available") is None


def test_parse_llm_json_response_strips_markdown_code_blocks():
    """Strips markdown code blocks (e.g., ```json ... ```) and parses valid JSON."""
    markdown_output = """```json
{
  "candidateName": "Alice Johnson",
  "email": "alice.j@example.com",
  "phone": "+1 555-0199",
  "jobTitle": "Data Scientist",
  "skills": ["Python", "Machine Learning", "SQL"],
  "experienceYears": 4.5,
  "education": ["M.S. in Computer Science"]
}
```"""
    parsed = parse_llm_json_response(markdown_output)
    assert parsed is not None
    assert parsed["candidateName"] == "Alice Johnson"
    assert parsed["email"] == "alice.j@example.com"
    assert parsed["jobTitle"] == "Data Scientist"
    assert "Python" in parsed["skills"]
    assert parsed["experienceYears"] == 4.5


def test_parse_llm_json_response_handles_em_dash_email_in_json():
    """Instructs or handles when email is given as em-dash in JSON, turning it to None."""
    llm_output_with_em_dash = """{
  "candidateName": "Bob Williams",
  "email": "—",
  "jobTitle": "DevOps Engineer",
  "skills": ["Docker", "Kubernetes"]
}"""
    parsed = parse_llm_json_response(llm_output_with_em_dash)
    assert parsed is not None
    assert parsed["candidateName"] == "Bob Williams"
    assert parsed["email"] is None  # Strictly None, em-dash forbidden
    assert parsed["jobTitle"] == "DevOps Engineer"


def test_parse_llm_json_response_safe_fallback_on_corrupted_json():
    """Returns None on corrupted JSON so the safe fallback structure is invoked."""
    corrupted_output = "I cannot generate valid JSON for this resume."
    assert parse_llm_json_response(corrupted_output) is None
    assert parse_llm_json_response("{invalid json missing quotes: 123}") is None


def test_structured_resume_extractor_safe_fallback():
    """Verifies that the safe fallback structure produces clean candidate details with ftfy."""
    extractor = StructuredResumeExtractor()
    sample_resume = """
    Johnâ€™s Doe
    Email: john.doe@techcorp.com
    Phone: (555) 123-4567
    
    Professional Experience
    Senior Software Engineer at ABC Tech (2020 - Present)
    - Developed high-performance distributed systems using Python and Go.
    - Optimized database queries in PostgreSQL.
    
    Education
    B.S. in Computer Science, State University
    
    Technical Skills
    Python, Go, PostgreSQL, Docker, AWS
    """
    fallback = extractor.get_safe_fallback_structure(sample_resume)
    assert fallback["candidateName"] in ("John'S Doe", "John's Doe", "John Doe", "Unknown")
    assert fallback["email"] == "john.doe@techcorp.com"
    assert "Senior Software Engineer" in fallback["jobTitle"] or "Software Engineer" in fallback["jobTitle"]
    assert fallback["source"] == "deterministic_fallback"


def test_databricks_prompt_schema():
    """Ensures Databricks prompt specifies the strict schema and em-dash prohibition."""
    assert "databricks-gpt-oss-120b" in DATABRICKS_EXTRACTION_PROMPT
    assert "STRICT EXTRACTION RULES" in DATABRICKS_EXTRACTION_PROMPT
    assert "STRICTLY FORBIDDEN from outputting placeholder em-dashes" in DATABRICKS_EXTRACTION_PROMPT
    assert "candidateName" in DATABRICKS_EXTRACTION_PROMPT
    assert "jobTitle" in DATABRICKS_EXTRACTION_PROMPT
    assert "email" in DATABRICKS_EXTRACTION_PROMPT
