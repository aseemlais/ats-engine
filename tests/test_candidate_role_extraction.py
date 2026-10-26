import pytest
from app.services.nlp_extractor import NLPExtractorService
from app.adapters.screening_adapter import ATSScreeningAdapter
from app.models.resume import ParsedResume

def test_current_role_extraction():
    extractor = NLPExtractorService()
    text = """
    Jane Doe
    jane@example.com
    
    EXPERIENCE
    Software Engineer at Startup Co
    2019 - 2021
    - Developed backend services
    
    Senior Software Engineer at Tech Global
    2021 - Present
    - Leading the distributed systems team
    """
    parsed = extractor.extract_structured("cand1", "cand1.txt", "h1", text, text)
    assert parsed.current_role == "Senior Software Engineer"

def test_most_recent_past_role_extraction():
    extractor = NLPExtractorService()
    text = """
    John Smith
    john@example.com
    
    EXPERIENCE
    Junior Analyst at Retail Analytics (2018 - 2020)
    Data Analyst at FinTech Corp (2020 - 2023)
    """
    parsed = extractor.extract_structured("cand2", "cand2.txt", "h2", text, text)
    assert parsed.current_role == "Data Analyst"

def test_chronological_multiple_roles():
    extractor = NLPExtractorService()
    text = """
    Alice Wonder
    alice@example.com
    
    WORK EXPERIENCE
    Associate Developer | 2016 - 2018
    Software Engineer | 2018 - 2021
    Engineering Lead | 2021 - 2024
    """
    parsed = extractor.extract_structured("cand3", "cand3.txt", "h3", text, text)
    assert parsed.current_role == "Engineering Lead"

def test_internship_title_preservation():
    extractor = NLPExtractorService()
    text = """
    Bob Builder
    bob@example.com
    
    INTERNSHIPS
    Machine Learning Intern at AI Research Labs
    June 2023 - August 2023
    - Built computer vision pipelines using PyTorch
    """
    parsed = extractor.extract_structured("cand4", "cand4.txt", "h4", text, text)
    assert parsed.current_role == "Machine Learning Intern"

def test_fresher_identification():
    extractor = NLPExtractorService()
    text = """
    Aarav Sharma
    aarav@example.com
    
    EDUCATION
    B.Tech in Computer Science, 2020 - 2024
    
    PROJECTS
    Automated ATS Resume Parser
    - Built using Python and FastAPI
    
    SKILLS
    Python, SQL, React
    """
    parsed = extractor.extract_structured("cand5", "cand5.txt", "h5", text, text)
    assert parsed.current_role == "Fresher"

def test_unclear_role():
    extractor = NLPExtractorService()
    text = """
    Random Person
    random@example.com
    Hobbies: Reading, Chess.
    """
    parsed = extractor.extract_structured("cand6", "cand6.txt", "h6", text, text)
    assert parsed.current_role == "Role not identified"

def test_missing_dates_supported_role():
    extractor = NLPExtractorService()
    text = """
    Charlie Brown
    charlie@example.com
    
    EXPERIENCE
    Cloud DevOps Engineer at CloudWorks
    - Managed AWS infrastructure and Kubernetes clusters
    """
    parsed = extractor.extract_structured("cand7", "cand7.txt", "h7", text, text)
    assert parsed.current_role == "Cloud DevOps Engineer"

def test_screening_adapter_returns_candidate_role_not_jd_title():
    adapter = ATSScreeningAdapter()
    
    resume_text = """
    Elena Rostova
    elena@example.com
    
    EXPERIENCE
    Senior Machine Learning Engineer at AI Systems Inc.
    2021 - Present
    - Built LLM serving infrastructure and PyTorch training pipelines
    
    Machine Learning Engineer at DataCorp
    2018 - 2021
    - Built recommendation algorithms
    
    SKILLS
    Python, PyTorch, Kubernetes, SQL
    """
    
    res = adapter.evaluate_candidate(
        job_title="Target JD Job Title - Senior Backend Architect",  # JD Title!
        job_description="Seeking a backend architect skilled in Python and SQL.",
        resume_text=resume_text
    )
    
    # Must be candidate's actual role, NOT the target JD job title
    assert res["currentRole"] == "Senior Machine Learning Engineer"
    assert res["jobRole"] == "Senior Machine Learning Engineer"
    assert res["currentRole"] != "Target JD Job Title - Senior Backend Architect"
