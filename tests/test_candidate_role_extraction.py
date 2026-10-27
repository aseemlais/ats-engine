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


def test_tool_or_skill_not_selected_as_role():
    extractor = NLPExtractorService()
    text = """
    Priya Patel
    priya@example.com

    SKILLS & TOOLS
    Languages: Python, Java, C++
    Tools: Jupyter Notebook, VS Code, Git, Docker, Postman

    WORK EXPERIENCE
    Generative AI Intern at Cognizant
    01/2024 - Present
    - Developed RAG systems using LangChain and Hugging Face transformers
    """
    parsed = extractor.extract_structured("cand_tool", "cand_tool.txt", "ht1", text, text)
    assert parsed.current_role == "Generative AI Intern"
    assert "Jupyter" not in parsed.current_role

def test_employment_date_on_separate_line_not_selected():
    extractor = NLPExtractorService()
    text = """
    Karthik Raman
    karthik@example.com

    INTERNSHIPS
    Python Full Stack Intern, Emglitz Technologies
    06/2025 – 06/2025
    - Developed full stack web applications using Python, Django, React
    """
    parsed = extractor.extract_structured("cand_date", "cand_date.txt", "hd1", text, text)
    assert parsed.current_role == "Python Full Stack Intern"
    assert "2025" not in parsed.current_role

def test_internship_with_company_and_title_on_separate_lines():
    extractor = NLPExtractorService()
    text = """
    Sneha Reddy
    sneha@example.com

    INTERNSHIP
    Embuzz Technologies Pvt Limited
    Embedded Systems Internship
    07/2024 – 08/2024
    - Programmed microcontrollers in C
    """
    parsed = extractor.extract_structured("cand_sep", "cand_sep.txt", "hs1", text, text)
    assert parsed.current_role == "Embedded Systems Intern"
    assert "Embuzz" not in parsed.current_role
    assert "2024" not in parsed.current_role

def test_project_title_not_selected_over_internship():
    extractor = NLPExtractorService()
    text = """
    Rohan Verma
    rohan@example.com

    ACADEMIC PROJECTS
    Hand Gesture Recognition (01/2024 – 03/2024)
    - Built a real-time gesture control pipeline using OpenCV and MediaPipe

    INTERNSHIP EXPERIENCE
    Data Science Intern at Innovate Analytics
    05/2023 – 07/2023
    - Performed customer segmentation and predictive modeling
    """
    parsed = extractor.extract_structured("cand_proj", "cand_proj.txt", "hp1", text, text)
    assert parsed.current_role == "Data Science Intern"
    assert "Hand Gesture" not in parsed.current_role

def test_concatenated_company_title_academy_cleaned():
    extractor = NLPExtractorService()
    text = """
    Vikas Krishnan
    vikas@example.com

    INTERNSHIPS
    AI Fullstack Engineer Intern, Future Intern Private Limited, Vinsup Skill Academy
    05/2024 - 07/2024
    - Built AI-powered web applications
    """
    parsed = extractor.extract_structured("cand_concat", "cand_concat.txt", "hc1", text, text)
    assert parsed.current_role == "AI Fullstack Engineer Intern"
    assert "Future Intern" not in parsed.current_role
    assert "Vinsup" not in parsed.current_role

def test_multiple_internships_recency_selection():
    extractor = NLPExtractorService()
    text = """
    Meera Nair
    meera@example.com

    INTERNSHIPS
    Web Development Intern at CodeLabs
    01/2023 - 03/2023

    Machine Learning Intern at NeuralTech
    06/2023 - 09/2023
    """
    parsed = extractor.extract_structured("cand_multi_int", "cand_multi_int.txt", "hm1", text, text)
    assert parsed.current_role == "Machine Learning Intern"
