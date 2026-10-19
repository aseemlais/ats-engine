import pytest
from app.services.nlp_extractor import NLPExtractorService
from app.utils.skill_normalizer import extract_skills_from_text, normalize_skill
from app.utils.degree_normalizer import normalize_degree, is_degree_satisfied

def test_skill_extraction_and_aliases():
    text = "Proficient in NLP, PyTorch, K8s, Postgres, Scikit-learn, C++, and AWS cloud architecture."
    skills = extract_skills_from_text(text)
    
    assert "nlp" in skills
    assert "pytorch" in skills
    assert "kubernetes" in skills # alias k8s -> kubernetes
    assert "postgresql" in skills # alias postgres -> postgresql
    assert "scikit-learn" in skills
    assert "c++" in skills
    assert "aws" in skills

def test_experience_extraction():
    extractor = NLPExtractorService()
    text = """
    Software Engineer at Acme Corp (2018 - 2022)
    Senior Engineer at Global Tech (2022 - Present)
    """
    years = extractor.extract_experience_years(text)
    assert years >= 6.0

def test_degree_normalization():
    assert normalize_degree("Master of Science in Computer Science") == "master"
    assert normalize_degree("B.Tech in Information Technology") == "bachelor"
    assert normalize_degree("Ph.D in Electrical Engineering") == "doctorate"

    # Hierarchy verification
    assert is_degree_satisfied(["master"], "bachelor") is True
    assert is_degree_satisfied(["bachelor"], "master") is False
