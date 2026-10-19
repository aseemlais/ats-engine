"""Skill / eligibility / experience / certification tests (pure Python, no embedding model)."""
from pathlib import Path
import types
import pytest

from app.config import config
from app.models.job import JobDescription
from app.services.certification_matcher import CertificationMatcherService
from app.services.experience_matcher import ExperienceMatcherService
from app.services.hard_filter import HardFilterService
from app.services.jd_parser import JDParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.skill_matcher import SkillMatcherService

FIX = Path(__file__).parent / "fixtures"
nlp = NLPExtractorService()
JD = JDParserService().parse_job_description((FIX / "synthetic_trainee_jd.txt").read_text(), job_id="T")


def resume(text, cid="C"):
    return nlp.extract_structured(cid, "t", "h", text, text)


RESUME = resume((FIX / "synthetic_trainee_resume.txt").read_text())


# ------------------------------------------------------------ skills & eligibility (SYNTHETIC trainee fixture)
def test_fixture_skills_extracted():
    assert {"sql", "python", "pandas", "numpy", "spark", "databricks", "aws", "cloud computing", "databases", "data engineering"} <= set(RESUME.skills)


def test_trainee_skill_detail():
    d = SkillMatcherService().evaluate_detailed(RESUME, JD)
    assert d["missing_required"] == [] and d["coverage"]["required"] == 1.0
    assert any("mysql / sql" in a for a in d["alternatives_satisfied"])
    assert d["score"] == pytest.approx(1.0)


def test_sql_or_mysql_either_satisfies_and_both_not_required():
    for have in (["python", "sql"], ["python", "mysql", "sql"]):                    # MySQL implies SQL on resumes
        r = types.SimpleNamespace(skills=have)
        assert SkillMatcherService().evaluate_detailed(r, JD)["missing_required"] == []
    only_mysql = resume("Skills: Python, MySQL")
    assert "sql" in only_mysql.skills
    assert SkillMatcherService().evaluate_detailed(only_mysql, JD)["missing_required"] == []


def test_missing_group_reported_as_one_slot_without_free_credit():
    r = types.SimpleNamespace(skills=["python"])
    d = SkillMatcherService().evaluate_detailed(r, JD)
    assert d["missing_required"] == ["mysql / sql (any one)"] and d["coverage"]["required"] == 0.5
    assert d["score"] < 0.5                                                          # preferred/supporting are also empty


def test_related_skill_does_not_satisfy_mandatory():
    r = types.SimpleNamespace(skills=["python", "postgresql"])      # a different SQL database is NOT listed as accepted
    assert SkillMatcherService().evaluate_detailed(r, JD)["missing_required"] == ["mysql / sql (any one)"]


def test_nonmandatory_skill_never_rejects_candidate():
    r = types.SimpleNamespace(skills=["python", "sql"], experience_years=0.0, degrees=["master"])
    for strict in (False, True):
        out = HardFilterService(enabled=True, strict=strict).evaluate(r, JD)
        assert out["eligible"] and out["missing_mandatory"] == []


def test_eligibility_separate_from_fit_score():
    r = types.SimpleNamespace(skills=["spark", "databricks", "aws"], experience_years=0.0, degrees=["master"])
    assert SkillMatcherService().evaluate_detailed(r, JD)["score"] > 0.1                # some fit evidence ...
    out = HardFilterService(enabled=True, strict=False).evaluate(r, JD)
    assert not out["eligible"] and "required technical skills" in out["reasons"][0]       # ... but still ineligible
    strict = HardFilterService(enabled=True, strict=True).evaluate(r, JD)
    assert set(strict["missing_mandatory"]) == {"mysql / sql (any one)", "python"}


# ------------------------------------------------------------ experience
def _r(work=0.0, intern=0.0, total=None, title=""):
    return types.SimpleNamespace(work_experience_years=work, internship_years=intern,
                                 experience_years=work + 0.5 * max(intern, 0) if total is None else total,
                                 section_texts={"experience": title, "internship": ""}, job_titles=[title] if title else [])


def test_no_requirement_means_not_applicable_and_no_fabricated_score():
    m = ExperienceMatcherService()
    assert not m.is_applicable(JD)
    d = m.evaluate_detailed(RESUME, JD)
    assert d["applicable"] is False and d["score"] is None


def test_explicit_minimum_scored_transparently():
    jd = JobDescription(title="Data Analyst", minimum_experience_years=4.0, raw_text="x")
    m = ExperienceMatcherService()
    assert m.evaluate_detailed(_r(work=2.0), jd)["score"] == pytest.approx(0.5)
    assert m.evaluate_detailed(_r(work=4.0), jd)["score"] == 1.0
    d = m.evaluate_detailed(_r(work=2.0, intern=1.0), jd)                    # 2 + 0.4*1 = 2.4 confirmed years
    assert d["confirmed_years"] == 2.4 and d["score"] == pytest.approx(0.6)


def test_maximum_range_and_undated_internship_and_projects():
    jd = JobDescription(title="Analyst", minimum_experience_years=2.0, maximum_experience_years=4.0, raw_text="x")
    m = ExperienceMatcherService()
    assert m.evaluate_detailed(_r(work=3.0), jd)["score"] == 1.0
    assert m.evaluate_detailed(_r(work=8.0), jd)["score"] == pytest.approx(0.5)      # above maximum: floor 0.5
    assert m.evaluate_detailed(_r(intern=-0.01), jd)["confirmed_years"] == 0.0         # unverified
    assert m.evaluate_detailed(RESUME, jd)["confirmed_years"] == pytest.approx(0.4 * RESUME.internship_years, abs=0.01)  # projects add nothing


def test_resume_extraction_internship_dated_and_not_professional():
    assert RESUME.work_experience_years == 0.0 and RESUME.internship_years == pytest.approx(0.17, abs=0.01)


# ------------------------------------------------------------ certifications
def _c(certs):
    return types.SimpleNamespace(certifications=certs, section_texts={}, cleaned_text="")


CERT_JD_PREF = JobDescription(title="Data Engineer", raw_text="Nice to have\n- SQL certification", required_skills=["sql"], preferred_skills=["databricks"])


def test_certification_levels_and_not_applicable():
    m = CertificationMatcherService()
    assert m.requirement_level(JD) == "none" and m.evaluate_detailed(RESUME, JD)["applicable"] is False
    assert m.requirement_level(CERT_JD_PREF) == "preferred"


def test_relevance_not_presence():
    m = CertificationMatcherService()
    assert m.evaluate_certifications(_c(["Yoga Teacher Training Certificate"]), CERT_JD_PREF) == 0.0     # unrelated
    assert m.evaluate_certifications(_c(["IBM SQL for Data Science"]), CERT_JD_PREF) == 0.5              # one relevant
    assert m.evaluate_certifications(_c(["IBM SQL for Data Science", "Databricks Fundamentals"]), CERT_JD_PREF) == 1.0
    assert m.evaluate_certifications(_c(["Coursera Digital Marketing", "Udemy Painting", "NPTEL Poetry"]), CERT_JD_PREF) == 0.0  # issuer names earn nothing
    assert m.evaluate_certifications(_c([]), CERT_JD_PREF) == 0.0


def test_legacy_cert_policy_restores_baseline(monkeypatch):
    monkeypatch.setattr(config, "na_policy", "legacy")
    assert CertificationMatcherService().evaluate_certifications(_c([]), JD) == 0.60
