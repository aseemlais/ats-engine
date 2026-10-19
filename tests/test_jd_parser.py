"""JD parser tests: title, negation, tiers, alternatives, experience, education. Pure Python, no model."""
from pathlib import Path
import pytest
from app.services.jd_parser import JDParserService

FIX = Path(__file__).parent / "fixtures"
parse = lambda t: JDParserService().parse_job_description(t, job_id="T")


@pytest.fixture(scope="module")
def trainee():
    return parse((FIX / "synthetic_trainee_jd.txt").read_text())


# ------------------------------------------------------------ diagnostic trainee JD (SYNTHETIC reconstruction)
def test_trainee_title_and_mandatory(trainee):
    assert trainee.title == "Data Engineering Trainee - Databricks"
    assert set(trainee.required_skills) == {"python", "sql", "mysql"} and set(trainee.mandatory_skills) == set(trainee.required_skills)
    assert ["mysql", "sql"] in trainee.skill_alternatives


def test_trainee_spark_not_mandatory_and_tiers(trainee):
    assert "spark" not in trainee.mandatory_skills and "spark" in trainee.nonmandatory_skills
    assert {"spark", "databricks", "cloud computing"} <= set(trainee.preferred_skills)
    assert "data engineering" in trainee.supporting_skills              # "Understanding of ... ETL/ELT" is concept-level
    assert ["aws", "azure", "gcp"] in trainee.skill_alternatives


def test_trainee_experience_and_education(trainee):
    assert trainee.minimum_experience_years == 0.0 and trainee.has_experience_requirement is False
    assert set(trainee.education_requirements) == {"bachelor", "master"}


# ------------------------------------------------------------ title
@pytest.mark.parametrize("text,title", [
    ("JOB DESCRIPTION\nTitle: Senior Data Analyst\nCompany: X", "Senior Data Analyst"),
    ("Job Title: Machine Learning Engineer\nWe are hiring.", "Machine Learning Engineer"),
    ("Data Engineering Trainee - Databricks\nAbout the role", "Data Engineering Trainee - Databricks"),
    ("About us\nWe are looking for a Backend Developer to join our team.", "Backend Developer"),
    ("Some text without any role words at all.", "Untitled Role"),
])
def test_title(text, title):
    assert parse(text).title == title


# ------------------------------------------------------------ negation / wording variations
@pytest.mark.parametrize("sentence", [
    "Experience with Spark is not mandatory",
    "Spark experience isn't required",
    "Spark is optional",
    "No prior Spark experience is necessary",
    "Knowledge of Spark is not a must",
    "Spark is not mandatory; training will be provided",
    "Spark experience is not required, we will train you",
])
def test_negated_skill_is_never_mandatory(sentence):
    jd = parse(f"Data Engineer\nRequirements\n- Python\n- {sentence}\n")
    assert "spark" not in jd.required_skills and "spark" not in jd.mandatory_skills
    assert "python" in jd.mandatory_skills


@pytest.mark.parametrize("sentence", ["Spark is mandatory", "Must have Spark", "Spark experience is required", "Spark (required)"])
def test_positive_mandatory_wording(sentence):
    assert "spark" in parse(f"Data Engineer\nRequirements\n- {sentence}\n").mandatory_skills


def test_not_mandatory_but_plus_is_preferred():
    jd = parse("Data Engineer\nRequirements\n- Python\n- Spark experience is not mandatory but a plus\n")
    assert "spark" in jd.preferred_skills and "spark" not in jd.required_skills


def test_preferred_sections_and_soft_wording():
    jd = parse("Analyst\nRequirements\n- SQL\n- Basic understanding of Tableau\nNice to have\n- AWS\n")
    assert jd.required_skills == ["sql"] and "tableau" in jd.supporting_skills and jd.preferred_skills == ["aws"]


def test_hard_wording_overrides_soft():
    assert "tableau" in parse("Analyst\nRequirements\n- Must have basic Tableau\n").required_skills


def test_skills_only_in_responsibilities_are_supporting():
    jd = parse("Analyst\nRequirements\n- SQL\nResponsibilities\n- Build Tableau dashboards\n")
    assert jd.required_skills == ["sql"] and "tableau" in jd.supporting_skills


def test_certification_only_lines_do_not_become_skills():
    jd = parse("Analyst\nRequirements\n- SQL\nNice to have\n- AWS certification\n")
    assert "aws" not in jd.preferred_skills


# ------------------------------------------------------------ alternatives
@pytest.mark.parametrize("line,group", [
    ("SQL or MySQL", ["mysql", "sql"]),
    ("Strong SQL/MySQL knowledge", ["mysql", "sql"]),
    ("either PostgreSQL or MySQL", ["mysql", "postgresql"]),
    ("AWS, Azure or GCP", ["aws", "azure", "gcp"]),
    ("Orchestration (Airflow, Redshift, or similar)", ["airflow", "redshift"]),
])
def test_alternative_groups(line, group):
    assert group in parse(f"Engineer\nRequirements\n- {line}\n").skill_alternatives


def test_and_lists_are_not_alternatives():
    jd = parse("Engineer\nRequirements\n- Python and SQL\n- Python, SQL and Tableau\n")
    assert jd.skill_alternatives == []


def test_python_then_sql_or_mysql_only_groups_the_pair():
    jd = parse("Engineer\nRequirements\n- Python, SQL or MySQL\n")
    assert ["mysql", "sql"] in jd.skill_alternatives and not any("python" in g for g in jd.skill_alternatives)


# ------------------------------------------------------------ experience
@pytest.mark.parametrize("line,mn,mx,app", [
    ("5+ years of experience in data analysis", 5.0, None, True),
    ("Minimum 3 years of experience", 3.0, None, True),
    ("2-4 years of relevant experience", 2.0, 4.0, True),
    ("3 to 5 years experience", 3.0, 5.0, True),
    ("At least 2 years experience, up to 6 years", 2.0, 6.0, True),
    ("Freshers welcome. No prior experience required", 0.0, None, False),
    ("Entry-level role", 0.0, None, False),
    ("Bachelor's degree (3 years) and strong SQL", 0.0, None, False),
    ("3-year bachelor's degree in Statistics", 0.0, None, False),
])
def test_experience_extraction(line, mn, mx, app):
    jd = parse(f"Analyst\nRequirements\n- {line}\n")
    assert (jd.minimum_experience_years, jd.maximum_experience_years, jd.has_experience_requirement) == (mn, mx, app)


def test_education_is_not_confused_with_experience_or_english_words():
    jd = parse("Analyst\nRequirements\n- Master's degree in Statistics\n- You will be working with product teams\n")
    assert jd.education_requirements == ["master"] and jd.minimum_experience_years == 0.0


# ---------------------------------------------------------------- negation scope (sentence/clause level)
@pytest.mark.parametrize("text,required,nonmandatory", [
    ("Spark experience is not mandatory. Python required.", ["python"], ["spark"]),
    ("Python and SQL are required. Spark experience is not mandatory.", ["python", "sql"], ["spark"]),
    ("Spark experience is not mandatory. Python and SQL are required.", ["python", "sql"], ["spark"]),
    ("Python is required; Spark is not mandatory.", ["python"], ["spark"]),
    ("Python is required but Spark is not mandatory.", ["python"], ["spark"]),
    ("Requirements\n- Python required. Spark experience is not mandatory.", ["python"], ["spark"]),
])
def test_negation_does_not_leak_across_sentences(text, required, nonmandatory):
    jd = JDParserService().parse_job_description(text)
    assert jd.required_skills == required
    assert jd.nonmandatory_skills == nonmandatory
    assert "spark" not in jd.required_skills


def test_not_mandatory_but_a_plus_in_one_clause_stays_preferred_and_does_not_split():
    jd = JDParserService().parse_job_description("Spark is not mandatory but a plus. Python required.")
    assert jd.required_skills == ["python"]
    assert jd.preferred_skills == ["spark"]


def test_abbreviation_is_not_a_sentence_boundary():
    jd = JDParserService().parse_job_description("Python (e.g. Pandas) required. Databricks is good to have.")
    assert jd.required_skills == ["pandas", "python"]
    assert jd.preferred_skills == ["databricks"]


def test_cert_only_sentence_does_not_hide_skill_in_same_line():
    jd = JDParserService().parse_job_description("Python required. AWS certification is a plus.")
    assert jd.required_skills == ["python"]


def test_clause_splitter_unit():
    from app.services.jd_parser import _clauses
    assert _clauses("Python required") == ["Python required"]
    assert _clauses("Python required. Spark not mandatory.") == ["Python required.", "Spark not mandatory."]
    assert _clauses("Python required, but Spark not mandatory") == ["Python required", "Spark not mandatory"]
    assert len(_clauses("Spark not mandatory but a plus")) == 1


def test_skill_listed_before_subsequent_negation_is_never_mandatory():
    text = (
        "Requirements\n"
        "- SQL\n"
        "- Python\n"
        "- PySpark\n"
        "- Databricks\n"
        "Prior experience in PySpark, Spark, or Databricks is not mandatory.\n"
    )
    jd = JDParserService().parse_job_description(text)
    assert set(jd.required_skills) == {"python", "sql"}
    assert set(jd.mandatory_skills) == {"python", "sql"}
    assert "spark" not in jd.mandatory_skills
    assert "databricks" not in jd.mandatory_skills
    assert {"spark", "databricks"} <= set(jd.nonmandatory_skills)

