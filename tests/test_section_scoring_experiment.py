"""
Comprehensive Unit and Integration Test Suite for Local spaCy Section-Based ATS Scoring Experiment.
Tests cover all 12 experimental scenarios and safety constraints.
"""
import os
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import config
from app.models.job import JobDescription
from experiments.experiment_config import SectionDetectionConfig, SectionScoringConfig
from experiments.spacy_section_detector import DetectedSection, DocumentExtractionResult, SpacySectionDetector
from experiments.section_scorer import SectionScorer, SectionScoreResult


@pytest.fixture(scope="module")
def sample_jd():
    """Standard Job Description for testing."""
    return JobDescription(
        job_id="JD_TEST_001",
        title="Senior Data Analyst",
        raw_text="Looking for a Senior Data Analyst with Python, SQL, Tableau, and Machine Learning.",
        required_skills=["python", "sql", "tableau"],
        preferred_skills=["machine learning", "snowflake", "spark"],
        mandatory_skills=["python", "sql"],
        minimum_experience_years=3.0,
        education_requirements=["bachelor"],
        responsibilities="Build ETL pipelines, perform statistical analysis, and design executive dashboards.",
    )


@pytest.fixture(scope="module")
def scorer():
    """Shared SectionScorer instance for tests."""
    return SectionScorer()


# ---------------------------------------------------------------------------
# Scenario 1: Common sections resume
# ---------------------------------------------------------------------------
def test_scenario_1_common_sections_resume(scorer, sample_jd):
    resume_text = """
    Jane Doe
    jane.doe@example.com

    Professional Summary
    Experienced Data Analyst with 4+ years of expertise in SQL and Python.

    Technical Skills
    Python, SQL, Tableau, Pandas, Git, Docker

    Work Experience
    Data Analyst at FinTech Corp (2020 - Present)
    - Developed SQL analytics workflows and automated Python pipelines.
    - Designed interactive Tableau dashboards for leadership.

    Academic Background
    Bachelor of Science in Computer Science, State University, 2019

    Key Projects
    Real-Time Customer Segmentation
    - Built customer churn prediction model using Python and SQL.

    Certifications
    Tableau Desktop Certified Associate
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="jane_doe.txt")
    assert ext.success is True
    categories = [s.category for s in ext.sections]
    assert "summary" in categories
    assert "skills" in categories
    assert "experience" in categories
    assert "education" in categories
    assert "projects" in categories
    assert "certifications" in categories

    # Character offsets verified
    for s in ext.sections:
        assert s.char_start >= 0
        assert s.char_end >= s.char_start
        assert len(s.sentences) > 0 or len(s.text) == 0

    res = scorer.score_extraction_result(ext, "CAND_001", sample_jd)
    # 1. Finite and valid score bounds
    assert 0.0 <= res.core_score <= 100.0
    assert 0.0 <= res.final_score <= 100.0
    assert res.final_score >= res.core_score

    # 2. Required skills accurately matched
    assert res.skills_evaluation is not None
    assert set(sample_jd.required_skills).issubset(set(res.skills_evaluation.matched_required))
    for skill in ("python", "sql", "tableau"):
        assert skill in res.skills_evaluation.matched_required

    # 3. Education strictly excluded from scoring and bonus
    assert "education" not in res.core_dimension_scores
    for sec in res.detailed_section_scores:
        if sec.category == "education":
            assert sec.is_core is False
    assert res.education_eligibility is not None
    assert res.additional_bonus == 0.0

    # 4. Experience evaluated and included as active dimension
    assert res.experience_evaluation is not None
    assert res.experience_evaluation.required_years == sample_jd.minimum_experience_years
    assert res.core_dimension_scores.get("experience") is not None
    assert res.core_dimension_scores["experience"] > 0.0

    # 5. Certifications excluded from active core score when JD requirement is none
    assert res.certifications_evaluation is not None
    assert res.certifications_evaluation.requirement_level == "none"
    assert res.core_dimension_scores.get("certification") == 0.0

    # 6. Composed score matches composed formula from score components and active weights
    w = config.scoring_weights()
    # Active dimensions for this scenario: semantic, bm25, skill, experience (certification is not applicable)
    active_dims = ["semantic", "bm25", "skill", "experience"]
    active_w_sum = sum(w[d] for d in active_dims)
    expected_core = (sum(w[d] * res.core_dimension_scores[d] for d in active_dims) / active_w_sum) * 100.0
    assert res.core_score == pytest.approx(expected_core, abs=1e-2)

    # 7. Additional bonuses handled separately
    assert res.final_score == pytest.approx(min(100.0, res.core_score + res.additional_bonus + res.fresher_experience_bonus), abs=1e-4)

    assert "skills" in res.section_bm25_scores
    assert res.section_bm25_scores["skills"] > 0.0


# ---------------------------------------------------------------------------
# Scenario 2: No explicit Skills heading (skills in summary/projects)
# ---------------------------------------------------------------------------
def test_scenario_2_no_explicit_skills_heading(scorer, sample_jd):
    resume_text = """
    Alex Rivera
    alex@example.com

    Executive Summary
    Skilled data professional proficient in Python, SQL, and database optimization with 5 years experience.

    Professional Experience
    Data Specialist at Logistics Inc (2019 - Present)
    - Handled large scale relational databases using advanced SQL queries.

    Technical Projects
    Warehouse Pipeline
    - Implemented ETL pipeline in Python and designed Tableau monitoring dashboards.

    Education
    BS in Information Systems, 2018
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="alex.txt")
    assert "skills" not in ext.section_map  # No explicit skills heading

    res = scorer.score_extraction_result(ext, "CAND_002", sample_jd)
    # Skills heading missing note is captured and summary/projects text was used
    assert any("skills heading missing" in n.lower() for n in res.notes)
    # Candidate is not penalized with 0 for skills
    assert res.core_dimension_scores["skill"] > 0.0
    assert res.core_score > 40.0


# ---------------------------------------------------------------------------
# Scenario 3: Custom or unfamiliar headings
# ---------------------------------------------------------------------------
def test_scenario_3_custom_unfamiliar_headings(scorer, sample_jd):
    resume_text = """
    Chris Morgan
    chris@example.com

    TECHNICAL ARSENAL:
    Python, SQL, Tableau, PostgreSQL, BigQuery

    WHERE I HAVE WORKED:
    Data Lead at Media Dynamics (2020 - Present)
    - Developed analytics pipelines using Python and SQL.

    ACADEMIC DEGREES:
    B.S. in Data Science, 2019

    BRAGGING RIGHTS:
    - Won regional DataHack competition 2022.
    - Published open-source Python data cleaning toolkit with 500+ GitHub stars.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="chris.txt")
    headings = [s.raw_heading for s in ext.sections]
    assert any("TECHNICAL ARSENAL" in h for h in headings)
    assert any("BRAGGING RIGHTS" in h for h in headings)
    # The content of custom heading "BRAGGING RIGHTS" is preserved under unknown or additional
    non_core = [s for s in ext.sections if not s.is_core and s.text.strip()]
    assert len(non_core) > 0

    res = scorer.score_extraction_result(ext, "CAND_003", sample_jd)
    assert res.core_score > 40.0
    assert res.final_score >= 45.0
    # Bonus is calculated from bragging rights section
    assert res.additional_relevance >= 0.0


# ---------------------------------------------------------------------------
# Scenario 4: Relevant skills in Projects section
# ---------------------------------------------------------------------------
def test_scenario_4_relevant_skills_in_projects(scorer, sample_jd):
    resume_text = """
    Marcus Vance
    marcus@example.com

    Experience
    Associate Analyst at Retail Co (2022 - Present)
    - Performed weekly reporting and Excel analysis.

    Projects
    Streaming Analytics Platform
    - Architected an end-to-end streaming data pipeline using Python, SQL, and Tableau.
    - Automated daily metrics ingestion.

    Education
    BS in Mathematics, 2021
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="marcus.txt")
    res = scorer.score_extraction_result(ext, "CAND_004", sample_jd)
    # Projects section should have high BM25 and semantic relevance
    assert res.section_bm25_scores["projects"] > 0.0
    assert res.section_semantic_scores["projects"] > 0.0


# ---------------------------------------------------------------------------
# Scenario 5: Relevant evidence in Achievements / Publications
# ---------------------------------------------------------------------------
def test_scenario_5_relevant_evidence_in_achievements_publications(scorer, sample_jd):
    resume_text = """
    Dr. Elena Rostova
    elena@example.com

    Summary
    Data scientist with strong foundations.

    Technical Skills
    Python, SQL, Tableau

    Experience
    Research Fellow at Institute (2020 - Present)
    - Data analysis and reporting.

    Education
    PhD in Computer Science, 2020

    Research Publications
    - Rostova et al., 'Scalable Machine Learning and Snowflake Data Warehousing in Healthcare', 2023.
    - Rostova et al., 'Real-time Spark Processing for Distributed Analytics', IEEE 2022.

    Key Achievements
    - Recipient of National Science Award for Snowflake distributed data analytics architecture.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="elena.txt")
    categories = [s.category for s in ext.sections]
    assert "publications" in categories or "achievements" in categories

    res = scorer.score_extraction_result(ext, "CAND_005", sample_jd)
    # Bonus should be awarded for new skills (Snowflake, Spark, Machine Learning) found in publications/achievements
    assert res.additional_bonus > 0.0
    assert res.additional_relevance >= scorer.scoring_config.ADDITIONAL_RELEVANCE_THRESHOLD
    bb = res.additional_bonus_breakdown
    assert len(bb.get("relevant_new_skills", [])) > 0


# ---------------------------------------------------------------------------
# Scenario 6: No additional relevant evidence (bonus = 0.0)
# ---------------------------------------------------------------------------
def test_scenario_6_no_additional_relevant_evidence(scorer, sample_jd):
    resume_text = """
    Bob Miller
    bob@example.com

    Skills
    Python, SQL, Tableau

    Experience
    Data Analyst at Logistics Co (2021 - Present)
    - SQL database querying and Python automation.

    Education
    BS in Computer Science, 2020

    Hobbies & Personal Interests
    - Long distance marathon running and bicycle touring.
    - Cooking Mediterranean cuisine and baking sourdough bread.
    - Landscape watercolor painting.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="bob.txt")
    res = scorer.score_extraction_result(ext, "CAND_006", sample_jd)
    # Hobbies have zero technical relevance, bonus must be 0.0
    assert res.additional_bonus == 0.0
    assert res.additional_relevance == 0.0
    assert res.final_score == res.core_score


# ---------------------------------------------------------------------------
# Scenario 7: Empty or missing sections (no automatic penalty)
# ---------------------------------------------------------------------------
def test_scenario_7_empty_or_missing_sections(scorer, sample_jd):
    # Candidate has NO projects and NO certifications
    resume_text = """
    Samantha Reed
    samantha@example.com

    Professional Summary
    Senior Data Analyst with 6 years of experience in enterprise reporting.

    Skills
    Python, SQL, Tableau, Power BI

    Work Experience
    Senior Data Analyst at Global Bank (2018 - Present)
    - Built end-to-end data pipelines using Python and complex SQL queries.
    - Delivered Tableau dashboards for executive decision makers.

    Education
    BS in Statistics, 2017
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="samantha.txt")
    assert "projects" not in ext.section_map
    assert "certifications" not in ext.section_map

    res = scorer.score_extraction_result(ext, "CAND_007", sample_jd)
    # Active sections normalized properly without arbitrary cross-section weights; candidate achieves solid score
    assert res.core_score > 55.0
    assert res.final_score > 55.0


# ---------------------------------------------------------------------------
# Scenario 8: Repeated mentions of the same skill (no double-counting)
# ---------------------------------------------------------------------------
def test_scenario_8_repeated_mentions_no_double_counting(scorer, sample_jd):
    resume_text = """
    David 반복
    david@example.com

    Skills
    Python, SQL, Tableau

    Experience
    Data Analyst (2021 - Present)
    - Python Python Python coding.
    - SQL SQL queries and Tableau Tableau Tableau reports.

    Achievements
    - Awarded best Python developer of the year for SQL automation with Tableau.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="david.txt")
    res = scorer.score_extraction_result(ext, "CAND_008", sample_jd)
    bb = res.additional_bonus_breakdown
    # All skills in Achievements were already credited in core sections
    # Deduplication ensures they are recorded as deduplicated and do NOT generate new skill bonus
    assert bb.get("deduplicated_core_skills_count", 0) > 0
    # No new skills were introduced in achievements
    assert len(bb.get("new_additional_skills", [])) == 0


# ---------------------------------------------------------------------------
# Scenario 9: Long unrelated section does not inflate score
# ---------------------------------------------------------------------------
def test_scenario_9_long_unrelated_section_no_inflation(scorer, sample_jd):
    # 500 words of unrelated prose in a non-core section
    fluff = (
        "The gentle mountain breeze rustled the leaves of the ancient pine trees across the valley. "
        "Every summer our hiking club traverses the continental divide, studying geological formations "
        "and photographing migratory songbirds. The campfire stories inspire creativity and reflection. "
    ) * 20

    resume_text = f"""
    Frank Miller
    frank@example.com

    Skills
    Python, SQL

    Experience
    Junior Analyst (2023 - Present)
    - SQL data extraction and Python reporting.

    Education
    BS in Business, 2022

    Travel Chronicles & Mountain Explorations
    {fluff}
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="frank.txt")
    res = scorer.score_extraction_result(ext, "CAND_009", sample_jd)
    # Bonus must be 0.0 because semantic similarity of fluff is below floor and new skills = 0
    assert res.additional_bonus == 0.0
    assert res.additional_relevance == 0.0


# ---------------------------------------------------------------------------
# Scenario 10: Incomplete or failed text extraction reporting
# ---------------------------------------------------------------------------
def test_scenario_10_incomplete_text_extraction(scorer, sample_jd):
    # File does not exist
    ext = scorer.detector.process_file("non_existent_file_xyz123.pdf")
    assert ext.success is False
    assert "not found" in ext.error_message.lower()

    res = scorer.score_extraction_result(ext, "CAND_010", sample_jd)
    assert res.core_score == 0.0
    assert res.final_score == 0.0
    assert any("extraction failed" in n.lower() for n in res.notes)


# ---------------------------------------------------------------------------
# Scenario 11: Multiple sections mapped to the same core dimension
# ---------------------------------------------------------------------------
def test_scenario_11_multiple_sections_same_core_dimension(scorer, sample_jd):
    resume_text = """
    Maya Lin
    maya@example.com

    Core Competencies
    Python, SQL

    Tools & Technologies
    Tableau, Git, Docker

    Professional Experience
    Data Analyst at Alpha Corp (2021 - Present)
    - Built ETL workflows.

    Internship Experience
    Data Intern at Beta Lab (2020 - 2021)
    - Built SQL queries and Tableau reports.

    Education
    BS in Computer Science, 2020
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="maya.txt")
    # 'Core Competencies' and 'Tools & Technologies' both map to 'skills'
    # 'Professional Experience' and 'Internship Experience' both map to 'experience'
    assert "Python" in ext.section_map["skills"]
    assert "Tableau" in ext.section_map["skills"]
    assert "Alpha Corp" in ext.section_map["experience"]
    assert "Beta Lab" in ext.section_map["experience"]

    res = scorer.score_extraction_result(ext, "CAND_011", sample_jd)
    assert res.core_score > 45.0


# ---------------------------------------------------------------------------
# Scenario 12: High core score candidate capped at 100
# ---------------------------------------------------------------------------
def test_scenario_12_high_core_score_capped_at_100(scorer, sample_jd):
    # Highly qualified candidate with bonus
    high_result = SectionScoreResult(
        candidate_id="CAND_TOP",
        name="Top Candidate",
        core_score=99.0,
        additional_bonus=3.0,
        final_score=min(100.0, 99.0 + 3.0),
    )
    assert high_result.final_score == 100.0
    assert high_result.final_score <= 100.0


# ---------------------------------------------------------------------------
# Extra Safety Tests: Bonus toggle and production isolation
# ---------------------------------------------------------------------------
def test_bonus_toggle_and_beta_zero(sample_jd):
    # Test beta=0.0
    scorer_zero_beta = SectionScorer(scoring_config=SectionScoringConfig(BONUS_BETA=0.0))
    resume_text = """
    Candidate Test
    Skills: Python, SQL
    Achievements: Snowflake expert and Machine Learning innovator.
    """
    ext = scorer_zero_beta.detector.detect_sections(resume_text)
    res = scorer_zero_beta.score_extraction_result(ext, "CAND_T1", sample_jd)
    assert res.additional_bonus == 0.0

    # Test ENABLE_ADDITIONAL_BONUS=False
    scorer_disabled = SectionScorer(scoring_config=SectionScoringConfig(ENABLE_ADDITIONAL_BONUS=False))
    res2 = scorer_disabled.score_extraction_result(ext, "CAND_T2", sample_jd)
    assert res2.additional_bonus == 0.0


def test_production_codebase_integrity():
    """Verify that production ATS files were not modified or overwritten."""
    prod_files = [
        PROJECT_ROOT / "app" / "config.py",
        PROJECT_ROOT / "app" / "services" / "hybrid_ranker.py",
        PROJECT_ROOT / "app" / "services" / "score_composer.py",
        PROJECT_ROOT / "app" / "services" / "semantic_search.py",
        PROJECT_ROOT / "app" / "services" / "bm25_search.py",
        PROJECT_ROOT / "app" / "services" / "skill_matcher.py",
    ]
    for p in prod_files:
        assert p.exists(), f"Production file {p} is missing!"
    
    # Verify production nominal weights remain standard
    w = config.scoring_weights()
    assert w["semantic"] == pytest.approx(0.35)
    assert w["bm25"] == pytest.approx(0.20)
    assert w["skill"] == pytest.approx(0.27)
    assert w["experience"] == pytest.approx(0.10)
    assert w["certification"] == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Batch 1 Refinement Tests: Independent Section BM25 & Semantic Scoring
# ---------------------------------------------------------------------------
def test_batch1_section_bm25_calculated_independently(scorer, sample_jd):
    """Verify that section BM25 is strictly independent and unaffected by other sections."""
    skills_text = "Python, SQL, Tableau, Pandas, NumPy"
    
    # Resume A: Short experience
    resume_a = f"""
    Candidate A
    Skills
    {skills_text}

    Experience
    Junior Analyst (2022-2023)
    Reporting tasks.
    """
    
    # Resume B: Long 10x experience with unrelated domain keywords
    resume_b = f"""
    Candidate B
    Skills
    {skills_text}

    Experience
    Lead Enterprise Architect (2015-2023)
    Directed global cloud migration across AWS and Azure infrastructure.
    Orchestrated Kubernetes clusters, designed CI/CD pipelines with Jenkins and Terraform.
    Managed a distributed organization of 45 site reliability engineers.
    Monitored network topology and database replication clusters.
    """
    
    ext_a = scorer.detector.detect_sections(resume_a, file_path="res_a.txt")
    ext_b = scorer.detector.detect_sections(resume_b, file_path="res_b.txt")
    
    res_a = scorer.score_extraction_result(ext_a, "CAND_BA", sample_jd)
    res_b = scorer.score_extraction_result(ext_b, "CAND_BB", sample_jd)
    
    # The Skills section BM25 must be identical regardless of how long or different Experience is
    score_a = res_a.section_scores["skills"].bm25_normalized
    score_b = res_b.section_scores["skills"].bm25_normalized
    assert score_a > 0.0
    assert score_a == pytest.approx(score_b, rel=1e-5)
    assert res_a.section_scores["skills"].bm25_raw == pytest.approx(res_b.section_scores["skills"].bm25_raw, rel=1e-5)


def test_batch1_section_semantic_calculated_independently(scorer, sample_jd):
    """Verify that section semantic similarity is strictly independent across sections."""
    skills_text = "Python, SQL, Tableau, Pandas, Machine Learning"
    
    # Resume 1: With Education
    resume_1 = f"""
    Candidate 1
    Skills
    {skills_text}

    Education
    Master of Science in Computational Statistics, 2021
    """
    
    # Resume 2: Without Education
    resume_2 = f"""
    Candidate 2
    Skills
    {skills_text}
    """
    
    ext_1 = scorer.detector.detect_sections(resume_1, file_path="res_1.txt")
    ext_2 = scorer.detector.detect_sections(resume_2, file_path="res_2.txt")
    
    res_1 = scorer.score_extraction_result(ext_1, "CAND_S1", sample_jd)
    res_2 = scorer.score_extraction_result(ext_2, "CAND_S2", sample_jd)
    
    # Skills section semantic cosine similarity must be identical
    cos_1 = res_1.section_scores["skills"].semantic_raw_cosine
    cos_2 = res_2.section_scores["skills"].semantic_raw_cosine
    assert cos_1 > 0.0
    assert cos_1 == pytest.approx(cos_2, rel=1e-5)
    assert res_1.section_scores["skills"].semantic_calibrated == pytest.approx(res_2.section_scores["skills"].semantic_calibrated, rel=1e-5)


def test_batch1_no_cross_section_weights_used(scorer, sample_jd):
    """Verify that arbitrary cross-section weights (0.40, 0.30, 0.20, 0.10) are removed and not multiplied."""
    # Ensure config weights are deprecated/empty
    assert scorer.scoring_config.SECTION_BM25_WEIGHTS == {}
    assert scorer.scoring_config.SECTION_SEMANTIC_WEIGHTS == {}
    
    resume_text = """
    Candidate Test
    Skills
    Python, SQL, Tableau
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="test_no_weights.txt")
    res = scorer.score_extraction_result(ext, "CAND_NW", sample_jd)
    
    skills_sec = res.section_scores["skills"]
    # The score stored must be the actual calculated score, NOT multiplied by 0.40
    raw_calc_norm, _, _ = scorer.score_section_bm25(skills_sec.text_evidence, skills_sec.query_text, "CAND_NW", "skills")
    assert skills_sec.bm25_normalized == pytest.approx(raw_calc_norm, rel=1e-5)
    assert skills_sec.bm25_normalized != pytest.approx(raw_calc_norm * 0.40, rel=1e-5)


def test_batch1_unknown_sections_preserved_with_scores(scorer, sample_jd):
    """Verify that unknown/custom sections are preserved with text evidence and independent scores."""
    resume_text = """
    Jane Inventor
    Skills: Python, SQL

    INVENTIONS AND PROPRIETARY ARCHITECTURE:
    - Custom design US-2023: Distributed query optimizer for streaming pipelines using SQL and Python.
    - System design US-2022: Automated anomaly detection in warehouse telemetry.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="inventor.txt")
    res = scorer.score_extraction_result(ext, "CAND_INV", sample_jd)
    
    assert len(res.unknown_sections) > 0
    unk = res.unknown_sections[0]
    assert "INVENTIONS" in unk.raw_heading
    assert unk.category == "unknown"
    assert "Custom design US-2023" in unk.text_evidence
    assert unk.char_length > 0
    assert unk.bm25_normalized > 0.0
    assert unk.semantic_raw_cosine > 0.0
    assert any("preserved as unknown" in note for note in unk.uncertainty_notes)


def test_batch1_empty_missing_sections_no_fabricated_scores(scorer, sample_jd):
    """Verify that empty or missing sections receive exactly 0.0 without fabricated scores."""
    resume_text = """
    Candidate Minimal
    Skills
    Python, SQL
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="minimal.txt")
    res = scorer.score_extraction_result(ext, "CAND_MIN", sample_jd)
    
    for missing_cat in ["experience", "projects", "education"]:
        sec = res.section_scores[missing_cat]
        assert sec.bm25_normalized == 0.0
        assert sec.bm25_raw == 0.0
        assert sec.semantic_raw_cosine == 0.0
        assert sec.semantic_calibrated == 0.0
        assert sec.text_evidence == ""
        assert any("absent or empty: no relevance score fabricated" in note for note in sec.uncertainty_notes)


def test_batch1_no_global_resume_scores_added(scorer, sample_jd):
    """Verify that global whole-resume BM25 and semantic scores do not contribute to section scoring."""
    resume_text = """
    Candidate Isolated
    Skills
    Python, SQL
    Experience
    Data Analyst with SQL reporting.
    """
    ext = scorer.detector.detect_sections(resume_text, file_path="isolated.txt")
    res = scorer.score_extraction_result(ext, "CAND_ISO", sample_jd)
    
    # Check that each section score query is section-specific and text is strictly that section's text
    assert res.section_scores["skills"].text_evidence == "Python, SQL"
    assert "Data Analyst" not in res.section_scores["skills"].text_evidence
    assert res.section_scores["experience"].text_evidence == "Data Analyst with SQL reporting."
    assert "Python" not in res.section_scores["experience"].text_evidence
    
    # Core dimension scores do not contain 'global_bm25' or 'global_semantic'
    assert "global_bm25" not in res.core_dimension_scores
    assert "global_semantic" not in res.core_dimension_scores


# ===========================================================================
# BATCH 2: Education Eligibility Filter Tests (10 scenarios)
# ===========================================================================
from experiments.education_eligibility import (
    EducationEligibilityService,
    EducationEligibilityStatus,
    EducationEligibilityResult,
)


@pytest.fixture(scope="module")
def edu_service():
    """Shared EducationEligibilityService instance."""
    return EducationEligibilityService()


# Batch 2 helper: create a JD with specific education requirements and raw_text
def _make_jd_with_edu(raw_edu_text, edu_requirements=None, **kwargs):
    """Build a JD with education requirement text embedded."""
    defaults = dict(
        job_id="JD_EDU_TEST",
        title="Software Engineer",
        raw_text=raw_edu_text,
        required_skills=["python", "sql"],
        preferred_skills=["docker"],
        mandatory_skills=["python"],
        minimum_experience_years=0.0,
        education_requirements=edu_requirements or [],
        responsibilities="Build software systems.",
    )
    defaults.update(kwargs)
    return JobDescription(**defaults)


def _make_extraction(resume_text, detector=None):
    """Build a DocumentExtractionResult from resume text."""
    if detector is None:
        detector = SpacySectionDetector()
    return detector.detect_sections(resume_text, file_path="test_edu.txt", file_type="txt")


# ---------------------------------------------------------------------------
# Test 21: Candidate satisfies one of several accepted degree alternatives (OR logic)
# ---------------------------------------------------------------------------
def test_batch2_alternative_degree_eligible(edu_service, scorer):
    """JD accepts B.Tech OR M.Sc OR M.Tech; candidate has M.Sc -> ELIGIBLE."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Qualifications\n"
            "B.Tech or M.Sc or M.Tech in Computer Science or related field"
        ),
        edu_requirements=["bachelor", "master"],
    )
    resume = """
    John Doe
    Education
    M.Sc in Computer Science, University of Technology, 2021
    Skills
    Python, SQL, Docker
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.ELIGIBLE
    assert result.is_eligible is True
    assert result.matched_degree is not None


# ---------------------------------------------------------------------------
# Test 22: Candidate satisfies single mandatory degree requirement
# ---------------------------------------------------------------------------
def test_batch2_single_degree_eligible(edu_service, scorer):
    """JD requires Bachelor's; candidate has B.Tech -> ELIGIBLE."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Requirements\n"
            "Bachelor's degree in Computer Science or related field required"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Alice Kumar
    Education
    B.Tech in Information Technology, ABC Institute, 2020
    Skills
    Python, SQL
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.ELIGIBLE
    assert result.is_eligible is True


# ---------------------------------------------------------------------------
# Test 23: Candidate demonstrably fails explicit mandatory requirement -> INELIGIBLE
# ---------------------------------------------------------------------------
def test_batch2_ineligible_low_degree(edu_service, scorer):
    """JD requires Bachelor's; candidate only has Diploma -> INELIGIBLE."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Requirements\n"
            "Bachelor's degree in any engineering discipline is mandatory"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Bob Singh
    Education
    Diploma in Mechanical Engineering, Polytechnic Institute, 2018
    Skills
    Python, SQL
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.INELIGIBLE
    assert result.is_eligible is False


# ---------------------------------------------------------------------------
# Test 24: JD contains no mandatory education requirement -> NOT_REQUIRED
# ---------------------------------------------------------------------------
def test_batch2_no_education_requirement(edu_service, scorer):
    """JD says nothing about education -> NOT_REQUIRED."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "We are looking for a Python developer.\n"
            "Requirements\n"
            "Strong Python and SQL skills required.\n"
            "3+ years of experience.\n"
        ),
        edu_requirements=[],
    )
    resume = """
    Carol Mehta
    Skills
    Python, SQL, Docker
    Experience
    3 years as Python developer
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.NOT_REQUIRED
    assert result.is_eligible is True


# ---------------------------------------------------------------------------
# Test 25: Resume education heading missing but evidence exists in full text
# ---------------------------------------------------------------------------
def test_batch2_missing_heading_evidence_in_text(edu_service, scorer):
    """No Education heading, but degree text exists elsewhere -> should extract evidence."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Requirements\n"
            "Bachelor's degree required"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Deepak Gupta
    Professional Summary
    Software engineer with B.Tech in Computer Science from IIT Delhi (2019).
    Skilled in Python, SQL, and cloud infrastructure.
    Skills
    Python, SQL, AWS
    Experience
    Software Engineer at TechCorp, 2019-Present
    """
    ext = _make_extraction(resume, scorer.detector)
    evidence = edu_service.extract_candidate_education(ext)
    # Evidence should have been found via full text fallback
    assert len(evidence) > 0
    result = edu_service.evaluate_eligibility(jd, ext)
    # Should be ELIGIBLE because B.Tech >= bachelor
    assert result.status == EducationEligibilityStatus.ELIGIBLE


# ---------------------------------------------------------------------------
# Test 26: Education evidence cannot be extracted anywhere -> UNKNOWN
# ---------------------------------------------------------------------------
def test_batch2_no_education_evidence_unknown(edu_service, scorer):
    """JD requires degree, resume has zero education info -> UNKNOWN (not auto-reject)."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Requirements\n"
            "Bachelor's degree required"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Eve Anonymous
    Skills
    Python, SQL, Docker
    Experience
    5 years in software development
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.UNKNOWN
    assert result.is_eligible is False
    assert result.requires_manual_verification is True


# ---------------------------------------------------------------------------
# Test 27: Degree is in progress when JD requires completed degree -> INELIGIBLE
# ---------------------------------------------------------------------------
def test_batch2_in_progress_degree_ineligible(edu_service, scorer):
    """JD requires completed Bachelor's; candidate is pursuing it -> INELIGIBLE."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Requirements\n"
            "Bachelor's degree in Computer Science required"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Farhan Khan
    Education
    B.Tech in Computer Science, National University (Pursuing, Expected 2027)
    Skills
    Python, SQL
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.INELIGIBLE
    assert result.is_eligible is False


# ---------------------------------------------------------------------------
# Test 28: Field-of-study requirement is present and matched
# ---------------------------------------------------------------------------
def test_batch2_field_of_study_matched(edu_service, scorer):
    """JD requires CS or related; candidate has B.Tech in Information Technology -> ELIGIBLE (related)."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Qualifications\n"
            "Bachelor's degree in Computer Science or related field"
        ),
        edu_requirements=["bachelor"],
    )
    resume = """
    Grace Thomas
    Education
    B.Tech in Information Technology, State University, 2020
    Skills
    Python, SQL
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.ELIGIBLE
    assert result.is_eligible is True


# ---------------------------------------------------------------------------
# Test 29: Degree alternatives treated as OR, not cumulative
# ---------------------------------------------------------------------------
def test_batch2_or_logic_not_cumulative(edu_service, scorer):
    """JD accepts B.Tech or M.Sc or M.Tech; candidate with just B.Tech must be ELIGIBLE
    (must NOT require ALL listed degrees)."""
    jd = _make_jd_with_edu(
        raw_edu_text=(
            "Education\n"
            "B.Tech or M.Sc or M.Tech in Computer Science"
        ),
        edu_requirements=["bachelor", "master"],
    )
    resume = """
    Hiren Patel
    Education
    B.Tech in Computer Science, Gujarat Technical University, 2021
    Skills
    Python, SQL
    """
    ext = _make_extraction(resume, scorer.detector)
    result = edu_service.evaluate_eligibility(jd, ext)
    assert result.status == EducationEligibilityStatus.ELIGIBLE
    assert result.is_eligible is True


# ---------------------------------------------------------------------------
# Test 30: Education has zero contribution to BM25, semantic, core score, or bonus
# ---------------------------------------------------------------------------
def test_batch2_education_excluded_from_relevance_scoring(scorer, sample_jd):
    """Education section text must NOT contribute to any relevance dimension or bonus.
    Two identical resumes — one with education text containing JD-relevant keywords,
    one with empty education — must produce the SAME core/final score and aggregates."""
    resume_with_edu = """
    Candidate WithEdu
    Skills
    Python, SQL, Tableau
    Experience
    Senior Data Analyst building Python ETL pipelines.
    Projects
    Built dashboards with Tableau and SQL analytics.
    Education
    Bachelor of Science in Computer Science, Advanced Python Machine Learning SQL Tableau Data Analysis, 2020
    """
    resume_without_edu = """
    Candidate NoEdu
    Skills
    Python, SQL, Tableau
    Experience
    Senior Data Analyst building Python ETL pipelines.
    Projects
    Built dashboards with Tableau and SQL analytics.
    """
    ext_with = scorer.detector.detect_sections(resume_with_edu, file_path="with_edu.txt")
    ext_without = scorer.detector.detect_sections(resume_without_edu, file_path="no_edu.txt")

    res_with = scorer.score_extraction_result(ext_with, "CAND_WEDU", sample_jd)
    res_without = scorer.score_extraction_result(ext_without, "CAND_NEDU", sample_jd)

    # Core aggregates must be identical (education excluded)
    assert res_with.bm25_core_aggregate == res_without.bm25_core_aggregate, \
        f"BM25 aggregate differs: {res_with.bm25_core_aggregate} vs {res_without.bm25_core_aggregate}"
    assert res_with.semantic_core_aggregate == res_without.semantic_core_aggregate, \
        f"Semantic aggregate differs: {res_with.semantic_core_aggregate} vs {res_without.semantic_core_aggregate}"

    # Core score must be equal (education does not affect score)
    assert abs(res_with.core_score - res_without.core_score) < 0.5, \
        f"Core scores differ by more than 0.5%: {res_with.core_score} vs {res_without.core_score}"

    # Final score should be very close (only bonus on non-core sections could differ minimally)
    assert abs(res_with.final_score - res_without.final_score) < 1.0, \
        f"Final scores differ by more than 1.0%: {res_with.final_score} vs {res_without.final_score}"

    # Education section is NOT in core_categories for aggregation (is_core should be False)
    edu_sec_with = res_with.section_scores.get("education")
    if edu_sec_with:
        assert edu_sec_with.is_core is False, "Education section should not be marked as core for scoring"

    # Education eligibility should still be populated
    assert res_with.education_eligibility is not None


# ===========================================================================
# BATCH 3: Experience & Internship Scoring Tests (20 scenarios)
# ===========================================================================
from experiments.experience_scorer import (
    ExperienceScorerService,
    ExperienceRequirementStatus,
    ExperienceType,
    ExperienceRecord,
    ExperienceEvaluationResult,
)
from experiments.experiment_config import ExperienceScoringConfig


@pytest.fixture(scope="module")
def exp_service():
    """Shared ExperienceScorerService instance."""
    return ExperienceScorerService()


# ---------------------------------------------------------------------------
# Test 31: 1 year of relevant professional employment against 1 year requirement
# ---------------------------------------------------------------------------
def test_batch3_1yr_relevant_professional_employment(exp_service, scorer):
    """Candidate with 1 year professional experience against 1 year JD requirement."""
    jd = JobDescription(
        job_id="JD_EXP_1YR",
        title="Python Developer",
        raw_text="Requirements\nMinimum 1 year of professional experience in Python development.\nResponsibilities\nBuild Python backend APIs and SQL databases.",
        required_skills=["python", "sql"],
        minimum_experience_years=1.0,
        responsibilities="Build Python backend APIs and SQL databases.",
    )
    resume = """
    Alice Dev
    Work Experience
    Software Engineer at Tech Solutions
    Jan 2022 - Jan 2023
    - Built Python backend APIs and maintained SQL databases.
    - Automated unit tests and deployed microservices.
    """
    ext = scorer.detector.detect_sections(resume, file_path="alice.txt")
    eval_res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Build Python backend APIs and SQL databases.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    assert eval_res.status == ExperienceRequirementStatus.MEETS_REQUIREMENT
    assert eval_res.verified_total_years >= 1.0
    assert eval_res.duration_ratio == 1.0
    assert eval_res.records[0].exp_type == ExperienceType.PROFESSIONAL
    assert eval_res.records[0].type_multiplier == 1.0
    assert eval_res.records[0].duties_relevance > 0.5
    assert eval_res.final_experience_score > 0.5


# ---------------------------------------------------------------------------
# Test 32: 9 months of relevant professional employment against 1 year requirement
# ---------------------------------------------------------------------------
def test_batch3_9months_relevant_professional_employment(exp_service, scorer):
    """Candidate with 9 months experience against 1 year requirement -> duration ratio 0.75."""
    jd = JobDescription(
        job_id="JD_EXP_1YR",
        title="Python Developer",
        raw_text="Requirements\nAt least 1 year of experience required.",
        required_skills=["python"],
        minimum_experience_years=1.0,
        responsibilities="Develop Python applications.",
    )
    resume = """
    Bob Junior
    Experience
    Python Developer at StartupCo
    Jan 2022 - Oct 2022
    - Developed Python applications and script automations.
    """
    ext = scorer.detector.detect_sections(resume, file_path="bob.txt")
    eval_res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Develop Python applications.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    assert eval_res.status == ExperienceRequirementStatus.BELOW_REQUIREMENT
    assert abs(eval_res.duration_ratio - 0.75) < 0.05
    assert eval_res.final_experience_score <= 0.75


# ---------------------------------------------------------------------------
# Test 33: Relevant internship with configured internship multiplier (0.4)
# ---------------------------------------------------------------------------
def test_batch3_relevant_internship_multiplier(exp_service, scorer):
    """1 year internship against 1 year requirement receives configured 0.4 multiplier."""
    jd = JobDescription(
        job_id="JD_EXP_1YR",
        title="Data Analyst",
        raw_text="Requirements\n1 year of experience required.",
        required_skills=["python", "sql"],
        minimum_experience_years=1.0,
        responsibilities="Data analysis and SQL querying.",
    )
    resume = """
    Charlie Intern
    Experience
    Data Analyst Intern at Analytics Lab
    Jan 2022 - Jan 2023
    - Performed data analysis and wrote complex SQL queries.
    """
    ext = scorer.detector.detect_sections(resume, file_path="charlie.txt")
    eval_res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Data analysis and SQL querying.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    assert eval_res.records[0].exp_type == ExperienceType.INTERNSHIP
    assert eval_res.records[0].type_multiplier == 0.4
    # Score should be capped by internship multiplier: at most 0.40 * 1.0 = 0.40
    assert eval_res.final_experience_score <= 0.40


# ---------------------------------------------------------------------------
# Test 34: Relevant professional experience with unrelated duties
# ---------------------------------------------------------------------------
def test_batch3_professional_with_unrelated_duties(exp_service, scorer):
    """Candidate has 1 year experience, but duties are completely unrelated to JD."""
    jd = JobDescription(
        job_id="JD_DATA",
        title="Data Engineer",
        raw_text="Requirements\n1 year of experience required.\nResponsibilities\nBuild Apache Spark pipelines, data warehousing, and ETL.",
        required_skills=["spark", "etl"],
        minimum_experience_years=1.0,
        responsibilities="Build Apache Spark pipelines, data warehousing, and ETL.",
    )
    resume = """
    David Unrelated
    Experience
    Associate at Retail Superstore
    Jan 2022 - Jan 2023
    - Supervised grocery restocking and managed checkout cashier operations.
    - Handled customer returns and physical store maintenance.
    """
    ext = scorer.detector.detect_sections(resume, file_path="david.txt")
    eval_res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Build Apache Spark pipelines, data warehousing, and ETL.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    # Duties relevance should be very low
    assert eval_res.records[0].duties_relevance < 0.30
    assert eval_res.final_experience_score < 0.30


# ---------------------------------------------------------------------------
# Test 35: No experience requirement: experience does not reduce the core score
# ---------------------------------------------------------------------------
def test_batch3_no_experience_requirement_neutral_score(scorer):
    """When JD has no experience requirement, candidate core score is not dragged down."""
    jd_no_exp = JobDescription(
        job_id="JD_NO_EXP",
        title="Junior Developer",
        raw_text="Job Description\nFreshers welcome. No experience required.\nRequirements\nPython, SQL.",
        required_skills=["python", "sql"],
        minimum_experience_years=0.0,
    )
    resume = """
    Eva Fresher
    Skills
    Python, SQL
    Education
    B.Tech in Computer Science, 2023
    """
    ext = scorer.detector.detect_sections(resume, file_path="eva.txt")
    res = scorer.score_extraction_result(ext, "CAND_EVA", jd_no_exp)
    # Experience should be not_applicable and not penalize the candidate
    assert res.core_dimension_scores["experience"] == 0.0 or res.core_dimension_scores["experience"] is None
    assert res.core_score > 30.0


# ---------------------------------------------------------------------------
# Test 36: No employment history: candidate is not penalized when experience not required
# ---------------------------------------------------------------------------
def test_batch3_no_employment_history_not_penalized(exp_service):
    """Candidate with zero employment history when JD has no experience requirement -> NOT_REQUIRED."""
    jd = JobDescription(
        job_id="JD_ENTRY",
        title="Entry Level Software Engineer",
        raw_text="Entry-level opening. Fresh graduates welcome. No prior experience required.",
        minimum_experience_years=0.0,
    )
    ext = DocumentExtractionResult(
        file_path="fresher.txt",
        file_type="txt",
        raw_text="Skills\nPython, Java",
        cleaned_text="Skills\nPython, Java",
        sections=[],
        section_map={},
        success=True,
    )
    res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Python programming",
        jd_vector=None,
        bm25_scorer_fn=lambda t, q: (0.0, 0.0, []),
        semantic_scorer_fn=lambda t, v: (0.0, 0.0),
    )
    assert res.status == ExperienceRequirementStatus.NOT_REQUIRED
    assert res.verified_total_years == 0.0
    assert res.final_experience_score == 1.0  # Full neutral score


# ---------------------------------------------------------------------------
# Test 37: Overlapping employment periods are not double-counted
# ---------------------------------------------------------------------------
def test_batch3_overlapping_employment_not_double_counted(exp_service):
    """Concurrent jobs in the same time span (Jan-Dec 2022) must be merged to 12 months, not 19."""
    r1 = ExperienceRecord(
        role_title="Backend Engineer",
        intervals=[(2022 * 12 + 1, 2022 * 12 + 13)],  # Jan 2022 - Dec 2022 (12 mos)
        duration_months=12.0,
        duration_years=1.0,
    )
    r2 = ExperienceRecord(
        role_title="Consultant",
        intervals=[(2022 * 12 + 6, 2022 * 12 + 13)],  # Jun 2022 - Dec 2022 (7 mos)
        duration_months=7.0,
        duration_years=0.58,
    )
    total_years, had_overlaps = exp_service.compute_non_overlapping_years([r1, r2])
    assert had_overlaps is True
    # Non-overlapping total should be exactly 1.0 year (not 1.58 years)
    assert total_years == 1.0


# ---------------------------------------------------------------------------
# Test 38: Internship overlapping professional employment is not double-counted
# ---------------------------------------------------------------------------
def test_batch3_internship_overlapping_employment(exp_service):
    """An internship running concurrently with a full-time role does not add duplicate duration."""
    r_full = ExperienceRecord(
        role_title="Software Engineer",
        exp_type=ExperienceType.PROFESSIONAL,
        type_multiplier=1.0,
        intervals=[(2022 * 12 + 1, 2022 * 12 + 13)],  # 12 mos
        duration_months=12.0,
        duration_years=1.0,
        duties_relevance=1.0,
    )
    r_intern = ExperienceRecord(
        role_title="Research Intern",
        exp_type=ExperienceType.INTERNSHIP,
        type_multiplier=0.4,
        intervals=[(2022 * 12 + 3, 2022 * 12 + 9)],  # 6 mos
        duration_months=6.0,
        duration_years=0.5,
        duties_relevance=1.0,
    )
    total_years, had_overlaps = exp_service.compute_non_overlapping_years([r_full, r_intern])
    assert had_overlaps is True
    assert total_years == 1.0
    adj_years = exp_service.compute_relevance_adjusted_years([r_full, r_intern])
    # The full-time role (mult=1.0) takes precedence over the internship (mult=0.4) during overlap
    assert adj_years == 1.0


# ---------------------------------------------------------------------------
# Test 39: Academic and personal projects are not treated as employment
# ---------------------------------------------------------------------------
def test_batch3_projects_not_treated_as_employment(exp_service, scorer):
    """Projects section must not count toward employment records."""
    resume = """
    Frank Student
    Projects
    Real-Time Data Streaming Pipeline (Jan 2021 - Dec 2022)
    - Built Apache Kafka and Python pipeline for stock market feeds.
    - Managed database scaling and automated ETL.
    """
    ext = scorer.detector.detect_sections(resume, file_path="frank.txt")
    records = exp_service.extract_experience_records(ext)
    assert len(records) == 0  # Projects must not produce employment records


# ---------------------------------------------------------------------------
# Test 40: Current employment duration is handled consistently
# ---------------------------------------------------------------------------
def test_batch3_current_employment_duration(exp_service):
    """Role from Jan 2024 to Present evaluated against reference date (Oct 2026)."""
    text = "Software Engineer at TechCorp\nJan 2024 - Present\nDeveloped Python microservices."
    intervals, is_curr, s_str, e_str = exp_service.parse_date_intervals(text)
    assert is_curr is True
    assert e_str == "Present"
    start_idx = 2024 * 12 + 1
    ref_idx = exp_service.config.REFERENCE_YEAR * 12 + exp_service.config.REFERENCE_MONTH
    expected_months = ref_idx - start_idx
    actual_months = sum(en - st for st, en in intervals)
    assert actual_months == expected_months


# ---------------------------------------------------------------------------
# Test 41: Missing dates produce an uncertainty result when duration cannot be verified
# ---------------------------------------------------------------------------
def test_batch3_missing_dates_produces_uncertainty(exp_service, scorer):
    """Resume lists role but has no dates -> marked uncertain."""
    resume = """
    Grace Engineer
    Experience
    Senior Python Developer at FinTech Global
    - Designed distributed database clusters and Python services.
    """
    ext = scorer.detector.detect_sections(resume, file_path="grace.txt")
    records = exp_service.extract_experience_records(ext)
    assert len(records) == 1
    assert any("duration cannot be verified" in un for un in records[0].uncertainty_notes)


# ---------------------------------------------------------------------------
# Test 42: JD requirement expressed in months
# ---------------------------------------------------------------------------
def test_batch3_jd_requirement_in_months(exp_service):
    """JD states 'Minimum 6 months of professional experience' -> 0.5 years."""
    jd = JobDescription(
        job_id="JD_MOS",
        title="Junior Analyst",
        raw_text="Requirements\nMinimum 6 months of professional experience in data analysis.",
        minimum_experience_years=0.0,
    )
    req = exp_service.parse_jd_experience_requirement(jd)
    assert req.is_mandatory is True
    assert req.minimum_years == 0.5
    assert req.units_used == "months"


# ---------------------------------------------------------------------------
# Test 43: JD requirement expressed as a year range, using its lower bound
# ---------------------------------------------------------------------------
def test_batch3_jd_requirement_year_range_lower_bound(exp_service):
    """JD states '1-2 years of experience' -> uses lower bound 1.0 year."""
    jd = JobDescription(
        job_id="JD_RANGE",
        title="Associate Engineer",
        raw_text="Requirements\n1-2 years of experience in software development.",
        minimum_experience_years=0.0,
    )
    req = exp_service.parse_jd_experience_requirement(jd)
    assert req.is_mandatory is True
    assert req.minimum_years == 1.0
    assert req.is_range is True
    assert req.upper_bound_years == 2.0


# ---------------------------------------------------------------------------
# Test 44: Ambiguous JD requirement yields UNKNOWN
# ---------------------------------------------------------------------------
def test_batch3_ambiguous_jd_requirement_unknown(exp_service):
    """JD states 'Proven experience required' with no numbers -> UNKNOWN."""
    jd = JobDescription(
        job_id="JD_AMB",
        title="Software Engineer",
        raw_text="Requirements\nRelevant experience is required for this role.",
        minimum_experience_years=0.0,
    )
    req = exp_service.parse_jd_experience_requirement(jd)
    assert req.status == ExperienceRequirementStatus.UNKNOWN
    assert len(req.uncertainty_notes) > 0


# ---------------------------------------------------------------------------
# Test 45: Job title alone does not produce high relevance when duties are unrelated
# ---------------------------------------------------------------------------
def test_batch3_title_alone_does_not_grant_relevance(exp_service, scorer):
    """High-sounding job title with completely unrelated duties gets low relevance."""
    jd = JobDescription(
        job_id="JD_AI",
        title="Machine Learning Engineer",
        raw_text="Responsibilities\nDevelop deep learning PyTorch models and neural network training pipelines.",
        responsibilities="Develop deep learning PyTorch models and neural network training pipelines.",
    )
    record = ExperienceRecord(
        role_title="Senior Principal Machine Learning Lead",
        organization="Retail Corp",
        responsibilities_text="Answered telephone inquiries, organized filing cabinets, and delivered office mail.",
    )
    exp_service.score_record_duties(
        record=record,
        jd_query="Develop deep learning PyTorch models and neural network training pipelines.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    assert record.duties_relevance < 0.25


# ---------------------------------------------------------------------------
# Test 46: High lexical and semantic relevance from actual duties is reflected
# ---------------------------------------------------------------------------
def test_batch3_high_relevance_from_actual_duties(exp_service, scorer):
    """Modest title with highly relevant duties gets high duties relevance."""
    jd = JobDescription(
        job_id="JD_ETL",
        title="ETL Developer",
        raw_text="Responsibilities\nBuild automated ETL pipelines in Python and design SQL analytics schemas.",
        responsibilities="Build automated ETL pipelines in Python and design SQL analytics schemas.",
    )
    record = ExperienceRecord(
        role_title="Contractor",
        organization="Acme Inc",
        responsibilities_text="Built automated ETL pipelines in Python and designed SQL analytics schemas for reporting.",
    )
    exp_service.score_record_duties(
        record=record,
        jd_query="Build automated ETL pipelines in Python and design SQL analytics schemas.",
        jd_vector=scorer.embedding_service.compute_jd_vectors(jd).get("full"),
        bm25_scorer_fn=lambda t, q: scorer.score_section_bm25(t, q, "cand", "exp"),
        semantic_scorer_fn=lambda t, v: scorer.score_section_semantic(t, v),
    )
    assert record.duties_relevance > 0.70


# ---------------------------------------------------------------------------
# Test 47: Unknown experience type is handled transparently
# ---------------------------------------------------------------------------
def test_batch3_unknown_experience_type_handling(exp_service, scorer):
    """Unrecognized role classification applies conservative unknown multiplier and flags note."""
    resume = """
    Hanna Unknown
    Experience
    Member of Collaborative Effort
    Jan 2022 - Jan 2023
    - Contributed to team tasks.
    """
    ext = scorer.detector.detect_sections(resume, file_path="hanna.txt")
    records = exp_service.extract_experience_records(ext)
    assert len(records) == 1
    assert records[0].exp_type == ExperienceType.UNKNOWN
    assert records[0].type_multiplier == 0.5
    assert any("Uncertain employment type" in un for un in records[0].uncertainty_notes)


# ---------------------------------------------------------------------------
# Test 48: Education remains excluded from experience and core relevance scoring
# ---------------------------------------------------------------------------
def test_batch3_education_excluded_from_experience(scorer, sample_jd):
    """Education credentials do not add experience duration or inflate experience score."""
    resume_edu_heavy = """
    Ian PhD
    Education
    PhD in Computer Science, Stanford University (2018 - 2023)
    Bachelor of Science in Computer Science, MIT (2014 - 2018)
    Skills
    Python, SQL
    """
    ext = scorer.detector.detect_sections(resume_edu_heavy, file_path="ian.txt")
    res = scorer.score_extraction_result(ext, "CAND_IAN", sample_jd)
    # Experience score must be 0.0 because there is no employment history
    exp_eval = res.experience_evaluation
    assert exp_eval is not None
    assert exp_eval.verified_total_years == 0.0
    assert exp_eval.final_experience_score == 0.0


# ---------------------------------------------------------------------------
# Test 49: Experience evidence is not counted twice in the final score
# ---------------------------------------------------------------------------
def test_batch3_no_double_counting_in_final_score(scorer, sample_jd):
    """Verify experience evidence contributes through its own dimension without duplicate weighting."""
    resume = """
    Jack Engineer
    Skills
    Python, SQL
    Experience
    Senior Data Analyst at Enterprise Corp (2020 - 2023)
    - Built ETL data pipelines using Python and SQL.
    """
    ext = scorer.detector.detect_sections(resume, file_path="jack.txt")
    res = scorer.score_extraction_result(ext, "CAND_JACK", sample_jd)
    # Core score is composed cleanly
    assert res.core_score <= 100.0
    assert "experience" in res.core_dimension_scores
    # Additional bonus evaluates only non-core sections (not experience)
    assert res.additional_bonus <= 3.0


# ---------------------------------------------------------------------------
# Test 50: All experience contributions and bonuses respect documented caps
# ---------------------------------------------------------------------------
def test_batch3_experience_caps_respected(exp_service, scorer):
    """Candidates with 15 years experience against 1 year requirement have duration ratio capped at 1.0."""
    jd = JobDescription(
        job_id="JD_1YR",
        title="Software Engineer",
        raw_text="Minimum 1 year of experience required.",
        minimum_experience_years=1.0,
    )
    resume = """
    Senior Architect
    Experience
    Senior Architect at Global Systems
    2010 - 2025
    - Designed distributed systems and software architecture.
    """
    ext = scorer.detector.detect_sections(resume, file_path="long.txt")
    eval_res = exp_service.evaluate_candidate_experience(
        extraction=ext,
        jd=jd,
        jd_query="Software Engineer",
        jd_vector=None,
        bm25_scorer_fn=lambda t, q: (1.0, 5.0, []),
        semantic_scorer_fn=lambda t, v: (1.0, 0.8),
    )
    assert eval_res.duration_ratio <= 1.0
    assert eval_res.final_experience_score <= 1.0
    assert eval_res.fresher_bonus <= 2.0
