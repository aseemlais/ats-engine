"""Stage 1 (P1): the BM25 query is built from parsed technical content only."""
from pathlib import Path
import pytest
from app.models.job import JobDescription
from app.services.bm25_query import build_bm25_query, MIN_QUERY_TOKENS
from app.services.bm25_search import BM25SearchService
from app.services.jd_parser import JDParserService
from app.utils.text_cleaner import tokenize_for_bm25

FIX = Path(__file__).parent / "fixtures"
parse = lambda t: JDParserService().parse_job_description(t, job_id="T")

BULLET = """Data Engineer
Requirements
- Python
- SQL or MySQL
- Experience with Airflow
Responsibilities
- Build and maintain ETL pipelines
- Optimise warehouse queries
"""
PROSE = ("Data Engineer\nWe need a data engineer. Python and SQL are required. Airflow experience is a plus. "
         "You will build ETL pipelines and optimise warehouse queries.\n")
ADMIN_BULLET = """
Compensation and process
- Salary range 8-12 LPA, night shift rotational
- Interview process: 3 rounds, notice period 30 days
- Location: Pune, hybrid, relocation assistance
"""
ADMIN_IN_RESPONSIBILITIES = "- Attend 3 interview rounds and negotiate salary (CTC) at joining\n- Work from office on rotational shift\n"


@pytest.mark.parametrize("base", [BULLET, PROSE])
def test_admin_additions_do_not_change_query_or_score(base):
    plain = build_bm25_query(parse(base))
    with_admin = build_bm25_query(parse(base + ADMIN_BULLET))
    assert plain.method == "structured" and with_admin.text == plain.text
    assert tokenize_for_bm25(with_admin.text) == tokenize_for_bm25(plain.text)
    svc = BM25SearchService(index_path=Path("/nonexistent/idx.pkl"))
    svc.build_index(["C"], ["python sql airflow etl pipelines warehouse engineer"], persist=False)
    assert svc.search(plain.text) == svc.search(with_admin.text)


def test_admin_lines_inside_responsibilities_are_filtered():
    jd = parse(BULLET.replace("- Optimise warehouse queries\n", "- Optimise warehouse queries\n" + ADMIN_IN_RESPONSIBILITIES))
    q = build_bm25_query(jd)
    for w in ("salary", "interview", "shift", "ctc"):
        assert w not in q.tokens
    assert q.lines_excluded_as_administrative == 2 and "warehouse" in q.tokens


def test_query_contains_all_skill_tiers():
    jd = parse("Analyst\nRequirements\n- SQL\n- Basic understanding of Tableau\nNice to have\n- AWS\n")
    q = build_bm25_query(JobDescription(**{**jd.model_dump(), "responsibilities": "Build reports and analyse trends with data"}))
    assert {"sql", "tableau", "aws"} <= set(q.tokens) and q.method == "structured"


def test_fallback_when_parser_output_is_empty_or_short():
    # imperfect parser output: only raw text, nothing parsed
    raw = "We need someone who builds data pipelines.\nSalary 10 LPA, hybrid, 2 interview rounds.\nStrong analytical mindset."
    q = build_bm25_query(JobDescription(raw_text=raw))
    assert q.method == "fallback_full_jd" and q.fallback_reason
    assert "pipelines" in q.tokens and "salary" not in q.tokens and "interview" not in q.tokens
    short = build_bm25_query(JobDescription(raw_text=raw, required_skills=["sql"]))
    assert short.method == "fallback_full_jd" and "shorter" in short.fallback_reason


def test_empty_everything_gives_empty_query():
    q = build_bm25_query(JobDescription(raw_text=""))
    assert q.text == "" and q.tokens == []


def test_no_job_specific_hardcoding():
    for skills in (["welding"], ["accounting", "tally"]):    # arbitrary terms flow through untouched
        q = build_bm25_query(JobDescription(raw_text="x", required_skills=skills, responsibilities="Perform daily work duties for clients"))
        assert set(skills) <= set(q.tokens)
