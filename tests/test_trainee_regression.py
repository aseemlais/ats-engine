"""Regression tests from the fresher/trainee diagnostic (JD wording: 'strong fundamentals' in a Mandatory section,
'Supporting Skills', 'not mandatory ... training', 'Certifications are not mandatory'). No embedding model needed."""
import types
from pathlib import Path
import numpy as np
import pytest
from app.config import config
from app.services.jd_parser import JDParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.skill_matcher import SkillMatcherService
from app.services.hybrid_ranker import HybridRankerService
from app.services.semantic_search import SemanticSearchService
from app.services.bm25_search import BM25SearchService
from app.services.certification_matcher import CertificationMatcherService

F = Path(__file__).resolve().parent / "fixtures"
JD = (F / "jd_databricks_trainee.txt").read_text()
RES = (F / "resume_fresher_databricks.txt").read_text()


def _jd(t=JD): return JDParserService().parse_job_description(t, "J")
def _res(t=RES): return NLPExtractorService().extract_structured("C", "t", "h", t, t)


def test_mandatory_section_beats_fundamentals_wording():
    jd = _jd()
    assert set(jd.required_skills) == {"python", "sql", "mysql"} and set(jd.mandatory_skills) == set(jd.required_skills)
    assert ["mysql", "sql"] in jd.skill_alternatives
    assert {"spark", "databricks"} <= set(jd.nonmandatory_skills) and not ({"spark", "databricks"} & set(jd.required_skills))
    assert {"data engineering", "spark", "databricks"} <= set(jd.supporting_skills)
    assert jd.experience_applicable is False and jd.education_requirements == []


def test_fresher_skill_score_and_no_missing_mysql():
    d = SkillMatcherService().evaluate_detailed(_res(), _jd())
    assert d["missing_required"] == [] and d["score"] > 0.9


def _rank(sem, jd_text=JD, res_text=RES, tmp=None):
    import shutil, tempfile
    idx = Path(tempfile.mkdtemp()) / "b.pkl"
    sem_svc = SemanticSearchService(embedding_service=types.SimpleNamespace(compute_jd_vectors=lambda j: {}), qdrant_db=object())
    sem_svc.compute_section_similarity = lambda a, b: sem
    rk = HybridRankerService(semantic_service=sem_svc, bm25_service=BM25SearchService(index_path=idx),
                             certification_service=CertificationMatcherService(None))
    r = _res(res_text)
    rk.hard_filter_service.enabled = False
    return rk.rank_candidates([r], {r.candidate_id: {"full": np.ones(2)}}, _jd(jd_text), top_k=1).results[0]


def test_fresher_experience_excluded_and_cert_bonus_bounded():
    b = _rank(0.55)
    assert b.experience_applicable is False and b.components["experience"]["effective_weight"] == 0.0
    assert b.certification_level == "none" and 0 < b.certification_bonus <= config.cert_bonus_none * 100 + 1e-9
    eff = sum(c["effective_weight"] for c in b.components.values())
    assert eff == pytest.approx(1.0)
    base = sum(c["contribution"] for c in b.components.values()) * 100   # components are rounded to 4 dp
    assert b.raw_ats_score == pytest.approx(min(100.0, base + b.certification_bonus), abs=0.1)


def test_unrelated_certs_give_no_bonus():
    unrelated = RES.split("CERTIFICATIONS")[0] + "CERTIFICATIONS\n- Google: Foundations of Gardening\n- IBM: Introduction to Pottery\n"
    assert _rank(0.55, res_text=unrelated).certification_bonus == 0.0


def test_decisions_use_unrounded_score():
    from app.services.shortlist_policy import recommend_single
    assert recommend_single(59.996, True)[1] == "Review"       # display 60.0 must not flip the decision
