"""Focused tests for the scoring-math fixes. No embedding model needed (a deterministic fake stands in
for it ONLY to exercise the ranker arithmetic; its similarity values are not realistic)."""
import hashlib, importlib, re, types, sys
from pathlib import Path

import numpy as np
import pytest

from app.config import config
from app.models.candidate import CandidateScoreBreakdown
from app.models.job import JobDescription
from app.services.bm25_search import BM25SearchService
from app.services.certification_matcher import certification_requirement_level as cert_level
from app.services.experience_matcher import ExperienceMatcherService
from app.services.jd_parser import JDParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.shortlist_policy import apply_batch_policy, percentile_ranks, recommend_single
from app.services.skill_matcher import SkillMatcherService
from app.utils.text_cleaner import tokenize_for_bm25

DATA = Path(__file__).resolve().parent.parent / "data" / "resumes"
JD_TXT = (DATA / "JD_Senior_Data_Analyst.txt").read_text(encoding="utf-8", errors="replace")
NAMES = ["Resume_1_PriyaNair", "Resume_2_DavidChen", "Resume_3_SarahJohnson"]
TXT = {n: (DATA / f"{n}.txt").read_text(encoding="utf-8", errors="replace") for n in NAMES}
nlp = NLPExtractorService()


def parse(text, cid="C"):
    return nlp.extract_structured(cid, "t", "h", text, text)


# ---------------------------------------------------------------- BM25
def _reference(n=40):
    """Synthetic reference corpus: unrelated + partially related documents."""
    ids, texts = [], []
    for i in range(n):
        base = "marketing campaign brand social media content event coordination copywriting"
        extra = " sql python reporting" if i % 5 == 0 else (" excel dashboards" if i % 7 == 0 else "")
        ids.append(f"REF{i}"); texts.append(base + extra + f" filler{i}")
    return ids, texts


@pytest.fixture
def svc(tmp_path):
    s = BM25SearchService(index_path=tmp_path / "bm25.pkl")
    ids, texts = _reference()
    s.build_index(ids, texts, persist=True)          # ingestion path writes the reference corpus
    return s


def test_single_candidate_positive_ordered_and_equals_batch(svc):
    single = {}
    for n in NAMES:
        svc.build_index(["CAND_SCREEN"], [TXT[n]], persist=False)
        single[n] = svc.search(JD_TXT)["CAND_SCREEN"]
    assert all(0.0 < v < 1.0 for v in single.values())
    assert single[NAMES[0]] < single[NAMES[1]] < single[NAMES[2]]
    svc.build_index(NAMES, [TXT[n] for n in NAMES], persist=False)
    batch = svc.search(JD_TXT)
    for n in NAMES:
        assert batch[n] == pytest.approx(single[n], abs=0.03)   # batch-independent (only N/avgdl shift slightly)


def test_weak_batch_is_not_stretched_to_one(svc):
    svc.build_index(NAMES[:1] + ["W2"], [TXT[NAMES[0]], "gardening and cooking hobbies"], persist=False)
    assert max(svc.search(JD_TXT).values()) < 0.5               # old code divided by the batch maximum -> 1.0


def test_screening_never_overwrites_persisted_index(svc):
    before = hashlib.md5(svc.index_path.read_bytes()).hexdigest()
    svc.build_index(["CAND_SCREEN"], [TXT[NAMES[1]]], persist=False)
    svc.search(JD_TXT)
    assert hashlib.md5(svc.index_path.read_bytes()).hexdigest() == before
    assert len(BM25SearchService(index_path=svc.index_path).candidate_ids) == 40


# ---------------------------------------------------------------- weighted sum / missing-data policy
class FakeEmb:
    def _v(self, text):
        v = np.zeros(384, dtype=np.float32)
        for t in tokenize_for_bm25(text or ""):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % 384] += 1
        n = np.linalg.norm(v)
        return v / n if n else v
    encode_text = lambda self, t: self._v(t)
    def compute_resume_vectors(self, r):
        d = {"full": self._v(r.cleaned_text)}
        for s in ("skills", "experience", "projects", "education"):
            d[s] = self._v(r.section_texts.get(s, "")) if r.section_texts.get(s, "").strip() else d["full"]
        return d
    def compute_jd_vectors(self, jd):
        f = self._v(jd.raw_text)
        return {"full": f, "skills": self._v(" ".join(jd.required_skills)), "experience": f, "projects": f, "education": f}


def _ranker(tmp_path):
    from app.services.hybrid_ranker import HybridRankerService
    from app.services.semantic_search import SemanticSearchService
    from app.services.certification_matcher import CertificationMatcherService
    emb = FakeEmb()
    sem = SemanticSearchService(embedding_service=emb, qdrant_db=object())
    bm = BM25SearchService(index_path=tmp_path / "b.pkl")
    return HybridRankerService(semantic_service=sem, bm25_service=bm, certification_service=CertificationMatcherService(emb)), emb


def _score(tmp_path, jd_text, name):
    rk, emb = _ranker(tmp_path)
    jd = JDParserService().parse_job_description(jd_text, job_id="J")
    r = parse(TXT[name], "X")
    resp = rk.rank_candidates([r], {"X": emb.compute_resume_vectors(r)}, jd, top_k=1)
    return resp.results[0], resp.weights_used


def _manual(b, w, cert_eff):
    return round(100 * (w["semantic"] * b.semantic_score + w["bm25"] * b.bm25_score + w["skill"] * b.skill_score
                        + w["experience"] * b.experience_score + w["certification"] * cert_eff), 2)


def test_weighted_sum_matches_formula_required_cert(tmp_path):
    jd = JD_TXT.replace("NICE TO HAVE\n- AWS certification", "NICE TO HAVE\n- Experience in e-commerce analytics")
    jd = jd.replace("REQUIREMENTS\n", "REQUIREMENTS\n- AWS certification required\n")
    b, w = _score(tmp_path, jd, NAMES[1])
    assert b.certification_level == "required"
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert b.final_score == pytest.approx(_manual(b, w, b.certification_score), abs=0.02)   # unchanged formula


def test_preferred_missing_cert_is_neutral_not_zero_not_bonus(tmp_path):
    b, w = _score(tmp_path, JD_TXT, NAMES[1])                     # AWS certification is under NICE TO HAVE
    assert b.certification_level == "preferred" and b.certification_score == 0.0
    avg = (w["semantic"] * b.semantic_score + w["bm25"] * b.bm25_score + w["skill"] * b.skill_score
           + w["experience"] * b.experience_score) / (1 - w["certification"])
    assert b.certification_effective_score == pytest.approx(avg, abs=1e-3)   # own average: no unearned credit
    assert b.final_score == pytest.approx(_manual(b, w, avg), abs=0.02)


def test_no_cert_mention_means_not_applicable_not_060(tmp_path):
    jd = re.sub(r"NICE TO HAVE\n- AWS certification\n", "NICE TO HAVE\n", JD_TXT)
    b, _ = _score(tmp_path, jd, NAMES[1])
    assert b.certification_level == "none" and b.certification_effective_score != pytest.approx(0.60)


def test_legacy_policy_restores_old_cert_behaviour(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "na_policy", "legacy")
    jd = re.sub(r"NICE TO HAVE\n- AWS certification\n", "NICE TO HAVE\n", JD_TXT)
    b, _ = _score(tmp_path, jd, NAMES[1])
    assert b.certification_score == pytest.approx(0.60) and b.certification_effective_score == pytest.approx(0.60)


@pytest.mark.parametrize("txt,lvl", [
    (JD_TXT, "preferred"),
    ("REQUIREMENTS\n- AWS Certified Solutions Architect required\n", "required"),
    ("Requirements:\n- PMP certification\n", "required"),
    ("Requirements:\n- Python\nBenefits:\n- Certification reimbursement\n", "none"),
    ("- 3 years of Python\n", "none"),
])
def test_certification_requirement_level(txt, lvl):
    assert cert_level(txt) == lvl


# ---------------------------------------------------------------- skills
def test_skill_alternatives_and_aliases_and_no_double_penalty():
    jd = JDParserService().parse_job_description(JD_TXT, job_id="J")
    assert ["airflow", "redshift"] in jd.skill_alternatives
    assert "tableau" in jd.mandatory_skills and set(jd.mandatory_skills) == set(jd.required_skills)   # Requirements section
    assert "aws" not in jd.preferred_skills                       # "AWS certification" belongs to the cert component
    sc, matched, missing, _ = SkillMatcherService().evaluate_skills(parse(TXT[NAMES[1]]), jd)
    assert "redshift" in matched and "airflow / redshift (any one)" not in missing
    assert {"tableau", "data engineering"} <= set(missing)        # genuine gaps are still reported
    assert sc == pytest.approx(7 / 10)                            # 10 slots: 9 singles + 1 group; no free preferred credit


def test_no_preferred_skills_gives_no_free_credit(monkeypatch):
    jd = JobDescription(required_skills=["python", "sql"], preferred_skills=[])
    r = types.SimpleNamespace(skills=["python"])
    assert SkillMatcherService().evaluate_skills(r, jd)[0] == pytest.approx(0.5)
    monkeypatch.setattr(config, "na_policy", "legacy")
    assert SkillMatcherService().evaluate_skills(r, jd)[0] == pytest.approx(0.8 * 0.5 + 0.2 * 0.5)


# ---------------------------------------------------------------- experience
def _bd(t):
    return nlp.extract_experience_breakdown(t, nlp.segment_sections(t))


def test_work_entry_followed_by_internship_is_not_misclassified():
    t = "J\nEXPERIENCE\nData Analyst - Acme (2021-2025)\n- x\nData Analyst Intern - Beta (2020-2021)\n- y\n"
    assert _bd(t)[:2] == (4.0, 1.0)
    t2 = "J\nEXPERIENCE\nData Analyst Intern - Beta (2020-2021)\n- y\nData Analyst - Acme (2021-2025)\n- x\n"
    assert _bd(t2)[:2] == (4.0, 1.0)                               # order independent


def test_overlapping_jobs_not_double_counted():
    t = "J\nEXPERIENCE\nAnalyst - A (2020-2024)\n- x\nConsultant - B (2022-2024)\n- y\n"
    assert _bd(t)[0] == 4.0
    t = "J\nEXPERIENCE\nAnalyst - A (2019-2021)\n- x\nSenior Analyst - B (2021-2024)\n- y\n"
    assert _bd(t)[0] == 5.0                                        # sequential jobs still add


def test_undated_internship_is_unverified():
    r = parse("J\nINTERNSHIP\nWorked on a machine learning project as an intern building models for churn\n")
    assert r.internship_years == -0.01 and r.experience_years == 0.0
    jd = JobDescription(title="Data Analyst", minimum_experience_years=2.0, raw_text="Data Analyst")
    assert ExperienceMatcherService().evaluate_experience(r, jd) < 0.2   # no credited duration


# ---------------------------------------------------------------- policy / config
@pytest.mark.parametrize("score,eligible,rec", [
    (72, True, "Shortlisted"), (60, True, "Shortlisted"), (59.96, True, "Review"), (45, True, "Review"),
    (44.9, True, "Not Shortlisted"), (80, False, "Not Shortlisted"), (50, False, "Not Shortlisted")])
def test_recommend_single(score, eligible, rec):
    assert recommend_single(score, eligible)[1] == rec


def _cands(scores, elig="Eligible"):
    return [CandidateScoreBreakdown(candidate_id=f"C{i}", final_score=s, eligibility=elig)
            for i, s in enumerate(sorted(scores, reverse=True))]


def test_best_available_topk_and_honest_scores():
    c = _cands([40, 38, 36, 35, 30, 20, 10, 5]); apply_batch_policy(c, k=5)
    assert [x.recommendation for x in c[:5]] == ["Best Available Match"] * 5
    assert [x.recommendation for x in c[5:]] == ["Not Shortlisted"] * 3
    assert all(x.raw_ats_score == x.final_score for x in c)
    c = _cands([30, 20]); apply_batch_policy(c, k=5)
    assert [x.recommendation for x in c] == ["Best Available Match"] * 2     # fewer than K
    c = _cands([75, 40]); apply_batch_policy(c, k=5)
    assert [x.recommendation for x in c] == ["Shortlisted", "Not Shortlisted"]
    c = _cands([50, 40], elig="Not Eligible"); apply_batch_policy(c, k=5)
    assert all(x.recommendation == "Not Shortlisted" for x in c)             # mandatory not waived


def test_percentiles():
    assert percentile_ranks([10, 20, 30]) == [16.7, 50.0, 83.3] and percentile_ranks([50, 50]) == [50.0, 50.0]


def test_threshold_env_and_validation(monkeypatch):
    import app.config as cfg
    monkeypatch.setenv("SHORTLIST_THRESHOLD", "65"); monkeypatch.setenv("REVIEW_THRESHOLD", "40")
    importlib.reload(cfg); assert (cfg.config.shortlist_threshold, cfg.config.review_threshold) == (65.0, 40.0)
    monkeypatch.setenv("REVIEW_THRESHOLD", "70")
    with pytest.raises(ValueError):
        importlib.reload(cfg)
    monkeypatch.delenv("SHORTLIST_THRESHOLD"); monkeypatch.delenv("REVIEW_THRESHOLD"); importlib.reload(cfg)
