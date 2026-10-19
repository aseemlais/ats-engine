"""Pipeline tests: chunking, section mapping, composition, ranker integration, degraded mode, eligibility,
BM25 lifecycle, configuration, API compatibility.

MOCK-BASED: every test that touches embeddings uses MockSentenceModel (a deterministic hashed bag-of-words encoder).
Its vectors are NOT real semantic embeddings; these tests verify plumbing/arithmetic (shapes, chunking, aggregation,
weights, flags) and say NOTHING about real-model semantic quality. Real-model tests live in test_embeddings.py /
test_ranking.py and are skipped when the model cannot load. Fixtures named synthetic_* are SYNTHETIC.
"""
import hashlib, os, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

from app.config import config
from app.models.candidate import CandidateScoreBreakdown, SearchResponse
from app.models.job import JobDescription
from app.services import embedding_service as es
from app.services.bm25_search import BM25SearchService
from app.services.certification_matcher import CertificationMatcherService
from app.services.embedding_service import EmbeddingService, EmbeddingUnavailableError, section_text
from app.services.hard_filter import HardFilterService
from app.services.hybrid_ranker import HybridRankerService
from app.services.jd_parser import JDParserService
from app.services.nlp_extractor import NLPExtractorService
from app.services.score_composer import certification_status, compose_score
from app.services.semantic_search import SemanticSearchService
from app.services.text_chunking import aggregate_vectors, chunk_text

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
DIM = 64
nlp = NLPExtractorService()
JD_TXT = (FIX / "synthetic_trainee_jd.txt").read_text()
RES_TXT = (FIX / "synthetic_trainee_resume.txt").read_text()
W = {"semantic": 0.40, "bm25": 0.27, "skill": 0.15, "experience": 0.10, "certification": 0.08}


class MockSentenceModel:
    """MOCK: hashed bag-of-words unit vectors. Counts calls so tests can assert chunking/caching behaviour."""
    def __init__(self, dim=DIM):
        self.dim, self.calls, self.texts_seen = dim, 0, []

    def encode(self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            self.texts_seen.append(t)
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        return out


def parse(text, cid="C"):
    return nlp.extract_structured(cid, "t", "h", text, text)


def mock_emb(model=None):
    return EmbeddingService(model=model or MockSentenceModel(), dimension=DIM)


def make_ranker(tmp_path, emb=None):
    emb = emb or mock_emb()
    sem = SemanticSearchService(embedding_service=emb, qdrant_db=object())
    bm = BM25SearchService(index_path=tmp_path / "bm.pkl")
    return HybridRankerService(semantic_service=sem, bm25_service=bm,
                               certification_service=CertificationMatcherService(emb)), emb


def rank(tmp_path, jd_text, resume_text, emb=None, vectors=True, **kw):
    rk, emb = make_ranker(tmp_path, emb)
    jd = JDParserService().parse_job_description(jd_text, job_id="J")
    r = parse(resume_text, "X")
    vm = {"X": emb.compute_resume_vectors(r)} if vectors else {}
    return rk.rank_candidates([r], vm, jd, top_k=1, **kw), jd, r


# =================================================================== chunking / aggregation (no model)
def test_chunk_empty_and_short():
    assert chunk_text("") == ([], False) and chunk_text("   \n ") == ([], False)
    assert chunk_text("a b c", 10, 2, 3) == (["a b c"], False)


def test_chunk_long_text_is_bounded_overlapping_and_flags_truncation():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks, trunc = chunk_text(text, 100, 20, 5)
    assert len(chunks) == 5 and trunc is True                       # bounded; the START of the text is kept
    assert chunks[0].split()[0] == "w0" and chunks[1].split()[0] == "w80"   # step = 100 - 20
    assert all(len(c.split()) <= 100 for c in chunks)
    chunks, trunc = chunk_text(text, 100, 20, 50)
    assert trunc is False and chunks[-1].split()[-1] == "w999"      # nothing dropped when under the cap


def test_chunk_invalid_params():
    for bad in ((0, 0, 1), (10, 10, 1), (10, 2, 0)):
        with pytest.raises(ValueError):
            chunk_text("a b", *bad)


def test_aggregate_vectors_unit_norm_weighted_and_errors():
    v = aggregate_vectors([np.array([1.0, 0.0]), np.array([0.0, 1.0])], [3, 1])
    assert np.linalg.norm(v) == pytest.approx(1.0) and v[0] > v[1]
    assert not aggregate_vectors([np.zeros(2)], None).any()         # all-zero stays zero (no NaN)
    with pytest.raises(ValueError):
        aggregate_vectors([])
    with pytest.raises(ValueError):
        aggregate_vectors([np.ones(2)], [0.0])


# =================================================================== embedding service with MOCK model
def test_section_text_maps_internship_into_experience():
    r = parse(RES_TXT)
    assert "Data Science Intern" in section_text(r, "experience")      # internship content is no longer invisible
    assert "Bronze-Silver-Gold" in section_text(r, "projects")
    assert "SQL" in section_text(r, "skills") and "M.Sc." in section_text(r, "education")


def test_resume_vectors_shapes_and_missing_section_falls_back_to_full():
    r = parse("Jane\nSKILLS\nPython, SQL\n")
    v = mock_emb().compute_resume_vectors(r)
    assert set(v) == {"full", "skills", "experience", "projects", "education"}
    assert all(x.shape == (DIM,) for x in v.values())
    assert np.array_equal(v["experience"], v["full"]) and np.array_equal(v["projects"], v["full"])   # no such section
    assert not np.array_equal(v["skills"], v["full"])


def test_empty_text_gives_zero_vector_not_error():
    e = mock_emb()
    assert not e.encode_text("").any() and e.encode_text("   ").shape == (DIM,)


def test_dimension_mismatch_is_reported():
    with pytest.raises(ValueError, match="dimension"):
        EmbeddingService(model=MockSentenceModel(dim=32), dimension=DIM).encode_text("hello world")


def test_long_text_is_chunked_not_truncated(monkeypatch):
    monkeypatch.setattr(config, "chunk_max_words", 50); monkeypatch.setattr(config, "chunk_overlap_words", 10)
    monkeypatch.setattr(config, "max_chunks_per_section", 4)
    m = MockSentenceModel(); e = mock_emb(m)
    v = e.encode_text(" ".join(f"tok{i}" for i in range(500)))
    assert len(m.texts_seen) == 4 and np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)   # capped at max_chunks
    assert any("tok0 " in t or t.startswith("tok0") for t in m.texts_seen)


def test_corpus_batched_matches_individual():
    rs = [parse(RES_TXT, "A"), parse("Bob\nSKILLS\nExcel\nEXPERIENCE\nAnalyst - X (2019-2022)\n- dashboards\n", "B")]
    e = mock_emb()
    batch = e.compute_corpus_vectors_batched(rs)
    for r in rs:
        single = e.compute_resume_vectors(r)
        for k in single:
            assert np.allclose(batch[r.candidate_id][k], single[k], atol=1e-6)
    assert e.compute_corpus_vectors_batched([]) == {}


def test_model_unavailable_raises_and_fails_fast_without_retry(monkeypatch):
    calls = {"n": 0}

    def boom(name):
        calls["n"] += 1
        raise OSError("403 Forbidden")
    monkeypatch.setattr(es, "SentenceTransformer", boom)
    e = EmbeddingService(dimension=DIM)
    for _ in range(3):
        with pytest.raises(EmbeddingUnavailableError, match="403"):
            e.encode_text("hello")
    assert calls["n"] == 1                                          # a blocked download is attempted ONCE


def test_jd_vectors_wording_depends_on_experience_requirement():
    m = MockSentenceModel(); e = mock_emb(m)
    jd = JDParserService().parse_job_description(JD_TXT, job_id="J")
    e.compute_jd_vectors(jd)
    seen = " | ".join(m.texts_seen)
    assert "0.0 years" not in seen and "internships" in seen          # fresher JD: no fabricated '0 years' query
    assert "databricks" in seen.lower() and "spark" in seen.lower()   # supporting skills are part of the skills query
    m2 = MockSentenceModel()
    mock_emb(m2).compute_jd_vectors(JobDescription(title="Data Engineer", raw_text="x", required_skills=["python"],
                                                   minimum_experience_years=3.0, experience_applicable=True))
    assert any("3 years experience" in t for t in m2.texts_seen)


def test_cosine_rejects_mismatched_or_missing_vectors():
    s = SemanticSearchService(embedding_service=mock_emb(), qdrant_db=object())
    with pytest.raises(ValueError):
        s.compute_cosine_similarity(np.ones(3), np.ones(4))
    with pytest.raises(ValueError):
        s.compute_cosine_similarity(None, np.ones(4))
    assert s.compute_cosine_similarity(np.zeros(4), np.ones(4)) == 0.0


# =================================================================== composition arithmetic
ALL = {c: "applied" for c in W}


def test_compose_contributions_sum_to_final_and_scale_invariant():
    sc = {"semantic": .5, "bm25": .2, "skill": .8, "experience": .6, "certification": .4}
    a = compose_score(sc, ALL, W)
    assert a["final"] == pytest.approx(sum(c["contribution"] for c in a["components"].values()), abs=1e-3)
    assert a["final"] == pytest.approx(.4 * .5 + .27 * .2 + .15 * .8 + .1 * .6 + .08 * .4, abs=1e-9)   # sum(W)=1.0
    b = compose_score(sc, ALL, {k: v * 7 for k, v in W.items()})
    assert b["final"] == pytest.approx(a["final"])                                                      # weights need not sum to 1


def test_not_applicable_component_is_excluded_and_weights_renormalised():
    sc = {"semantic": .5, "bm25": .2, "skill": .8, "experience": None, "certification": None}
    st = {**ALL, "experience": "not_applicable", "certification": "not_applicable"}
    r = compose_score(sc, st, W)
    active = W["semantic"] + W["bm25"] + W["skill"]
    assert r["final"] == pytest.approx((.4 * .5 + .27 * .2 + .15 * .8) / active)
    assert sum(c["effective_weight"] for c in r["components"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["components"]["experience"]["status"] == "not_applicable" and r["components"]["experience"]["contribution"] == 0.0


def test_no_active_components_is_zero_and_scores_are_clipped():
    assert compose_score({}, {}, W)["final"] == 0.0
    r = compose_score({"semantic": 5.0, "bm25": -2.0}, {"semantic": "applied", "bm25": "applied"}, W)
    assert 0.0 <= r["final"] <= 1.0


def test_certification_policy_levels():
    assert certification_status("none", 0.0, 0.5) == "not_applicable"
    assert certification_status("required", 0.0, 0.5) == "applied"            # absent mandatory cert scores 0 (genuine miss)
    assert certification_status("preferred", 0.0, 0.5) == "neutral"           # absence is not penalised
    assert certification_status("preferred", 0.9, 0.5) == "applied"           # held relevant cert can only help
    assert certification_status("preferred", 0.4, 0.5) == "neutral"           # ...and never drags the score down


def test_preferred_certificate_benefit_is_capped_by_its_weight():
    base = {"semantic": .5, "bm25": .5, "skill": .5, "experience": .5}
    st = {**{c: "applied" for c in base}, "certification": "applied"}
    with_cert = compose_score({**base, "certification": 1.0}, st, W)["final"]
    without = compose_score({**base, "certification": None}, {**st, "certification": "neutral"}, W)["final"]
    assert 0 < with_cert - without <= W["certification"]


# =================================================================== ranker integration (diagnostic synthetic fixture)
def test_diagnostic_fixture_end_to_end_with_mock_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cert_bonus_none", 0.0)   # isolate the pure weighted sum from the optional certification bonus
    resp, jd, r = rank(tmp_path, JD_TXT, RES_TXT)
    b = resp.results[0]
    # JD interpretation
    assert jd.title == "Data Engineering Trainee - Databricks"
    assert {"python", "sql"} <= set(jd.required_skills) or any("sql" in g for g in jd.skill_alternatives)
    assert "spark" not in jd.required_skills and "databricks" not in jd.required_skills
    assert jd.has_experience_requirement is False
    # experience: not applicable -> excluded, not scored as a fresher penalty
    assert b.experience_applicable is False and b.components["experience"]["status"] == "not_applicable"
    assert b.components["certification"]["status"] in ("not_applicable", "neutral")
    assert b.mandatory_requirements_met is True and b.eligibility == "Eligible" and b.missing_mandatory_skills == []
    assert any("sql" in a.lower() for a in b.accepted_alternatives)
    assert b.degraded is False and b.scoring_mode == "full"
    # contributions add up to the final score; active weights sum to 1
    assert sum(c["contribution"] for c in b.components.values()) * 100 == pytest.approx(b.final_score, abs=0.05)
    assert sum(c["effective_weight"] for c in b.components.values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= b.final_score <= 100.0


def test_fresher_is_not_penalised_when_jd_has_no_experience_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cert_bonus_none", 0.0)   # isolate the pure weighted sum from the optional certification bonus
    b = rank(tmp_path, JD_TXT, RES_TXT)[0].results[0]
    manual = sum(W[c] * s for c, s in (("semantic", b.semantic_score), ("bm25", b.bm25_score), ("skill", b.skill_score))) \
        / (W["semantic"] + W["bm25"] + W["skill"])
    assert b.final_score == pytest.approx(manual * 100, abs=0.1)     # experience weight does not enter the formula


def test_explicit_experience_requirement_is_scored(tmp_path):
    jd = JD_TXT.replace("Freshers welcome. No prior experience required.", "Minimum 3 years of experience required.")
    # (a) the fresher is rejected by the experience pre-filter (< 40% of the minimum) - an explicit requirement is enforced
    resp, parsed, _ = rank(tmp_path, jd, RES_TXT)
    assert parsed.minimum_experience_years == 3.0 and resp.results == [] and resp.hard_filtered_count == 0
    # (b) a 2-year analyst passes the filter and is scored transparently: min(2/3, 1) + role bonus, never above 1
    two_years = "Jane\nSKILLS\nSQL, Python\nEXPERIENCE\nData Analyst - Acme (2023-2025)\n- built dashboards\n"
    (tmp_path / "x").mkdir()
    b = rank(tmp_path / "x", jd, two_years)[0].results[0]
    assert b.experience_applicable is True and b.components["experience"]["status"] == "applied"
    assert 0.6 <= b.experience_score <= 1.0 and b.required_experience_years == 3.0


def test_deterministic_output(tmp_path):
    a = rank(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path, JD_TXT, RES_TXT)[0].results[0].model_dump()
    (tmp_path / "b").mkdir()
    b = rank(tmp_path / "b", JD_TXT, RES_TXT)[0].results[0].model_dump()
    assert a == b


def test_batch_score_independent_of_unrelated_candidates_beyond_bm25_tolerance(tmp_path):
    rk, emb = make_ranker(tmp_path)
    # stable reference corpus (what ingestion persists); WITHOUT one, a lone resume is its own corpus and its
    # BM25 score legitimately differs from its in-batch score (documented cold-start limitation)
    ref = [f"marketing brand campaign social media filler{i}" + (" sql python reporting" if i % 5 == 0 else "") for i in range(40)]
    rk.bm25_service.build_index([f"REF{i}" for i in range(40)], ref, persist=True)
    jd = JDParserService().parse_job_description(JD_TXT, job_id="J")
    target = parse(RES_TXT, "T")
    other = [parse(f"Person {i}\nSKILLS\nGardening, Cooking\nEXPERIENCE\nChef - Cafe (2018-2022)\n- cooked food {i}\n", f"O{i}") for i in range(5)]
    vm = {r.candidate_id: emb.compute_resume_vectors(r) for r in [target] + other}
    alone = rk.rank_candidates([target], {"T": vm["T"]}, jd, top_k=10).results[0]
    batch = [c for c in rk.rank_candidates([target] + other, vm, jd, top_k=10).results if c.candidate_id == "T"][0]
    assert batch.semantic_score == alone.semantic_score and batch.skill_score == alone.skill_score
    assert batch.bm25_score == pytest.approx(alone.bm25_score, abs=0.03) and batch.final_score == pytest.approx(alone.final_score, abs=1.0)


# =================================================================== degraded mode
class _Unavailable(EmbeddingService):
    def __init__(self):
        super().__init__(dimension=DIM)

    @property
    def model(self):
        raise EmbeddingUnavailableError("Embedding model unavailable: test")


def test_model_unavailable_fails_loudly_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "allow_degraded_semantic", False)
    rk, _ = make_ranker(tmp_path, _Unavailable())
    jd = JDParserService().parse_job_description(JD_TXT, job_id="J")
    with pytest.raises(EmbeddingUnavailableError):
        rk.rank_candidates([parse(RES_TXT, "X")], {}, jd)


def test_degraded_mode_is_explicit_deterministic_and_excludes_semantic(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "cert_bonus_none", 0.0)   # isolate the pure weighted sum from the optional certification bonus
    monkeypatch.setattr(config, "allow_degraded_semantic", True)
    rk, _ = make_ranker(tmp_path, _Unavailable())
    jd = JDParserService().parse_job_description(JD_TXT, job_id="J")
    resp = rk.rank_candidates([parse(RES_TXT, "X")], {}, jd)
    b = resp.results[0]
    assert b.degraded and b.scoring_mode == "degraded_no_semantic" and resp.degraded_count == 1
    assert b.components["semantic"]["status"] == "unavailable" and b.semantic_score == 0.0
    manual = (W["bm25"] * b.bm25_score + W["skill"] * b.skill_score) / (W["bm25"] + W["skill"])
    assert b.final_score == pytest.approx(manual * 100, abs=0.1)       # no fabricated semantic value
    assert "degraded" in b.explanation.lower()
    again = rk.rank_candidates([parse(RES_TXT, "X")], {}, jd).results[0]
    assert again.final_score == b.final_score


def test_candidate_without_vectors_is_flagged_degraded_not_zero_scored(tmp_path):
    resp, _, _ = rank(tmp_path, JD_TXT, RES_TXT, vectors=False)
    b = resp.results[0]
    assert b.degraded and b.components["semantic"]["status"] == "unavailable"


# =================================================================== eligibility vs ranking
JD_MANDATORY = "Backend Developer\nRequirements\n- Python\n- Docker\nGood to have\n- Kubernetes\n"


def test_missing_mandatory_is_reported_even_in_lenient_mode(tmp_path):
    b = rank(tmp_path, JD_MANDATORY, "Sam\nSKILLS\nPython, Flask\n")[0].results[0]
    assert b.mandatory_requirements_met is False and "docker" in b.missing_mandatory_skills
    assert b.eligibility == "Eligible"                  # lenient default: rejected only when NONE of the mandatory skills is held
    assert "NOT met" in b.explanation


def test_strict_mode_marks_ineligible_but_still_scores(monkeypatch, tmp_path):
    rk, emb = make_ranker(tmp_path)
    rk.hard_filter_service = HardFilterService(enabled=False, strict=True)       # filtering off: candidate is scored, not dropped
    jd = JDParserService().parse_job_description(JD_MANDATORY, job_id="J")
    r = parse("Sam\nSKILLS\nPython, Flask\n", "X")
    b = rk.rank_candidates([r], {"X": emb.compute_resume_vectors(r)}, jd).results[0]
    assert b.eligibility == "Not Eligible" and any("docker" in x.lower() for x in b.eligibility_reasons)
    assert b.final_score > 0 and b.recommendation == "Not Shortlisted"           # fit score never overrides a failed mandatory


def test_optional_skills_never_affect_eligibility(tmp_path):
    jd = JDParserService().parse_job_description(JD_MANDATORY, job_id="J")
    r = parse("Sam\nSKILLS\nPython, Docker\n", "X")                                # lacks preferred Kubernetes only
    rep = HardFilterService(enabled=True, strict=True).assess(r, jd)
    assert rep["eligible"] and rep["mandatory_requirements_met"] and rep["missing_mandatory"] == []


# =================================================================== BM25 lifecycle
def test_bm25_persistence_update_and_empty_guard(tmp_path):
    p = tmp_path / "i.pkl"
    s = BM25SearchService(index_path=p)
    s.build_index(["A", "B"], ["python sql etl", "marketing brand"], persist=True)
    h = hashlib.md5(p.read_bytes()).hexdigest()
    s.build_index(["Z"], ["only one doc"], persist=False)
    assert hashlib.md5(p.read_bytes()).hexdigest() == h                            # transient request never overwrites it
    s.build_index([], [], persist=True)
    assert hashlib.md5(p.read_bytes()).hexdigest() == h                            # empty corpus never overwrites it either
    s2 = BM25SearchService(index_path=p)
    assert s2.candidate_ids == ["A", "B"]
    s2.build_index(["A", "B", "C"], ["python sql etl", "marketing brand", "databricks spark"], persist=True)   # ingestion update
    assert BM25SearchService(index_path=p).candidate_ids == ["A", "B", "C"]


def test_bm25_scores_bounded_zero_for_no_overlap_and_empty_query(tmp_path):
    s = BM25SearchService(index_path=tmp_path / "i.pkl")
    s.build_index(["A", "B"], ["python sql etl pipelines", "gardening cooking"], persist=False)
    sc = s.search("python sql etl pipelines")
    assert 0.0 <= sc["B"] <= sc["A"] <= 1.0 and sc["B"] == 0.0
    assert s.search("") == {}


# =================================================================== configuration precedence / validation
def _cfg_subprocess(env_extra, code):
    env = {**os.environ, **env_extra, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))}
    import pydantic
    if getattr(pydantic, "IS_OFFLINE_SHIM", False):
        env["PYTHONPATH"] += os.pathsep + str(Path(__file__).parent / "_offline_shims")
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(ROOT))


def test_env_var_overrides_default_and_request_override_overrides_env():
    out = _cfg_subprocess({"SEMANTIC_WEIGHT": "0.5"}, "from app.config import config; print(config.scoring_weights()['semantic'], config.scoring_weights({'semantic': 0.1})['semantic'])")
    assert out.stdout.split() == ["0.5", "0.1"], out.stderr


def test_invalid_configuration_is_rejected():
    assert _cfg_subprocess({"SEMANTIC_WEIGHT": "-1"}, "import app.config").returncode != 0
    assert _cfg_subprocess({"CHUNK_OVERLAP_WORDS": "500"}, "import app.config").returncode != 0


def test_scoring_weights_helper_ignores_unknown_and_none():
    w = config.scoring_weights({"semantic": 0.9, "bogus": 1, "bm25": None})
    assert w["semantic"] == 0.9 and "bogus" not in w and w["bm25"] == config.bm25_weight


# =================================================================== API compatibility
OLD_FIELDS = {"candidate_id", "name", "rank", "final_score", "semantic_score", "bm25_score", "skill_score", "experience_score",
              "education_score", "certification_score", "matched_required_skills", "missing_required_skills",
              "matched_preferred_skills", "candidate_experience_years", "required_experience_years", "education_tier",
              "explanation", "raw_ats_score", "percentile_rank", "eligibility", "fit", "recommendation",
              "certification_level", "certification_effective_score", "skill_applicable"}
ADAPTER_KEYS = {"candidateName", "email", "phone", "currentRole", "experience", "skills", "educationDetails", "atsScore",
                "overallScore", "fit", "recommendation", "recommendationReason", "summary", "matchedSkills",
                "relevantExperience", "education", "skillGaps", "strengths", "concerns", "rawAtsScore", "eligibility",
                "eligibilityReason", "thresholds", "scoreBreakdown"}


def test_breakdown_and_response_keep_all_previous_fields():
    assert OLD_FIELDS <= set(CandidateScoreBreakdown.model_fields if hasattr(CandidateScoreBreakdown, "model_fields") else CandidateScoreBreakdown.__shim_fields__)
    names = set(SearchResponse.model_fields if hasattr(SearchResponse, "model_fields") else SearchResponse.__shim_fields__)
    assert {"job_id", "total_indexed_resumes", "hard_filtered_count", "retrieved_count", "final_count",
            "execution_time_seconds", "weights_used", "results"} <= names


def test_adapter_payload_is_backward_compatible_and_adds_explainability(tmp_path):
    from app.adapters.screening_adapter import ATSScreeningAdapter
    rk, emb = make_ranker(tmp_path)
    ad = ATSScreeningAdapter(nlp_extractor=nlp, jd_parser=JDParserService(), hybrid_ranker=rk, embedding_service=emb)
    out = ad.evaluate_candidate("Data Engineering Trainee - Databricks", JD_TXT, RES_TXT)
    assert ADAPTER_KEYS <= set(out)
    assert {"mandatoryRequirementsMet", "missingMandatorySkills", "acceptedAlternatives", "degraded"} <= set(out)
    assert out["scoreBreakdown"]["experienceApplicable"] is False and out["scoreBreakdown"]["scoringMode"] == "full"
    assert out["eligibility"] == "Eligible" and out["mandatoryRequirementsMet"] is True


def test_adapter_fails_loudly_or_degrades_explicitly(tmp_path, monkeypatch):
    from app.adapters.screening_adapter import ATSScreeningAdapter
    rk, _ = make_ranker(tmp_path, _Unavailable())
    ad = ATSScreeningAdapter(nlp_extractor=nlp, jd_parser=JDParserService(), hybrid_ranker=rk, embedding_service=_Unavailable())
    monkeypatch.setattr(config, "allow_degraded_semantic", False)
    with pytest.raises(EmbeddingUnavailableError):
        ad.evaluate_candidate("T", JD_TXT, RES_TXT)
    monkeypatch.setattr(config, "allow_degraded_semantic", True)
    out = ad.evaluate_candidate("T", JD_TXT, RES_TXT)
    assert out["degraded"] is True and out["scoreBreakdown"]["scoringMode"] == "degraded_no_semantic"


def test_bm25_cold_start_without_reference_corpus_is_batch_dependent_documented(tmp_path):
    """Documents a KNOWN LIMITATION: with no persisted reference corpus the score depends on the batch."""
    s = BM25SearchService(index_path=tmp_path / "none.pkl")
    jd = "python sql etl pipelines databricks"
    s.build_index(["T"], ["python sql etl pipelines"], persist=False)
    alone = s.search(jd)["T"]
    s.build_index(["T"] + [f"O{i}" for i in range(5)], ["python sql etl pipelines"] + ["gardening cooking"] * 5, persist=False)
    assert abs(s.search(jd)["T"] - alone) > 0.01          # cold start: seed the corpus via ingestion for stable scores
