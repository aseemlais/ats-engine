"""BM25 query-mode tests: `structured` (baseline, default) vs `filtered` (generic boilerplate removed from free text).
Pure Python, no embedding model. Corpus statistics come from a temporary reference index, never data/bm25_index.pkl.
These tests prove MECHANICAL properties (nothing technical is dropped, generic text is ignored, tokenisation is consistent,
no duplication). They do not, and cannot, show better ranking accuracy: there are no human relevance labels."""
import tempfile
from pathlib import Path
import pytest
from app.config import config
from app.models.job import JobDescription
from app.services.bm25_query import (GENERIC_TERMS, QUERY_MODES, MIN_QUERY_TOKENS, build_bm25_query, filter_generic)
from app.services.bm25_search import BM25SearchService
from app.services.jd_parser import JDParserService
from app.utils.skill_normalizer import ALIAS_TO_CANONICAL, SKILL_TAXONOMY
from app.utils.text_cleaner import tokenize_for_bm25
from tests.fixtures.bm25_study.data import JD_MARKETING, JD_NURSE, JD_SENIOR_DE, SYN_RESUMES

FIX = Path(__file__).parent / "fixtures"
parse = lambda t: JDParserService().parse_job_description(t, job_id="T")
TRAINEE_FULL = (FIX / "jd_user_trainee_full.txt").read_text(encoding="utf-8")
ALL_JDS = {"trainee_full": TRAINEE_FULL, "trainee_short": (FIX / "jd_databricks_trainee.txt").read_text(encoding="utf-8"),
           "senior_de": JD_SENIOR_DE, "nurse": JD_NURSE, "marketing": JD_MARKETING}

PROSE_GENERIC = ("Data Engineer\nPython and SQL are required. Spark experience is not mandatory.\n"
                 "Responsibilities\n- Build ETL pipelines with Spark\n- Collaborate with stakeholders in a fast-paced environment\n"
                 "- Demonstrate strong communication and ownership\n")


def _ref(extra_docs=None):
    """Temporary reference index (>= 30 documents so statistics are 'ok') from synthetic resumes plus filler."""
    docs = {k: t for k, (_d, t) in SYN_RESUMES.items()}
    docs.update({f"FILL_{i}": f"filler{i} alpha{i % 7} beta{i % 5} gamma{i % 3} python sql data report" for i in range(24)})
    docs.update(extra_docs or {})
    svc = BM25SearchService(index_path=Path(tempfile.mkdtemp()) / "t.pkl")
    svc.build_index(list(docs), list(docs.values()), persist=True)
    return svc, docs


# ------------------------------------------------------------------ configuration / contract
def test_default_mode_is_baseline_structured_and_modes_are_registered():
    assert config.bm25_query_mode == "structured" and QUERY_MODES == ("structured", "filtered")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        build_bm25_query(parse(PROSE_GENERIC), mode="nonsense")


def test_mode_none_reads_config_and_restores():
    jd = parse(PROSE_GENERIC)
    old = config.bm25_query_mode
    try:
        config.bm25_query_mode = "filtered"
        assert build_bm25_query(jd).mode == "filtered"
        config.bm25_query_mode = "structured"
        assert build_bm25_query(jd).mode == "structured" and build_bm25_query(jd).removed_terms == []
    finally:
        config.bm25_query_mode = old


def test_structured_mode_is_byte_identical_to_the_previous_behaviour():
    """Baseline preserved: explicit structured mode equals default, and never records removed terms."""
    for name, t in ALL_JDS.items():
        jd = parse(t); a, b = build_bm25_query(jd), build_bm25_query(jd, "structured")
        assert (a.text, a.method, a.skill_terms) == (b.text, b.method, b.skill_terms) and b.removed_terms == [], name


# ------------------------------------------------------------------ generic text
def test_filter_removes_generic_words_and_keeps_technical_ones():
    txt, removed = filter_generic("Collaborate with cross-functional stakeholders on Spark and Airflow ETL pipelines", [])
    assert "spark" in txt.split() and "airflow" in txt.split() and "etl" in txt.split() and "pipelines" in txt.split()
    assert {"collaborate", "cross-functional", "stakeholders"} <= set(removed) and not (set(txt.split()) & GENERIC_TERMS)


def test_generic_only_line_filters_to_nothing():
    txt, removed = filter_generic("Passionate self-motivated team player with excellent communication", [])
    assert set(txt.split()) <= {"player"}                                               # 'player' is not a stop word by design
    assert {"passionate", "self-motivated", "team", "excellent", "communication"} <= set(removed)


def test_no_taxonomy_skill_or_alias_token_is_ever_in_the_generic_list():
    """Guard: the list can never silently delete a technical term."""
    toks = set()
    for canon, aliases in SKILL_TAXONOMY.items():
        for a in [canon] + list(aliases):
            toks.update(tokenize_for_bm25(a, query=True))
    toks.update(tokenize_for_bm25(" ".join(ALIAS_TO_CANONICAL), query=True))
    assert not (toks & GENERIC_TERMS), sorted(toks & GENERIC_TERMS)


def test_ambiguous_domain_words_are_not_in_the_generic_list():
    """Found by auditing removed terms: these have real meaning in some domain (campaign performance, continuous delivery,
    growth marketing, social worker, company law, recruiting candidates, sales opportunities, dynamic programming ...)."""
    ambiguous = {"performance", "delivery", "deliver", "growth", "environment", "worker", "company", "candidate", "candidates",
                 "opportunity", "opportunities", "dynamic", "driven", "efficient", "reliable", "competitive", "awareness",
                 "professional", "smart", "solid", "work", "working", "workplace", "career", "best", "aware"}
    assert not (ambiguous & GENERIC_TERMS), sorted(ambiguous & GENERIC_TERMS)
    jd = parse(JD_MARKETING)
    assert "performance" in build_bm25_query(jd, "filtered").tokens          # "campaign performance" must survive


def test_added_boilerplate_does_not_change_filtered_query():
    base = parse(PROSE_GENERIC)
    extra = parse(PROSE_GENERIC + "- Be a passionate, motivated team player with a positive attitude and strong leadership\n")
    a, b = build_bm25_query(base, "filtered"), build_bm25_query(extra, "filtered")
    assert a.tokens == b.tokens or set(b.tokens) <= set(a.tokens) | {"player"}
    sa, sb = build_bm25_query(base, "structured"), build_bm25_query(extra, "structured")
    assert set(sb.tokens) - set(sa.tokens)                       # the baseline IS sensitive to such additions (documents the gap)


# ------------------------------------------------------------------ technical requirements are never dropped
@pytest.mark.parametrize("name", list(ALL_JDS))
def test_all_parsed_skill_tiers_survive_filtering(name):
    jd = parse(ALL_JDS[name]); q = build_bm25_query(jd, "filtered")
    toks = set(q.tokens)
    for tier in (jd.required_skills, jd.preferred_skills, jd.supporting_skills):
        for skill in tier:
            assert set(tokenize_for_bm25(skill, query=True)) <= toks, (name, skill)


def test_alternative_skill_groups_survive_filtering():
    jd = parse(TRAINEE_FULL)
    assert ["mysql", "sql"] in jd.skill_alternatives
    toks = set(build_bm25_query(jd, "filtered").tokens)
    assert {"sql", "mysql", "python"} <= toks


def test_filtered_removes_only_listed_generic_terms():
    for name, t in ALL_JDS.items():
        jd = parse(t); s, f = build_bm25_query(jd, "structured"), build_bm25_query(jd, "filtered")
        dropped = set(s.tokens) - set(f.tokens)
        assert dropped <= GENERIC_TERMS, (name, sorted(dropped - GENERIC_TERMS))


def test_technical_responsibility_terms_are_kept():
    jd = parse(JD_SENIOR_DE); f = set(build_bm25_query(jd, "filtered").tokens)
    assert {"etl", "pipelines", "monitoring", "alerting", "optimise", "scalable", "production"} <= f


# ------------------------------------------------------------------ negation
def test_negated_skill_stays_in_both_queries_exactly_as_baseline():
    jd = parse("Data Engineer\nPython and SQL are required. Spark experience is not mandatory. Airflow is a plus.\n"
               "Responsibilities\n- Build ETL pipelines\n- Support the data platform\n")
    s, f = build_bm25_query(jd, "structured"), build_bm25_query(jd, "filtered")
    assert "spark" in jd.nonmandatory_skills and "spark" in s.tokens and "spark" in f.tokens
    assert {"python", "sql", "airflow"} <= set(f.tokens)
    assert set(s.tokens) - set(f.tokens) <= GENERIC_TERMS                              # only generic words may differ from the baseline


# ------------------------------------------------------------------ query length / fallback
@pytest.mark.parametrize("name", list(ALL_JDS))
def test_filtered_is_never_longer_than_structured(name):
    jd = parse(ALL_JDS[name])
    assert len(build_bm25_query(jd, "filtered").tokens) <= len(build_bm25_query(jd, "structured").tokens)


def test_no_responsibility_text_means_identical_queries():
    jd = parse(TRAINEE_FULL)
    assert jd.responsibilities.strip() == ""
    assert build_bm25_query(jd, "filtered").text == build_bm25_query(jd, "structured").text


def test_filter_cannot_push_a_query_below_the_minimum_silently():
    jd = JobDescription(**{**parse("Analyst\nRequirements\n- SQL\n").model_dump(), "responsibilities": "Strong team collaboration and communication"})
    q = build_bm25_query(jd, "filtered")
    if q.method == "structured":
        assert len(q.tokens) >= MIN_QUERY_TOKENS
    else:
        assert q.method == "fallback_full_jd" and q.fallback_reason


def test_fallback_path_is_filtered_too():
    raw = "We need someone who builds data pipelines in Spark.\nSalary 10 LPA, hybrid.\nStrong team player with excellent communication.\n"
    jd = JobDescription(**{**parse("x").model_dump(), "raw_text": raw, "responsibilities": "", "required_skills": [], "preferred_skills": [], "supporting_skills": []})
    s, f = build_bm25_query(jd, "structured"), build_bm25_query(jd, "filtered")
    assert s.method == f.method == "fallback_full_jd"
    assert "communication" in s.tokens and "communication" not in f.tokens and "spark" in f.tokens
    assert "salary" not in f.tokens                                                   # administrative filtering still applies


# ------------------------------------------------------------------ unrelated and generic-only resumes
def test_unrelated_resumes_score_far_below_the_on_domain_resume_in_both_modes():
    """Unrelated documents may still share an incidental word (a mechanical engineer also 'designs'), so the claim is relative."""
    svc, docs = _ref()
    jd = parse(JD_SENIOR_DE)
    ids = ["SYN_DE_SENIOR_1", "SYN_ACC_1", "SYN_MECH_1"]
    for mode in QUERY_MODES:
        r = svc.score_documents(build_bm25_query(jd, mode).text, ids, [docs[i] for i in ids])
        good = r.normalized["SYN_DE_SENIOR_1"]
        assert r.normalized["SYN_ACC_1"] < 0.1 * good and r.normalized["SYN_MECH_1"] < 0.25 * good, (mode, r.normalized)


def test_generic_only_resume_is_penalised_by_filtering_and_domain_resume_wins():
    svc, docs = _ref()
    for jd_text, good in ((JD_SENIOR_DE, "SYN_DE_SENIOR_1"), (JD_NURSE, "SYN_NURSE_1"), (JD_MARKETING, "SYN_MKT_1")):
        jd = parse(jd_text)
        ids = [good, "SYN_GENERIC_2"]; txt = [docs[i] for i in ids]
        s = svc.score_documents(build_bm25_query(jd, "structured").text, ids, txt)
        f = svc.score_documents(build_bm25_query(jd, "filtered").text, ids, txt)
        assert f.raw["SYN_GENERIC_2"] < 0.5 * s.raw["SYN_GENERIC_2"], jd_text[:20]      # generic-only evidence at least halves (bound set after seeing 0.10-0.22)
        assert f.normalized[good] > f.normalized["SYN_GENERIC_2"]                         # on-domain resume wins under filtering


def test_filtered_scores_are_not_inflated_for_the_on_domain_resume():
    """Filtering must not manufacture score: the raw evidence for a resume can only fall or stay equal."""
    svc, docs = _ref()
    for jd_text in (JD_SENIOR_DE, JD_NURSE, JD_MARKETING, TRAINEE_FULL):
        jd = parse(jd_text); ids = list(SYN_RESUMES); txt = [docs[i] for i in ids]
        s = svc.score_documents(build_bm25_query(jd, "structured").text, ids, txt)
        f = svc.score_documents(build_bm25_query(jd, "filtered").text, ids, txt)
        for i in ids:
            assert f.raw[i] <= s.raw[i] + 1e-9


def test_scores_independent_of_batch_composition_in_both_modes():
    svc, docs = _ref()
    jd = parse(JD_SENIOR_DE)
    for mode in QUERY_MODES:
        q = build_bm25_query(jd, mode).text
        alone = svc.score_documents(q, ["SYN_DE_SENIOR_1"], [docs["SYN_DE_SENIOR_1"]]).normalized["SYN_DE_SENIOR_1"]
        batch = svc.score_documents(q, list(SYN_RESUMES), [docs[i] for i in SYN_RESUMES]).normalized["SYN_DE_SENIOR_1"]
        assert alone == pytest.approx(batch)


# ------------------------------------------------------------------ tokenisation consistency / duplication
@pytest.mark.parametrize("name", list(ALL_JDS))
@pytest.mark.parametrize("mode", list(QUERY_MODES))
def test_query_text_retokenises_to_itself(name, mode):
    q = build_bm25_query(parse(ALL_JDS[name]), mode)
    toks = tokenize_for_bm25(q.text, query=True)
    assert tokenize_for_bm25(" ".join(toks), query=True) == toks


def test_compound_and_context_gated_tokens_survive_filtering():
    jd = JobDescription(**{**parse("x").model_dump(), "required_skills": ["python"], "preferred_skills": [], "supporting_skills": [],
                           "responsibilities": "Strong R programming for analysis\nMaintain CI/CD and PL/SQL jobs with excellent communication\nBuild dashboards in Power BI"})
    q = build_bm25_query(jd, "filtered"); toks = q.tokens
    assert {"r", "programming", "ci/cd", "pl/sql", "powerbi"} <= set(toks) and "sql" not in toks
    assert not ({"strong", "excellent", "communication"} & set(toks))
    assert tokenize_for_bm25(" ".join(toks), query=True) == toks


def test_query_is_tokenised_with_the_same_tokenizer_as_documents():
    q = build_bm25_query(parse(JD_SENIOR_DE), "filtered")
    doc = " ".join(q.tokens)
    assert set(tokenize_for_bm25(doc)) >= set(q.tokens) - {"r", "c"}                  # document-mode tokenizer yields the same terms


def test_repeated_query_terms_and_duplicate_candidates_do_not_inflate_or_duplicate():
    svc, docs = _ref()
    q = build_bm25_query(parse(JD_SENIOR_DE), "filtered").text
    once = svc.score_documents(q, ["SYN_DE_SENIOR_1"], [docs["SYN_DE_SENIOR_1"]])
    rep = svc.score_documents((q + "\n") * 4, ["SYN_DE_SENIOR_1"], [docs["SYN_DE_SENIOR_1"]])
    assert rep.normalized == once.normalized and rep.raw == once.raw
    dup = svc.score_documents(q, ["SYN_DE_SENIOR_1", "SYN_NURSE_1", "SYN_DE_SENIOR_1"],
                              [docs["SYN_DE_SENIOR_1"], docs["SYN_NURSE_1"], docs["SYN_DE_SENIOR_1"]])
    assert sorted(dup.normalized) == ["SYN_DE_SENIOR_1", "SYN_NURSE_1"]               # one entry per candidate
    assert dup.normalized["SYN_NURSE_1"] == pytest.approx(
        svc.score_documents(q, ["SYN_NURSE_1"], [docs["SYN_NURSE_1"]]).normalized["SYN_NURSE_1"])


def test_scoring_with_either_mode_does_not_mutate_the_reference_index():
    svc, docs = _ref()
    before = (len(svc.reference_corpus), svc.index_path.stat().st_mtime_ns)
    for mode in QUERY_MODES:
        svc.score_documents(build_bm25_query(parse(JD_NURSE), mode).text, ["X"], ["patient care icu"])
    assert (len(svc.reference_corpus), svc.index_path.stat().st_mtime_ns) == before


def test_ranker_diagnostics_expose_query_mode_and_removed_terms():
    from app.services.hybrid_ranker import HybridRankerService as HybridRanker
    q = build_bm25_query(parse(PROSE_GENERIC), "filtered")
    svc, docs = _ref()
    res = svc.score_documents(q.text, ["SYN_DE_SENIOR_1"], [docs["SYN_DE_SENIOR_1"]])
    ranker = HybridRanker.__new__(HybridRanker); ranker.bm25_service = svc
    d = ranker._bm25_diagnostics(q, res, "SYN_DE_SENIOR_1")
    assert d["query_mode"] == "filtered" and "communication" in d["generic_terms_removed"] and "query_terms" in d
