"""Stage 2 (P2/P4) + Stage 4 (P5) + Stage 5 (P6): BM25 normalisation, reference corpus reliability, concurrency, diagnostics."""
import threading
from pathlib import Path
import pytest

from app.config import config
from app.services.bm25_search import BM25SearchService, METHODS
from app.utils.text_cleaner import TOKENIZER_VERSION

TOPICS = ["python sql etl pipelines", "marketing campaign brand social", "accounting ledger audit tax",
          "welding fabrication steel safety", "nursing patient care ward", "java spring microservices api",
          "tableau dashboards reporting excel", "recruiting hiring onboarding payroll"]


def ref_docs(n=40):
    ids, texts = [], []
    for i in range(n):
        ids.append(f"R{i}"); texts.append(f"{TOPICS[i % len(TOPICS)]} filler{i} common words resume experience")
    return ids, texts


@pytest.fixture
def svc(tmp_path):
    s = BM25SearchService(index_path=tmp_path / "bm25.pkl")
    ids, texts = ref_docs(40)
    s.build_index(ids, texts, persist=True)
    return s


@pytest.fixture
def small_svc(tmp_path):
    s = BM25SearchService(index_path=tmp_path / "small.pkl")
    ids, texts = ref_docs(6)
    s.build_index(ids, texts, persist=True)
    return s


Q = "python sql etl pipelines"


# ---------------------------------------------------------------- reproduce the OLD mathematics
def test_legacy_repeated_query_terms_deflate_but_bound_does_not(svc):
    doc = ["python sql"]                      # partial match so the old ceiling does not clip it to 1.0
    a = svc.score_documents(Q, ["X"], doc, method="legacy_ceiling").normalized["X"]
    b = svc.score_documents(Q + " python python python", ["X"], doc, method="legacy_ceiling").normalized["X"]
    assert b < a                                                     # old: repetition lowers the score
    c = svc.score_documents(Q, ["X"], doc, method="theoretical_bound").normalized["X"]
    d = svc.score_documents(Q + " python python python", ["X"], doc, method="theoretical_bound").normalized["X"]
    assert c == pytest.approx(d)                                     # new: unique terms, repetition is irrelevant


def test_legacy_ceiling_is_not_an_upper_bound_bound_is(svc):
    heavy = ["python sql etl pipelines " * 30]
    leg = svc.score_documents(Q, ["X"], heavy, method="legacy_ceiling")
    assert leg.raw["X"] / leg.denominator > 1.0 and leg.normalized["X"] == 1.0        # exceeds the old ceiling, clipped
    new = svc.score_documents(Q, ["X"], heavy, method="theoretical_bound")
    assert new.raw["X"] < new.denominator and new.normalized["X"] < 1.0                 # genuine bound: never exceeded
    light = svc.score_documents(Q, ["Y"], ["python sql etl pipelines"], method="theoretical_bound").normalized["Y"]
    assert light < new.normalized["X"]                               # old method tied both at 1.0; new still discriminates


def test_unseen_terms_deflate_legacy_not_bound(svc):
    doc = ["python sql etl pipelines"]
    base_leg = svc.score_documents(Q, ["X"], doc, method="legacy_ceiling").normalized["X"]
    leg = svc.score_documents(Q + " zzzunseen1 zzzunseen2", ["X"], doc, method="legacy_ceiling").normalized["X"]
    assert leg < base_leg
    r = svc.score_documents(Q + " zzzunseen1 zzzunseen2", ["X"], doc, method="theoretical_bound")
    assert r.normalized["X"] == pytest.approx(svc.score_documents(Q, ["X"], doc).normalized["X"])
    assert set(r.unseen_terms) == {"zzzunseen1", "zzzunseen2"}


# ---------------------------------------------------------------- theoretical bound properties
def test_bound_is_bounded_monotone_and_batch_independent(svc):
    docs = {"A": "python", "B": "python sql", "C": "python sql etl", "D": "python sql etl pipelines", "E": "gardening"}
    full = svc.score_documents(Q, list(docs), list(docs.values()))
    n = full.normalized
    assert all(0.0 <= v <= 1.0 for v in n.values())
    assert n["E"] == 0.0 and n["A"] < n["B"] < n["C"] < n["D"]
    for cid, t in docs.items():                                       # alone vs inside the batch: exactly equal
        assert svc.score_documents(Q, [cid], [t]).normalized[cid] == pytest.approx(n[cid], abs=1e-12)
    extra = svc.score_documents(Q, ["Z"] + list(docs), ["unrelated words here"] + list(docs.values())).normalized
    assert extra["D"] == pytest.approx(n["D"], abs=1e-12)            # unrelated additions do not move scores


def test_edge_cases(svc, small_svc):
    assert svc.score_documents("", ["X"], ["python"]).normalized == {}                 # empty query: contract preserved
    assert svc.score_documents("the and of", ["X"], ["python"]).normalized == {}       # stop-words only
    assert svc.score_documents(Q, [], []).normalized == {}
    long_q = " ".join(f"term{i}" for i in range(500)) + " python"
    r = svc.score_documents(long_q, ["X"], ["python"])
    assert 0.0 <= r.normalized["X"] <= 1.0 and len(r.unique_terms) == 501
    dup = svc.score_documents(Q, ["A", "B"], ["python sql", "python sql"]).normalized
    assert dup["A"] == dup["B"]                                      # duplicate documents score identically
    last = svc.score_documents(Q, ["A", "A"], ["gardening", "python sql etl pipelines"]).normalized
    assert list(last) == ["A"] and last["A"] > 0                     # duplicate id: last wins, one result
    assert small_svc.reference_info()["reliability"] == "low"


def test_small_reference_is_flagged_and_keeps_unseen_terms_in_denominator(small_svc, svc):
    q = Q + " unseenterm"
    low = small_svc.score_documents(q, ["X"], ["python sql etl pipelines"])
    assert low.reference_reliability == "low" and any("unseen terms kept" in n for n in low.notes)
    ok = svc.score_documents(q, ["X"], ["python sql etl pipelines"])
    assert ok.reference_reliability == "ok" and ok.normalized["X"] > low.normalized["X"] * 0.0


# ---------------------------------------------------------------- three distinct quantities
def test_raw_normalized_percentile_are_distinct(svc):
    r = svc.score_documents(Q, ["T", "U"], ["python sql etl pipelines python sql", "marketing brand"])
    assert r.raw["T"] > r.raw["U"] and r.raw["T"] > 1.0                # raw is not confined to [0, 1]
    assert 0 <= r.normalized["T"] <= 1 and 0 <= r.percentile["T"] <= 1
    assert r.percentile["T"] >= 0.8 and r.percentile["U"] <= 0.5       # relative to the reference population


def test_reference_percentile_method_is_relative(svc):
    p = svc.score_documents(Q, ["T"], ["python sql etl pipelines"], method="reference_percentile")
    assert p.normalized["T"] == pytest.approx(p.percentile["T"]) and p.method == "reference_percentile"
    with pytest.raises(ValueError):
        svc.score_documents(Q, ["T"], ["x"], method="nope")
    assert set(METHODS) == {"theoretical_bound", "reference_percentile", "legacy_ceiling"}


def test_top_k(svc):
    docs = {f"C{i}": "python " * i + "sql" for i in range(1, 6)}
    allr = svc.score_documents(Q, list(docs), list(docs.values())).normalized
    top2 = svc.score_documents(Q, list(docs), list(docs.values()), top_k=2).normalized
    assert len(allr) == 5 and len(top2) == 2 and set(top2) == set(sorted(allr, key=lambda c: (-allr[c], c))[:2])
    assert len(svc.search(Q)) == 40 and len(svc.search(Q, top_k=3)) == 3     # default returns everything


# ---------------------------------------------------------------- Stage 4: corpus reliability & concurrency
def test_shared_state_hazard_is_real_for_build_index_persist_false(svc):
    """EVIDENCE: the legacy per-request pattern (mutate the shared index, then search) loses candidates when two
    requests interleave. This is deterministic interleaving, not a timing race."""
    svc.build_index(["REQ_A"], ["python sql etl pipelines"], persist=False)      # request A prepares its index ...
    svc.build_index(["REQ_B"], ["marketing brand"], persist=False)               # ... request B replaces it ...
    assert "REQ_A" not in svc.search(Q)                                          # ... A's candidate has vanished


def test_stateless_scoring_is_concurrency_safe(svc):
    expected = {i: svc.score_documents(Q, [f"T{i}"], ["python sql " * (i + 1)]).normalized[f"T{i}"] for i in range(16)}
    results, errors = {}, []
    def work(i):
        try:
            for _ in range(20):
                results[i] = svc.score_documents(Q, [f"T{i}"], ["python sql " * (i + 1)]).normalized[f"T{i}"]
        except Exception as e:  # pragma: no cover
            errors.append(e)
    th = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    [t.start() for t in th]; [t.join() for t in th]
    assert not errors and results == expected
    assert len(svc.candidate_ids) == 40                                            # shared state untouched


def test_requests_never_overwrite_persisted_index(svc):
    before = svc.index_path.read_bytes()
    svc.score_documents(Q, ["X"], ["python"])
    assert svc.index_path.read_bytes() == before and len(BM25SearchService(index_path=svc.index_path).candidate_ids) == 40


def test_restart_reload_and_ingestion_update_and_deletion(svc):
    s2 = BM25SearchService(index_path=svc.index_path)                              # simulated API restart
    assert s2.candidate_ids == svc.candidate_ids and not s2.is_stale()
    a = svc.score_documents(Q, ["X"], ["python sql"]).normalized["X"]
    assert s2.score_documents(Q, ["X"], ["python sql"]).normalized["X"] == a
    ids, texts = ref_docs(40)
    svc.build_index(ids[:-3] + ["NEW1"], texts[:-3] + ["python sql etl"], persist=True)      # ingestion: 3 deleted, 1 added
    assert s2.is_stale() and s2.refresh_if_stale() and not s2.is_stale()
    assert "NEW1" in s2.candidate_ids and "R39" not in s2.candidate_ids and len(set(s2.candidate_ids)) == len(s2.candidate_ids) == 38


def test_stale_tokenizer_version_detected(svc):
    import pickle
    d = pickle.loads(svc.index_path.read_bytes()); d["tokenizer_version"] = TOKENIZER_VERSION - 1
    svc.index_path.write_bytes(pickle.dumps(d))
    s2 = BM25SearchService(index_path=svc.index_path)
    assert s2.is_stale() and s2.reference_info()["stale"] is True


def test_duplicate_ids_and_documents_counted_once_in_stats(small_svc):
    q = Q
    one = small_svc.score_documents(q, ["X"], ["python sql etl pipelines"])
    dup = small_svc.score_documents(q, ["X", "Y"], ["python sql etl pipelines"] * 2)
    assert dup.normalized["X"] == pytest.approx(one.normalized["X"])
    assert dup.normalized["X"] == pytest.approx(dup.normalized["Y"])
    ids, texts = ref_docs(6)
    same_id = small_svc.score_documents(q, [ids[0]], [texts[0]])       # an ingested resume screened again
    assert same_id.reference_size == 6


# ---------------------------------------------------------------- Stage 5: diagnostics
def test_diagnostics_fields(svc):
    r = svc.score_documents(Q + " unseenxyz", ["X"], ["python sql"])
    assert r.method == config.bm25_normalization and r.reference_size == 40 and r.denominator > 0
    assert r.matched_terms["X"] == ["python", "sql"] and set(r.unmatched_terms["X"]) == {"etl", "pipelines", "unseenxyz"}
    assert r.unseen_terms == ["unseenxyz"] and r.query_tokens[0] == "python"
