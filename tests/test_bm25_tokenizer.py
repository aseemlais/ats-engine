"""Stage 3: tokenisation regression + Stage 4 corpus reliability checks that need the tokenizer or the shipped data."""
import sqlite3, threading
from pathlib import Path
import pytest

from app.config import config
from app.services.bm25_search import BM25SearchService
from app.utils.text_cleaner import TOKENIZER_VERSION, tokenize_for_bm25 as tok

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- compound / punctuated skills
def test_ci_cd_is_one_token_in_all_spellings():
    for s in ("CI/CD", "ci / cd", "CI-CD", "CICD"):
        assert tok(f"experience with {s} pipelines") == ["experience", "ci/cd", "pipelines"], s
    assert "ci" not in tok("CI/CD") and "cd" not in tok("CI/CD")


def test_plsql_is_not_sql_and_tsql_is_distinct():
    assert tok("PL/SQL") == ["pl/sql"]
    assert "sql" not in tok("PL/SQL developer")                 # related, NOT identical
    assert tok("T-SQL") == ["t-sql"] and "sql" not in tok("T-SQL")
    assert tok("SQL") == ["sql"]


def test_other_slash_compounds():
    assert tok("TCP/IP") == ["tcp/ip"]
    assert tok("ETL/ELT") == ["etl", "elt"]                      # two separate technologies/practices stay separate


def test_existing_special_tokens_unchanged():
    assert tok("C++ C# .NET Node.js") == ["c++", "c#", ".net", "node.js"]
    assert tok("scikit-learn pre-trained end-to-end") == ["scikit-learn", "pre-trained", "end-to-end"]


# ---------------------------------------------------------------- single-letter languages
def test_r_and_c_kept_in_programming_context():
    assert "r" in tok("Skills: Python, R, SQL")
    assert "r" in tok("the R language and R programming")
    assert "c" in tok("Languages: C, Java")
    assert "c" in tok("ANSI C and C programming")
    assert "r" in tok("python, r, sql")                          # lower-case needs two tech neighbours in documents


def test_r_and_c_not_created_from_noise():
    for noise in ("Plan C", "Vitamin C", "R&D team", "option B/C", "grade a student", "Section C of the policy", "R. Sharma"):
        t = tok(noise)
        assert "r" not in t and "c" not in t and "a" not in t, (noise, t)


def test_query_mode_keeps_r_from_skill_lists_but_documents_stay_strict():
    assert tok("python r sql", query=True) == ["python", "r", "sql"]
    assert tok("r sql python", query=True)[0] == "r"
    assert tok("R", query=True) == ["r"]
    assert "r" not in tok("python r team")                       # lower-case, one tech neighbour, document mode


# ---------------------------------------------------------------- aliases and distinct technologies
def test_same_technology_spellings_are_merged():
    assert tok("k8s") == tok("Kubernetes") == ["kubernetes"]
    assert tok("Postgres") == tok("PostgreSQL") == ["postgresql"]
    assert tok("Power BI") == tok("PowerBI") == ["powerbi"]
    assert tok("sklearn") == tok("scikit learn") == tok("scikit-learn") == ["scikit-learn"]
    assert tok("NodeJS") == tok("node.js") == ["node.js"]
    assert tok("C sharp") == tok("csharp") == tok("C#") == ["c#"]
    assert tok("cpp") == tok("C++") == ["c++"]


@pytest.mark.parametrize("a,b", [("java", "javascript"), ("spark", "pyspark"), ("sql", "mysql"), ("mysql", "postgresql"),
                                 ("sql", "pl/sql"), ("c", "c++"), ("c++", "c#"), ("react", "react.js"), ("sql", "t-sql")])
def test_distinct_technologies_are_never_merged(a, b):
    ctx = {"c": "Languages: C, Java"}.get(a, a)
    assert tok(ctx, query=True)[-1 if a != "c" else 1] != tok(b, query=True)[0]


def test_false_match_is_gone_end_to_end(tmp_path):
    svc = BM25SearchService(index_path=tmp_path / "i.pkl")
    ids = [f"R{i}" for i in range(40)]
    # the reference must CONTAIN the query terms, otherwise they are "unseen" and every score is trivially 0
    spice = ["SQL reporting", "Java backend", "CI/CD pipelines", "Python scripts", "JavaScript frontend", "Skills: Python, R, SQL", "PL/SQL packages", "marketing"]
    texts = [f"filler{i} {spice[i % len(spice)]} generic resume words" for i in range(40)]
    svc.build_index(ids, texts)
    docs = {"PLSQL": "Oracle PL/SQL developer packages triggers", "SQL": "SQL analyst reporting queries",
            "R": "Statistician: Python, R, SQL", "NOISE": "Plan C and Vitamin C R&D", "JS": "JavaScript developer"}
    res = svc.score_documents("sql", list(docs), list(docs.values()), method="theoretical_bound")
    assert res.raw["SQL"] > 0 and res.raw["R"] > 0 and res.raw["PLSQL"] == 0
    r2 = svc.score_documents("python r sql", list(docs), list(docs.values()))
    assert "r" in r2.matched_terms["R"] and "r" not in r2.matched_terms["NOISE"]
    assert svc.score_documents("java", list(docs), list(docs.values())).raw["JS"] == 0
    cicd = svc.score_documents("ci/cd", ["A", "B"], ["we use CI/CD daily", "continuous cd player ci"])
    assert cicd.unseen_terms == [] and cicd.raw["A"] > 0 and cicd.raw["B"] == 0
    assert svc.score_documents("sql", list(docs), list(docs.values())).unseen_terms == []


def test_search_does_not_retokenise_stored_tokens(tmp_path):
    svc = BM25SearchService(index_path=tmp_path / "i.pkl")
    svc.build_index([f"F{i}" for i in range(35)] + ["T"], [f"filler{i} words" for i in range(35)] + ["Skills: Python, R, SQL"])
    assert svc.reference_corpus["T"].count("r") == 1
    assert "r" in svc.corpus_tokens[-1]
    assert svc.search("r", method="theoretical_bound")["T"] > 0


# ---------------------------------------------------------------- Stage 4: shipped data / persistence / concurrency
def test_shipped_index_matches_metadata_db_and_current_tokenizer():
    db, idx = ROOT / "data" / "ats_metadata.db", ROOT / "data" / "bm25_index.pkl"
    if not (db.exists() and idx.exists()):
        pytest.skip("shipped data/ files not present")
    ids = [r[0] for r in sqlite3.connect(db).execute("select candidate_id from resumes")]
    svc = BM25SearchService(index_path=idx)
    assert len(ids) == len(set(ids)), "duplicate candidate ids in the metadata DB"
    assert len(svc.candidate_ids) == len(set(svc.candidate_ids)), "a resume is counted twice in the index"
    assert set(ids) == set(svc.candidate_ids)
    assert svc.tokenizer_version_loaded == TOKENIZER_VERSION and not svc.is_stale()


def test_persist_reload_roundtrip_and_double_build_is_idempotent(tmp_path):
    p = tmp_path / "i.pkl"; a = BM25SearchService(index_path=p)
    ids, texts = [f"D{i}" for i in range(35)], [f"python doc{i} etl" for i in range(35)]
    a.build_index(ids, texts); a.build_index(ids, texts)         # same ids twice: still 35, not 70
    assert len(a.candidate_ids) == 35
    b = BM25SearchService(index_path=p)
    assert b.candidate_ids == a.candidate_ids and b.corpus_tokens == a.corpus_tokens
    assert b.score_documents("python etl", ["D0"], ["python etl"]).raw == a.score_documents("python etl", ["D0"], ["python etl"]).raw


def test_top_k_exact_counts_and_order(tmp_path):
    svc = BM25SearchService(index_path=tmp_path / "i.pkl")
    ids = [f"D{i}" for i in range(35)]
    svc.build_index(ids, [("python " * (i % 5 + 1)) + f"doc{i}" for i in range(35)])
    full = svc.search("python", method="theoretical_bound")
    assert len(full) == 35
    for k in (0, 1, 5, 35, 100):
        got = svc.search("python", top_k=k, method="theoretical_bound")
        assert len(got) == min(k, 35)
    best = svc.search("python", top_k=3, method="theoretical_bound")
    assert set(best) == set(sorted(full, key=lambda c: (-full[c], c))[:3])


def test_concurrent_scoring_while_index_is_rewritten(tmp_path):
    p = tmp_path / "i.pkl"; svc = BM25SearchService(index_path=p)
    ids, texts = [f"D{i}" for i in range(40)], [f"python sql doc{i}" for i in range(40)]
    svc.build_index(ids, texts)
    expected = svc.score_documents("python sql", ["X"], ["python sql"]).raw["X"]
    errors, results = [], []

    def scorer():
        try:
            for _ in range(30):
                results.append(svc.score_documents("python sql", ["X"], ["python sql"]).raw["X"])
        except Exception as e:                                   # pragma: no cover - failure path
            errors.append(e)

    def writer():
        try:
            for _ in range(10):
                svc.build_index(ids, texts, persist=True); svc.refresh_if_stale()
        except Exception as e:                                   # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=scorer) for _ in range(4)] + [threading.Thread(target=writer)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors and len(results) == 120
    assert all(abs(r - expected) < 1e-9 for r in results)        # identical corpus rewritten => identical scores
