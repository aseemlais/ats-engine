# TEST REPORT - checkpoint 10 addendum
`python tests/run_offline.py` (offline runner, pydantic/pytest shims) -> **233 passed, 0 failed, 4 skipped, 0 import errors** (checkpoint 09: 188/0/4).
New: `tests/test_bm25_query_modes.py` (45 tests: generic text, technical requirements, alternatives, negation, query length/fallback, unrelated and
generic-only resumes, batch independence, tokenisation idempotence, duplication, reference immutability, diagnostics keys). Mutation checks: filter disabled -> 7 fail;
skills added to stop-list -> 2 fail. Not run here (packages/network unavailable): real pytest, FastAPI routes, real embedding model, PDF parsing.
The comparison script is mechanical evidence only; no human labels exist.

---

# TEST REPORT - checkpoint 05 addendum

Same environment limits as below (offline runner with a pydantic shim; no pytest/fastapi/sentence-transformers/PyMuPDF).
`python tests/run_offline.py` -> **136 passed, 0 failed, 4 skipped, 0 import errors** (was 126/0/4).
The 10 new `test_jd_parser.py` tests (negation scope, contrast clauses, abbreviations, cert-only clause, splitter unit)
were also run against the cp04 parser: **all 10 fail there**, so they are genuine regression tests.
The 4 skips are unchanged (2 real-model embedding tests, real-model ranking, PDF). Still unverified: real embeddings,
FastAPI routes / 503 mapping, Qdrant, PDF parsing, Streamlit, Docker. Please run `python -m pytest tests -v` locally.

---

# TEST REPORT

## Environment (this sandbox)
Python 3.12.3, no network. **Not installed:** pytest, pydantic, fastapi, rank-bm25, sentence-transformers (+ model), qdrant-client, PyMuPDF.
Installed and used: numpy, pandas, scikit-learn, joblib, python-dotenv, pydantic-free stdlib. No installs/downloads were retried.

## How the tests were run
`python tests/run_offline.py` - a standard-library runner written for this session. Because pytest and pydantic are missing it used
**clearly labelled shims** (`tests/_offline_shims/pydantic.py`, `pytest.py`; the run header prints `shims in use: ['pydantic']`;
pytest itself was imported from the shim for `approx/raises/fixture/parametrize`). `tests/conftest.py` stubs `sentence_transformers`,
`qdrant_client`, `fitz` only when they cannot be imported. **Consequence:** results validate the project's scoring logic, NOT real pydantic
validation behaviour, real pytest collection, Qdrant, PDF parsing, FastAPI routing or the real embedding model.
**Please run `python -m pytest tests -v` on your machine (with the real packages) before relying on these results.**

## Result (final tree, run twice, identical)
`126 passed, 0 failed, 4 skipped, 0 import errors`

| File | Passed | Skipped | Notes |
|---|---:|---:|---|
| test_jd_parser.py | 41 | 0 | title, negation, tiers, alternatives, experience ranges (from cp02) |
| test_matching.py | 14 | 0 | skills, eligibility, experience, certification (from cp02) |
| test_scoring_fix.py | 27 | 0 | BM25 stability/persistence, NA policy, shortlist policy (cp01/cp02, still green after ranker rewrite) |
| test_pipeline.py | 38 | 0 | **new**: chunking, section mapping, composer arithmetic, ranker integration, degraded mode, eligibility, BM25 lifecycle, config precedence, API compatibility |
| test_nlp.py / test_bm25.py / test_parser.py | 3 / 1 / 2 | 0 / 0 / 1 | PDF test skipped: PyMuPDF missing |
| test_embeddings.py / test_ranking.py | 0 | 2 / 1 | **REAL-MODEL** tests skipped: `sentence_transformers` not installed |

## Mock vs real
All embedding-related tests in `test_pipeline.py` use `MockSentenceModel` (hashed bag-of-words vectors) - they check shapes, dimension
validation, chunk counts/bounds, pooling, fallbacks, caching of load failures, weights and flags. They produce **no** evidence about
semantic quality. **Blocked / not run:** real-model integration (`test_embeddings.py`, `test_ranking.py`), PDF parsing, API routes (`app.main`,
`app.server`, only syntax-checked with `ast.parse`), Streamlit UI, Docker.

## Coverage of the requested areas
JD title/negation (41) - mandatory vs preferred, SQL/MySQL, Databricks/Spark/PySpark/cloud (parser + matching + diagnostic test) -
experience N/A, explicit requirement, overlap, internship vs employment (matching + scoring_fix + pipeline) - certification policies
(matching + pipeline) - BM25 single vs batch (scoring_fix; pipeline) - index persistence/update/empty guard (pipeline) - chunking/aggregation,
missing model, vector dimensions, long/empty text (pipeline) - bounds and active-weight normalisation, eligibility vs ranking, determinism,
config precedence (subprocess), API/adapter compatibility (pipeline).
Measured, not asserted from memory: BM25 for the synthetic fixture is 0.3274 alone vs 0.3258 in a batch of four (rebuilt 64-document corpus).

## Not tested / gaps
Real semantic scores; weights vs labelled data (none exists); the previous "0.62 alone / 1.0 in batch" BM25 behaviour could not be reproduced
(rank-bm25 unavailable) - it is taken from the earlier audit. `.env` file loading is exercised only through environment variables.
