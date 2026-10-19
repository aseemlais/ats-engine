# CHANGELOG (checkpoint 05 addendum - negation scope fix)

Base = checkpoint_04_tested_project (unchanged reference). One defect fixed, found while verifying cp04.

## Defect: negation / hard / preferred wording leaked across sentences in `app/services/jd_parser.py`
Every classification flag (negation, "required", "preferred", trainable, certification-only) was evaluated on the whole
LINE. In prose JDs (several sentences on one line) `"Python and SQL are required. Spark experience is not mandatory."`
produced `required=[]`, `nonmandatory=[python, spark, sql]` - mandatory skills were lost, so mandatory enforcement and
the default hard filter silently did nothing. Bulleted JDs (the only form in the test fixture) were unaffected.
Same root cause: `"Python required. AWS certification is a plus."` skipped the whole line as certification-only.

Fix: new `_clauses()` splits each line into clauses (sentence ends, `;`, and but/however/although/though/whereas only
when BOTH sides name a skill, so "X is not mandatory but a plus" is not split; abbreviations such as e.g./i.e./etc. never
split). Classification, negation, tiers, cert-only skip and alternative-group detection now run per clause. Experience,
education and responsibilities extraction are unchanged (still line level).

Files: `app/services/jd_parser.py` (modified), `tests/test_jd_parser.py` (+10 tests).
No other file changed. Default weights, config, BM25, embeddings: untouched.

---

# CHANGELOG

Baseline = the ORIGINAL project (uploaded as `1792304031111__pytest_cache.zip`, which despite its name is the whole
original project tree incl. `__pycache__`). Starting point = `checkpoint_02_parser_and_matching.zip`, which was verified
to be a strict superset of `checkpoint_01_recovered_source.zip` (same project version; 12 files differ, 5 files added, none lost)
and of the original (checkpoint 1 = original + the first session's BM25 / ranker / shortlist-policy / config changes).

## Recovery findings
| Reported change | Present in cp01 | Present in cp02 | Status |
|---|---|---|---|
| BM25 absolute scale, no batch-max, `persist=False` for requests | yes | yes | verified (tests) |
| NA_POLICY / BEST_AVAILABLE_K / shortlist policy | yes | yes | verified |
| JD parser rewrite (title, negation, tiers, alternatives, experience range) | no | yes | verified (41 parser tests) |
| Skill taxonomy: Databricks, cloud, DB groups, SQL/MySQL | no | yes | verified |
| Experience matcher: absent requirement => non-applicable | no | yes (matcher only) | **ranker ignored it** -> fixed here |
| Certification matcher: capped, relevance-based | no | yes | verified |
| Semantic section mapping + chunking | no | no | **unrecovered -> implemented here** |
| Weighted formula update | no | no | **not implemented upstream -> done here (defaults deliberately unchanged)** |

## Changes made in this session (original -> now: only what this session added on top of cp02)
1. **Ranker integration** (`hybrid_ranker.py`, new `score_composer.py`): experience N/A was computed by the matcher but the
   ranker still weighted it (as 0.0). Now a single documented formula renormalises over active components and reports per-component
   status/weight/contribution. Equivalent to the previous neutral-imputation formula whenever weights sum to 1.
2. **Eligibility & explainability** (`hard_filter.py`, `candidate.py`, `explainability.py`, adapter): `assess()` reports unmet mandatory
   skills even in lenient mode; results expose `mandatory_requirements_met`, `missing_mandatory_skills`, `eligibility_reasons`,
   `accepted_alternatives`, `components`, `experience_applicable`, `degraded`, `scoring_mode`. All old fields retained.
3. **Semantic section mapping + chunking** (`embedding_service.py`, new `text_chunking.py`): internship text feeds the experience
   section; long text is chunked (180 words, 30 overlap, max 8 chunks, length-weighted mean pooling, truncation flagged);
   JD queries no longer say "0.0 years experience" and include supporting (trainable) skills; dimension validation.
4. **Missing-model behaviour**: `EmbeddingUnavailableError`; load failure attempted once then fails fast (no repeated downloads);
   default = fail loudly (HTTP 503); `ALLOW_DEGRADED_SEMANTIC=true` = explicit, flagged, deterministic degraded scoring.
   Previously a missing vector silently scored semantic = 0.0.
5. **Config** (`config.py`, `.env.example`): `.env` now actually loaded (it never was, although documented); precedence documented;
   weights/threshold/chunk/cosine validation; `scoring_weights()` helper; request overrides include certification default.
   **Default weights unchanged** - see "Weights" below.
6. **BM25 data repair**: shipped `data/bm25_index.pkl` held ONE document (`CAND_SCREEN`) = the reported overwrite defect persisted in data.
   Rebuilt from `data/ats_metadata.db` (64 resumes) via new `scripts/rebuild_bm25_index.py`. Removed unused `rank-bm25` from requirements.
7. **Tests/tooling**: `tests/test_pipeline.py` (38 tests), `tests/run_offline.py` + labelled shims, conftest import stubs, real-model/PDF tests
   now *skip with a reason* instead of erroring, `test_upload_endpoint.py` (a manual script with a hard-coded `C:\Users\...` path that
   crashed collection) is now import-safe; `scripts/diagnostic_report.py`.

## Weights (decision)
Proposed 0.35/0.20/0.27/0.10/0.08 was **not adopted**: no labelled data exists to validate it, and BM25 is now on an absolute (not
batch-max) scale, which removes the main motivation (score compression/inflation). Defaults stay 0.40/0.27/0.15/0.10/0.08; the proposal is
documented as an opt-in profile in `.env.example`. `NA_POLICY=neutral_imputation` == renormalising active weights; `legacy` restores old
certification/skill behaviour; `BEST_AVAILABLE_K` only labels the top-K eligible as "Best Available Match" when nobody reaches the shortlist threshold.

## Files
Modified vs cp02: `.env.example, requirements.txt, data/bm25_index.pkl, app/{config.py, server.py}, app/api/search_routes.py,
app/adapters/screening_adapter.py, app/models/candidate.py, app/services/{embedding_service, explainability, hard_filter, hybrid_ranker,
semantic_search}.py, tests/{conftest, test_embeddings, test_parser, test_ranking, test_upload_endpoint}.py, README.md`.
Added: `app/services/{score_composer, text_chunking}.py, scripts/{diagnostic_report, rebuild_bm25_index}.py, tests/{_helpers, run_offline, test_pipeline}.py,
tests/_offline_shims/{pydantic,pytest}.py, CHANGELOG.md, TEST_REPORT.md, DIAGNOSTIC_REPORT.md`. Deleted: none (original-only dirs `data/benchmarks`,
`data/temp_uploads` were empty directories absent from the checkpoints).
Modified by cp01/cp02 relative to the original (carried forward, verified): `bm25_search, hybrid_ranker, config, screening_adapter, candidate/job models,
jd_parser, skill_matcher, skill_normalizer, experience_matcher, certification_matcher, education_matcher (any listed degree accepted), hard_filter,
nlp_extractor (alias expansion), shortlist_policy (new), requirements (new)`.

## Known limitations
* Real-model semantic quality is **unvalidated** (model unavailable here); the cosine calibration (floor 0.15 / ceiling 0.70) is a heuristic.
* BM25 scale is conservative (reference ceiling = every JD term present); typical resumes score low. Without a persisted reference corpus
  (fresh install) a lone resume's BM25 differs from its in-batch score; run ingestion or `rebuild_bm25_index.py` first.
* Default lenient hard filter rejects only candidates with NONE of the mandatory skills; others are flagged (`mandatory_requirements_met=false`),
  not rejected. Set `STRICT_HARD_FILTER=true` to reject on any miss.
* Qdrant `full` vectors stored before this change came from truncated text; re-ingest to refresh (ranking recomputes section vectors per request).
* FastAPI routes/Streamlit were only syntax-checked here (packages unavailable); the 503 mapping is untested.
