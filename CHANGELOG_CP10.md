# Checkpoint 10 changes (base: checkpoint_09_bm25_tokenisation.zip)
Changed: `app/services/bm25_query.py` (+`filtered` mode, `GENERIC_TERMS`, `filter_generic`, `BM25Query.mode/removed_terms`; `build_bm25_query(jd, mode=None)`),
`app/config.py` (+`bm25_query_mode`, env `BM25_QUERY_MODE`, default `structured`, validated), `.env.example` (documented option),
`app/services/hybrid_ranker.py` (2 additive keys in `bm25_diagnostics`: `query_mode`, `generic_terms_removed`),
`scripts/evaluate_bm25_labelled.py` (`--query-mode both|structured|filtered`; still refuses to run without labels).
Added: `tests/test_bm25_query_modes.py` (45 tests), `tests/fixtures/bm25_study/` (synthetic JDs/resumes), `tests/fixtures/jd_user_trainee_full.txt`,
`tests/fixtures/resume_user_trainee_full.txt`, `scripts/bm25_query_strategy_compare.py`, `reports/BM25_QUERY_STRATEGY_COMPARISON.md` (generated),
`reports/BM25_QUERY_STRATEGY_REPORT.md`, this file.
Unchanged on purpose: hybrid weights (0.40/0.27/0.15/0.10/0.08 defaults), score composer, API models/contracts, normalisation code, tokenizer v2,
`data/bm25_index.pkl` (byte-identical), default query behaviour (`structured` == previous behaviour, asserted by test).
Tests: `python tests/run_offline.py` -> 233 passed, 0 failed, 4 skipped (same 4 as before: real embedding model x3, PDF). Was 188 at checkpoint 09.
