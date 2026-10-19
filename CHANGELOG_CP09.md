# Checkpoint 09 changes (base: checkpoint_08_merged_trainee_fixes.zip)
Changed: `app/utils/text_cleaner.py` (tokenizer v2; TOKENIZER_VERSION 1 -> 2), `app/services/bm25_search.py` (query-mode
tokenising; `pretokenized` flag so `search()` no longer re-tokenises stored tokens), `app/services/bm25_query.py` (query-mode),
`data/bm25_index.pkl` (rebuilt from the 64 DB resumes with tokenizer v2).
Added: `tests/test_bm25_tokenizer.py` (23 tests), `scripts/bm25_normalization_compare.py`, `scripts/evaluate_bm25_labelled.py`,
`evaluation/` (template + README), `reports/`, this file.
Unchanged on purpose: `app/config.py` (weights), hybrid ranker, score composer, API models, normalisation code.
Tokenizer v2: `ci/cd` (also `ci-cd`, `cicd`), `pl/sql`, `tcp/ip`, `power bi`, `scikit-learn`, `c sharp` become single tokens
(`pl/sql` no longer yields `sql`); `R`/`C` kept only next to a programming word; same-technology spellings merged
(k8s, postgres, cpp, csharp, sklearn, powerbi, js, nodejs...). Distinct technologies are never merged.
Any index built with tokenizer v1 must be rebuilt: `python scripts/rebuild_bm25_index.py` (the service flags it via `is_stale()`).
