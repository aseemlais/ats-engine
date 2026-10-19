# BM25 normalisation: method comparison (Checkpoint 09)

**Finding on arrival:** Checkpoint 08 already contained the three methods, the stateless scorer, the structured query and the
diagnostics (`theoretical_bound` is the default). Stage 2 was therefore *verified and compared*, not rewritten; the only real
defects found were in tokenisation (Stage 3).

Setup: reference corpus = shipped index (64 real resumes, reliability "ok"); JD = `tests/fixtures/jd_databricks_trainee.txt`;
candidate = `tests/fixtures/resume_fresher_databricks.txt` plus all 64 stored resumes. Reproduce:
`python scripts/bm25_normalization_compare.py` (full tables: `reports/BM25_NORMALIZATION_COMPARISON.md`).
These are mechanical properties, **not** accuracy: there are no human labels.

| property (fresher resume) | theoretical_bound | reference_percentile | legacy_ceiling |
|---|---|---|---|
| normalised score | 0.302 | 1.000 | 0.503 |
| best other stored resume | 0.287 | 1.000 (tie) | 0.478 |
| admin text via query builder | unchanged | unchanged | unchanged |
| admin text, builder bypassed | 0.302 -> 0.177 | 1.000 -> 0.984 | 0.503 -> 0.194 |
| query repeated x5 | unchanged | unchanged | 0.503 -> 0.347 (deflates) |
| resume keyword-stuffed x30 | 0.302 -> 0.295 (saturates) | unchanged | 0.503 -> 0.491 |
| unrelated documents | 0.000 | 0.000 | 0.000 |
| query length 3 / 6 / 10 / 24 terms | 0.22 / 0.55 / 0.67 / 0.30 | 0.52 / 0.89 / 1.0 / 1.0 | 0.37 / 0.95 / 1.0 / 0.51 |

## Choice: `theoretical_bound` (already the default; kept)
* **Mathematically consistent:** the denominator is the tf->infinity limit of the BM25+ sum, so raw <= bound always; no clipping,
  score in [0, 1], absolute and independent of which resumes are scored together.
* **Repeated query terms do not change it** (legacy deflates by 30% when the query is repeated). Keyword stuffing saturates.
* **Percentile saturates:** in a 64-resume mostly non-technical reference, any decent technical match lands at ~1.0, so it cannot
  separate a good from an excellent candidate, and its meaning changes whenever the reference corpus changes. Kept only as a
  diagnostic (`percentile_vs_reference`) and an experimental method.
* **Legacy** is not an upper bound (clipped to 1.0 at 10 terms in the table above) and is sensitive to query repetition.

## Why the strong fresher scores ~0.30 (and why it was not "fixed")
14 of 24 unique query terms matched (python, sql, databricks, spark, etl, ...); unmatched include `mysql` and generic
responsibility words (`maintain`, `simple`, `under`, `guidance`, `write`, `support`). A term present once earns about half of its
per-term bound, so ~58% coverage x ~0.5 = ~0.3. This is what the metric means ("fraction of the theoretical maximum lexical
evidence"), not a defect. Scores were not inflated and hybrid weights are untouched (`app/config.py` is byte-identical to the
supplied ZIP). **Remaining cause of low scores** is query dilution by generic verbs from responsibility lines (Stage 1 content,
not changed here): a stopword/verb filter for the *responsibility* part of the BM25 query is the next sensible step, to be judged
on labelled data.

## Limitations
No labelled data, so ranking quality is unmeasured. Query-length sensitivity is inherent to "fraction of maximum evidence":
a longer query lowers the score of a candidate who covers only part of it. The percentile method was tested on one JD only.
