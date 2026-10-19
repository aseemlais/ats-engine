# BEFORE / AFTER DIAGNOSTIC (SYNTHETIC fixtures)

Fixtures: `tests/fixtures/synthetic_trainee_jd.txt`, `synthetic_trainee_resume.txt` - **reconstructed from the written description; the real JD and
resume were not supplied.** "Before" = original project code run in this sandbox on the same fixtures; "after" = this release.

| Item | Before (original) | After |
|---|---|---|
| Title | `Software Engineer` (fallback) | `Data Engineering Trainee - Databricks` |
| Required skills | aws, azure, data engineering, data science, gcp, mysql, python, spark, sql (everything required; both SQL **and** MySQL) | mysql, python, sql - with `[mysql, sql]` accepted as alternatives (either satisfies) |
| Preferred / supporting | none | preferred: aws, azure, gcp, cloud computing, databricks, spark; supporting: data engineering. "Spark experience is not mandatory" is recorded in `nonmandatory_skills` and never required |
| Accepted alternatives | none | `[mysql, sql]`, `[aws, azure, gcp]` |
| Skill score (candidate) | 0.6333, missing azure/gcp/mysql | 1.0, missing none; matched via `mysql / sql (any one): satisfied by sql` |
| Experience policy | fresher scored 0.7225 with no requirement in the JD | not applicable (`experience_applicable=false`, excluded from the formula) |
| Certification policy | not run (needs the embedding package) | JD has none -> not applicable; relevance of held certificates (0.75) is informational only |
| Mandatory eligibility | n/a | Eligible; mandatory requirements met; no missing mandatory skills |
| BM25 | batch-dependent (earlier audit) | 0.3274 alone, 0.3258 in a 4-resume batch (64-doc reference corpus) |
| Semantic | not run | **could not run (model unavailable)** |

## Component scores available here (degraded run, `scripts/diagnostic_report.py`)
semantic: unavailable | bm25 0.3274 (effective weight 0.6429) | skill 1.0 (0.3571) | experience: not applicable | certification: not applicable.
Degraded score = **56.76** (BM25 + skills only). It is *not* a full-model score and must not be compared with one; the full-model
overall score for this candidate **could not be validated**. The earlier unverified "~67%" is not a target and was not used.
