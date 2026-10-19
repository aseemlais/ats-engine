# BM25 query-dilution investigation (Checkpoint 10)

Base: `checkpoint_09_bm25_tokenisation.zip` (you called it Checkpoint 08; the archive and its CHANGELOG_CP09.md say 09, and it
already contained everything you listed: three normalisation methods with `theoretical_bound` default, tokenizer v2, index/persistence tests,
diagnostics and labelled-evaluation scripts). None of those were rewritten. Hybrid weights, API models and normalisation code are untouched.

## 1. Decision

| item | outcome |
|---|---|
| New option | `BM25_QUERY_MODE=structured` (**default, = existing baseline, byte-identical**) or `filtered` |
| What `filtered` does | removes a fixed list of 87 clearly generic words (stakeholders, fast-paced, collaborate, excellent, ...) from the **free responsibility text only**. Parsed required/preferred/supporting skills and any taxonomy mention are never removed |
| Why not default | there are **no human relevance labels**, so "better ranking" cannot be claimed. The evidence below is mechanical |
| Raw full-JD query | rejected as a production strategy (see 3.1); kept only as a comparison baseline in the script |

## 2. What the existing code does (inspected)

* `jd_parser.py` -> `JobDescription` (required / preferred / supporting / nonmandatory skills, `responsibilities` = text under a Responsibilities-type header).
* `bm25_query.build_bm25_query` -> query = all three skill tiers (canonical taxonomy names) + responsibility lines with administrative lines (salary, interview, shift, ...) dropped; falls back to admin-filtered full JD if fewer than 4 tokens.
* `bm25_search.score_documents` -> BM25+ over **unique** query terms; statistics from the reference index only (batch-independent); `theoretical_bound` normalisation. Repeated query terms count once; duplicate candidate ids collapse to one.
* Dilution mechanism: every unique query term with non-zero IDF adds `idf*(delta+k1+1)` to the denominator and can add raw evidence for any resume that happens to contain it. Generic words in responsibility text therefore (a) enlarge the denominator and (b) give generic-sounding resumes raw score without any technical evidence.

## 3. Results (from `reports/BM25_QUERY_STRATEGY_COMPARISON.md`, regenerated after the list was pruned)

Corpus, identical for every row: 64 real resumes (DB, identifiers only) + the full trainee resume + 16 **synthetic** resumes (nursing, marketing, data-engineering, accounting, mechanical, 3 "generic-only" resumes made of boilerplate). Reference index built from that same corpus in a temp file.

| JD | strategy | unique terms | generic share of denominator* | top-10 overlap vs structured (64 real resumes only) | generic-only resumes: mean percentile (higher = ranks higher) |
|---|---|---|---|---|---|
| Fresher technical (your Databricks trainee JD, real) | raw | 175 | 12.7% | 4/10 | 81.9% |
| | structured | 9 | 0% | - | 11.1% |
| | filtered | 9 (identical) | 0% | 10/10 | 11.1% |
| Experienced technical (Senior Data Engineer, synthetic JD) | raw | 78 | 23.9% | 8/10 | 97.5% |
| | structured | 46 | 29.3% | - | 97.5% |
| | filtered | 33 | 0% | 8/10 | 71.2% |
| Senior Data Analyst (repo JD) | raw | 105 | 5.7% | 8/10 | 21.8% |
| | structured | 41 | 4.9% | - | 22.2% |
| | filtered | 39 | 0% | 8/10 | 6.2% |
| Other domain: nursing (synthetic JD) | raw | 70 | 16.8% | 6/10 | 95.1% |
| | structured | 35 | 26.4% | - | 96.3% |
| | filtered | 24 | 0% | 6/10 | 72.4% |
| Other domain: marketing (synthetic JD) | raw | 61 | 24.8% | 8/10 | 97.5% |
| | structured | 33 | 29.9% | - | 97.5% |
| | filtered | 22 | 0% | 6/10 | 63.4% |

\* share of the normalisation denominator contributed by words on the generic list (IDF-weighted). The list defines this metric, so it is descriptive, not independent evidence.
Query-build time (once per request): structured 0.02-0.15 ms, filtered 0.03 ms for a JD with no responsibility text and 1.5-2.3 ms with it; end-to-end scoring of 81 documents is 46-58 ms for all three strategies, so runtime is not a deciding factor.

### 3.1 What the data supports
1. **Raw full JD is clearly worse as a lexical query.** For your trainee JD the fixture resume gets 0.115 normalised (raw JD) vs 0.638 (structured), the top-10 changes 6 of 10 names, and generic-only resumes rank in the 82nd percentile. Salary, shift, interview-process and attribute text dominate. The existing structured query is the right base.
2. **For your trainee JD, `filtered` changes nothing.** That JD has no Responsibilities section, so structured = filtered = `data engineering mysql python sql cloud computing databricks spark`. Whatever that resume's BM25 component was, this work cannot change it.
3. **Where the JD has responsibility prose, the baseline is demonstrably sensitive to generic words.** Under `structured`, three generic-only synthetic resumes rank 2nd-4th for the Senior DE JD and 3rd-5th for nursing; 82-89% of their raw score comes from the removed words. Under `filtered` that evidence drops to roughly 10-22% of baseline and the on-domain resume wins by a wide margin (test bound: at least halves, set after seeing 0.10-0.22).
4. **It also affects real resumes**: for the marketing JD, two real (technology) resumes leave the top-10 because 54% of their score came from generic words; for nursing, one (52.5%).
5. **Filtering does not inflate scores.** Raw evidence can only fall or stay equal for every document (asserted by test). Median normalised score of the real resumes fell for the nursing (0.025 -> 0.016) and marketing (0.032 -> 0.024) JDs and rose only slightly for the senior DE JD (0.041 -> 0.043); the rise is a denominator effect, not an accuracy gain.

### 3.2 What the data does NOT support
* **No ranking-accuracy claim.** AUC rows in the generated file use synthetic positives I defined; they test the mechanism, not hiring quality.
* **The first version of the stop-list was partly circular**: my synthetic generic-only resumes used the same boilerplate words I had listed, which made the filter look better (generic-only resumes at the 7-8th percentile). Auditing the removed terms showed the list also removed words with real domain meaning (`performance` in "campaign performance", `delivery`, `growth`, `worker`, `company`, `candidates`, `opportunities`, `dynamic`). I pruned those (a guard test now forbids them) and the effect shrank to the numbers above (63-72nd percentile). The generic-only resumes still rank in the upper-middle because residual words remain (`environment`, `take`, `skills`, ...): the filter is deliberately conservative and does not eliminate dilution.
* Rank order of the long tail of real resumes moves moderately (Kendall tau vs structured on the 64 real resumes: 0.75 senior DE, 0.96 analyst, 0.69 nursing, 0.64 marketing; for nursing/marketing those resumes are all irrelevant anyway). Whether any of those moves are improvements is unknown without labels.
* Synthetic JDs/resumes are small templates I wrote; real resume text is noisier.

## 4. Requirement-preservation checks (all tested)
* Every required/preferred/supporting skill token survives in the filtered query for all five JDs (`test_all_parsed_skill_tiers_survive_filtering`).
* Alternative groups (SQL/MySQL) survive; negated skills (Spark "not mandatory") stay in the query exactly as in the baseline.
* The only tokens `filtered` ever drops relative to `structured` are on the generic list (`test_filtered_removes_only_listed_generic_terms`); no taxonomy skill/alias token is on the list (`test_no_taxonomy_skill_or_alias_token_is_ever_in_the_generic_list`); ambiguous domain words are asserted absent.
* Technical responsibility terms (etl, pipelines, monitoring, alerting, scalable, production, "campaign performance") are kept.
* Tokenisation: both modes' final query text re-tokenises to itself (idempotent) for every JD; compounds (`ci/cd`, `pl/sql` without a spurious `sql`, `powerbi`) and context-gated `R` survive; the same tokenizer/version as documents is used. Duplicate candidate ids yield one result and do not change other candidates' scores; repeated query terms do not inflate; scoring never mutates the reference index.
* Mutation checks: disabling the filter makes 7 tests fail; adding skills to the stop-list makes 2 fail.

## 5. Other findings (not changed; need your decision)
1. **JD parser marks `data engineering` as mandatory for your trainee JD** (parsed required = data engineering, mysql, python, sql) although the JD says only SQL/MySQL and Python are mandatory. Cause (isolated): bare bullets under "General Technical Requirements" become *required* even after the line "basic understanding or familiarity". This affects the skill component and eligibility, not BM25. **Highest-priority follow-up.**
2. **The lexical query uses canonical skill names**, so surface forms in the JD never reach BM25: `etl`, `elt`, `pipelines`, `pyspark` are absent from your trainee query (they map to `data engineering` / `spark`), although the resume contains them. The tokenizer deliberately keeps `spark` and `pyspark` distinct, so a resume saying only "PySpark" would not match `spark`.
3. **Non-taxonomy domains lose their requirement lines**: the nursing query contains responsibility text but not "BLS", "ACLS", "RN licence" (Requirements section is not used for free text). Title extraction also returns "Untitled Role" for "Registered Nurse - ICU".
4. Your 73.1% cannot be reproduced here (needs the embedding model). Against the shipped 64-resume reference, the pasted resume gets BM25 raw 21.46 / normalised 0.658 / 100th reference percentile under `structured` (and `filtered`, identical) vs 0.128 for the raw JD. Pasted text may differ slightly from your PDF-parsed text.

## 6. Exact commands
Run from the project root (Windows PowerShell shown; same on Linux/macOS without the activate step).
```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest tests -v                       # full suite with real pytest/pydantic
python -m pytest tests/test_bm25_query_modes.py -v
python tests\run_offline.py                     # offline runner (no packages needed); sandbox result: 233 passed, 0 failed, 4 skipped
python scripts\bm25_query_strategy_compare.py   # regenerates reports\BM25_QUERY_STRATEGY_COMPARISON.md
$env:BM25_QUERY_MODE = "filtered"; uvicorn app.main:app --port 8000     # opt in for a session (default: structured)
python scripts\evaluate_bm25_labelled.py evaluation\labelled_pairs.csv --query-mode both   # needs >=30 human-labelled pairs
```
No rebuild of `data/bm25_index.pkl` is needed: the index stores resume tokens only and query mode does not affect it.

## 7. Before enabling `filtered` in production
Label at least 30 JD/resume pairs across several JDs (see `evaluation/README.md`), run the last command above, and enable `filtered` only if it is not worse than `structured` on held-out JDs.
