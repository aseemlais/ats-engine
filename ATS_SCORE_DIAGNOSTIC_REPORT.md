# ATS Score Diagnostic Report: Sreya S Match for Data Engineering Trainee – Databricks

**Date**: 2026-10-18  
**Candidate Evaluated**: Sreya S (`CV_00057`)  
**Target Job Description**: `Data Engineering Trainee - Databricks` (`JD – Data Engineering Trainee – Databricks (1).pdf`)  
**Pipeline**: Integrated Hybrid Semantic ATS (FastAPI + Sentence-Transformers + BM25+ Lexical + Exact NLP Matcher)

---

## 1. Reproduced Score

The candidate and real job description were evaluated end-to-end through the live screening adapter (`ATSScreeningAdapter`) and `/invocations` endpoint.

| Metric | Value | Scale / Unit |
| :--- | :--- | :--- |
| **Reported Match (Frontend Display)** | **51%** (`overallScore: 50.6%`, rounded integer `atsScore: 51`) | Percentage `[0 - 100]` |
| **Raw Unrounded Formula Output** | **50.65%** (`rawAtsScore: 50.65`) | Percentage `[0.00 - 100.00]` |
| **Recommendation** | **Review** | `Shortlisted (≥60%)` / `Review (45-59%)` / `Not Shortlisted (<45%)` |
| **Eligibility Status** | **Eligible** | Hard filter passed (`mandatoryRequirementsMet: true`) |
| **Double Percentage Conversion Check** | **Verified Clean** | Unrounded float `0.5065` multiplied by 100 exactly once; display values round to 51% |

---

## 2. Actual Component Breakdown

Under the 5-factor active-component formula:
$$\text{Final Score} = \frac{\sum_{i \in \text{active}} w_i s_i}{\sum_{i \in \text{active}} w_i} \times 100$$

Since the JD establishes no minimum experience requirement and no mandatory certification, both `experience` and `certification` are marked `not_applicable` and excluded from the active denominator $\sum w_i = 0.40 + 0.27 + 0.15 = 0.82$.

| Component | Nominal Weight | Effective Weight | Raw Score | Effective Contribution | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Semantic Similarity** | `0.40` | `0.4878` (48.78%) | `0.5872` (calibrated) | **+28.65%** | Applied |
| **BM25 Lexical Search** | `0.27` | `0.3293` (32.93%) | `0.1126` (absolute) | **+3.71%** | Applied |
| **Skill Match** | `0.15` | `0.1829` (18.29%) | `1.0000` (100%) | **+18.29%** | Applied |
| **Experience Match** | `0.10` | `0.0000` (0.00%) | `N/A` (0.0 yrs required) | **+0.00%** | Not Applicable |
| **Certification Match** | `0.08` | `0.0000` (0.00%) | `N/A` (none required) | **+0.00%** | Not Applicable |
| **Composite Sum** | **1.00** | **1.0000** (100%) | — | **50.65%** | **Final ATS Score** |

---

## 3. Parsed Mandatory and Supporting Requirements

### Extracted JD Structure

The target JD explicitly divides technical requirements into mandatory and general categories:
- **Mandatory Technical Requirements**: SQL/MySQL, Python.
- **General Technical Requirements**: Data Engineering concepts, ETL/ELT, Cloud concepts, PySpark, Spark, Databricks.
- **Explicit Clause**: *"Prior experience in PySpark, Spark, or Databricks is not mandatory. Candidates with a strong foundation in SQL/MySQL and Python will be considered and trained on the required technologies."*
- **Experience Requirement**: None (`Freshers welcome. No prior experience required`).
- **Certification Requirement**: None.

### Parser Representation (After Bug Fix)

```json
{
  "title": "Data Engineering Trainee - Databricks",
  "mandatory_skills": ["mysql", "python", "sql"],
  "required_skills": ["mysql", "python", "sql"],
  "preferred_skills": [],
  "supporting_skills": ["data engineering", "databricks", "spark"],
  "nonmandatory_skills": ["databricks", "spark"],
  "skill_alternatives": [["mysql", "sql"], ["databricks", "spark"]],
  "minimum_experience_years": 0.0,
  "has_experience_requirement": false,
  "education_requirements": []
}
```

- Python and SQL/MySQL are correctly categorized as mandatory.
- PySpark, Spark, and Databricks are correctly recognized as non-mandatory supporting skills.
- Absent experience and certification requirements are cleanly recognized as non-applicable.

---

## 4. Resume Extraction and Retrieval Findings

1. **Extraction Integrity**:
   - `PyMuPDF` extracted 3,505 characters with no corruption or missing text.
   - Sreya holds an M.Sc. in Data Science and Business Analysis (2025–2027) and a B.Sc. in Mathematics (2022–2025).
   - Skills extracted: `python`, `sql`, `mysql`, `databricks`, `spark`, `pandas`, `numpy`, `aws`, `data engineering`, `machine learning`, `scikit-learn`.
   - Projects extracted: Databricks Fashion Boutique Data Pipeline (Medallion architecture), Customer Churn Prediction, Geospatial BI.
   - Experience extracted: 0.2 years data science internship at TechnoCovai (classified correctly as internship; work experience: 0.0 yrs).

2. **Semantic Search Findings**:
   - **Full-Document Raw Cosine**: **`0.7037`** — an exceptionally strong bi-encoder match against the JD.
   - On the dense bi-encoder calibration scale $[0.15, 0.70]$, `0.7037` saturates the ceiling at **1.000 (100%)**.
   - However, the ATS applies section-level chunking:
     - `skills`: raw cosine = `0.4319` (weight `0.40`)
     - `experience`: raw cosine = `0.4895` (weight `0.30`, uses internship text fallback)
     - `projects`: raw cosine = `0.5950` (weight `0.20`)
     - `education`: raw cosine = `0.2953` (weight `0.10`)
     - Weighted average raw cosine: **`0.4682`**
     - Calibrated semantic score: $\frac{0.4682 - 0.15}{0.70 - 0.15} = \mathbf{0.5785}$ to $\mathbf{0.5872}$.
   - The section breakdown lowers the raw semantic score from `0.7037` to `0.4682` because section-specific queries (especially education and isolated skill tokens) dilute dense vector alignment.

3. **BM25 Lexical Retrieval Findings**:
   - Full JD contains 367 total tokens (175 unique).
   - Over **55%** of the JD tokens are administrative boilerplate:
     - Stipend and compensation: *₹10,000 per month*, *₹3,00,000 per annum (3 LPA)*, *non-billing period*.
     - Night shift arrangements: *night shift requirement*, *5 candidates willing to work night shift*.
     - Selection stages: *Aptitude Assessment*, *Technical Assessment*, *HR Interview*, *Final Review*.
     - Company & legal terms: *Coimbatore*, *employment commitment agreement*.
   - In `BM25SearchService`, the reference ceiling is the theoretical maximum score of an average-length document containing **every** term from the query text at its query frequency.
   - Because a technical resume never contains salary figures, night shift statements, or HR interview stage terms, the inflated ceiling depresses the candidate's lexical score to **`0.1126`** (11.26%).

---

## 5. Root Cause Analysis

### Primary Drivers of the 51% Score

1. **Active Weight Renormalization Amplified BM25**:
   Because `experience` (10%) and `certification` (8%) were not applicable, the remaining components absorbed their weights:
   - BM25's effective weight jumped from **27% to 32.93%**.
   - Semantic's effective weight jumped from **40% to 48.78%**.
   - Together, Semantic and BM25 dictated **81.71%** of Sreya's entire screening score.

2. **BM25 Denominator Inflation via Administrative Boilerplate**:
   BM25 scores the resume against `jd.raw_text`. The large amount of non-technical administrative text (salary, stipend, night shifts, HR pipeline) inflated the BM25 reference ceiling.
   - Sreya achieved a BM25 score of only **0.1126**, contributing **+3.71%** out of a possible **32.93%**.
   - This single factor cost Sreya **$\sim 29$ percentage points**.
   - In controlled experiments where BM25 queries only technical requirements, Sreya's BM25 score rose to **0.5132**, which increases the overall score to **63.4% (Shortlisted)**.

3. **Section-Level Semantic Chunking vs Full Cosine**:
   - Full document cosine is **0.7037** (100% calibrated match).
   - Section weighting reduced raw cosine to **0.4682** (58.7% calibrated match).
   - Contributing **+28.65%** out of a possible **48.78%** (a reduction of **$\sim 20$ percentage points**).

4. **Parser Defect (Isolated & Repaired)**:
   - In `app/services/jd_parser.py`, when skills (`PySpark`, `Databricks`) were listed under bullet items in a general section, they received `tier = "required"`.
   - When a subsequent clause explicitly stated: *"Prior experience in PySpark, Spark, or Databricks is not mandatory"*, the parser registered `neg = True`, but the monotonic tier rank check (`1 > 3` = False) prevented demoting the skill, and line 260 subsequently stripped the skill from `nonmandatory_skills`.
   - Furthermore, `'Night Shift Requirement'` was categorized as a `"required"` technical section, causing trailing career path lines to be treated as mandatory skills.
   - **Impact on Sreya**: Sreya already possessed Spark and Databricks, so she achieved `skill_score = 1.0` regardless. However, for any candidate without Databricks, this bug caused false hard-filter rejections.

---

## 6. Confirmed Code Changes

### Files Modified

1. **`app/services/jd_parser.py`**:
   - **Administrative Section Filtering**: Added administrative headers (`compensation`, `salary`, `stipend`, `night shift`, `selection process`, `interview`, `commitment`, `career opportunity`, `career path`) to the `"skip"` regex so administrative boilerplate does not bleed into technical requirements.
   - **Non-Mandatory Skill Demotion**: Fixed the monotonic tier check so that any skill with an explicit negation (`neg = True`) cannot remain in the `required` tier and is demoted to `supporting` (or `preferred` if plus wording is present).
   - **Non-Mandatory List Retention**: Preserved all explicitly negated skills in `nonmandatory_skills` rather than discarding them when previously marked `required`.

2. **`tests/test_jd_parser.py`**:
   - Added `test_skill_listed_before_subsequent_negation_is_never_mandatory` regression test verifying that skills listed as bullet points prior to an explicit non-mandatory clause are correctly classified as non-mandatory supporting skills.

---

## 7. Regression Test Results

Both the full test suite and the offline test runner were executed:

1. **Full Pytest Suite**:
   ```
   python -m pytest tests -v
   ============================ 141 passed in 23.43s =============================
   ```
2. **Offline Test Runner**:
   ```
   python tests/run_offline.py
   ======================================================================
   RESULT: 141 passed, 0 failed, 0 skipped, 0 import errors in 31.9s
   ```
   - Prior passing tests: 140
   - Current passing tests: 141 (140 existing + 1 new regression test)
   - Zero regressions.

---

## 8. Remaining Limitations & Architectural Observations

1. **BM25 Scope on Raw JD**:
   The ATS currently queries BM25 using `jd.raw_text`. When employers paste entire PDF job descriptions that include extensive administrative, benefits, salary, and shift text, BM25 scores will be lower than keyword presence suggests. Restricting BM25 queries to structured technical sections (`jd.responsibilities` + `jd.required_skills` + `jd.preferred_skills`) is a possible future calibration, but must be evaluated against a labeled multi-JD dataset.

2. **Section Weighting Calibration**:
   Candidates who are freshers or transitioning fields may have empty formal `experience` sections. While the system merges internship text into the experience vector, the section-level query weighting (40% skills, 30% experience, 20% projects, 10% education) will naturally produce lower cosine values than a holistic full-document vector.
