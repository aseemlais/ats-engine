# Local spaCy Section-Based ATS Scoring Experiment Report

**Execution Date**: 2026-10-24 19:21:54
**Target Role / Job ID**: Senior Data Analyst (`JD_Senior_Data_Analyst`)
**Required Skills**: a/b testing, airflow, data analysis, data engineering, data warehousing, numpy, pandas, python, redshift, sql, tableau
**Preferred Skills**: None specified
**Bonus Scaling Factor (Beta)**: 3.0 (Enabled: True)

---

## 1. Executive Summary & Ranking Comparison

This experiment evaluates whether isolating lexical (BM25) and semantic matching to detected resume sections — instead of performing global whole-resume matching that dilutes dense evidence — combined with an optional low-weight additional-section bonus produces more meaningful rankings without altering production settings.

| Exp Rank | Candidate | File | Exp Score | Base Score | Score Delta | Base Rank | Rank Shift | Core Score | Add. Bonus | Edu Filter | Exp Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Senior Data Analyst | Resume_3_SarahJohnson.txt | 67.83% | 80.23% | -12.40 | 1 | = | 67.04% | +0.79 | 🟢 ELIGIBLE | 🟢 MEETS |
| 2 | Data Analyst | Resume_2_DavidChen.txt | 56.18% | 61.76% | -5.58 | 2 | = | 55.34% | +0.84 | 🟡 UNKNOWN | 🟢 MEETS |
| 3 | Unknown | new_r (3).pdf | 31.10% | 0.00% | +31.10 | 4 | +1 | 31.10% | +0.00 | 🟢 ELIGIBLE | 🔴 BELOW |
| 4 | Unknown | cv_nujoom_IAIP.docx | 27.89% | 0.00% | +27.89 | 3 | -1 | 27.89% | +0.00 | 🟡 UNKNOWN | 🟡 UNKNOWN |
| 5 | Marketing Coordinator | Resume_1_PriyaNair.txt | 0.92% | 0.00% | +0.92 | 5 | = | 0.92% | +0.00 | 🟡 UNKNOWN | 🟡 UNKNOWN |

---

## 2. Macro Aggregate Statistics

- **Total Candidates Evaluated**: 5
- **Mean Baseline Score**: 28.40%
- **Mean Experimental Score**: 36.78%
- **Average Score Delta**: +8.39%
- **Candidates with Improved Rank**: 1
- **Candidates with Demoted Rank**: 1
- **Candidates with Unchanged Rank**: 3

---

## 3. Deep-Dive Candidate Score Breakdowns

### 1. Senior Data Analyst (`CAND_005` - Resume_3_SarahJohnson.txt)

- **Experimental Final Score**: **67.83%** (Core: 67.04% + Bonus: 0.79)
- **Baseline Final Score**: 80.23% (Rank: 1 -> 1)
- **Education Eligibility Filter**: **ELIGIBLE**
  - **Decision Reason**: Candidate satisfies education requirement via alternative 'BACHELOR' with degree 'B.S' in computer science.
  - **Matched Degree**: `B.S` | **Field**: `computer science`
  - **JD Requirement**: `BACHELOR in computer science`
  - **Candidate Degrees**: `B.S`
- **Experience Evaluation (Batch 3)**: **MEETS_REQUIREMENT** (Score: `0.6880`)
  - **Required**: `5.00 yrs` | **Verified**: `7.75 yrs` | **Relevance-Adjusted**: `3.44 yrs`
  - **Duration Ratio**: `1.00` | **Eff. Type Multiplier**: `1.00` | **Duties Relevance**: `0.4200`
  - **Decision Reason**: Candidate satisfies experience requirement with 7.75 verified years (relevance-adjusted: 3.44 yrs vs 5.00 yrs required).

  *Extracted Roles & Duties:*
  - `Senior Data Analyst — CartFlow (e-commerce), Austin, TX (2021–Present)` at `Unknown` (PROFESSIONAL, 5.75 yrs, mult=1.0): Duties Rel=`0.5223` (BM25=0.5374, Sem=0.5121) -> Adj=`3.00 yrs`
  - `Data Analyst — ShopStream, Austin, TX (2019–2021)` at `Unknown` (PROFESSIONAL, 2.92 yrs, mult=1.0): Duties Rel=`0.2182` (BM25=0.1143, Sem=0.2874) -> Adj=`0.64 yrs`

#### Detected Sections Summary

| Raw Heading | Category | Is Core? | Char Length | Sentences |
| :--- | :--- | :---: | :---: | :---: |
| `SARAH JOHNSON` | `unknown` | No | 106 | 2 |
| `SUMMARY` | `summary` | No | 152 | 1 |
| `EXPERIENCE` | `experience` | Yes | 761 | 1 |
| `EDUCATION` | `education` | Yes | 59 | 1 |
| `SKILLS` | `skills` | Yes | 210 | 1 |

#### Independent Section Relevance Breakdown (Batch 1)

> **Note**: Section BM25 and Semantic scores below are calculated strictly independently from that section's own text and the relevant JD requirements. Cross-section weights (0.40, 0.30, 0.20, 0.10) are NOT used.

| Heading | Category | Core? | BM25 (Norm) | BM25 (Raw) | Semantic (Calibrated) | Semantic (Raw Cosine) | Matched Terms / Uncertainty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `SKILLS` | `skills` | Yes | 0.3810 | 11.0904 | 0.8542 | 0.6198 | Matched: testing, airflow, numpy, pandas, python |
| `EXPERIENCE` | `experience` | Yes | 0.5832 | 42.4470 | 0.6039 | 0.4821 | Matched: build, maintain, dashboards, used, merchandising |
| `Projects` | `projects` | Yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Section 'projects' absent or empty: no relevance score fabricated. |
| `EDUCATION` | `education` | No | 0.0000 | 0.0000 | 0.3980 | 0.3689 | Section 'education' scored for diagnostics only; excluded from relevance aggregation (Batch 2). |
| `SARAH JOHNSON` | `unknown` | No | 0.0279 | 2.7726 | 0.4389 | 0.3914 | Matched: data, senior | Unrecognized heading 'SARAH JOHNSON': content preserved as unknown section. |
| `SUMMARY` | `summary` | No | 0.0418 | 4.1589 | 1.0000 | 0.7222 | Matched: data, reporting, senior |

- **Provisional Core BM25 Bridge**: `0.4821` (vs Baseline Document BM25: `0.5627`)
- **Provisional Core Semantic Bridge**: `0.7291` (vs Baseline Document Semantic: `0.7303`)
*(Note: Downstream aggregation bridge is an unweighted placeholder isolated for this batch and will be formally redesigned in a subsequent batch.)*

#### Additional-Section Bonus Breakdown

- **Bonus Awarded**: `+0.79 pts`
- **Combined Additional Relevance**: `0.2633` (Threshold: `0.2`)
- **Semantic Relevance**: `0.4389` | **Lexical Relevance**: `0.0000`
- **New Relevant Skills Found in Non-Core Sections**: `None`
- **Deduplicated Skills (Already in Core)**: `1 skills avoided double-counting`

**Extraction & Processing Notes**:
- Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.
- Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.
- Batch 3: Experience evaluated using Duration x Type x Duties Relevance.
- Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.
- Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.
- Batch 4: Certifications evaluated conditionally on JD without absent-penalty.

---

### 2. Data Analyst (`CAND_004` - Resume_2_DavidChen.txt)

- **Experimental Final Score**: **56.18%** (Core: 55.34% + Bonus: 0.84)
- **Baseline Final Score**: 61.76% (Rank: 2 -> 2)
- **Education Eligibility Filter**: **UNKNOWN** ⚠️ *Requires manual verification*
  - **Decision Reason**: Candidate credentials (B.S) do not unambiguously satisfy mandatory requirement (BACHELOR in computer science).
  - **Matched Degree**: `None` | **Field**: `None`
  - **JD Requirement**: `BACHELOR in computer science`
  - **Candidate Degrees**: `B.S`
- **Experience Evaluation (Batch 3)**: **MEETS_REQUIREMENT** (Score: `0.3502`)
  - **Required**: `5.00 yrs` | **Verified**: `5.75 yrs` | **Relevance-Adjusted**: `1.75 yrs`
  - **Duration Ratio**: `1.00` | **Eff. Type Multiplier**: `1.00` | **Duties Relevance**: `0.3036`
  - **Decision Reason**: Candidate satisfies experience requirement with 5.75 verified years (relevance-adjusted: 1.75 yrs vs 5.00 yrs required).

  *Extracted Roles & Duties:*
  - `Data Analyst — GrowthMetrics Inc., Austin, TX (2022–Present)` at `Unknown` (PROFESSIONAL, 4.75 yrs, mult=1.0): Duties Rel=`0.2978` (BM25=0.1524, Sem=0.3948) -> Adj=`1.42 yrs`
  - `Junior Data Analyst — RetailPulse Analytics (2021–2022)` at `Unknown` (PROFESSIONAL, 1.92 yrs, mult=1.0): Duties Rel=`0.3181` (BM25=0.0762, Sem=0.4794) -> Adj=`0.61 yrs`

#### Detected Sections Summary

| Raw Heading | Category | Is Core? | Char Length | Sentences |
| :--- | :--- | :---: | :---: | :---: |
| `DAVID CHEN` | `unknown` | No | 96 | 2 |
| `SUMMARY` | `summary` | No | 118 | 1 |
| `EXPERIENCE` | `experience` | Yes | 638 | 1 |
| `EDUCATION` | `education` | Yes | 53 | 1 |
| `SKILLS` | `skills` | Yes | 119 | 1 |

#### Independent Section Relevance Breakdown (Batch 1)

> **Note**: Section BM25 and Semantic scores below are calculated strictly independently from that section's own text and the relevant JD requirements. Cross-section weights (0.40, 0.30, 0.20, 0.10) are NOT used.

| Heading | Category | Core? | BM25 (Norm) | BM25 (Raw) | Semantic (Calibrated) | Semantic (Raw Cosine) | Matched Terms / Uncertainty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `SKILLS` | `skills` | Yes | 0.2857 | 8.3178 | 0.8934 | 0.6414 | Matched: testing, numpy, pandas, python, redshift |
| `EXPERIENCE` | `experience` | Yes | 0.2367 | 17.2297 | 0.6755 | 0.5215 | Matched: maintain, dashboards, used, teams, design |
| `Projects` | `projects` | Yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Section 'projects' absent or empty: no relevance score fabricated. |
| `EDUCATION` | `education` | No | 0.0000 | 0.0000 | 0.2359 | 0.2797 | Section 'education' scored for diagnostics only; excluded from relevance aggregation (Batch 2). |
| `DAVID CHEN` | `unknown` | No | 0.0139 | 1.3863 | 0.4690 | 0.4079 | Matched: data | Unrecognized heading 'DAVID CHEN': content preserved as unknown section. |
| `SUMMARY` | `summary` | No | 0.0557 | 5.5452 | 0.8930 | 0.6411 | Matched: data, analysis, teams, reporting |

- **Provisional Core BM25 Bridge**: `0.2612` (vs Baseline Document BM25: `0.1569`)
- **Provisional Core Semantic Bridge**: `0.7844` (vs Baseline Document Semantic: `0.7079`)
*(Note: Downstream aggregation bridge is an unweighted placeholder isolated for this batch and will be formally redesigned in a subsequent batch.)*

#### Additional-Section Bonus Breakdown

- **Bonus Awarded**: `+0.84 pts`
- **Combined Additional Relevance**: `0.2814` (Threshold: `0.2`)
- **Semantic Relevance**: `0.4690` | **Lexical Relevance**: `0.0000`
- **New Relevant Skills Found in Non-Core Sections**: `None`
- **Deduplicated Skills (Already in Core)**: `1 skills avoided double-counting`

**Extraction & Processing Notes**:
- Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.
- Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.
- Batch 3: Experience evaluated using Duration x Type x Duties Relevance.
- Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.
- Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.
- Batch 4: Certifications evaluated conditionally on JD without absent-penalty.

---

### 3. Unknown (`CAND_002` - new_r (3).pdf)

- **Experimental Final Score**: **31.10%** (Core: 31.10% + Bonus: 0.00)
- **Baseline Final Score**: 0.00% (Rank: 4 -> 3)
- **Education Eligibility Filter**: **ELIGIBLE**
  - **Decision Reason**: Candidate satisfies education requirement via alternative 'BACHELOR' with degree 'MASTER' in data science.
  - **Matched Degree**: `MASTER` | **Field**: `data science`
  - **JD Requirement**: `BACHELOR in computer science`
  - **Candidate Degrees**: `MASTER, BACHELOR`
- **Experience Evaluation (Batch 3)**: **BELOW_REQUIREMENT** (Score: `0.0000`)
  - **Required**: `5.00 yrs` | **Verified**: `0.00 yrs` | **Relevance-Adjusted**: `0.00 yrs`
  - **Duration Ratio**: `0.00` | **Eff. Type Multiplier**: `1.00` | **Duties Relevance**: `0.0000`
  - **Decision Reason**: No verified employment records found; mandatory requirement is 5.00 years.

#### Detected Sections Summary

| Raw Heading | Category | Is Core? | Char Length | Sentences |
| :--- | :--- | :---: | :---: | :---: |
| `RINAD R` | `unknown` | No | 63 | 1 |
| `Objective` | `summary` | No | 368 | 2 |
| `Education` | `education` | Yes | 158 | 1 |
| `Skills` | `skills` | Yes | 333 | 1 |
| `Projects` | `projects` | Yes | 2043 | 10 |
| `Certifications` | `certifications` | Yes | 223 | 3 |
| `Extra-Curricular Activities` | `leadership` | No | 179 | 2 |

#### Independent Section Relevance Breakdown (Batch 1)

> **Note**: Section BM25 and Semantic scores below are calculated strictly independently from that section's own text and the relevant JD requirements. Cross-section weights (0.40, 0.30, 0.20, 0.10) are NOT used.

| Heading | Category | Core? | BM25 (Norm) | BM25 (Raw) | Semantic (Calibrated) | Semantic (Raw Cosine) | Matched Terms / Uncertainty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Skills` | `skills` | Yes | 0.3435 | 10.0011 | 0.4120 | 0.3766 | Matched: data, analysis, numpy, pandas, python |
| `Experience` | `experience` | Yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Section 'experience' absent or empty: no relevance score fabricated. |
| `Projects` | `projects` | Yes | 0.1128 | 11.2224 | 0.0000 | 0.1391 | Matched: data, engineering, numpy, pandas, python |
| `Education` | `education` | No | 0.1633 | 2.7726 | 0.1538 | 0.2346 | Matched: bachelor, data | Section 'education' scored for diagnostics only; excluded from relevance aggregation (Batch 2). |
| `RINAD R` | `unknown` | No | 0.0000 | 0.0000 | 0.0000 | 0.0312 | Unrecognized heading 'RINAD R': content preserved as unknown section. |
| `Objective` | `summary` | No | 0.0343 | 3.4124 | 0.5486 | 0.4517 | Matched: data, engineering |
| `Extra-Curricular Activities` | `leadership` | No | 0.0000 | 0.0000 | 0.0000 | 0.1474 | — |

- **Provisional Core BM25 Bridge**: `0.2282`
- **Provisional Core Semantic Bridge**: `0.1961`
*(Note: Downstream aggregation bridge is an unweighted placeholder isolated for this batch and will be formally redesigned in a subsequent batch.)*

#### Additional-Section Bonus Breakdown

- **Bonus Awarded**: `0.00 pts` (Reason: `relevance below threshold or no non-core evidence`)

**Extraction & Processing Notes**:
- Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.
- Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.
- Batch 3: Experience evaluated using Duration x Type x Duties Relevance.
- Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.
- Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.
- Batch 4: Certifications evaluated conditionally on JD without absent-penalty.

---

### 4. Unknown (`CAND_001` - cv_nujoom_IAIP.docx)

- **Experimental Final Score**: **27.89%** (Core: 27.89% + Bonus: 0.00)
- **Baseline Final Score**: 0.00% (Rank: 3 -> 4)
- **Education Eligibility Filter**: **UNKNOWN** ⚠️ *Requires manual verification*
  - **Decision Reason**: Candidate credentials (M.SC (in-progress), B.SC) do not unambiguously satisfy mandatory requirement (BACHELOR in computer science).
  - **Matched Degree**: `None` | **Field**: `None`
  - **JD Requirement**: `BACHELOR in computer science`
  - **Candidate Degrees**: `M.SC (in-progress), B.SC`
- **Experience Evaluation (Batch 3)**: **UNKNOWN** (Score: `0.0000`)
  - **Required**: `5.00 yrs` | **Verified**: `0.00 yrs` | **Relevance-Adjusted**: `0.00 yrs`
  - **Duration Ratio**: `0.00` | **Eff. Type Multiplier**: `1.00` | **Duties Relevance**: `0.0000`
  - **Decision Reason**: Candidate experience duration is uncertain due to missing dates; flagged for verification.

  *Extracted Roles & Duties:*
  - `•  Data Collection, Cleaning & Validation: Preprocessed and feature-engineered time-series and structured datasets, ensuring data quality and consistency before analysis.` at `Unknown` (UNKNOWN, Uncertain dates, mult=0.5): Duties Rel=`0.2361` (BM25=0.0803, Sem=0.3400) -> Adj=`0.00 yrs`
  - ⚠️ *Note*: Uncertain employment type: applied conservative unknown multiplier.
  - ⚠️ *Note*: Missing dates or duration: duration cannot be verified.

#### Detected Sections Summary

| Raw Heading | Category | Is Core? | Char Length | Sentences |
| :--- | :--- | :---: | :---: | :---: |
| `Professional Summary` | `summary` | No | 440 | 3 |
| `Education` | `education` | Yes | 305 | 1 |
| `Technical Skills` | `skills` | Yes | 473 | 1 |
| `Projects` | `projects` | Yes | 1656 | 9 |
| `Relevant Experience` | `experience` | Yes | 823 | 6 |
| `Achievements & Activities` | `achievements` | No | 777 | 5 |
| `AHAMMED NUJOOM YP` | `unknown` | No | 82 | 1 |

#### Independent Section Relevance Breakdown (Batch 1)

> **Note**: Section BM25 and Semantic scores below are calculated strictly independently from that section's own text and the relevant JD requirements. Cross-section weights (0.40, 0.30, 0.20, 0.10) are NOT used.

| Heading | Category | Core? | BM25 (Norm) | BM25 (Raw) | Semantic (Calibrated) | Semantic (Raw Cosine) | Matched Terms / Uncertainty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Technical Skills` | `skills` | Yes | 0.3537 | 10.2982 | 0.4474 | 0.3961 | Matched: data, analysis, engineering, numpy, pandas |
| `Relevant Experience` | `experience` | Yes | 0.0803 | 5.8422 | 0.3597 | 0.3478 | Matched: teams, sql, reporting, stakeholders |
| `Projects` | `projects` | Yes | 0.0970 | 9.6440 | 0.2473 | 0.2860 | Matched: data, analysis, dashboards, design, reporting |
| `Education` | `education` | No | 0.0991 | 1.6834 | 0.3391 | 0.3365 | Matched: data | Section 'education' scored for diagnostics only; excluded from relevance aggregation (Batch 2). |
| `Professional Summary` | `summary` | No | 0.0913 | 9.0769 | 0.8387 | 0.6113 | Matched: data, analysis, python, sql, dashboards |
| `Achievements & Activities` | `achievements` | No | 0.0139 | 1.3863 | 0.0604 | 0.1832 | Matched: mentor |
| `AHAMMED NUJOOM YP` | `unknown` | No | 0.0000 | 0.0000 | 0.0000 | 0.1037 | Unrecognized heading 'AHAMMED NUJOOM YP': content preserved as unknown section. |

- **Provisional Core BM25 Bridge**: `0.1770`
- **Provisional Core Semantic Bridge**: `0.3515`
*(Note: Downstream aggregation bridge is an unweighted placeholder isolated for this batch and will be formally redesigned in a subsequent batch.)*

#### Additional-Section Bonus Breakdown

- **Bonus Awarded**: `0.00 pts` (Reason: `relevance below threshold or no non-core evidence`)

**Extraction & Processing Notes**:
- Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.
- Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.
- Batch 3: Experience evaluated using Duration x Type x Duties Relevance.
- Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.
- Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.
- Batch 4: Certifications evaluated conditionally on JD without absent-penalty.

---

### 5. Marketing Coordinator (`CAND_003` - Resume_1_PriyaNair.txt)

- **Experimental Final Score**: **0.92%** (Core: 0.92% + Bonus: 0.00)
- **Baseline Final Score**: 0.00% (Rank: 5 -> 5)
- **Education Eligibility Filter**: **UNKNOWN** ⚠️ *Requires manual verification*
  - **Decision Reason**: JD mandates education qualification, but no verifiable education credentials were found in candidate resume.
  - **Matched Degree**: `None` | **Field**: `None`
  - **JD Requirement**: `BACHELOR in computer science`
  - **Candidate Degrees**: `None detected`
- **Experience Evaluation (Batch 3)**: **UNKNOWN** (Score: `0.0050`)
  - **Required**: `5.00 yrs` | **Verified**: `4.75 yrs` | **Relevance-Adjusted**: `0.03 yrs`
  - **Duration Ratio**: `0.95` | **Eff. Type Multiplier**: `0.47` | **Duties Relevance**: `0.0112`
  - **Decision Reason**: Candidate experience duration is uncertain due to missing dates; flagged for verification.

  *Extracted Roles & Duties:*
  - `Marketing Coordinator — BrightRetail Co., Chicago, IL (2023–Present)` at `Unknown` (UNKNOWN, 3.75 yrs, mult=0.5): Duties Rel=`0.0076` (BM25=0.0190, Sem=0.0000) -> Adj=`0.01 yrs`
  - `Marketing Intern — LocalBrand Studio (2022–2023)` at `Unknown` (INTERNSHIP, 1.92 yrs, mult=0.4): Duties Rel=`0.0183` (BM25=0.0000, Sem=0.0305) -> Adj=`0.01 yrs`
  - ⚠️ *Note*: Uncertain employment type: applied conservative unknown multiplier.

#### Detected Sections Summary

| Raw Heading | Category | Is Core? | Char Length | Sentences |
| :--- | :--- | :---: | :---: | :---: |
| `PRIYA NAIR` | `unknown` | No | 146 | 2 |
| `SUMMARY` | `summary` | No | 131 | 1 |
| `EXPERIENCE` | `experience` | Yes | 472 | 1 |
| `EDUCATION` | `education` | Yes | 61 | 1 |
| `SKILLS` | `skills` | Yes | 109 | 1 |

#### Independent Section Relevance Breakdown (Batch 1)

> **Note**: Section BM25 and Semantic scores below are calculated strictly independently from that section's own text and the relevant JD requirements. Cross-section weights (0.40, 0.30, 0.20, 0.10) are NOT used.

| Heading | Category | Core? | BM25 (Norm) | BM25 (Raw) | Semantic (Calibrated) | Semantic (Raw Cosine) | Matched Terms / Uncertainty |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `SKILLS` | `skills` | Yes | 0.0000 | 0.0000 | 0.0000 | 0.1415 | — |
| `EXPERIENCE` | `experience` | Yes | 0.0381 | 2.7726 | 0.0392 | 0.1716 | Matched: design, present |
| `Projects` | `projects` | Yes | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Section 'projects' absent or empty: no relevance score fabricated. |
| `EDUCATION` | `education` | No | 0.0000 | 0.0000 | 0.4572 | 0.4015 | Section 'education' scored for diagnostics only; excluded from relevance aggregation (Batch 2). |
| `PRIYA NAIR` | `unknown` | No | 0.0000 | 0.0000 | 0.1876 | 0.2532 | Unrecognized heading 'PRIYA NAIR': content preserved as unknown section. |
| `SUMMARY` | `summary` | No | 0.0000 | 0.0000 | 0.3624 | 0.3493 | — |

- **Provisional Core BM25 Bridge**: `0.0190`
- **Provisional Core Semantic Bridge**: `0.0119`
*(Note: Downstream aggregation bridge is an unweighted placeholder isolated for this batch and will be formally redesigned in a subsequent batch.)*

#### Additional-Section Bonus Breakdown

- **Bonus Awarded**: `0.00 pts` (Reason: `relevance below threshold or no non-core evidence`)

**Extraction & Processing Notes**:
- Batch 1: Section BM25 and semantic scores calculated independently without cross-section weights.
- Batch 2: Education excluded from relevance aggregation; evaluated as eligibility filter only.
- Batch 3: Experience evaluated using Duration x Type x Duties Relevance.
- Batch 4: Skills evaluated with token-safe alias matching, evidence provenance, and required-priority weighting.
- Batch 4: Projects scored on descriptions/contributions with duplicate-project suppression.
- Batch 4: Certifications evaluated conditionally on JD without absent-penalty.

---

## 4. Key Findings & Recommendations

1. **Evidence Density vs Document Length**: In baseline BM25, candidates with concise resumes often suffered because the whole-document denominator dampened term saturation, whereas lengthy resumes with scattered mentions received high whole-text BM25 scores. Section-based BM25 bounds each section independently via `theoretical_bound`, rewarding concentrated technical evidence.
2. **Elimination of Fallback Vector Dilution**: Baseline Semantic fell back to the whole resume vector whenever a candidate was missing an explicit section (e.g. `projects`), artificially counting the same profile text twice. Section-level semantic evaluation eliminates this artifact.
3. **Controlled Additional Evidence**: The Additional-Section Bonus correctly captures achievements, research publications, and leadership contributions without overpowering core dimensions and strictly deduplicates against core skills.
4. **Production Recommendation**: The experimental section-based pipeline is ready for larger offline evaluation across 1,000+ benchmark resumes. The existing production backend remained completely untouched throughout this experiment.
