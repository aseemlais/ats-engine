# Local spaCy Section-Based ATS Scoring Experiment

## 1. Overview & Objectives

This experiment evaluates an alternative, section-oriented scoring methodology for candidate resume ranking.

### Motivation
In the existing Hybrid Semantic ATS, BM25 lexical relevance and semantic similarity are computed globally across the entire resume while simultaneously calculating section-level scores. This whole-document evaluation can:
1. **Dilute dense evidence**: Concise candidates with highly relevant technical experience in their work history can be penalized by the global document length denominator.
2. **Double-count evidence**: Scattered keywords across multiple sections can inflate global BM25, while missing sections fall back to whole-resume embeddings.
3. **Overly restrict edge-case candidates**: Candidates with strong evidence formatted under non-standard headings (e.g. *Technical Arsenal*, *Research & Publications*, *Bragging Rights*) risk having that evidence discarded or downweighted.

### The Section-Based Approach
1. **spaCy Layout-Aware Extraction**: Text is extracted from PDF, DOCX, and TXT files and segmented into structured sections using an alias-driven heading recognition engine.
2. **Section-Level BM25**: Evaluates BM25 lexical relevance strictly within detected sections using section-specific queries and normalizes via `theoretical_bound` over seen query terms. No global whole-document BM25 score contributes to the final score.
3. **Section-Level Semantic Relevance**: Computes Sentence Transformer cosine similarity for each detected section independently. If a section is absent, it does not fall back to the whole resume vector.
4. **Core ATS Aggregation**: Preserves existing production weights (`semantic=0.35, bm25=0.20, skill=0.27, experience=0.10, certification=0.08`) and the existing `compose_score` normalization formula.
5. **Additional-Section Relevance Bonus**: Introduces an optional, low-weight, deduplicated bonus for relevant evidence found in non-core sections (*Achievements*, *Publications*, *Leadership*, *Custom/Unknown*).

---

## 2. Critical Safety Guarantee: Isolated Local Experiment

This experiment was developed and executed strictly as an isolated, offline evaluation:
- **Zero Production Modifications**: No files in `app/`, `frontend/`, or `server/` were modified.
- **Zero Network Deployments**: No code was pushed to GitHub, Hugging Face Spaces, Render, Vercel, or any remote server.
- **Zero Paid AI Services**: Runs entirely on free local open-source libraries (`spaCy 3.8`, `sentence-transformers/all-MiniLM-L6-v2` on CPU).
- **Zero Database Mutations**: No changes were made to Supabase, Qdrant indexes, or production environments.

---

## 3. Directory Structure

```
experiments/
├── experiment_config.py           # Configuration for heading patterns, section weights, and bonus parameters
├── spacy_section_detector.py      # spaCy-powered document extractor & heading classifier
├── section_scorer.py              # Section-level BM25 & semantic scorer + core composer + bonus calculator
├── section_scoring_experiment.py  # CLI runner for side-by-side baseline vs experimental ranking
├── README.md                      # This documentation guide
└── CHANGELOG_EXPERIMENT.md        # File catalog and implementation log

reports/
└── section_scoring_comparison_report.md  # Detailed evaluation report comparing candidate rankings

tests/
└── test_section_scoring_experiment.py    # Pytest test suite covering 12 required scenarios + safety checks
```

---

## 4. Mathematical Formulation (Batch 1 Refined)

### A. Independent Per-Section BM25 Lexical Relevance
Arbitrary cross-section weights ($0.40, 0.30, 0.20, 0.10$) have been **strictly removed**.
For every detected section $s$ (e.g. skills, experience, projects, education, unknown):
$$\text{BM25}_{\text{raw}}(s) = \sum_{t \in Q_s} \text{IDF}(t) \cdot \frac{f(t, D_s) \cdot (k_1 + 1)}{f(t, D_s) + k_1 \cdot \left(1 - b + b \cdot \frac{|D_s|}{\text{avgdl}}\right)}$$
Normalized via `theoretical_bound` over seen query terms to map the lexical evidence into $[0.0, 1.0]$:
$$\text{BM25}_{\text{norm}}(s) = \min\left(1.0, \frac{\text{BM25}_{\text{raw}}(s)}{\sum_{t \in Q_s, t \in \text{IDF}} \text{IDF}(t) \cdot (\delta + k_1 + 1)}\right)$$
- When section $s$ is absent or empty, $\text{BM25}_{\text{norm}}(s) = 0.0$ and $\text{BM25}_{\text{raw}}(s) = 0.0$. No scores are fabricated.
- Crucially, $\text{BM25}_{\text{norm}}(s)$ is **never multiplied** by any other section's percentage weight.

### B. Independent Per-Section Semantic Relevance
For every detected section $s$:
$$\text{Sim}_{\text{raw}}(s) = \cos(\mathbf{v}_{D_s}, \mathbf{v}_{Q_s})$$
Calibrated to $[0, 1]$ via production bounds ($\text{floor}=0.15, \text{ceil}=0.70$):
$$\text{Sim}_{\text{calibrated}}(s) = \min\left(1.0, \max\left(0.0, \frac{\text{Sim}_{\text{raw}}(s) - 0.15}{0.70 - 0.15}\right)\right)$$
- If section $s$ is absent or empty, $\text{Sim}_{\text{raw}}(s) = 0.0$ and $\text{Sim}_{\text{calibrated}}(s) = 0.0$.
- Each section's semantic similarity is calculated purely against its own relevant JD query vector without cross-section blending.

### C. Provisional Downstream Aggregation Bridge
As Batch 1 focuses on independent section-level scoring without yet redesigning the final aggregation formula:
- An isolated provisional bridge computes the unweighted mean of present core sections solely to supply the existing downstream `compose_score` pipeline.
- This prevents silent replacement weighting while keeping per-section calculations 100% independent until the aggregation redesign batch.

### D. Additional-Section Relevance Bonus
For non-core sections $s_{\text{add}} \in \{\text{achievements}, \text{publications}, \text{leadership}, \text{unknown}\}$:
$$\text{Relevance}_{\text{add}} = 0.60 \cdot \text{Sem}_{\text{add}} + 0.40 \cdot \text{Lex}_{\text{add}}^{\text{dedup}}$$
where $\text{Lex}_{\text{add}}^{\text{dedup}}$ only rewards skills **not** already credited in core sections.
If $\text{Relevance}_{\text{add}} \ge \text{threshold}$ (default $0.20$):
$$\text{Bonus} = \beta \cdot \text{Relevance}_{\text{add}}$$
$$\text{ExperimentalFinalScore} = \min(100.0, \text{CoreScore} + \text{Bonus})$$

---

## 5. Usage Instructions

### Running the Comparison CLI
Run the experiment against candidate resumes in a folder:
```powershell
python experiments/section_scoring_experiment.py `
  --resumes data/resumes `
  --jd data/resumes/JD_Senior_Data_Analyst.txt `
  --output reports/section_scoring_comparison_report.md `
  --beta 3.0
```

#### CLI Arguments:
- `--resumes`: Path to a single resume file or directory containing PDF, DOCX, or TXT resumes.
- `--jd`: Path to a Job Description text file.
- `--output`: Filepath to write the Markdown comparison report (default: `reports/section_scoring_comparison_report.md`).
- `--beta`: Scaling factor $\beta$ for the additional bonus (default: `3.0`). Set to `0.0` to disable bonus.
- `--no-bonus`: Convenience flag to disable additional section bonus completely.

---

## 6. Running the Test Suite

Execute the 20-test verification suite covering all 12 edge-case scenarios + safety checks + Batch 1 independent scoring verifications:
```powershell
python -m pytest tests/test_section_scoring_experiment.py -v
```

### Covered Scenarios:
1. Common sections resume with standard headings.
2. Missing explicit Skills heading (skills in summary/projects).
3. Custom/unfamiliar headings (*Technical Arsenal*, *Where I've Been*).
4. Relevant skills inside Projects section.
5. Relevant evidence in Achievements / Publications.
6. Non-relevant evidence in additional sections (Bonus = 0.0).
7. Empty or missing sections (no automatic failure penalty).
8. Repeated mentions of the same skill (strict deduplication).
9. Long, wordy unrelated section (no score inflation).
10. Incomplete or failed text extraction reporting.
11. Multiple sections mapped to the same core dimension.
12. High core score candidate capped at 100.0.
13. Bonus toggling and $\beta=0.0$ behavior.
14. Production codebase integrity verification.
15. **Batch 1**: Section BM25 calculated strictly independently.
16. **Batch 1**: Section semantic similarity calculated strictly independently.
17. **Batch 1**: No cross-section percentage weights used in calculations.
18. **Batch 1**: Unknown/custom sections preserved with text evidence and scores.
19. **Batch 1**: Empty or missing sections receive exactly 0.0 without fabricated scores.
20. **Batch 1**: Global whole-resume BM25 and semantic scores are not separately added.


---

## 7. Batch 2: Education as an Eligibility Filter Only

### Overview
In Batch 2, education has been decoupled completely from relevance scoring:
- **Zero Relevance Score Contribution**: Education has 0% weight and 0 points in section-level BM25 aggregation, semantic aggregation, core relevance scoring, and additional bonus.
- **Hard Filter Logic**: Education is evaluated strictly as an initial eligibility filter returning four deterministic statuses:
  - `ELIGIBLE`: Candidate satisfies mandatory education requirement.
  - `INELIGIBLE`: Candidate demonstrably fails mandatory degree tier, field of study, or completion criteria. Segregated from normal eligible candidate ranking.
  - `UNKNOWN`: Evidence is ambiguous or missing (flagged for manual verification; candidate is NOT automatically rejected).
  - `NOT_REQUIRED`: JD does not establish a mandatory degree requirement.

### Alternative Qualifications (OR Logic)
Alternative qualifications (e.g. `B.Tech OR M.Sc. OR M.Tech in Computer Science or related field`) are evaluated using `OR` logic:
- Satisfying any one accepted alternative qualifies the candidate as `ELIGIBLE`.
- Degrees are non-cumulative; candidates are not penalized for lacking other listed alternatives.

### Heading Fallback
If the resume omits an explicit `Education` heading, the pipeline scans the full extracted resume text for degree keywords and institutions before declaring `UNKNOWN` or `INELIGIBLE`.

### Batch 2 Covered Scenarios (Tests 21–30):
21. Candidate satisfies one of several accepted degree alternatives (OR logic).
22. Candidate satisfies single mandatory degree requirement.
23. Candidate demonstrably fails explicit mandatory requirement (`INELIGIBLE`).
24. JD contains no mandatory education requirement (`NOT_REQUIRED`).
25. Resume education heading missing but evidence exists in text.
26. Education evidence cannot be extracted anywhere (`UNKNOWN` with manual verification flag).
27. Degree is in progress when JD requires completed degree (`INELIGIBLE`).
28. Field-of-study requirement is present and matched.
29. Degree alternatives are treated as OR, not cumulative.
30. Education has zero contribution to BM25, semantic, core score, or bonus.

---

## 8. Batch 3: Experience and Internship Scoring

### Overview
Batch 3 introduces a transparent, explainable experience scoring component that evaluates candidate work history using three decoupled factors:
1. **Relevant Duration**: Total calendar duration of verified employment.
2. **Experience Type**: Distinguishes full-time professional employment, internships, and other work.
3. **Duties Relevance**: Measures how closely the candidate's actual duties and responsibilities match the JD, calculated strictly on duties text (not job titles) using BM25 lexical matching (0.40) and Sentence Transformer semantic cosine similarity (0.60).

### Key Architectural Principles
- **Eligibility vs. Relevance**: Experience eligibility (does the candidate meet the JD's minimum duration requirement?) is evaluated and reported separately from the continuous experience relevance score.
- **Projects Segregation**: Academic, personal, and open-source projects are strictly excluded from experience duration and cannot be counted as professional employment or internships.
- **Title Infiltration Prevention**: Job titles alone (e.g. "Senior Principal Architect") do not grant relevance; scoring is driven by actual demonstrated responsibilities.
- **No Double-Counting (Interval Merging)**: Overlapping concurrent employments or concurrent internships/jobs are projected onto a calendar month grid, preventing duplicate credit. Each month earns $\max(\text{type\_multiplier} \times \text{duties\_relevance})$.
- **Neutral Treatment for Freshers**: When the JD specifies no required experience or welcomes freshers (`NOT_REQUIRED`), candidates without experience are not penalized. Their experience dimension is marked `not_applicable` in the core score aggregation.
- **Fresher Experience Bonus**: Freshers with relevant internships or work experience can earn a separate, capped bonus (up to +2.0 points) that never compensates for failed education or missing mandatory skills.

### Mathematical Formulation

$$\text{Experience Score} = \min\left(1.0, \frac{\text{Relevance-Adjusted Non-Overlapping Years}}{\text{JD Required Years}}\right) = \text{Duration Ratio} \times \text{Type Multiplier} \times \text{Duties Relevance}$$

Where:
- $\text{Duration Ratio} = \min\left(1.0, \frac{\text{Total Verified Non-Overlapping Years}}{\text{JD Required Years}}\right)$
- $\text{Duties Relevance} = 0.60 \times \text{Semantic Sim}(\text{duties}, \text{JD}) + 0.40 \times \text{BM25}_{\text{norm}}(\text{duties}, \text{JD})$
- Configurable Type Multipliers:
  - `PROFESSIONAL`: $1.0$
  - `INTERNSHIP`: $0.4$
  - `OTHER`: $0.7$
  - `UNKNOWN`: $0.5$ (with extraction uncertainty flag)

### Experience Eligibility Statuses
- `MEETS_REQUIREMENT`: Candidate has verified duration $\ge$ required duration.
- `BELOW_REQUIREMENT`: Candidate has verified duration $<$ required duration.
- `NOT_REQUIRED`: JD does not require minimum experience or welcomes freshers.
- `UNKNOWN`: Experience requirements or candidate evidence are ambiguous.

### Batch 3 Covered Scenarios (Tests 31–50):
31. Candidate meets exact required duration.
32. Candidate exceeds required duration (capped at 1.0 / 100%).
33. Candidate has less than required duration (proportional duration ratio).
34. Professional employment receives higher weight than internship.
35. Candidate has only internships for an internship-friendly role.
36. High-duration low-relevance role receives lower score than lower-duration high-relevance role.
37. Candidate with high relevance in duties scores higher than one with matching title but generic duties.
38. Overlapping employment periods are not double-counted.
39. Internship overlapping professional employment is merged without inflating duration.
40. Open-source or academic projects are NOT counted as employment experience.
41. JD requires experience range (e.g. 3-5 years) evaluated against lower bound (3 years).
42. JD does not require experience (`NOT_REQUIRED`); candidate with zero experience is not penalized.
43. Ambiguous experience requirement in JD handled gracefully as `UNKNOWN`.
44. Candidate with ambiguous experience dates handled gracefully with uncertainty note.
45. Incomplete experience record (missing org or responsibilities) handled with uncertainty.
46. Experience scoring parameters are fully configurable via `ExperienceScoringConfig`.
47. Maximum experience score caps are strictly respected ($\le 1.0$).
48. Fresher experience bonus applies when experience not required and does not compensate for failed education.
49. Title alone does not determine relevance score; duties text drives scoring.
50. Multi-role career progression evaluated correctly per role.

---

## 7. Batch 4: Skills, Projects, and Certifications Evaluation & Bonus Repair

### Key Architectural Principles
- **Skills Normalization & Provenance**: Uses token-safe matching and configurable aliases (e.g., `scikit-learn`/`sklearn`, `postgresql`/`postgres`). Cross-section provenance tracking prevents duplicate counting across Skills, Experience, and Projects.
- **Education Exclusion**: Education sections are strictly excluded from core skill evidence and relevance scores.
- **Required vs Preferred Priority**: Missing required skills heavily penalize the skills dimension. A multitude of preferred skills cannot compensate for missing core required skills.
- **Projects Relevance & Duplication**: Project scoring evaluates descriptions and contributions using lexical BM25 and calibrated semantic similarity (not titles alone). Token-overlap Jaccard similarity neutralizes duplicate/repeated project descriptions.
- **Conditional Certifications**: Certifications are evaluated conditionally on JD requirements/preferences. If not required, absence causes no score deduction.
- **Additional Bonus Repair & Deduplication**: Fresh ephemeral BM25 instances prevent index contamination when evaluating new skills in additional sections (`achievements`, `publications`, `leadership`, `custom/unknown`). Skills already credited in core sections are strictly excluded.

