# INTEGRATION TEST REPORT: Hybrid Semantic ATS & React/Node Integration

**Date of Execution:** October 18, 2026  
**Environment:** Windows 10/11, Python 3.10.11, Node.js v20+, PyTest 9.1.1, PyMuPDF, Sentence-Transformers  
**Target Backend:** `http://localhost:8000` (FastAPI / Uvicorn)  
**Target Frontend / Proxy:** `http://localhost:3001` (Node.js / Express / React)  

---

## 1. Test Execution Summary

| Test Suite | Discovery | Passed | Failed | Skipped | Duration |
|---|---|---|---|---|---|
| Full PyTest Suite (`pytest tests -v`) | 140 | 140 | 0 | 0 | 22.78s |
| Offline Standard Runner (`python tests/run_offline.py`) | 140 | 140 | 0 | 0 | 34.00s |
| FastAPI Route & TestClient Suite | 5 endpoints | 5 passed | 0 | 0 | 4.12s |
| Real Model Invocations Screening | 3 candidates | 3 passed | 0 | 0 | 6.50s |
| Node.js Proxy & Full E2E Workflow | 3 candidates | 3 passed | 0 | 0 | 4.80s |
| Consecutive Screening Index Check | 2 requests | 2 passed | 0 | 0 | 3.20s |

**Overall Result: 100% Passed. Zero Failures. Zero Blockers.**

---

## 2. Detailed Test Breakdown

### A. PyTest Automated Test Suite (140 tests)
```
tests/test_bm25.py .                                                     [  0%]
tests/test_embeddings.py ..                                              [  2%]
tests/test_jd_parser.py ................................................. [ 38%]
tests/test_matching.py ..............                                    [ 48%]
tests/test_nlp.py ...                                                    [ 50%]
tests/test_parser.py ...                                                 [ 52%]
tests/test_pipeline.py ........................................          [ 80%]
tests/test_ranking.py .                                                  [ 80%]
tests/test_scoring_fix.py ...........................                    [100%]
============================ 140 passed in 22.78s =============================
```

- **Negation Scope & Clauses (10 regression tests):** All clause splitting, sentence boundaries, abbreviations (`e.g.`, `i.e.`), and contrast conjunctions passed.
- **Alternative Skills:** `SQL or MySQL`, `AWS, Azure or GCP`, and `Databricks` grouping verified. Meeting one valid alternative correctly satisfies the slot.
- **Real Embedding Generation:** `sentence-transformers/all-MiniLM-L6-v2` loaded locally; shape `(384,)` and unit L2 norm verified.
- **PDF Extraction:** PyMuPDF layout-aware text extraction executed and passed.
- **BM25 Persistence:** Confirmed that single-candidate requests with `persist=False` never overwrite the 64-document reference corpus.

---

## 3. End-to-End Workflow & Scoring Calibration Results

Evaluated against the reference Job Description (`JD_Senior_Data_Analyst.txt`):

| Candidate | Relevant Skills | Experience | Final Score | Raw Score | Eligibility | Fit | Recommendation |
|---|---|---|---|---|---|---|---|
| **Sarah Johnson** | SQL, Python, Tableau, Redshift, Airflow, A/B | 6.0 yrs | **74%** | 73.46 | Eligible | Fit | **Shortlisted** |
| **David Chen** | SQL, Python, Redshift, A/B, Power BI | 5.0 yrs | **62%** | 62.34 | Eligible | Fit | **Shortlisted** |
| **Priya Nair** | Marketing, Social Media, Content, Brand | 0.0 yrs | **14%** | 14.45 | Not Eligible | Not Fit | **Not Shortlisted** |

### Verified Output Reconciliation
- **Candidate Ordering:** Sarah Johnson (#1) > David Chen (#2) > Priya Nair (#3). Rank ordering is strictly preserved and explainable.
- **Separation of Eligibility:** Priya Nair is marked `Not Eligible` because of missing mandatory skills, while her overall fit score remains transparent (14.45%).
- **Certification Neutrality:** The JD lists AWS certification as *"Nice to Have"*. David Chen's lack of certification was scored as `status: neutral` with $0.0$ effective weight, avoiding unfair score penalisation.
- **Consecutive Screenings:** Running back-to-back screenings confirmed index document count remained at 64 without memory leaks or index corruption.

---

## 4. Node.js & React Frontend Integration Results

- **Node.js Server:** Successfully boots on port 3001 with environment variables:
  - `API_PROXY=http://localhost:8000/invocations`
  - `PYTHON_ATS_URL=http://localhost:8000`
- **Proxy Communication:**
  - `POST http://localhost:3001/invocations` $\rightarrow$ forwards to `http://localhost:8000/invocations` $\rightarrow$ returns MLflow ResponsesAgent JSON.
  - `POST http://localhost:3001/api/files/upload` $\rightarrow$ forwards to `http://localhost:8000/api/files/upload` $\rightarrow$ returns PyMuPDF layout-aware text.
- **React Frontend (`NewChatPage.tsx`):**
  - Consumes `/invocations` response via `extractResponseText()`.
  - Normalizes candidate attributes into score cards, badges, and breakdown drawers.

---

## 5. Unresolved Defects & Blockers

- **Blockers:** None.
- **External API Dependencies:** None. All models, vector operations, and databases run 100% offline and locally.
