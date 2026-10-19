# INTEGRATION CHANGELOG: Hybrid Semantic ATS & React/Node Integration

**Date:** October 18, 2026  
**Integration Target:** Updated Hybrid Semantic ATS Backend (`checkpoint_05_negation_scope_fix`) integrated with existing React + Node.js Application (`e2e-chatbot-app-next`).

---

## 1. Summary of Changes Already Present in Checkpoint 05

Checkpoint 05 introduced critical fixes on top of earlier checkpoints (CP01-CP04):

### A. Clause-Level Job Description Negation-Scope Fix (`app/services/jd_parser.py`)
- **Problem Fixed:** Earlier parser evaluated requirement classification flags (negation, "required", "preferred", trainable, certification-only) across entire lines. Prose sentences such as *"Python and SQL are required. Spark experience is not mandatory."* resulted in `required=[]` and `nonmandatory=[python, spark, sql]`, stripping mandatory skills and silently breaking hard filtering.
- **Solution:** Implemented `_clauses()` splitter which breaks text into clauses (sentence terminations `. `, `;`, and contrast conjunctions `but`/`however`/`although`/`though`/`whereas` only when both sides contain skills, preserving abbreviations such as `e.g.`, `i.e.`, `etc.`). Classification, negation, tiers, cert-only skipping, and alternative-group detection now operate per clause.

### B. Dynamic Score Composition & Active-Weight Renormalisation (`app/services/score_composer.py`)
- Implemented `compose_score()` using the formula:
  $$\text{final\_score} = \frac{\sum w_i \cdot s_i}{\sum w_{i,\text{active}}}$$
- Distinguishes absent/not-applicable criteria (e.g., JD has no certification or experience requirements) from failed requirements. Absent criteria are excluded from the denominator so candidates are not unfairly penalised.

### C. Configurable Shortlist Policy (`app/services/shortlist_policy.py`)
- Centralized policy for Shortlisted ($\ge 60\%$), Review ($45\% - 59\%$), and Not Shortlisted ($< 45\%$).
- Implemented `recommend_single(score, eligible)` and `BEST_AVAILABLE_K` fallback to highlight best candidates when no candidates reach the shortlist threshold.

### D. Section-Level Chunking & Embedding Safeguards (`app/services/embedding_service.py`, `app/services/text_chunking.py`)
- Integrated length-bounded sliding-window chunking (180 words, 30 overlap, max 8 chunks) with length-weighted pooling.
- Handled missing models gracefully via `EmbeddingUnavailableError` with fail-loud HTTP 503 or explicit `ALLOW_DEGRADED_SEMANTIC=true`.

### E. Self-Contained BM25+ & Corpus Protection (`app/services/bm25_search.py`)
- Native implementation of BM25+ (Lv & Zhai, 2011) with query-match ceiling normalization.
- Enforced `persist=False` during interactive and per-request ranking so temporary single-document indexes never overwrite the persistent corpus index.

---

## 2. Changes Newly Made During Integration

### A. Windows UTF-8 Encoding Safeguard (`tests/test_scoring_fix.py`)
- **Issue:** On Windows, `Path.read_text()` defaults to CP1252 (Windows ANSI). The sample resumes contained UTF-8 en-dashes (`\u2013` in `2022–Present`). CP1252 decoded `\xe2\x80\x93` as `â€“`, creating word characters that broke date regex boundary matches and caused experience extraction to return `0.0`.
- **Fix:** Added `encoding="utf-8", errors="replace"` to all `read_text()` invocations in `test_scoring_fix.py`.

### B. Offline Runner Shim Integration (`tests/run_offline.py`)
- **Issue:** When running in an environment with real `pytest` installed, `run_offline.py` failed with `KeyError` and `AttributeError: module 'pytest' has no attribute '_Skip'` because it looked for shim-specific attributes on the real pytest library.
- **Fix:** Configured `run_offline.py` to prioritize `_offline_shims` for its lightweight execution while leaving `python -m pytest tests -v` for full pytest execution.

### C. Flexible Prompt Parsing in Screening Adapter (`app/adapters/screening_adapter.py`)
- **Enhancement:** Enhanced regex pattern in `parse_screening_prompt()` to seamlessly support both the standard frontend prompt template (`JOB DESCRIPTION AND REQUIREMENTS:`, `CANDIDATE RESUME:`) and alternative prompt formats (`Job Description:`, `Resume:`, `JD:`).

### D. Server Error Handling & ResponsesAgent Compatibility (`app/server.py`)
- Unified `/invocations` endpoint to catch `EmbeddingUnavailableError` and return HTTP 503 if the local embedding model is unavailable and degraded mode is not enabled.
- Preserved all response fields required by `NewChatPage.tsx` (`atsScore`, `rawAtsScore`, `fit`, `recommendation`, `matchedSkills`, `strengths`, `concerns`, `scoreBreakdown`).

---

## 3. Files Modified, Added, or Preserved

| File Path | Action | Description |
|---|---|---|
| `app/services/jd_parser.py` | Updated | Clause-level negation fix & alternative skills |
| `app/services/score_composer.py` | Added | Active-component weight renormalisation |
| `app/services/shortlist_policy.py` | Added | Shortlisting thresholds & policy logic |
| `app/services/text_chunking.py` | Added | Sliding-window text chunking |
| `app/services/requirements.py` | Added | Skill grouping & requirement slots |
| `app/services/bm25_search.py` | Updated | Native BM25+ with persistent corpus protection |
| `app/services/embedding_service.py` | Updated | Chunked embeddings & fail-fast handling |
| `app/services/hard_filter.py` | Updated | Eligibility distinct from fit score |
| `app/services/hybrid_ranker.py` | Updated | Integrated score composer & NA policies |
| `app/services/certification_matcher.py`| Updated | Requirement-level awareness (required/preferred/none) |
| `app/services/experience_matcher.py` | Updated | Distinguishes absent requirement from failure |
| `app/adapters/screening_adapter.py` | Updated | Full backward-compatible response fields |
| `app/server.py` | Updated | FastAPI `/invocations`, `/api/files/upload`, CORS |
| `app/config.py` | Updated | Config precedence: request > env > `.env` > code |
| `tests/test_scoring_fix.py` | Fixed | Windows UTF-8 text file loading fix |
| `tests/run_offline.py` | Fixed | Consistent offline runner shim loading |
| `data/bm25_index.pkl` | Preserved | Stable 64-document corpus index |
| `data/ats_metadata.db` | Preserved | Candidate metadata SQLite store |

---

## 4. API Contract Verification

### Resume Upload (`POST /api/files/upload`)
- **Caller:** `NewChatPage.tsx` and Node.js `files.ts`
- **Request:** `multipart/form-data` with `file`
- **Response:**
  ```json
  {
    "success": true,
    "pathname": "Resume_2_DavidChen.txt",
    "contentType": "text/plain",
    "resumeText": "DAVID CHEN\nData Analyst..."
  }
  ```

### Screening Invocations (`POST /invocations`)
- **Caller:** `NewChatPage.tsx` via Node.js `/invocations` proxy
- **Request:**
  ```json
  {
    "input": [
      {
        "role": "user",
        "content": "JOB TITLE:\nSenior Data Analyst\n\nJOB DESCRIPTION AND REQUIREMENTS:\n...\n\nCANDIDATE RESUME:\n..."
      }
    ]
  }
  ```
- **Response:**
  ```json
  {
    "output": [
      {
        "type": "message",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": "{\"candidateName\": \"David Chen\", \"atsScore\": 62, ...}" }]
      }
    ],
    "output_text": "{\"candidateName\": \"David Chen\", \"atsScore\": 62, ...}",
    "text": "{\"candidateName\": \"David Chen\", \"atsScore\": 62, ...}"
  }
  ```

---

## 5. Known Limitations & Deliberately Unchanged Behaviors

1. **Formula Weights:** The provisional weights (0.40 Semantic, 0.27 BM25, 0.15 Skills, 0.10 Experience, 0.08 Certifications) were strictly preserved as requested.
2. **Cosine Similarity Calibration:** SBERT MiniLM bi-encoders naturally produce similarity scores between 0.15 and 0.70; monotonic calibration $(x - 0.15) / 0.55$ maps this to $0.0 - 1.0$ ordinally.
3. **Local Embedding Execution:** Uses local PyTorch CPU execution (`sentence-transformers/all-MiniLM-L6-v2`); requires no external paid API keys.
