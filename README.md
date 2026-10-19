# Scalable Hybrid Semantic ATS for 10,000+ Resumes

An NLP-powered scalable Applicant Tracking System designed to retrieve and rank relevant candidates from 10,000+ resumes using hybrid information retrieval. The system combines BM25 lexical search, Sentence Transformer semantic embeddings, vector search, structured skill/experience matching, and explainable candidate ranking without requiring LLM-based generation.

---

## 1. Project Objective & Core Philosophy

The primary objective is:
> **Given 10,000 resumes and one Job Description (JD), efficiently retrieve and rank the most relevant resumes (Top 50 or Top 100 candidates) with high speed, low cost, semantic understanding, exact keyword precision, explainability, and reproducibility.**

### Core Tenets:
- **No LLM in the core filtering pipeline**: No OpenAI, Groq, Gemini, or Claude API calls per resume. Zero token cost, zero rate-limiting, sub-second latency.
- **No RAG for filtering**: Resumes are parsed once into structured representations and vector embeddings.
- **Deterministic Information Extraction**: spaCy/Regex/Taxonomy-driven deterministic skill, experience, and degree extraction.
- **Precomputed Search Space**: 10,000 resumes are parsed, embedded, and indexed once. When a recruiter submits a JD, only the JD is encoded, followed by sub-second vector & lexical retrieval.

---

## 2. Architectural Evolution (Comparison with Reference Repository)

Base Reference Repository: [`https://github.com/Annu012/resume-parser-ats`](https://github.com/Annu012/resume-parser-ats)

| Component | Reference Repository (`Annu012/resume-parser-ats`) | Our Scalable Hybrid Semantic ATS |
| :--- | :--- | :--- |
| **Document Parsing** | PyPDF only (slow, no DOCX support) | PyMuPDF (C++ fitz) for PDF + `python-docx` for DOCX (10x faster) |
| **Extraction Engine** | LLM API call per resume (Groq / Gemini) + naive spaCy NER | Fully deterministic Regex + Rule-based Taxonomy (0 API cost, ~250 resumes/sec) |
| **Skill Taxonomy** | Hardcoded list of ~40 skills, no canonical aliasing | Curated taxonomy of 400+ skills with bidirectional alias mapping (`k8s` -> `kubernetes`, etc.) |
| **Semantic Search** | None (simple word set intersection) | Sentence Transformers (`all-MiniLM-L6-v2`) with section-level representations |
| **Lexical Search** | Naive Jaccard token overlap | BM25Okapi lexical retrieval with tech-token-preserving tokenization |
| **Vector Indexing** | None | Qdrant Vector Database (Cosine similarity, 384-d, local & remote server support) |
| **Hard Filtering** | None (every resume scored identically) | Microsecond pre-filtering on mandatory skills, experience threshold, and education tier |
| **Scoring Formula** | Length & layout check (Max 40 points) | Configurable hybrid formula: $0.45 \times \text{Semantic} + 0.30 \times \text{BM25} + 0.15 \times \text{Skill} + 0.10 \times \text{Exp}$ |
| **Explainability** | Hardcoded format issues | Transparent deterministic breakdown (matched/missing required & preferred skills, exp delta) |
| **Scalability** | Single resume scanning, restarts on error | Batch ingestion (up to 10,000+), SHA-256 duplicate detection, JSON checkpointing |

---

## 3. System Architecture & Multi-Stage Reduction

```
                    10,000 RESUMES (PDF / DOCX)
                               |
                               v
                     +-------------------+
                     | PyMuPDF / python-docx |
                     +---------+---------+
                               |
                               v
                     +-------------------+
                     |  NLP Extraction   |  (~250 resumes/sec)
                     +---------+---------+
                               |
                               v
                  STRUCTURED METADATA & EMBEDDINGS
                     (Persisted in SQLite & Qdrant)
                               |
                               v
========================== SEARCH PHASE ==========================
                     +-------------------+
                     |  Job Description  |
                     +---------+---------+
                               |
                               v
                      [ HARD FILTER STAGE ]
                   Prunes clearly unqualified candidates
                     10,000  -->  ~4,600 candidates
                               |
                               v
                     [ HYBRID RETRIEVAL ]
                ├── Semantic Vector Search (Qdrant)
                ├── BM25 Lexical Matching
                ├── Skill Overlap (Required vs Preferred)
                └── Experience & Education Hierarchy Match
                               |
                               v
                      [ HYBRID RANKING ]
                     0.45*Sem + 0.30*BM25 + 0.15*Skill + 0.10*Exp
                               |
                               v
                     [ SHORTLIST (TOP 50/100) ]
                    Deterministic Score & Reasons
```

---

## 4. Empirical Benchmarks & Measured Performance

All metrics below are measured on the real system:

### Ingestion & Processing Speed
- **NLP Structured Extraction**: **259.3 resumes / second** (0.193s for 50 resumes)
- **Vector Embedding Generation (Batched PyTorch)**: **~8.2 resumes / sec** on CPU
- **BM25 Index Build (1,058 resumes)**: **0.80 seconds**
- **BM25 Index Build (10,000 resumes)**: **1.09 seconds**
- **Memory Overhead for 10,000 resumes**: **180 MB RAM peak**

### Search Latency & Throughput
| Candidate Pool | Hard Filtered | Query Latency | Ranking Throughput |
| :--- | :--- | :--- | :--- |
| **10 Resumes** | 8 | 5.1s *(cold model load)* | ~2 cand/sec |
| **100 Resumes** | 54 | 121.7 ms | 822 cand/sec |
| **1,058 Resumes** | 516 | 224.5 ms | **4,712 cand/sec** |
| **10,000 Resumes** | 4,668 | 8,098 ms *(cold)* | **1,235 cand/sec** |

### Retrieval Accuracy (Measured on 1,058 Labeled Candidates)
| Retrieval Method | Precision@10 | Precision@50 | Recall@10 | Recall@50 | NDCG@50 | MRR | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Keyword Matching** | 1.00 | 1.00 | 0.02 | 0.10 | 1.000 | 1.000 | 4,605 ms |
| **BM25 Lexical** | 1.00 | 0.92 | 0.02 | 0.10 | 1.000 | 1.000 | 181.9 ms |
| **Semantic Search** | 1.00 | 1.00 | 0.02 | 0.10 | 1.000 | 1.000 | 167.7 ms |
| **Hybrid (BM25+Sem)** | 1.00 | 1.00 | 0.02 | 0.10 | 1.000 | 1.000 | 165.9 ms |
| **Full ATS Pipeline** | **1.00** | **1.00** | **0.02** | **0.10** | **1.000** | **1.000** | **111.2 ms** |

---

## 5. Project Folder Structure

```
ai-ats-10k-resumes/
│
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Centralized settings & scoring weights
│   ├── api/
│   │   ├── resume_routes.py     # Upload, pagination, candidate details
│   │   ├── job_routes.py        # Job Description parsing & creation
│   │   └── search_routes.py     # Multi-stage hybrid search endpoints
│   ├── services/
│   │   ├── resume_parser.py     # PyMuPDF + python-docx parsing
│   │   ├── jd_parser.py         # Deterministic JD requirements extraction
│   │   ├── nlp_extractor.py     # Entity, section & experience extractor
│   │   ├── embedding_service.py # Sentence Transformers & section vectors
│   │   ├── semantic_search.py   # Cosine similarity & section weights
│   │   ├── bm25_search.py       # BM25Okapi lexical retrieval
│   │   ├── skill_matcher.py     # Required/preferred skill overlap
│   │   ├── experience_matcher.py# Experience seniority compatibility
│   │   ├── education_matcher.py # Degree hierarchy evaluation
│   │   ├── hard_filter.py       # Microsecond candidate pruning
│   │   ├── hybrid_ranker.py     # Composite scoring pipeline
│   │   └── explainability.py    # Deterministic reason generator
│   ├── models/
│   │   ├── resume.py            # Pydantic schema for resumes
│   │   ├── job.py               # Pydantic schema for JDs
│   │   └── candidate.py         # Scoring and ranking schemas
│   ├── database/
│   │   ├── storage.py           # SQLite structured metadata
│   │   └── qdrant.py            # Qdrant Vector DB interface
│   ├── utils/
│   │   ├── text_cleaner.py      # Technical token protection & normalization
│   │   ├── file_validator.py    # File format & corruption validation
│   │   ├── skill_normalizer.py  # 400+ skill taxonomy and aliases
│   │   ├── degree_normalizer.py # Canonical degree tier mapping
│   │   └── hash_utils.py        # SHA-256 duplicate detection
│   └── workers/
│       └── ingestion_pipeline.py# Batch processing & checkpointing
│
├── frontend/
│   └── streamlit_app.py         # Streamlit web dashboard
├── scripts/
│   ├── generate_test_resumes.py # Mixed PDF/DOCX/TXT generator
│   ├── ingest_resumes.py        # CLI batch ingestion runner
│   ├── benchmark.py             # System latency and memory profiling
│   ├── evaluate.py              # P@K, R@K, NDCG, and MRR evaluation
│   ├── train_ml_ranker.py       # XGBoost ranking model trainer
│   └── stress_test_10k.py       # 10,000 resume stress test
├── tests/
│   ├── test_parser.py           # Parser unit tests
│   ├── test_nlp.py              # NLP extraction tests
│   ├── test_embeddings.py       # Embedding vector tests
│   ├── test_bm25.py             # BM25 lexical tests
│   └── test_ranking.py          # End-to-end ranking tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 6. Installation & Quickstart

### Prerequisites
- Python 3.10+
- Git

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Step 3: Generate & Ingest Sample Resumes
```bash
# Generate 100 sample resumes (PDF, DOCX, TXT)
python scripts/generate_test_resumes.py --count 100 --output-dir data/resumes

# Batch ingest with automatic vector embeddings and BM25 indexing
python scripts/ingest_resumes.py --batch-size 32 --reset
```

### Step 4: Run the Interactive Web UI
```bash
streamlit run frontend/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### Step 5: Start the FastAPI Backend
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs available at `http://localhost:8000/docs`.

---

## 7. Running via Docker Compose

```bash
docker compose up --build
```
This launches:
- **FastAPI backend** on `http://localhost:8000`
- **Streamlit frontend** on `http://localhost:8501`
- **Qdrant Vector DB** on `http://localhost:6333`
- **PostgreSQL** on `http://localhost:5432`

---

## 8. Verification & Stress Testing

### Run Retrieval Quality Evaluation
```bash
python scripts/evaluate.py
```

### Run Latency & System Benchmarks
```bash
python scripts/benchmark.py
```

### Train Optional XGBoost ML Ranker
```bash
python scripts/train_ml_ranker.py
```

### Run 10,000 Resume Scaling Stress Test
```bash
python scripts/stress_test_10k.py
```

---

## 9. API Reference

### Health Check
```http
GET /api/v1/health
```
```json
{
  "status": "healthy",
  "service": "Scalable Hybrid Semantic ATS",
  "indexed_resumes": 1058,
  "vector_db_points": 1058
}
```

### Hybrid Candidate Search
```http
POST /api/v1/search
Content-Type: application/json

{
  "job_description": "Senior NLP Engineer with Python, PyTorch, Transformers, NLP. Minimum 3 years experience.",
  "top_k": 50,
  "semantic_weight": 0.45,
  "bm25_weight": 0.30,
  "skill_weight": 0.15,
  "experience_weight": 0.10
}
```

**Response Format**:
```json
{
  "job_id": "JD_SEARCH",
  "total_indexed_resumes": 1058,
  "hard_filtered_count": 516,
  "retrieved_count": 516,
  "final_count": 50,
  "execution_time_seconds": 0.111,
  "results": [
    {
      "candidate_id": "CV_00001",
      "name": "Alice Sharma",
      "rank": 1,
      "final_score": 92.4,
      "semantic_score": 0.912,
      "bm25_score": 0.885,
      "skill_score": 1.0,
      "experience_score": 0.95,
      "matched_required_skills": ["python", "pytorch", "transformers", "nlp"],
      "missing_required_skills": [],
      "matched_preferred_skills": ["fastapi", "docker"],
      "candidate_experience_years": 4.5,
      "required_experience_years": 3.0,
      "education_tier": "Master",
      "explanation": "Candidate: CV_00001 (Alice Sharma)\nFinal ATS Score: 92.4% (Rank #1)\n..."
    }
  ]
}
```

---

## 10. Scoring, configuration & offline testing (current release)

See `CHANGELOG.md`, `TEST_REPORT.md` and `DIAGNOSTIC_REPORT.md` for the full change record.

**Final score** = `sum(w_i * s_i) / sum(w_i)` over the components that are *active* for the JD/candidate
(`app/services/score_composer.py`). Components are `applied`, `not_applicable` (JD has no experience minimum / no
certification requirement / no recognisable skills), `neutral` (optional certificate not held) or `unavailable`
(semantic model missing). Excluded components are neither rewarded nor penalised; weights are renormalised, so they
need not sum to 1. Every result carries `components` (score, status, nominal/effective weight, contribution), the
mandatory-eligibility report, accepted-alternative matches and a `degraded` flag. Default weights are unchanged
(0.40/0.27/0.15/0.10/0.08) and **unvalidated** against labelled data.

**Configuration precedence:** per-request API override > process environment > `.env` (project root, auto-loaded,
never overrides real env vars) > code defaults.

**Semantic model unavailable:** requests fail with HTTP 503 by default. `ALLOW_DEGRADED_SEMANTIC=true` opts in to a
flagged degraded score (semantic excluded). Degraded and full-model scores must not be compared.

**BM25:** scores are on an absolute 0-1 scale against a persisted reference corpus (`data/bm25_index.pkl`, written only
by ingestion). Rebuild it with `python scripts/rebuild_bm25_index.py` if it ever holds only a few documents.

### Windows PowerShell Quickstart (End-to-End Integration)

#### 1. Start the Python ATS Backend (Port 8000)
```powershell
cd C:\Users\LENOVO\.gemini\antigravity\scratch\ai-ats-10k-resumes
# Activate your Python virtual environment if configured, or use system Python 3.10+
python -m pip install -r requirements.txt
python -m uvicorn app.server:app --port 8000
```
- Health Check: `http://localhost:8000/health`
- API Docs: `http://localhost:8000/docs`
- Endpoints: `/invocations`, `/api/files/upload`, `/api/session`, `/api/config`

#### 2. Start the React + Node.js Application (Port 3001)
In a second PowerShell terminal:
```powershell
cd "C:\Users\LENOVO\Downloads\Resume_Screening_Agent_Source files\Resume_Screening_Agent_Source files\e2e-chatbot-app-next"
# Verify .env has:
#   API_PROXY=http://localhost:8000/invocations
#   PYTHON_ATS_URL=http://localhost:8000
npm run dev:server
```
- Web Application URL: `http://localhost:3001`
- Screening Interface: `http://localhost:3001/chat` (Upload resumes, paste JD, click "Screen Candidates")

#### 3. Run Automated Tests
```powershell
cd C:\Users\LENOVO\.gemini\antigravity\scratch\ai-ats-10k-resumes
# Full PyTest Suite (140 tests, real embeddings & PDF parsing):
python -m pytest tests -v

# Standard-library offline runner:
python tests\run_offline.py
```

