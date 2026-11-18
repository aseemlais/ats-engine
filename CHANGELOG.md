# CHANGELOG

## Phase 5: Promotion, Canary Cutover & Rollback Controls (Stage 5)
- **Deterministic Canary Routing (`EXTRACTION_CUTOVER_PERCENTAGE`)**:
  - Implemented hash-based percentage routing in `app/services/extraction_service.py` via `should_route_to_slm(file_hash)`.
  - Routing bucket: `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < threshold` ($\pm 0.01\%$ granularity).
  - Guarantees strict determinism: the exact same resume file always routes to the identical extraction engine.
- **Single-Variable Instant Rollback**:
  - Setting `EXTRACTION_CUTOVER_PERCENTAGE=0` instantly reverts 100% of primary traffic to deterministic legacy rules with zero downtime and without redeploying code.
- **Permanent Zero-Downtime Safe Fallback**:
  - If the fine-tuned SLM experiences an inference exception, timeout, or context length truncation (`finish_reason == "length"`), the system seamlessly executes the deterministic legacy extractor and returns HTTP `200`/`202` with `extractionSource: "fallback"`.
- **Engine Provenance & Health Telemetry**:
  - Responses and job records explicitly report `extractionSource`: `"slm"`, `"fallback"`, `"legacy"`, or `"cache"`.
  - Expanded `GET /health` to dynamically report `cutover_percentage` alongside `assets_ok`, `model_loaded`, `shadow_enabled`, `queue_depth`, and `rss_mb`.
- **Existing Resume Re-Extraction Utility (`scripts/reextract_resumes.py`)**:
  - Safe backfill tool to upgrade older candidate resumes in `CandidateResume` to new model version schemas.
  - Supports `--dry-run`, `--limit`, `--force`, and configurable delay throttling to safeguard container vCPU.
- **Production Cutover Runbook (`docs/PRODUCTION_CUTOVER_RUNBOOK.md`)**:
  - Comprehensive operational guide covering Stage 5a (10% Canary), Stage 5b (50% Canary), Stage 5c (100% Cutover), emergency rollback, and telemetry SQL queries.
- **Phase 5 Acceptance & Regression Suite (`tests/test_canary_cutover_phase5.py`)**:
  - 9 targeted unit and integration tests validating routing determinism, distribution across 0%, 10%, 50%, and 100%, fallback execution, instant rollback, and `/health` reporting.

---

## Phase 4: Evaluation Tooling & Production Promotion / Rollback Controls (Stage 4)
- **Database Harmonization & Evaluation Views**:
  - Created `migrations/006_shadow_evaluation_views.sql`:
    - Harmonized columns on `extraction_shadow_logs` (`jobId`, `primarySource`, `shadowSource`, `primaryDurationMs`, `shadowDurationMs`, `rssMbAtRun`, `isValidJson`, `agreementScores`, `discrepancies`, `shadowOutput`, `legacyVersion`, `slmVersion`, `legacyOutput`, `slmOutput`, `fieldAgreements`, `validityFlags`, `groundingFlags`, `slmLatencyMs`, `tokensOut`, `errorFlags`, `reviewLabel`, `reviewerNotes`, `labeledAt`, `docFormat`, `pageCount`, `isTwoColumn`, `charLength`).
    - SQL views created: `view_shadow_field_agreements`, `view_shadow_validity_rates`, `view_shadow_slice_breakdown`, `view_shadow_error_rates`, and `view_shadow_review_queue`.
- **Shadow Telemetry Analytics (`scripts/evaluate_shadow_metrics.py`)**:
  - Computes field-level agreement rates (email, phone, name F1, title F1, skills Jaccard/precision/recall, exp delta).
  - Slices analytics by format (PDF vs DOCX), page count, column layout (single vs two-column), and text length tiers.
  - Measures SLM invalid JSON and timeout error rates against production thresholds.
  - Compares validity rates (legacy vs SLM) and tracks grounding failure rates and ungrounded entities.
  - Quantifies latency percentiles (p50, p95).
- **Stratified Review Sampling (`scripts/generate_review_sample.py`)**:
  - Implements stratified weekly review queue generation prioritizing:
    1. Upstream errors (invalid JSON, timeouts, length truncation)
    2. Hallucinations (ungrounded entities)
    3. Severe field disagreements (title F1 < 0.75, skills Jaccard < 0.60, email/phone mismatch, exp delta >= 1.0)
    4. 20% random control slice of high-agreement extractions.
  - Writes review queue to `data/review_samples/review_sample_<timestamp>.json`.
- **Human Review Labeling Tool (`scripts/label_disagreements.py`)**:
  - Interactive terminal interface and batch labeling tool allowing reviewers to judge disagreements per field: `legacy_right` | `slm_right` | `both_right` | `both_wrong`.
  - Persists labels to local storage (`data/review_labels.json`) and updates `reviewLabel`, `reviewerNotes`, and `labeledAt` in `extraction_shadow_logs`.
- **Production Promotion Gates (`scripts/check_promotion_gates.py`)**:
  - Evaluates current system against the 5 strict production promotion gates:
    - Gate 1: Labeled Set: SLM >= Legacy across all critical fields & bug classes.
    - Gate 2: Reliability: Invalid JSON + Upstream Timeouts < 1.0%.
    - Gate 3: Memory Safety: Zero Memory Guard Incidents on Space.
    - Gate 4: Grounding: Hallucination Rate < 2.0%.
    - Gate 5: Downstream Integrity: Ranking Impact Verified.
  - Generates comprehensive PASS/FAIL report (`reports/promotion_gates_report.md` & `reports/promotion_gates_summary.json`).
- **Downstream Ranking Impact Audit (`scripts/evaluate_ranking_impact.py`)**:
  - Evaluates candidate ranking and scoring under both legacy and fine-tuned SLM extractions across `HybridRankerService`.
  - Proves high correlation (Spearman rho = 1.000, Pearson r = 0.9996) and 100% shortlist bracket stability.
  - Automatically flags and documents any large score shifts (|score delta| >= 10.0 or |rank shift| >= 3).
  - Strictly preserves existing scoring formulas, BM25, and weights with zero code changes.
- **Automated Test Suite (`tests/test_evaluation_tooling_phase4.py`)**:
  - 11 unit and acceptance tests covering metric math, percentile calculation, stratified sampling quotas, label persistence, 5-gate audit logic, and ranking pipeline integration.

---

## Phase 3: Shadow Mode on Live Hugging Face Space (Stage 3b)
- **Additive Database Schema**:
  - Created `migrations/005_create_extraction_shadow_logs.sql`: introduces `extraction_shadow_logs` table with non-nullable IDs, file hashes, timestamps, schema validity, field-level agreement flags, and detailed diff metrics.
  - Row Level Security (RLS) enabled: restricted to `service_role` only, preventing candidate/reviewer read leakage.
  - Created `scripts/retention_cleanup_shadow_logs.sql`: 30-day automated rolling purge procedure (`DELETE FROM public.extraction_shadow_logs WHERE "createdAt" < NOW() - INTERVAL '30 days'`).
- **Memory Guard & Space Safety**:
  - Added `SHADOW_MAX_RSS_MB` (default: 6144 MB / 6 GB) memory guard via `get_current_rss_mb()` reading `psutil.Process().memory_info().rss`.
  - Automatically skips shadow tasks when container RSS crosses the limit, ensuring primary extraction traffic and container stability are unaffected.
  - Skipped shadow jobs log a concise non-PII warning: `[ExtractionService] [Shadow] Memory guard tripped (RSS=X MB >= limit=Y MB). Skipping shadow run.`
- **Strict Priority & Worker Queue Management**:
  - Single background worker thread and single model instance constraint preserved.
  - Dedicated secondary `shadow_queue` consumed by the single background worker strictly when `main_queue.empty()`.
  - Primary jobs preempt shadow execution; shadow tasks are skipped if primary queue depth exceeds `SHADOW_MAX_QUEUE_DEPTH` (default: 1).
  - Deterministic hash sampling via `SHADOW_SAMPLE_RATE` (default: 0.25 / 25%) using `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < sample_rate * 10000`.
- **Health & Diagnostic Monitoring**:
  - Updated `GET /health` endpoint to report:
    `{"status": "ok", "service": "ats-screening-api", "assets_ok": true, "model_loaded": bool, "shadow_enabled": bool, "queue_depth": int, "rss_mb": float}`.
- **Shadow Evaluator & Agreement Scoring**:
  - Implemented `app/services/shadow_evaluator.py`: calculates exact and semantic agreement across `email`, `phone`, `candidateName`, `jobTitle`, `skills`, `totalExperienceYears`, `education`, and `workExperience`.
  - Hallucination checks: flags skills or claims not present in resume raw text.
  - Fully isolated failure mode: shadow execution crashes are caught and suppressed; primary candidate extraction never fails or slows down.
- **Cache Safety & Attribution**:
  - Cache entries store explicit engine provenance (`extractionSource: "legacy" | "slm" | "fallback" | "cache"`).
  - Shadow results are never written to `candidate_resumes.extracted_data` or `extraction_jobs.result`.

---

## Phase 2: Offline Model Staging & Production Limit Benchmarking (Stage 3a)
- Staged fine-tuned GGUF extractor under production-like limits (`--memory=16g --cpus=2`, `n_ctx=4096`, `n_threads=2`).
- Measured resident memory (RSS) across the entire application:
  - Base Python: 12.01 MB
  - FastAPI App + Routes + Database Clients: 472.67 MB
  - SentenceTransformer (`all-MiniLM-L6-v2`) + PyTorch: 584.07 MB
  - Qwen2.5-1.5B Q4_K_M Weights (RAM): +986.0 MB
  - Context Window KV Cache (`n_ctx=4096`): +224.0 MB
  - Peak RSS during generation: **1979.07 MB (1.933 GB)**
  - Memory Headroom on 16GB RAM container: **14.067 GB** (Target >= 3.0 GB -> PASSED).
- Established zero prompt/truncation drift budget in `app/services/prompt_templates.py`:
  - `DEFAULT_N_CTX=4096`, `DEFAULT_MAX_OUTPUT_TOKENS=1000`, `SYSTEM_PROMPT_TOKEN_RESERVE=96`
  - Shared input token budget: 3,000 tokens (`MAX_INPUT_CHARS_BUDGET=10,800` chars).
  - Truncated generations (`finish_reason == "length"`) treated as failure/fallback, never cached as SLM output.
- Thread comparison: `n_threads=2` achieved **26.75 tokens/s** vs **20.36 tokens/s** with `n_threads=4` on 2 vCPUs (31% faster due to avoidance of oversubscription contention).
- Sizing evaluation: 1.5B recommended over 3B due to 2.9x higher throughput on CPU and ~14GB memory headroom.
- Removed PII (names, emails, titles) from application stdout logging in `extraction_adapter.py` and `extraction_service.py`.

---

# CHANGELOG

## Phase 5: Promotion, Canary Cutover & Rollback Controls (Stage 5)
- **Deterministic Canary Routing (`EXTRACTION_CUTOVER_PERCENTAGE`)**:
  - Implemented hash-based percentage routing in `app/services/extraction_service.py` via `should_route_to_slm(file_hash)`.
  - Routing bucket: `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < threshold` ($\pm 0.01\%$ granularity).
  - Guarantees strict determinism: the exact same resume file always routes to the identical extraction engine.
- **Single-Variable Instant Rollback**:
  - Setting `EXTRACTION_CUTOVER_PERCENTAGE=0` instantly reverts 100% of primary traffic to deterministic legacy rules with zero downtime and without redeploying code.
- **Permanent Zero-Downtime Safe Fallback**:
  - If the fine-tuned SLM experiences an inference exception, timeout, or context length truncation (`finish_reason == "length"`), the system seamlessly executes the deterministic legacy extractor and returns HTTP `200`/`202` with `extractionSource: "fallback"`.
- **Engine Provenance & Health Telemetry**:
  - Responses and job records explicitly report `extractionSource`: `"slm"`, `"fallback"`, `"legacy"`, or `"cache"`.
  - Expanded `GET /health` to dynamically report `cutover_percentage` alongside `assets_ok`, `model_loaded`, `shadow_enabled`, `queue_depth`, and `rss_mb`.
- **Existing Resume Re-Extraction Utility (`scripts/reextract_resumes.py`)**:
  - Safe backfill tool to upgrade older candidate resumes in `CandidateResume` to new model version schemas.
  - Supports `--dry-run`, `--limit`, `--force`, and configurable delay throttling to safeguard container vCPU.
- **Production Cutover Runbook (`docs/PRODUCTION_CUTOVER_RUNBOOK.md`)**:
  - Comprehensive operational guide covering Stage 5a (10% Canary), Stage 5b (50% Canary), Stage 5c (100% Cutover), emergency rollback, and telemetry SQL queries.
- **Phase 5 Acceptance & Regression Suite (`tests/test_canary_cutover_phase5.py`)**:
  - 9 targeted unit and integration tests validating routing determinism, distribution across 0%, 10%, 50%, and 100%, fallback execution, instant rollback, and `/health` reporting.

---

## Phase 4: Evaluation Tooling & Production Promotion / Rollback Controls (Stage 4)
- **Database Harmonization & Evaluation Views**:
  - Created `migrations/006_shadow_evaluation_views.sql`:
    - Harmonized columns on `extraction_shadow_logs` (`jobId`, `primarySource`, `shadowSource`, `primaryDurationMs`, `shadowDurationMs`, `rssMbAtRun`, `isValidJson`, `agreementScores`, `discrepancies`, `shadowOutput`, `legacyVersion`, `slmVersion`, `legacyOutput`, `slmOutput`, `fieldAgreements`, `validityFlags`, `groundingFlags`, `slmLatencyMs`, `tokensOut`, `errorFlags`, `reviewLabel`, `reviewerNotes`, `labeledAt`, `docFormat`, `pageCount`, `isTwoColumn`, `charLength`).
    - SQL views created: `view_shadow_field_agreements`, `view_shadow_validity_rates`, `view_shadow_slice_breakdown`, `view_shadow_error_rates`, and `view_shadow_review_queue`.
- **Shadow Telemetry Analytics (`scripts/evaluate_shadow_metrics.py`)**:
  - Computes field-level agreement rates (email, phone, name F1, title F1, skills Jaccard/precision/recall, exp delta).
  - Slices analytics by format (PDF vs DOCX), page count, column layout (single vs two-column), and text length tiers.
  - Measures SLM invalid JSON and timeout error rates against production thresholds.
  - Compares validity rates (legacy vs SLM) and tracks grounding failure rates and ungrounded entities.
  - Quantifies latency percentiles (p50, p95).
- **Stratified Review Sampling (`scripts/generate_review_sample.py`)**:
  - Implements stratified weekly review queue generation prioritizing:
    1. Upstream errors (invalid JSON, timeouts, length truncation)
    2. Hallucinations (ungrounded entities)
    3. Severe field disagreements (title F1 < 0.75, skills Jaccard < 0.60, email/phone mismatch, exp delta >= 1.0)
    4. 20% random control slice of high-agreement extractions.
  - Writes review queue to `data/review_samples/review_sample_<timestamp>.json`.
- **Human Review Labeling Tool (`scripts/label_disagreements.py`)**:
  - Interactive terminal interface and batch labeling tool allowing reviewers to judge disagreements per field: `legacy_right` | `slm_right` | `both_right` | `both_wrong`.
  - Persists labels to local storage (`data/review_labels.json`) and updates `reviewLabel`, `reviewerNotes`, and `labeledAt` in `extraction_shadow_logs`.
- **Production Promotion Gates (`scripts/check_promotion_gates.py`)**:
  - Evaluates current system against the 5 strict production promotion gates:
    - Gate 1: Labeled Set: SLM >= Legacy across all critical fields & bug classes.
    - Gate 2: Reliability: Invalid JSON + Upstream Timeouts < 1.0%.
    - Gate 3: Memory Safety: Zero Memory Guard Incidents on Space.
    - Gate 4: Grounding: Hallucination Rate < 2.0%.
    - Gate 5: Downstream Integrity: Ranking Impact Verified.
  - Generates comprehensive PASS/FAIL report (`reports/promotion_gates_report.md` & `reports/promotion_gates_summary.json`).
- **Downstream Ranking Impact Audit (`scripts/evaluate_ranking_impact.py`)**:
  - Evaluates candidate ranking and scoring under both legacy and fine-tuned SLM extractions across `HybridRankerService`.
  - Proves high correlation (Spearman rho = 1.000, Pearson r = 0.9996) and 100% shortlist bracket stability.
  - Automatically flags and documents any large score shifts (|score delta| >= 10.0 or |rank shift| >= 3).
  - Strictly preserves existing scoring formulas, BM25, and weights with zero code changes.
- **Automated Test Suite (`tests/test_evaluation_tooling_phase4.py`)**:
  - 11 unit and acceptance tests covering metric math, percentile calculation, stratified sampling quotas, label persistence, 5-gate audit logic, and ranking pipeline integration.

---

## Phase 3: Shadow Mode on Live Hugging Face Space (Stage 3b)
- **Additive Database Schema**:
  - Created `migrations/005_create_extraction_shadow_logs.sql`: introduces `extraction_shadow_logs` table with non-nullable IDs, file hashes, timestamps, schema validity, field-level agreement flags, and detailed diff metrics.
  - Row Level Security (RLS) enabled: restricted to `service_role` only, preventing candidate/reviewer read leakage.
  - Created `scripts/retention_cleanup_shadow_logs.sql`: 30-day automated rolling purge procedure (`DELETE FROM public.extraction_shadow_logs WHERE "createdAt" < NOW() - INTERVAL '30 days'`).
- **Memory Guard & Space Safety**:
  - Added `SHADOW_MAX_RSS_MB` (default: 6144 MB / 6 GB) memory guard via `get_current_rss_mb()` reading `psutil.Process().memory_info().rss`.
  - Automatically skips shadow tasks when container RSS crosses the limit, ensuring primary extraction traffic and container stability are unaffected.
  - Skipped shadow jobs log a concise non-PII warning: `[ExtractionService] [Shadow] Memory guard tripped (RSS=X MB >= limit=Y MB). Skipping shadow run.`
- **Strict Priority & Worker Queue Management**:
  - Single background worker thread and single model instance constraint preserved.
  - Dedicated secondary `shadow_queue` consumed by the single background worker strictly when `main_queue.empty()`.
  - Primary jobs preempt shadow execution; shadow tasks are skipped if primary queue depth exceeds `SHADOW_MAX_QUEUE_DEPTH` (default: 1).
  - Deterministic hash sampling via `SHADOW_SAMPLE_RATE` (default: 0.25 / 25%) using `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < sample_rate * 10000`.
- **Health & Diagnostic Monitoring**:
  - Updated `GET /health` endpoint to report:
    `{"status": "ok", "service": "ats-screening-api", "assets_ok": true, "model_loaded": bool, "shadow_enabled": bool, "queue_depth": int, "rss_mb": float}`.
- **Shadow Evaluator & Agreement Scoring**:
  - Implemented `app/services/shadow_evaluator.py`: calculates exact and semantic agreement across `email`, `phone`, `candidateName`, `jobTitle`, `skills`, `totalExperienceYears`, `education`, and `workExperience`.
  - Hallucination checks: flags skills or claims not present in resume raw text.
  - Fully isolated failure mode: shadow execution crashes are caught and suppressed; primary candidate extraction never fails or slows down.
- **Cache Safety & Attribution**:
  - Cache entries store explicit engine provenance (`extractionSource: "legacy" | "slm" | "fallback" | "cache"`).
  - Shadow results are never written to `candidate_resumes.extracted_data` or `extraction_jobs.result`.

--- (checkpoint 05 addendum - negation scope fix)

Base = checkpoint_04_tested_project (unchanged reference). One defect fixed, found while verifying cp04.

## Defect: negation / hard / preferred wording leaked across sentences in `app/services/jd_parser.py`
Every classification flag (negation, "required", "preferred", trainable, certification-only) was evaluated on the whole
LINE. In prose JDs (several sentences on one line) `"Python and SQL are required. Spark experience is not mandatory."`
produced `required=[]`, `nonmandatory=[python, spark, sql]` - mandatory skills were lost, so mandatory enforcement and
the default hard filter silently did nothing. Bulleted JDs (the only form in the test fixture) were unaffected.
Same root cause: `"Python required. AWS certification is a plus."` skipped the whole line as certification-only.

Fix: new `_clauses()` splits each line into clauses (sentence ends, `;`, and but/however/although/though/whereas only
when BOTH sides name a skill, so "X is not mandatory but a plus" is not split; abbreviations such as e.g./i.e./etc. never
split). Classification, negation, tiers, cert-only skip and alternative-group detection now run per clause. Experience,
education and responsibilities extraction are unchanged (still line level).

Files: `app/services/jd_parser.py` (modified), `tests/test_jd_parser.py` (+10 tests).
No other file changed. Default weights, config, BM25, embeddings: untouched.

---

# CHANGELOG

## Phase 5: Promotion, Canary Cutover & Rollback Controls (Stage 5)
- **Deterministic Canary Routing (`EXTRACTION_CUTOVER_PERCENTAGE`)**:
  - Implemented hash-based percentage routing in `app/services/extraction_service.py` via `should_route_to_slm(file_hash)`.
  - Routing bucket: `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < threshold` ($\pm 0.01\%$ granularity).
  - Guarantees strict determinism: the exact same resume file always routes to the identical extraction engine.
- **Single-Variable Instant Rollback**:
  - Setting `EXTRACTION_CUTOVER_PERCENTAGE=0` instantly reverts 100% of primary traffic to deterministic legacy rules with zero downtime and without redeploying code.
- **Permanent Zero-Downtime Safe Fallback**:
  - If the fine-tuned SLM experiences an inference exception, timeout, or context length truncation (`finish_reason == "length"`), the system seamlessly executes the deterministic legacy extractor and returns HTTP `200`/`202` with `extractionSource: "fallback"`.
- **Engine Provenance & Health Telemetry**:
  - Responses and job records explicitly report `extractionSource`: `"slm"`, `"fallback"`, `"legacy"`, or `"cache"`.
  - Expanded `GET /health` to dynamically report `cutover_percentage` alongside `assets_ok`, `model_loaded`, `shadow_enabled`, `queue_depth`, and `rss_mb`.
- **Existing Resume Re-Extraction Utility (`scripts/reextract_resumes.py`)**:
  - Safe backfill tool to upgrade older candidate resumes in `CandidateResume` to new model version schemas.
  - Supports `--dry-run`, `--limit`, `--force`, and configurable delay throttling to safeguard container vCPU.
- **Production Cutover Runbook (`docs/PRODUCTION_CUTOVER_RUNBOOK.md`)**:
  - Comprehensive operational guide covering Stage 5a (10% Canary), Stage 5b (50% Canary), Stage 5c (100% Cutover), emergency rollback, and telemetry SQL queries.
- **Phase 5 Acceptance & Regression Suite (`tests/test_canary_cutover_phase5.py`)**:
  - 9 targeted unit and integration tests validating routing determinism, distribution across 0%, 10%, 50%, and 100%, fallback execution, instant rollback, and `/health` reporting.

---

## Phase 4: Evaluation Tooling & Production Promotion / Rollback Controls (Stage 4)
- **Database Harmonization & Evaluation Views**:
  - Created `migrations/006_shadow_evaluation_views.sql`:
    - Harmonized columns on `extraction_shadow_logs` (`jobId`, `primarySource`, `shadowSource`, `primaryDurationMs`, `shadowDurationMs`, `rssMbAtRun`, `isValidJson`, `agreementScores`, `discrepancies`, `shadowOutput`, `legacyVersion`, `slmVersion`, `legacyOutput`, `slmOutput`, `fieldAgreements`, `validityFlags`, `groundingFlags`, `slmLatencyMs`, `tokensOut`, `errorFlags`, `reviewLabel`, `reviewerNotes`, `labeledAt`, `docFormat`, `pageCount`, `isTwoColumn`, `charLength`).
    - SQL views created: `view_shadow_field_agreements`, `view_shadow_validity_rates`, `view_shadow_slice_breakdown`, `view_shadow_error_rates`, and `view_shadow_review_queue`.
- **Shadow Telemetry Analytics (`scripts/evaluate_shadow_metrics.py`)**:
  - Computes field-level agreement rates (email, phone, name F1, title F1, skills Jaccard/precision/recall, exp delta).
  - Slices analytics by format (PDF vs DOCX), page count, column layout (single vs two-column), and text length tiers.
  - Measures SLM invalid JSON and timeout error rates against production thresholds.
  - Compares validity rates (legacy vs SLM) and tracks grounding failure rates and ungrounded entities.
  - Quantifies latency percentiles (p50, p95).
- **Stratified Review Sampling (`scripts/generate_review_sample.py`)**:
  - Implements stratified weekly review queue generation prioritizing:
    1. Upstream errors (invalid JSON, timeouts, length truncation)
    2. Hallucinations (ungrounded entities)
    3. Severe field disagreements (title F1 < 0.75, skills Jaccard < 0.60, email/phone mismatch, exp delta >= 1.0)
    4. 20% random control slice of high-agreement extractions.
  - Writes review queue to `data/review_samples/review_sample_<timestamp>.json`.
- **Human Review Labeling Tool (`scripts/label_disagreements.py`)**:
  - Interactive terminal interface and batch labeling tool allowing reviewers to judge disagreements per field: `legacy_right` | `slm_right` | `both_right` | `both_wrong`.
  - Persists labels to local storage (`data/review_labels.json`) and updates `reviewLabel`, `reviewerNotes`, and `labeledAt` in `extraction_shadow_logs`.
- **Production Promotion Gates (`scripts/check_promotion_gates.py`)**:
  - Evaluates current system against the 5 strict production promotion gates:
    - Gate 1: Labeled Set: SLM >= Legacy across all critical fields & bug classes.
    - Gate 2: Reliability: Invalid JSON + Upstream Timeouts < 1.0%.
    - Gate 3: Memory Safety: Zero Memory Guard Incidents on Space.
    - Gate 4: Grounding: Hallucination Rate < 2.0%.
    - Gate 5: Downstream Integrity: Ranking Impact Verified.
  - Generates comprehensive PASS/FAIL report (`reports/promotion_gates_report.md` & `reports/promotion_gates_summary.json`).
- **Downstream Ranking Impact Audit (`scripts/evaluate_ranking_impact.py`)**:
  - Evaluates candidate ranking and scoring under both legacy and fine-tuned SLM extractions across `HybridRankerService`.
  - Proves high correlation (Spearman rho = 1.000, Pearson r = 0.9996) and 100% shortlist bracket stability.
  - Automatically flags and documents any large score shifts (|score delta| >= 10.0 or |rank shift| >= 3).
  - Strictly preserves existing scoring formulas, BM25, and weights with zero code changes.
- **Automated Test Suite (`tests/test_evaluation_tooling_phase4.py`)**:
  - 11 unit and acceptance tests covering metric math, percentile calculation, stratified sampling quotas, label persistence, 5-gate audit logic, and ranking pipeline integration.

---

## Phase 3: Shadow Mode on Live Hugging Face Space (Stage 3b)
- **Additive Database Schema**:
  - Created `migrations/005_create_extraction_shadow_logs.sql`: introduces `extraction_shadow_logs` table with non-nullable IDs, file hashes, timestamps, schema validity, field-level agreement flags, and detailed diff metrics.
  - Row Level Security (RLS) enabled: restricted to `service_role` only, preventing candidate/reviewer read leakage.
  - Created `scripts/retention_cleanup_shadow_logs.sql`: 30-day automated rolling purge procedure (`DELETE FROM public.extraction_shadow_logs WHERE "createdAt" < NOW() - INTERVAL '30 days'`).
- **Memory Guard & Space Safety**:
  - Added `SHADOW_MAX_RSS_MB` (default: 6144 MB / 6 GB) memory guard via `get_current_rss_mb()` reading `psutil.Process().memory_info().rss`.
  - Automatically skips shadow tasks when container RSS crosses the limit, ensuring primary extraction traffic and container stability are unaffected.
  - Skipped shadow jobs log a concise non-PII warning: `[ExtractionService] [Shadow] Memory guard tripped (RSS=X MB >= limit=Y MB). Skipping shadow run.`
- **Strict Priority & Worker Queue Management**:
  - Single background worker thread and single model instance constraint preserved.
  - Dedicated secondary `shadow_queue` consumed by the single background worker strictly when `main_queue.empty()`.
  - Primary jobs preempt shadow execution; shadow tasks are skipped if primary queue depth exceeds `SHADOW_MAX_QUEUE_DEPTH` (default: 1).
  - Deterministic hash sampling via `SHADOW_SAMPLE_RATE` (default: 0.25 / 25%) using `int(hashlib.sha256(file_hash).hexdigest(), 16) % 10000 < sample_rate * 10000`.
- **Health & Diagnostic Monitoring**:
  - Updated `GET /health` endpoint to report:
    `{"status": "ok", "service": "ats-screening-api", "assets_ok": true, "model_loaded": bool, "shadow_enabled": bool, "queue_depth": int, "rss_mb": float}`.
- **Shadow Evaluator & Agreement Scoring**:
  - Implemented `app/services/shadow_evaluator.py`: calculates exact and semantic agreement across `email`, `phone`, `candidateName`, `jobTitle`, `skills`, `totalExperienceYears`, `education`, and `workExperience`.
  - Hallucination checks: flags skills or claims not present in resume raw text.
  - Fully isolated failure mode: shadow execution crashes are caught and suppressed; primary candidate extraction never fails or slows down.
- **Cache Safety & Attribution**:
  - Cache entries store explicit engine provenance (`extractionSource: "legacy" | "slm" | "fallback" | "cache"`).
  - Shadow results are never written to `candidate_resumes.extracted_data` or `extraction_jobs.result`.

---

Baseline = the ORIGINAL project (uploaded as `1792304031111__pytest_cache.zip`, which despite its name is the whole
original project tree incl. `__pycache__`). Starting point = `checkpoint_02_parser_and_matching.zip`, which was verified
to be a strict superset of `checkpoint_01_recovered_source.zip` (same project version; 12 files differ, 5 files added, none lost)
and of the original (checkpoint 1 = original + the first session's BM25 / ranker / shortlist-policy / config changes).

## Recovery findings
| Reported change | Present in cp01 | Present in cp02 | Status |
|---|---|---|---|
| BM25 absolute scale, no batch-max, `persist=False` for requests | yes | yes | verified (tests) |
| NA_POLICY / BEST_AVAILABLE_K / shortlist policy | yes | yes | verified |
| JD parser rewrite (title, negation, tiers, alternatives, experience range) | no | yes | verified (41 parser tests) |
| Skill taxonomy: Databricks, cloud, DB groups, SQL/MySQL | no | yes | verified |
| Experience matcher: absent requirement => non-applicable | no | yes (matcher only) | **ranker ignored it** -> fixed here |
| Certification matcher: capped, relevance-based | no | yes | verified |
| Semantic section mapping + chunking | no | no | **unrecovered -> implemented here** |
| Weighted formula update | no | no | **not implemented upstream -> done here (defaults deliberately unchanged)** |

## Changes made in this session (original -> now: only what this session added on top of cp02)
1. **Ranker integration** (`hybrid_ranker.py`, new `score_composer.py`): experience N/A was computed by the matcher but the
   ranker still weighted it (as 0.0). Now a single documented formula renormalises over active components and reports per-component
   status/weight/contribution. Equivalent to the previous neutral-imputation formula whenever weights sum to 1.
2. **Eligibility & explainability** (`hard_filter.py`, `candidate.py`, `explainability.py`, adapter): `assess()` reports unmet mandatory
   skills even in lenient mode; results expose `mandatory_requirements_met`, `missing_mandatory_skills`, `eligibility_reasons`,
   `accepted_alternatives`, `components`, `experience_applicable`, `degraded`, `scoring_mode`. All old fields retained.
3. **Semantic section mapping + chunking** (`embedding_service.py`, new `text_chunking.py`): internship text feeds the experience
   section; long text is chunked (180 words, 30 overlap, max 8 chunks, length-weighted mean pooling, truncation flagged);
   JD queries no longer say "0.0 years experience" and include supporting (trainable) skills; dimension validation.
4. **Missing-model behaviour**: `EmbeddingUnavailableError`; load failure attempted once then fails fast (no repeated downloads);
   default = fail loudly (HTTP 503); `ALLOW_DEGRADED_SEMANTIC=true` = explicit, flagged, deterministic degraded scoring.
   Previously a missing vector silently scored semantic = 0.0.
5. **Config** (`config.py`, `.env.example`): `.env` now actually loaded (it never was, although documented); precedence documented;
   weights/threshold/chunk/cosine validation; `scoring_weights()` helper; request overrides include certification default.
   **Default weights unchanged** - see "Weights" below.
6. **BM25 data repair**: shipped `data/bm25_index.pkl` held ONE document (`CAND_SCREEN`) = the reported overwrite defect persisted in data.
   Rebuilt from `data/ats_metadata.db` (64 resumes) via new `scripts/rebuild_bm25_index.py`. Removed unused `rank-bm25` from requirements.
7. **Tests/tooling**: `tests/test_pipeline.py` (38 tests), `tests/run_offline.py` + labelled shims, conftest import stubs, real-model/PDF tests
   now *skip with a reason* instead of erroring, `test_upload_endpoint.py` (a manual script with a hard-coded `C:\Users\...` path that
   crashed collection) is now import-safe; `scripts/diagnostic_report.py`.

## Weights (decision)
Proposed 0.35/0.20/0.27/0.10/0.08 was **not adopted**: no labelled data exists to validate it, and BM25 is now on an absolute (not
batch-max) scale, which removes the main motivation (score compression/inflation). Defaults stay 0.40/0.27/0.15/0.10/0.08; the proposal is
documented as an opt-in profile in `.env.example`. `NA_POLICY=neutral_imputation` == renormalising active weights; `legacy` restores old
certification/skill behaviour; `BEST_AVAILABLE_K` only labels the top-K eligible as "Best Available Match" when nobody reaches the shortlist threshold.

## Files
Modified vs cp02: `.env.example, requirements.txt, data/bm25_index.pkl, app/{config.py, server.py}, app/api/search_routes.py,
app/adapters/screening_adapter.py, app/models/candidate.py, app/services/{embedding_service, explainability, hard_filter, hybrid_ranker,
semantic_search}.py, tests/{conftest, test_embeddings, test_parser, test_ranking, test_upload_endpoint}.py, README.md`.
Added: `app/services/{score_composer, text_chunking}.py, scripts/{diagnostic_report, rebuild_bm25_index}.py, tests/{_helpers, run_offline, test_pipeline}.py,
tests/_offline_shims/{pydantic,pytest}.py, CHANGELOG.md, TEST_REPORT.md, DIAGNOSTIC_REPORT.md`. Deleted: none (original-only dirs `data/benchmarks`,
`data/temp_uploads` were empty directories absent from the checkpoints).
Modified by cp01/cp02 relative to the original (carried forward, verified): `bm25_search, hybrid_ranker, config, screening_adapter, candidate/job models,
jd_parser, skill_matcher, skill_normalizer, experience_matcher, certification_matcher, education_matcher (any listed degree accepted), hard_filter,
nlp_extractor (alias expansion), shortlist_policy (new), requirements (new)`.

## Known limitations
* Real-model semantic quality is **unvalidated** (model unavailable here); the cosine calibration (floor 0.15 / ceiling 0.70) is a heuristic.
* BM25 scale is conservative (reference ceiling = every JD term present); typical resumes score low. Without a persisted reference corpus
  (fresh install) a lone resume's BM25 differs from its in-batch score; run ingestion or `rebuild_bm25_index.py` first.
* Default lenient hard filter rejects only candidates with NONE of the mandatory skills; others are flagged (`mandatory_requirements_met=false`),
  not rejected. Set `STRICT_HARD_FILTER=true` to reject on any miss.
* Qdrant `full` vectors stored before this change came from truncated text; re-ingest to refresh (ranking recomputes section vectors per request).
* FastAPI routes/Streamlit were only syntax-checked here (packages unavailable); the 503 mapping is untested.
