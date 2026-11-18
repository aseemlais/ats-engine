# ATS Resume Extractor: Production Promotion & Rollout Runbook (Stage 5)

This runbook defines the operational procedures for rolling out the fine-tuned small language model (SLM) resume extractor from **Shadow Mode (Stage 3b)** into **Production Canary (Stage 5)** and **Full Production**, along with instant rollback and re-extraction procedures.

---

## 1. Architectural Guardrails & Principles

1. **Deterministic Canary Routing**:
   - Primary routing is governed by `EXTRACTION_CUTOVER_PERCENTAGE` (0 to 100).
   - Routing uses `SHA-256(file_hash) % 10000` to deterministically allocate traffic. The same resume file will **always** route to the exact same engine across retries and restarts.
2. **Instant One-Variable Rollback**:
   - Setting `EXTRACTION_CUTOVER_PERCENTAGE=0` immediately redirects 100% of live extraction traffic back to legacy heuristic rules without restarting the server or redeploying code.
3. **Permanent Zero-Downtime Fallback**:
   - If the SLM experiences an inference timeout, unparseable output, or hits the context window limit (`finish_reason="length"`), it automatically executes the deterministic legacy extractor. The response returns HTTP `200`/`202` with `extractionSource: "fallback"`. Client traffic never fails.
4. **Hardware & Memory Protection**:
   - Hugging Face Spaces free-tier limits: 2 vCPUs, 16GB RAM.
   - Exactly **1 model instance** and **1 background worker thread** process jobs sequentially via `asyncio.to_thread`.
   - The psutil memory guard (`SHADOW_MAX_RSS_MB=6144`, 6GB) automatically disables non-essential shadow comparisons if RSS crosses the safety threshold, ensuring primary candidate evaluation never starves.

---

## 2. Step-by-Step Canary Rollout Stages

```
   ┌────────────────────────────────────────────────────────┐
   │ Stage 0: 0% Cutover (Shadow Mode Active)               │
   │ EXTRACTION_CUTOVER_PERCENTAGE=0                        │
   │ 100% Legacy Primary, SLM Shadow Sampled (25%)          │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 5a: 10% Canary Cutover                           │
   │ EXTRACTION_CUTOVER_PERCENTAGE=10                       │
   │ 10% SLM Primary, 90% Legacy Primary                    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 5b: 50% Canary Cutover                           │
   │ EXTRACTION_CUTOVER_PERCENTAGE=50                       │
   │ 50% SLM Primary, 50% Legacy Primary                    │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │ Stage 5c: 100% Full Production                         │
   │ EXTRACTION_CUTOVER_PERCENTAGE=100                      │
   │ (or USE_NEW_EXTRACTOR=true)                            │
   │ 100% SLM Primary, Automatic Legacy Fallback Active     │
   └────────────────────────────────────────────────────────┘
```

### Stage 5a: 10% Canary (Observation Window: 24–48 Hours)
1. **Set Environment Variable** on the Hugging Face Space:
   ```env
   EXTRACTION_CUTOVER_PERCENTAGE=10
   ```
2. **Verify Space Health**:
   ```bash
   curl -s https://<your-space-url>/health | jq .
   ```
   Confirm output reflects:
   ```json
   {
     "status": "healthy",
     "assets_ok": true,
     "model_loaded": true,
     "cutover_percentage": 10.0,
     "queue_depth": 0,
     "rss_mb": 1980.5
   }
   ```
3. **Monitor Error Rates via SQL**:
   Run in Supabase SQL Editor:
   ```sql
   SELECT * FROM public.view_shadow_error_rates;
   ```
   - Target: `invalid_json_rate_pct < 1.0%`
   - Target: `timeout_rate_pct < 1.0%`
   - Target: `grounding_failure_rate_pct < 2.0%`

---

### Stage 5b: 50% Canary (Observation Window: 24–48 Hours)
1. **Increase Cutover Percentage**:
   ```env
   EXTRACTION_CUTOVER_PERCENTAGE=50
   ```
2. **Run Promotion Gate Audit Script**:
   ```powershell
   python scripts/check_promotion_gates.py --export-report
   ```
   Confirm all 5 gates report `🟢 PASS`.
3. **Audit Field-Level Agreement**:
   ```sql
   SELECT * FROM public.view_shadow_field_agreements;
   ```
   - Confirm candidate names and contact info maintain $\ge 99\%$ accuracy.

---

### Stage 5c: 100% Production Promotion
1. **Set 100% Cutover**:
   ```env
   EXTRACTION_CUTOVER_PERCENTAGE=100
   ```
   *(Optionally set `USE_NEW_EXTRACTOR=true`)*
2. **Confirm 100% Health Status**:
   ```bash
   curl -s https://<your-space-url>/health | jq .cutover_percentage
   # Returns: 100.0
   ```
3. **Verify Downstream Ranking Stability**:
   ```powershell
   python scripts/evaluate_ranking_impact.py --export-report
   ```

---

## 3. Emergency Instant Rollback Procedure

If any anomaly occurs (e.g. latency spike $> 30$s, elevated error rate, or unexpected ranking anomalies):

### Step 1: Trigger Immediate Rollback
On the Hugging Face Space Settings $\rightarrow$ Environment Variables:
```env
EXTRACTION_CUTOVER_PERCENTAGE=0
```
*(Ensure `USE_NEW_EXTRACTOR=false`)*

### Step 2: Verification
```bash
curl -s https://<your-space-url>/health | jq .cutover_percentage
# Returns: 0.0
```
- **Result**: 100% of incoming candidate resumes are instantly processed by the fast, deterministic legacy extractor.
- **Downtime**: 0 seconds.

---

## 4. Re-Extracting Existing Resumes with New Model Version

To upgrade older candidate profiles in `CandidateResume` to the fine-tuned model version without service disruption:

### Step 1: Dry-Run Preview
```powershell
python scripts/reextract_resumes.py --limit 20
```
*Previews extraction deltas, detected skills count, and execution time without modifying the database.*

### Step 2: Execute Re-Extraction
```powershell
python scripts/reextract_resumes.py --limit 100 --apply --delay 0.2
```
*Processes in batches of 100 with a 200ms throttle between items to prevent vCPU spikes. Updates `CandidateResume.extractedData` and marks `extractorVersion` atomically.*

---

## 5. Summary of Telemetry Views & Tools

| Tool / SQL View | Purpose |
| :--- | :--- |
| `GET /health` | Real-time container diagnostics (model loaded, RSS MB, queue depth, cutover percentage). |
| `view_shadow_error_rates` | Live monitoring of JSON invalidity, timeouts, and grounding failures. |
| `view_shadow_field_agreements` | Agreement rates for email, phone, name F1, title F1, skills Jaccard, exp delta. |
| `scripts/check_promotion_gates.py` | Automated PASS/FAIL check against the 5 production promotion gates. |
| `scripts/reextract_resumes.py` | Safe backfill utility to re-extract older resumes to new model version. |
