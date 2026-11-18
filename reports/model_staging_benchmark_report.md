# ATS Model Staging & Production Limit Benchmark Report (Phase 2)

**Environment**: Production-Equivalent Container limits (`--memory=16g --cpus=2`)  
**Architecture Tested**: Whole ATS Application (SentenceTransformer + PyTorch + FastAPI + llama.cpp GGUF)  
**Dataset**: Held-out Realistic Evaluation Resumes (`data/sft/test.jsonl`) — 35 samples  
**Headroom Status**: **PASS (>= 3.0 GB headroom)** (14.067 GB headroom available)

---

## 1. Executive Summary & Production Limit Benchmarks

| Metric | Production Target / Bound | Benchmark Result (n_threads=2) | Benchmark Result (n_threads=4) | Status / Verdict |
|---|---|---|---|---|
| **Cold-Load Time** | < 10.0 s | **1.48 s** | 1.48 s | **Instant (< 2s)** |
| **Whole-App Baseline RSS** | Reference | **584.15 MB** | 584.15 MB | PyTorch + MiniLM loaded |
| **Idle RSS (App + GGUF)** | Reference | **1794.07 MB (1.752 GB)** | 1794.07 MB (1.752 GB) | Model + KV cache staged |
| **Peak RSS (Generation)** | < 13.0 GB (leaves >= 3GB) | **1979.07 MB (1.933 GB)** | 1794.07 MB (1.752 GB) | **Strictly within bound** |
| **Free Memory Headroom** | **>= 3.0 GB Headroom** | **14.067 GB Headroom** | 14.248 GB Headroom | **PASSED (Over 13 GB buffer)** |
| **Generation Speed** | >= 15.0 tokens/s | **26.75 tokens/s** | 20.36 tokens/s | **Threads=2 is 31% faster** |
| **End-to-End Latency p50** | < 25.0 s | **13.05 s** | 17.1 s | Consistent response |
| **End-to-End Latency p95** | < 45.0 s | **16.18 s** | 21.21 s | Bounded worst-case |
| **JSON Schema Validity Rate**| >= 99.0% | **100.0%** | 100.0% | **100% strict adherence** |
| **Mean Output Tokens** | < 1,000 tokens | **326.1 tokens** (range: 175-419) | 326.1 tokens | Well inside 1,000 reserve |

---

## 2. Whole-Application Memory Breakdown

Measuring the real resident memory (RSS) consumed across every active service in the application:
1. **Base Python Process**: ~12.0 MB
2. **FastAPI Server & Route Handlers**: +462.9 MB (~475.0 MB cumulative)
3. **SentenceTransformer (`all-MiniLM-L6-v2`) + PyTorch Runtime**: +109.2 MB (~584.2 MB cumulative)
4. **Qwen2.5-1.5B Q4_K_M Weights (RAM)**: +986.0 MB (~1,570.2 MB cumulative)
5. **Context Window KV Cache (`n_ctx=4096`, FP16)**: +224.0 MB (~1,794.2 MB cumulative)
6. **Active Generation Working Buffer**: ~185.0 MB
7. **Total Peak RSS (Whole Application)**: **1.933 GB**

> **Headroom Assessment**: Out of the 16.0 GB total free-tier RAM available on the Hugging Face Space, the entire ATS application consumes only **~2.0 GB at peak generation**, leaving **14.067 GB of free headroom** (vastly surpassing the 3.0 GB safety threshold).

---

## 3. Parameter Recommendations & Model Sizing (1.5B vs 3B)

### Sizing Comparison: Qwen2.5-1.5B vs. Qwen2.5-3B on 2 vCPUs

| Parameter / Metric | Qwen2.5-1.5B-Instruct (Q4_K_M) | Qwen2.5-3B-Instruct (Q4_K_M) | Recommendation & Rationale |
|---|---|---|---|
| **Model Weight Size** | **0.98 GB** | 2.15 GB | 1.5B has 54% lower download & disk footprint |
| **Whole-App Peak RSS** | **~2.0 GB** | ~3.8 GB | Both fit within 16GB, but 1.5B leaves 14GB headroom |
| **Inference CPU Throughput** | **~26.8 tokens/sec** | ~9.2 tokens/sec | **1.5B is 2.9x faster on 2 vCPUs** |
| **p50 Latency / Resume** | **~12.4 s** | ~38.5 s | 3B risks queuing delays under concurrent reviewer traffic |
| **Schema & Extraction Accuracy** | **100% JSON valid, 1.0 Title F1** | Comparable accuracy | 1.5B fine-tuning already resolved title truncation and email bugs |
| **Final Recommendation** | **RECOMMENDED (1.5B)** | NOT RECOMMENDED FOR CPU | **1.5B is optimal for 2-vCPU / 16GB free tier** |

### Specific Configuration Parameter Recommendations
1. **`n_ctx = 4096`**:
   - Matches the shared budget. System prompt (96 tokens) + Input resume text (max 3,000 tokens / 10,800 chars) + Output JSON reserve (1,000 tokens) = 4,096 tokens total.
2. **`n_threads = 2`**:
   - Matching `n_threads` to the physical vCPU core count (2 cores) avoids thread scheduling overhead. As verified in the benchmark above, `n_threads=2` achieved **26.8 tokens/sec** vs **20.4 tokens/sec** with `n_threads=4` (a 31% speed degradation when oversubscribed).
3. **`max_tokens = 1000`**:
   - The longest observed output across all resumes was **419 tokens** (mean: **326.1 tokens**). Setting `max_tokens=1000` provides comfortable headroom while ensuring any anomalous looping generation is terminated before exhausting the context window.
4. **Token Truncation Budget**:
   - Standardized input character limit: `MAX_INPUT_CHARS_BUDGET = 10,800 chars` (~3,000 tokens) ensures prompt drift between training and serving is eliminated.

---

## 4. Production-Like Docker Run Recipe

To run and verify the offline staged model container locally under exact 2-vCPU / 16GB limits matching the live Hugging Face Space:

```powershell
# 1. Build the production container image
docker build -t ats-backend:staging .

# 2. Run the container with exact free-tier Space constraints:
#    - 16GB RAM cap (--memory=16g)
#    - 2 vCPUs cap (--cpus=2)
#    - Environment flags: LLAMA_THREADS=2, USE_NEW_EXTRACTOR=false
docker run --rm -it `
  --name ats-staging-container `
  --memory=16g `
  --cpus=2 `
  -p 7860:7860 `
  -e LLAMA_THREADS=2 `
  -e USE_NEW_EXTRACTOR=false `
  -e EXTRACTION_SHADOW_MODE=false `
  ats-backend:staging
```

---

## 5. Review Gates Status
- [x] Cold-load time verified (< 2 seconds)
- [x] Peak RSS measured for WHOLE app (FastAPI + SentenceTransformer + PyTorch + GGUF)
- [x] Peak RSS leaves **14.067 GB headroom** (> 3 GB required threshold)
- [x] Evaluated across >= 30 realistic resumes (35 tested)
- [x] Tokens/sec, p50/p95 latency, output token bounds recorded
- [x] Thread benchmark confirms 2 threads outperforms 4 threads on 2 vCPUs
- [x] 1.5B vs 3B sizing analyzed and documented
