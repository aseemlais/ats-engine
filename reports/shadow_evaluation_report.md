# ATS Resume Extraction: Phase 4 Shadow Mode Evaluation Report
**Generated:** 2026-11-18T05:36:23.183600+00:00 | **Total Shadow Evaluations:** 1

## 1. Field-Level Agreement Rates (Legacy vs SLM)
| Field | Metric Type | Score / Rate |
| :--- | :--- | :--- |
| **Email** | Exact String Match | **100.0%** |
| **Phone** | Normalized Digits Match | **100.0%** |
| **Candidate Name** | Token F1 | **1.0** |
| **Job Title** | Token F1 | **0.8** |
| **Skills** | Jaccard Overlap | **0.8333** (Prec: 0.8333, Rec: 1.0) |
| **Total Experience** | Mean Absolute Delta | **0.0 years** |
| **Education Count** | Exact Match | **100.0%** |
| **Work Experience Count** | Exact Match | **100.0%** |

## 2. Validity-Rate Comparison (Legacy vs Fine-Tuned SLM)
| Field Check | Legacy Extractor | Fine-Tuned SLM | Delta |
| :--- | :--- | :--- | :--- |
| **Valid RFC Email** | 100.0% | 100.0% | 0.0% |
| **Valid Phone (>= 7 digits)** | 100.0% | 100.0% | 0.0% |
| **Plausible Name (Non-blank/Valid)** | 100.0% | 100.0% | 0.0% |
| **Dates Parse Successfully** | 100.0% | 100.0% | 0.0% |

## 3. Reliability, Grounding & Error Rates
| Quality Gate | Value | Target Status |
| :--- | :--- | :--- |
| **Invalid JSON Rate** | 0.0% (0 / 1) | PASS (<1%) |
| **Upstream Timeout Rate** | 0.0% (0 / 1) | PASS (<1%) |
| **Combined Error Rate** | 0.0% | PASS (<1%) |
| **Grounding Failure Rate** | 100.0% | WARN |
| **Overall Grounded Extractions** | 0.0% | Supported by Resume Text |

## 4. Latency Profile (Single Free-Tier vCPU Worker)
- **Median Latency (p50):** 1420.5 ms
- **95th Percentile (p95):** 1420.5 ms
- **Mean Latency:** 1420.5 ms
- **Range:** 1420.5 ms – 1420.5 ms

## 5. Slice Breakdown (Format, Layout & Length)
| Slice | Samples | Title F1 | Skills Jaccard | Avg Latency |
| :--- | :--- | :--- | :--- | :--- |
| PDF | 2p | 1-Col | medium (2k-5k) | 1 | 0.8 | 0.8333 | 1420.5 ms |